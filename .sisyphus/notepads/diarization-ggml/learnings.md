# Learnings — diarization-ggml

## Conventions
- C++17 standard for diarization-ggml (matches embedding-ggml)
- All PLDA/VBx math MUST use `double` (float64) to match numpy
- Follow whisper.cpp model/state separation pattern
- Shared ggml submodule at ../ggml

## Validated Parameters
- Segmentation: 10s chunks, 1s step, 589 frames/chunk, 7 powerset classes
- Embedding: 256-dim, 16kHz, cosine metric, min 400 samples
- VBx: threshold=0.6, Fa=0.07, Fb=0.8, maxIters=20
- PLDA: lda_dim=128
- Pipeline: segmentation.min_duration_off=0.0
- Warm-up: (0.0, 0.0) — no warm-up for this model
- Frame step: 0.016875s (270 samples at 16kHz)

## Build System Refactoring (Task: extract library targets)
- `segmentation-core` STATIC library: model.cpp, sincnet.cpp, lstm.cpp
  - PUBLIC includes: src/, ../ggml/include
  - PUBLIC links: ggml, Accelerate framework
  - PUBLIC defs: ACCELERATE_NEW_LAPACK, GGML_USE_METAL (when enabled)
- `embedding-core` STATIC library: model.cpp, fbank.cpp
  - PUBLIC includes: src/, kaldi-native-fbank/, ../ggml/include
  - PUBLIC links: ggml, kaldi-native-fbank-core, Accelerate framework
  - PUBLIC defs: ACCELERATE_NEW_LAPACK
- `embedding-coreml` remains separate — optional CoreML bridge, linked by exe only
- Use PUBLIC on library link/include/defs so consumers (like diarization-ggml) inherit them transitively
- Replaced file(GLOB SOURCES) with explicit file lists for library targets
- Executables link ONLY against their respective -core library (single dependency)
- Both libraries produce .a archives in build root (libsegmentation-core.a, libembedding-core.a)

## PLDA Converter (convert_plda.py)
- Binary format: PLDA magic (4B) + version uint32 (4B) + 6 float64 arrays = 398,344 bytes total
- Mixed precision in source .npz: mean1=float64, mean2=float32, lda=float32, plda arrays=float64
- All converted to float64 for output consistency
- Eigendecomposition (eigh) pre-computed in Python — exact 0.0 error vs vbx_setup reference
- Validation .npz includes test embeddings + expected xvec/plda transform outputs (seed=42)
- glob.glob() used to resolve wildcard paths from HuggingFace cache
- Post-eigendecomposition plda_tr and plda_psi are reversed ([::-1]) — critical for C++ consumer

## Project Scaffold (Task 3: CMakeLists.txt + stubs)
- `if(NOT TARGET ggml)` guards added to segmentation-ggml, embedding-ggml, AND diarization-ggml CMakeLists.txt
- `if(NOT TARGET kaldi-native-fbank-core)` guard added to embedding-ggml AND diarization-ggml CMakeLists.txt
- Guards prevent "add_subdirectory called multiple times" when diarization-ggml includes both sub-projects
- diarization-ggml links PRIVATE against segmentation-core and embedding-core (not PUBLIC — it's an executable)
- Accelerate framework linked separately for PLDA/VBx linear algebra (not inherited from sub-projects since they use PUBLIC but diarization links PRIVATE)
- Headers in include/ use `namespace diarization {}` except diarization.h (top-level structs)
- All source stubs compile with empty function bodies; bool stubs return false
- CLI uses simple argc/argv parsing (matches segmentation-ggml pattern, no getopt)
- `--help` returns exit code 0; missing args returns exit code 1

## Aggregation (Task 6: overlap-add + speaker count + to_diarization)
- `closest_frame(t, sw)` = `lround((t - sw.start - 0.5*sw.duration) / sw.step)` — matches pyannote.core.SlidingWindow.closest_frame exactly
- CRITICAL: Python constructs frames SlidingWindow with `start=chunks.start` (not frame_window.start). Must use `chunk_window.start` as frames.start in C++.
- The 0.5*duration offset in closest_frame cancels with the 0.5*frames.duration added to arguments in inference.py:596, but both must be present for correctness
- `std::lround` matches `np.rint` for non-half values; in practice chunk_start/frame_step never lands exactly on 0.5
- aggregate_chunks: NaN handling via `std::isnan()` check per element, clean value used for accumulation
- compute_speaker_count: sum across speakers → aggregate(num_classes=1) → lround to int
- to_diarization: aggregate(skip_average=true) → argsort descending → select top-count speakers per frame
- No header changes needed — existing signatures sufficient, closest_frame is file-static helper

## Hungarian Algorithm + Constrained Assignment (Task 7)
- Hungarian algorithm adapted from scipy's `rectangular_lsap.cpp` (BSD-3-Clause)
- Based on: Crouse 2016 "On implementing 2D rectangular assignment algorithms" (shortest augmenting path)
- Handles rectangular matrices natively (transposes if nc < nr)
- maximize mode: negate cost matrix then minimize
- `assign_embeddings()` signature changed from original stub: added `soft_clusters` output vector and renamed `centroids` from `std::vector<float>&` output
- `constrained_argmax()` added as new public function: applies Hungarian per chunk to soft_clusters
- `cosine_distance()` added as public function: double precision, handles zero-norm gracefully
- NaN replacement in constrained_argmax uses global minimum of soft_clusters (matches `np.nanmin`)
- soft_clusters = 2.0 - cosine_distance (similarity measure, range [0, 2])

## Embedding Extraction (Task 8: extract_embeddings)
- `extract_embeddings()` function in global namespace (not `namespace diarization`) matching diarization.h convention
- Audio cropping: chunk_start = chunk_index * 16000 (1s step), chunk_length = 160000 (10s at 16kHz), zero-padded beyond audio end
- Fbank computed once per chunk via `embedding::compute_fbank()` → ~998 frames for 10s (snip_edges=true, 10ms shift, 25ms window)
- Frame mapping: fbank (~998 frames) → seg (589 frames) via `seg_frame = (long long)ft * 589 / num_fbank_frames` — integer arithmetic avoids float rounding
- Masking: zero out entire fbank frame (80 bins) where speaker is inactive. CMN already applied by compute_fbank; no re-computation after masking (approximation since CoreML/GGML TSTP pooling doesn't receive separate mask)
- NaN handling: all-zero mask → fill 256-dim embedding with `std::nanf("")`
- `#ifdef EMBEDDING_USE_COREML` selects between CoreML (`embedding_coreml_encode`) and GGML (`embedding::model_infer`) inference
- CoreML header at `coreml_bridge.h` (include path from `embedding-coreml` target's PUBLIC include of `src/coreml/`)
- Forward declarations of `embedding_coreml_context`, `embedding::embedding_model`, `embedding::embedding_state` in diarization.h to avoid pulling in heavy headers
- Binarized segmentation layout: [num_chunks, num_frames_per_chunk, num_speakers] row-major float
- Output embeddings layout: [num_chunks, num_speakers, 256] row-major float

## Filter Embeddings + AHC Clustering (Task 9)
- filter_embeddings: single_active_mask = (frame_sum == 1.0f) — exact float comparison safe since segmentations are binarized to 0.0/1.0
- filter_embeddings: num_clean[s] accumulated in single pass over frames per chunk; stack-allocated float[16] avoids heap allocation
- filter_embeddings: active_threshold = min_active_ratio * num_frames (0.2 * 589 = 117.8)
- ahc_cluster: centroid linkage implemented directly — no external library (fastcluster etc.)
- ahc_cluster: node numbering: 0..n-1 = original points, n..2n-2 = merged clusters (scipy convention)
- ahc_cluster: new centroid = (size_i * cent_i + size_j * cent_j) / (size_i + size_j) — weighted average
- ahc_cluster: fcluster uses top-down DFS (not union-find) to correctly handle centroid linkage inversions (non-monotonic dendrogram)
- ahc_cluster: fcluster DFS: propagate same label if merge_dist <= threshold, assign new labels if > threshold
- ahc_cluster: contiguous renumbering via sort + lower_bound (matches np.unique return_inverse)
- No header changes needed — clustering.h signatures already matched the stubs

## VBx Clustering Core Loop (Task 10)
- VBx is a Variational Bayes x-vector clustering algorithm — iterative Bayesian inference refining speaker responsibilities
- Port from vbx.py:27-155 (cluster_vbx + VBx functions)
- Initialization: one-hot from AHC clusters → scale by init_smoothing=7.0 → row-wise softmax
- Pi initialized as uniform: 1/S for each speaker
- Pre-loop constants: G (T,) per-frame log-likelihood constant, V (D,) = sqrt(Phi), rho (T,D) = X ⊙ V
- Core loop equations from Landini et al. paper:
  - Eq. (17): invL (S,D) = 1/(1 + Fa/Fb * Σγ * Φ) — per-speaker inverse Lambda
  - Eq. (16): α (S,D) = Fa/Fb * invL ⊙ (γᵀ @ ρ) — speaker models
  - Eq. (23): log_p (T,S) = Fa * (ρ@αᵀ - 0.5*(invL+α²)·Φ + G) — log-likelihoods
  - Eq. (25): ELBO = Σlog p(x) + Fb/2 * Σ(log(invL) - invL - α² + 1) — convergence criterion
- CBLAS operations used:
  - cblas_dgemm CblasTrans/NoTrans for γᵀ@ρ (S,D) and ρ@αᵀ (T,S)
  - cblas_dgemv for (invL+α²)@Φ: (S,D)@(D,)→(S,)
- Numerically stable logsumexp and softmax implemented as static helpers
- Convergence: ELBO improvement < 1e-4 or maxIters=20
- No header changes needed — VBxResult struct and vbx_cluster signature already correct
- LSP (clangd) shows false errors due to missing compile_commands.json include paths; cmake build succeeds cleanly

## Full Pipeline Orchestration (Task 11: diarize() + RTTM output)
- Both `segmentation-ggml/src/model.h` and `embedding-ggml/src/model.h` are exported as PUBLIC includes — `#include "model.h"` finds only the first in search path order
- Resolved by using explicit relative paths: `#include "../../segmentation-ggml/src/model.h"` and `#include "../../embedding-ggml/src/model.h"` — both have different include guards so they coexist
- WAV loading copied from embedding-ggml/src/main.cpp (static function, not shared library code)
- Pipeline frees models eagerly: segmentation freed after all chunks processed, embedding freed after extraction, PLDA is stack-allocated vectors (auto-cleaned)
- Error handling uses goto labels: `cleanup` (frees both models + seg), `cleanup_emb` (frees emb only, used for no-speaker early exit returning true)
- VBx failure path (`return false`) is safe because both models are already freed by that point
- Python VBxClustering computes centroids from soft VBx gamma assignments (not hard AHC clusters) — differs from BaseClustering.assign_embeddings() which uses hard mean
- Centroid formula: `W = gamma[:, pi > 1e-7]`, `centroids = W.T @ filtered_emb / W.sum(0).T`
- Did NOT use `diarization::assign_embeddings()` for VBx path because it computes centroids from hard assignments; instead computed centroids + cosine distance + constrained_argmax manually
- Reconstruct step: clustered_seg initialized with NaN; for each (chunk, cluster), take max of binarized segmentation across local speakers assigned to that cluster
- RTTM conversion: contiguous runs of 1.0 in discrete_diarization → segments with start = frame_idx * FRAME_STEP, duration = num_frames * FRAME_STEP
- RTTM format: `SPEAKER <uri> 1 <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>` with .3f precision
- URI extracted from audio filename without extension
- num_chunks formula: `max(1, 1 + ceil((audio_duration - 10.0) / 1.0))` — ceil variant processes slightly more audio than pyannote's floor variant but results are equivalent
- Memory optimization: seg_logits freed after powerset conversion, audio_samples freed after embedding extraction

## Test Suite (Task 12: compare_pipeline.py + compare_rttm.py)
- `compare_rttm.py`: standalone RTTM→DER comparison using `pyannote.metrics.diarization.DiarizationErrorRate`
- `compare_pipeline.py`: orchestrates Python pipeline (via hook) and C++ binary, compares outputs per stage
- Python pipeline hook signature: `hook(step_name, step_artifact, **kwargs)` — `step_artifact` may have `.data` (SlidingWindowFeature) or be ndarray/Tensor
- Pipeline hook step names from speaker_diarization.py: "segmentation", "speaker_counting", "embeddings", "discrete_diarization"
- RTTM format: `SPEAKER <uri> 1 <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>` with .3f precision
- DiarizationErrorRate(detailed=True) returns dict with keys: "diarization error rate", "missed detection", "false alarm", "confusion", "total"
- C++ binary CLI: `./diarization-ggml seg.gguf emb.gguf audio.wav --plda plda.bin --coreml emb.mlpackage -o output.rttm`
- Stage-by-stage comparison is structured but currently only RTTM end-to-end runs (C++ `--dump-stage` not yet implemented)
- `sys.path.insert(0, script_dir)` used to allow `compare_pipeline.py` to import from `compare_rttm.py` as sibling module
- Tolerance budget encoded in TOLERANCES dict: cosine for seg/emb, exact for powerset/count/ahc, max_abs_diff for plda/vbx, DER% for rttm

## Task 13: End-to-End Validation
- DER = 0.14% (with Python segmentation logits) — PASS (threshold: 1.0%)
- DER = 0.00% with 0.25s collar — perfect match
- 13 segments, 2 speakers — matches Python exactly
- Only difference: one segment duration differs by 0.033s (2 frames) — false alarm
- FRAME_DURATION must be 0.0619375 (model receptive field duration), NOT 0.016875 (step)
- RTTM segment start times must use frame midpoints: `start + frame_idx * step + 0.5 * duration`
  - Python's Binarize class uses `timestamps = [frames[i].middle for i in range(num_frames)]`
  - C++ was using frame starts, causing systematic -0.031s offset on all segments
- `--seg-logits <path>` flag added to bypass C++ segmentation model for validation
- Binary format: flat float32, shape (num_chunks, 589, 7), row-major

## Segmentation Model Precision Issue (Known Limitation)
- C++ GGML segmentation model produces different powerset argmax results than Python F32 model
- Root cause: NOT just F16 weight precision — even with F32 weights + F32 SincNet conv, results differ
- The LSTM custom op (cblas_sgemm) accumulates floating-point errors differently than PyTorch's LSTM
- Effect: C++ model picks overlap classes (e.g., {speaker 0, speaker 1}) instead of single-speaker classes
  - Python chunk 0: single_active=195, only speakers 1,2 active
  - C++ chunk 0: single_active=75, ALL 3 speakers active
- This causes filter_embeddings() to return 0 embeddings (all below 117.8 threshold)
- Workaround: use `--seg-logits` to load Python-generated segmentation logits
- Fixes attempted but insufficient:
  1. F32 GGUF weights (LSTM + Linear + Classifier) — still 0 filtered embeddings
  2. F32 SincNet conv (manual im2col with GGML_TYPE_F32) — still 0 filtered embeddings
  3. Hybrid F16/F32 (SincNet F16, rest F32) — still 0 filtered embeddings
- The issue is fundamental to the GGML LSTM implementation vs PyTorch LSTM
- Future fix options: port PyTorch's exact LSTM implementation, or use softmax-based powerset conversion

## compare_pipeline.py seg-logits bypass (Task 14)
- `--seg-logits` / `--no-seg-logits` flag added via `argparse.BooleanOptionalAction` (default: enabled)
- Logit extraction uses `pipeline._segmentation.model` to get raw seg model before powerset conversion
- Audio loaded via `torchaudio.load()`, resampled to 16kHz if needed
- Chunks: 10s windows (160000 samples) with 1s step (16000 samples), zero-padded
- Output format: flat float32 binary, shape (num_chunks, 589, 7), saved via `ndarray.tofile()`
- C++ binary receives `--seg-logits <path>` to bypass GGML segmentation model
- `extract_seg_logits` param added to `run_python_pipeline()`, `seg_logits_path` param added to `run_cpp_pipeline()`
- `patch_torch_load()` is called before pipeline load (already existed), torchaudio import added alongside torch

## Tensor Layout Fix (CORRECTS "Segmentation Model Precision Issue" above)
- The "GGML Segmentation Model Precision Issue" documented above was a MISDIAGNOSIS
- REAL root cause: tensor layout mismatch in segmentation model output
  - GGML model outputs logits in [class, frame] memory order (ne[0]=589 contiguous)
  - powerset_to_multilabel() expects [frame, class] order (7 class values contiguous per frame)
- Fix: 13-line in-place transpose after each model_infer() call in diarization.cpp (~line 426)
- F16 precision in ggml_conv_1d (SincNet) is NOT an issue:
  - Cosine similarity 0.9999 between F16 and F32 conv outputs
  - 100% argmax match between F16 and F32
  - Identical DER (0.28%) with original ggml_conv_1d
- F32 weight support in LSTM/SincNet was unnecessary — reverted from git history
- The `--seg-logits` bypass remains available as optional debugging tool, not required
- Pipeline runs fully native C++ end-to-end: DER 0.28%, 2 speakers, 14 segments
