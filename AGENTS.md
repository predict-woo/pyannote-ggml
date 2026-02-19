# AGENTS.md — AI Agent Guidelines for pyannote-audio C++ Port

This file provides context and guidelines for AI agents working on the `diarization-ggml` project — a full C++ port of the `pyannote/speaker-diarization-community-1` pipeline.

## Project Overview

Native C++ speaker diarization + transcription pipeline on Apple Silicon:
- **Diarization** (offline): 39x real-time — SincNet/BiLSTM segmentation + WeSpeaker embedding + VBx clustering
- **Diarization** (streaming): Byte-identical to offline via `streaming_push` / `streaming_finalize`
- **Transcription + Diarization** (integrated): ~14.6x real-time — Whisper transcription + pyannote streaming diarization + WhisperX-style segment-level speaker alignment
- **Speaker alignment**: WhisperX algorithm — assigns speakers at Whisper sentence level (not per-token), robust 2.55% confusion rate

## Repository Structure

```
pyannote.cpp/
├── models/                             # Neural network model implementations
│   ├── segmentation-ggml/              # Segmentation model (SincNet + BiLSTM)
│   │   ├── src/
│   │   │   ├── model.cpp               # GGML inference, graph building
│   │   │   ├── sincnet.cpp             # SincNet conv layers
│   │   │   ├── lstm.cpp                # BiLSTM with cblas_sgemm optimization
│   │   │   └── coreml/                 # CoreML bridge (Obj-C++)
│   │   ├── convert.py                  # PyTorch → GGUF converter
│   │   ├── convert_coreml.py           # PyTorch → CoreML converter
│   │   └── tests/
│   │       └── test_accuracy.py        # GGML vs PyTorch comparison
│   │
│   └── embedding-ggml/                 # Speaker embedding model (ResNet34)
│       ├── src/
│       │   ├── model.cpp               # GGML inference
│       │   ├── fbank.cpp               # Mel filterbank features
│       │   └── coreml/                 # CoreML bridge
│       ├── convert_coreml.py           # PyTorch → CoreML converter
│       └── tests/
│
├── diarization-ggml/                   # Full diarization + transcription pipeline
│   ├── include/
│   │   ├── diarization.h               # DiarizationResult, DiarizationConfig
│   │   └── transcriber_types.h         # TranscribeSegment, TranscribeWord, TranscribeResult
│   ├── src/
│   │   ├── diarization.cpp             # Offline diarization pipeline
│   │   ├── powerset.cpp                # Powerset → multilabel conversion
│   │   ├── plda.cpp                    # PLDA scoring
│   │   ├── vbx.cpp                     # VBx clustering
│   │   ├── clustering.cpp              # fastcluster AHC wrapper
│   │   ├── fastcluster/                # Daniel Müllner's O(n²) AHC
│   │   ├── streaming.cpp               # Streaming diarization (~640 lines)
│   │   ├── streaming.h                 # streaming_init/push/recluster/finalize/free
│   │   ├── streaming_state.h           # StreamingConfig, StreamingState
│   │   ├── pipeline.cpp                # Integrated transcription+diarization pipeline
│   │   ├── pipeline.h                  # pipeline_init/push/finalize/free, callback
│   │   ├── transcriber.cpp             # Whisper worker thread
│   │   ├── transcriber.h               # TranscriberConfig, transcriber API
│   │   ├── aligner.cpp                 # WhisperX-style segment-level speaker alignment
│   │   ├── aligner.h                   # AlignedSegment, AlignedWord, align_segments()
│   │   ├── silence_filter.cpp          # VAD-based silence compression
│   │   ├── silence_filter.h
│   │   ├── audio_buffer.cpp            # FIFO with absolute frame tracking
│   │   ├── audio_buffer.h
│   │   ├── segment_detector.cpp        # Speech boundary detection from VAD chunks
│   │   ├── segment_detector.h
│   │   ├── main_transcribe.cpp         # CLI: `transcribe` binary
│   │   └── main.cpp                    # CLI: `diarization-ggml` binary
│   ├── tests/
│   │   ├── compare_rttm.py             # DER computation (GROUND TRUTH)
│   │   ├── compare_pipeline.py         # Stage-by-stage validation
│   │   ├── test_aligner.cpp            # Aligner unit tests
│   │   ├── test_transcriber.cpp        # Transcriber unit tests
│   │   ├── test_pipeline.cpp           # Pipeline integration test
│   │   ├── test_silence_filter.cpp
│   │   ├── test_audio_buffer.cpp
│   │   └── test_segment_detector.cpp
│   └── plda.gguf                       # Pre-computed PLDA model (GGUF)
│
├── whisper.cpp/                        # whisper.cpp (local copy, NOT a submodule — source modified)
│   ├── src/whisper.cpp                 # Modified: DTW + flash attention coexistence
│   ├── models/                         # Whisper GGUF models
│   │   └── ggml-large-v3-turbo-q5_0.bin
│   └── build/                          # whisper.cpp build directory
│
├── tools/                              # Viewer tools
│   ├── whisper-viewer.html             # Karaoke-style Whisper JSON viewer
│   ├── transcript-viewer.html          # Speaker-labeled transcript viewer
│   └── rttm-viewer.html               # RTTM timeline viewer
│
├── samples/                            # Test audio files
│   ├── sample.wav                      # Short test (2 speakers, ~30s)
│   └── long_sample.wav                 # Long test (6 speakers, ~45min Korean)
│
├── ggml/                               # GGML submodule (tensor library)
└── .sisyphus/                          # Work tracking
    ├── plans/                          # Task plans (READ ONLY)
    └── notepads/                       # Learnings, decisions, issues
```

## Build Commands

```bash
# Segmentation only
cd models/segmentation-ggml && cmake -B build && cmake --build build

# Embedding only  
cd models/embedding-ggml && cmake -B build && cmake --build build

# Full pipeline with CoreML
cd diarization-ggml
cmake -B build -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON
cmake --build build

# Generate CoreML models (run once)
cd models/segmentation-ggml && ../../.venv/bin/python3 convert_coreml.py
cd models/embedding-ggml && ../../.venv/bin/python3 convert_coreml.py
```

## Testing Protocol

**CRITICAL: Run BOTH tests after ANY code change that could affect numerical output.**

### Test 1: Segmentation Accuracy
```bash
cd /Users/andyye/dev/pyannote-audio
.venv/bin/python3 models/segmentation-ggml/tests/test_accuracy.py
```
Must pass: cosine > 0.995, max_err < 1.0

### Test 2: Full Pipeline DER (GROUND TRUTH)
```bash
cd /Users/andyye/dev/pyannote-audio/diarization-ggml

./build/bin/diarization-ggml \
  ../models/segmentation-ggml/segmentation.gguf \
  ../models/embedding-ggml/embedding.gguf \
  ../samples/sample.wav \
  --plda plda.gguf \
  --coreml ../models/embedding-ggml/embedding.mlpackage \
  --seg-coreml ../models/segmentation-ggml/segmentation.mlpackage \
  -o /tmp/test.rttm

cd /Users/andyye/dev/pyannote-audio
.venv/bin/python3 diarization-ggml/tests/compare_rttm.py /tmp/test.rttm /tmp/py_reference.rttm --threshold 1.0
```
Must show: 2 speakers, 13 segments, DER ≤ 1.0%

### Generate Python Reference (if missing)
```bash
.venv/bin/python3 -c "
import torch
torch.load = lambda *a, **k: torch._load(*a, **{**k, 'weights_only': False})
from pyannote.audio import Pipeline
p = Pipeline.from_pretrained('pyannote/speaker-diarization-community-1')
d = p('samples/sample.wav')
with open('/tmp/py_reference.rttm', 'w') as f:
    for t, _, s in d.itertracks(yield_label=True):
        f.write(f'SPEAKER sample 1 {t.start:.3f} {t.end-t.start:.3f} <NA> <NA> {s} <NA> <NA>\n')
"
```

## Key Technical Decisions

### Numerical Precision
- **GGML uses F16 weights** — cosine similarity ~0.9999 vs F32, acceptable
- **PLDA/VBx uses float64** — must match numpy exactly for correct clustering
- **CoreML uses F16 compute** — faster, power-efficient, accuracy verified

### LSTM Optimization
- Pre-convert F16→F32 weights once at init (not per-inference)
- Use `cblas_sgemm` for batched input transform (W_ih @ input for all timesteps)
- Use `cblas_sgemv` for per-timestep hidden state (W_hh @ h_t)
- Stack arrays instead of std::vector for small buffers

### AHC Clustering
- Replaced naive O(n³) centroid linkage with fastcluster O(n²)
- 3184 embeddings: 10+ min → 0.8s
- Uses `cutree_cdist()` with threshold=0.6

### CoreML Integration
- Segmentation: raw waveform (160000 samples) → logits (589, 7)
- Embedding: fbank features (T, 80) → embedding (256)
- Both use `MLComputeUnitsAll` (CPU + GPU + Neural Engine)
- Runtime model compilation from .mlpackage

### Streaming Architecture
The streaming API (`streaming.h`, `streaming.cpp`, `streaming_state.h`) processes audio incrementally. `streaming_push` accepts ~1s of audio at a time and returns VAD chunks. `streaming_recluster` runs full clustering on demand. `streaming_finalize` produces byte-identical output to the offline pipeline.

**Key files**:
- `streaming.h` — public API: `streaming_init`, `streaming_push`, `streaming_recluster`, `streaming_finalize`, `streaming_free`
- `streaming.cpp` — full implementation (~640 lines)
- `streaming_state.h` — `StreamingConfig` (model paths) and `StreamingState` (all accumulated state)
- `diarization_stream.h` — C API header (declared but NOT implemented)

**Three data stores in StreamingState**:

| Store | Lifetime | Why |
|---|---|---|
| `audio_buffer` | Sliding window (~10s) | Trimmed after each chunk via `samples_trimmed` offset. Only need current 10s window. |
| `embeddings` | Grows forever | `[N × 256]`, 3 per chunk (NaN for silent speakers). Recluster needs ALL embeddings for `soft_clusters` and `constrained_argmax`. |
| `binarized` | Grows forever | `[num_chunks × 589 × 3]`, per-frame binary speaker activity. Used for: filtering silent embeddings, marking inactive speakers as -2, computing speaker_count across overlapping chunks, rebuilding the global timeline (`clustered_seg`) from local→global speaker mapping. |

**`process_one_chunk()` — the shared workhorse** (static helper):
- Called by both `streaming_push` and `streaming_finalize`
- Crops 160K samples from `audio_buffer` at buffer-relative offset `chunks_processed * STEP_SAMPLES - samples_trimmed`
- Zero-pads if window extends past available audio (important for finalize's last chunks)
- Runs segmentation CoreML → powerset decode → binarize
- Computes fbank, then for each of 3 local speakers: masks fbank by speaker activity, runs embedding CoreML (or stores NaN if silent)
- Appends to `binarized` and `embeddings`
- After processing, trims `audio_buffer` front (next chunk starts 1s later)

**Audio buffer trimming**:
- `samples_trimmed` tracks how many absolute samples have been erased from buffer front
- After processing chunk N, everything before absolute position `(N+1) * STEP_SAMPLES` is dead (9s overlap means next chunk starts 1s later)
- `process_one_chunk` erases the dead prefix and updates `samples_trimmed`
- All indexing is buffer-relative: `chunk_start = chunks_processed * STEP_SAMPLES - samples_trimmed`
- `streaming_push` and `streaming_finalize` convert buffer size to absolute via `buffer.size() + samples_trimmed`

**`streaming_push` flow**:
1. Append samples to `audio_buffer`
2. Compute `total_abs_samples = buffer.size() + samples_trimmed`
3. While `total_abs_samples >= chunks_processed * STEP_SAMPLES + CHUNK_SAMPLES`: call `process_one_chunk`, build VADChunk (OR of 3 local speakers), recompute loop condition
4. First 10 pushes (10s) return empty — first chunk needs full 10s window
5. After that, each push returns 1 new VADChunk

**`streaming_recluster` flow** (mirrors offline pipeline exactly):
1. `filter_embeddings` — remove NaN embeddings (silent speakers)
2. L2-normalize filtered embeddings (double precision) for AHC
3. PLDA transform (256-dim → 128-dim)
4. AHC clustering (threshold=0.6)
5. VBx refinement (FA=0.07, FB=0.8, max 20 iters)
6. Compute centroids from VBx gamma (weighted average of filtered embeddings)
7. `soft_clusters` — cosine similarity of ALL embeddings (not just filtered!) against centroids
8. `constrained_argmax` — assign each embedding to cluster, no two local speakers in same chunk get same global speaker
9. Mark inactive local speakers as -2
10. Reconstruct timeline: `clustered_seg` → `to_diarization` → extract segments

**`streaming_finalize` flow**:
1. Compute offline chunk count: `max(1, 1 + ceil((duration - 10.0) / 1.0))`
2. Process remaining partial chunks (zero-padded) to match offline count
3. Call `streaming_recluster` on complete data
4. Set `finalized = true`

**Constants** (defined at top of `streaming.cpp`):

| Constant | Value |
|---|---|
| SAMPLE_RATE | 16000 |
| CHUNK_SAMPLES | 160000 (10s) |
| STEP_SAMPLES | 16000 (1s hop, 9s overlap) |
| FRAMES_PER_CHUNK | 589 |
| NUM_POWERSET_CLASSES | 7 |
| NUM_LOCAL_SPEAKERS | 3 |
| EMBEDDING_DIM | 256 |
| FBANK_NUM_BINS | 80 |

Reconstruction constants (inside `streaming_recluster`):

| Constant | Value |
|---|---|
| CHUNK_DURATION | 10.0s |
| CHUNK_STEP | 1.0s |
| FRAME_DURATION | 0.0619375s |
| FRAME_STEP | 0.016875s |

### Integration Pipeline
The transcription+diarization path is a 7-stage streaming pipeline:
1. **VAD Silence Filter** - compresses long silence to <=2s while preserving transitions.
2. **Audio Buffer** - FIFO filtered-audio store with absolute frame accounting.
3. **Segmentation** - pyannote streaming VAD drives segment-end detection.
4. **Transcription** - Whisper worker transcribes buffered audio (20s minimum trigger, ~30s cap).
5. **Alignment** - WhisperX-style segment-level speaker assignment by maximum overlap.
6. **Finalize** - flushes all stages and runs final recluster + alignment.
7. **Callback** - emits incremental speaker-labeled word segments.

**Key files**:
- `pipeline.h` / `pipeline.cpp`
- `silence_filter.h` / `silence_filter.cpp`
- `audio_buffer.h` / `audio_buffer.cpp`
- `segment_detector.h` / `segment_detector.cpp`
- `transcriber.h` / `transcriber.cpp`
- `aligner.h` / `aligner.cpp`
- `main_transcribe.cpp`

**Build command (integrated pipeline)**:
```bash
cd diarization-ggml
cmake -B build -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON -DWHISPER_COREML=ON
cmake --build build
```

**Test commands**:
```bash
# Unit tests
cd diarization-ggml
./build/bin/test_silence_filter
./build/bin/test_audio_buffer
./build/bin/test_segment_detector
./build/bin/test_transcriber
./build/bin/test_aligner
./build/bin/test_pipeline

# Integration test
python3 tests/test_integration.py \
  --build-dir build \
  --seg-model ../models/segmentation-ggml/segmentation.gguf \
  --emb-model ../models/embedding-ggml/embedding.gguf \
  --whisper-model ../whisper.cpp/models/ggml-base.en.bin \
  --plda plda.gguf
```

**Integration constants** (`pipeline.cpp`):

| Constant | Value |
|---|---|
| MIN_SEGMENT_DURATION | 20s |
| MAX_WHISPER_SAMPLES | 30s (480000 samples) |
| SAMPLE_RATE | 16000 |

### WhisperX-Style Speaker Alignment
Speaker assignment uses the WhisperX algorithm at **segment level** (not per-token). This was a deliberate design choice — Whisper's natural sentence-level segments (2-10s) provide robust temporal boundaries for speaker matching, while individual token timestamps (50ms) are too noisy.

**Algorithm** (implemented in `aligner.cpp`):
1. Sort diarization segments by start time
2. For each Whisper `TranscribeSegment` (a sentence):
   - Compute overlap duration with every diarization segment
   - Sum overlap per speaker
   - Assign the speaker with maximum total overlap
3. If no overlap (gap in diarization): find nearest diarization segment by midpoint distance
4. All words within a segment inherit the segment's speaker
5. Consecutive same-speaker segments are merged into `AlignedSegment`

**Data flow**:
```
Whisper output:  TranscribeSegment { start, end, text, words[] }
                        ↓
                align_segments()  ←  DiarizationResult (from streaming_recluster)
                        ↓
                 AlignedSegment { speaker, start, duration, words[] }
```

**Key types** (in `transcriber_types.h`):
- `TranscribeWord` — individual word with start/end timestamps
- `TranscribeSegment` — Whisper sentence with start/end and child words
- `TranscribeResult` — vector of TranscribeSegments from one Whisper run

**Why NOT per-token alignment**:
- DTW word-level timestamps are noisy (~50ms tokens straddling diarization boundaries)
- Sentence-level segments (2-10s) overlapping with multi-second diarization segments = robust match
- DTW adds computation cost for no speaker-assignment benefit
- DTW is disabled in the transcriber (no `dtw_token_timestamps` context params)

### Pipeline Audio Management
The pipeline uses a **submission queue** to prevent dropped audio when Whisper is busy:
- `enqueue_audio_chunk()` adds audio to a `std::deque<PendingSubmission>` queue
- `try_submit_next()` only submits to Whisper if it's not already processing
- After each Whisper result, the next queued chunk is submitted
- `pipeline_finalize()` drains the queue completely

**Re-alignment on every recluster**: All accumulated `TranscribeSegment`s are stored in `all_transcribe_segments`. Every time `streaming_recluster` runs (after each Whisper result), ALL segments are re-aligned against the latest diarization. This ensures early segments get corrected speaker labels as more data arrives. The final `streaming_finalize` + re-alignment produces the definitive output.

### Silence Filter Behavior Without VAD
Without `--vad-model`, `vad_ctx == nullptr`, so `detect_probability` returns 1.0 for any non-zero audio. The silence filter is effectively a no-op — all audio passes through unchanged. This means streaming diarization gets the same audio as offline.

## Common Pitfalls

### 1. Per-element accuracy ≠ Pipeline accuracy
The segmentation accuracy test (cosine similarity) can pass while the full DER test fails. Small numerical differences compound through powerset→speaker_count→clustering. **Always run both tests.**

### 2. GGML tensor layout
GGML is column-major: `ne[0]` is the contiguous dimension. A tensor with shape `[589, 7, 1]` has 589 elements contiguous, then 7 rows. This is transposed from PyTorch's row-major layout.

### 3. CoreML output precision
CoreML may output F16 even when configured for F32 compute. The bridge code handles F16→F32 conversion. Check `output.dataType` before reading.

### 4. Segmentation model input
The segmentation model expects raw waveform `(1, 1, 160000)` for CoreML, but `(160000, 1, 1)` for GGML (column-major). The transpose happens in the bridge.

### 5. Streaming recluster mutates state (FIXED)
Previously, `streaming_recluster` overwrote `state->chunk_idx`, `state->local_speaker_idx`, and `state->embeddings` with filtered versions, making push→recluster→push→recluster cycles unsafe. This has been **fixed**: recluster now uses local variables for filtering and keeps `state->embeddings`, `state->chunk_idx`, and `state->local_speaker_idx` unfiltered (3 entries per chunk) so subsequent `push()` calls can safely append to them. Push→recluster→push→recluster cycles now work correctly.

### 6. Streaming requires CoreML
The streaming code has no GGML-only fallback. Both segmentation and embedding `#else` branches print errors and return false/continue. Any streaming work must be compiled with `-DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON`.

### 7. Audio buffer uses sliding window with offset tracking
`audio_buffer` is NOT indexed by absolute position. `samples_trimmed` tracks how many samples were erased from the front. All buffer access uses `chunks_processed * STEP_SAMPLES - samples_trimmed` as the base offset. If you add code that reads from `audio_buffer`, you must account for this offset.

## Performance Targets

| Component | Target | Achieved |
|-----------|--------|----------|
| Segmentation (CoreML) | <25ms/chunk | 12ms ✓ |
| Embedding (CoreML) | <15ms/chunk | 13ms ✓ |
| AHC clustering | <1s for 3000 embeddings | 0.8s ✓ |
| Full pipeline | Real-time or better | 39x RT ✓ |
| Integrated (transcribe+diarize) | Real-time or better | ~14.6x RT ✓ |

Integrated pipeline tested on 45min Korean meeting audio (6 speakers, `long_sample.wav`):
- 2713s audio processed in 186s (RTF=0.069)
- 701 Whisper segments → 186 aligned speaker segments
- 977 diarization segments from `streaming_finalize`
- Speaker confusion rate: 2.55% (compared against offline diarization RTTM)

## Dependencies

- **GGML**: Tensor library (submodule)
- **whisper.cpp**: Whisper inference (local copy, NOT a submodule — source modified for DTW+flash attention coexistence)
- **kaldi-native-fbank**: Mel filterbank extraction
- **fastcluster**: O(n²) hierarchical clustering (BSD, vendored)
- **Accelerate.framework**: BLAS/LAPACK on macOS
- **CoreML.framework**: Neural Engine inference on macOS
- **coremltools**: Python package for model conversion

## Git Conventions

- **Never force-push** to main
- **Never modify** `.sisyphus/plans/*.md` (read-only task tracking)
- **Append only** to `.sisyphus/notepads/*.md`
- Commit messages: `type(scope): description`
  - `feat`: new feature
  - `fix`: bug fix
  - `perf`: performance improvement
  - `refactor`: code restructuring
  - `test`: test additions/changes
  - `build`: build system changes

## Useful Commands

```bash
# Quick benchmark
./build/bin/segmentation-ggml segmentation.gguf --test --audio ../samples/sample.wav

# Profile diarization
./build/bin/diarization-ggml ... 2>&1 | grep -E "Timing|ms"

# Check GGML Metal backend
GGML_METAL_LOG_LEVEL=1 ./build/bin/diarization-ggml ...
```
