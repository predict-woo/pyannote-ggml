# AGENTS.md — AI Agent Guidelines for pyannote-audio C++ Port

This file provides context and guidelines for AI agents working on the `diarization-ggml` project — a full C++ port of the `pyannote/speaker-diarization-community-1` pipeline.

## Project Overview

Native C++ speaker diarization pipeline achieving **39x real-time** on Apple Silicon:
- **Segmentation**: SincNet + BiLSTM + Linear → 7-class powerset logits (CoreML, 12ms/chunk)
- **Embedding**: WeSpeaker ResNet34 → 256-dim speaker embeddings (CoreML, 13ms/chunk)
- **Clustering**: VBx with PLDA + fastcluster AHC (CPU, O(n²))

## Repository Structure

```
pyannote-audio/
├── segmentation-ggml/          # Segmentation model (SincNet + BiLSTM)
│   ├── src/
│   │   ├── model.cpp           # GGML inference, graph building
│   │   ├── sincnet.cpp         # SincNet conv layers
│   │   ├── lstm.cpp            # BiLSTM with cblas_sgemm optimization
│   │   └── coreml/             # CoreML bridge (Obj-C++)
│   ├── convert.py              # PyTorch → GGUF converter
│   ├── convert_coreml.py       # PyTorch → CoreML converter
│   └── tests/
│       └── test_accuracy.py    # GGML vs PyTorch comparison
│
├── embedding-ggml/             # Speaker embedding model (ResNet34)
│   ├── src/
│   │   ├── model.cpp           # GGML inference
│   │   ├── fbank.cpp           # Mel filterbank features
│   │   └── coreml/             # CoreML bridge
│   ├── convert_coreml.py       # PyTorch → CoreML converter
│   └── tests/
│
├── diarization-ggml/           # Full diarization pipeline
│   ├── src/
│   │   ├── diarization.cpp     # Pipeline orchestration
│   │   ├── powerset.cpp        # Powerset → multilabel conversion
│   │   ├── plda.cpp            # PLDA scoring
│   │   ├── vbx.cpp             # VBx clustering
│   │   ├── clustering.cpp      # fastcluster AHC wrapper
│   │   └── fastcluster/        # Daniel Müllner's O(n²) AHC
│   ├── tests/
│   │   ├── compare_rttm.py     # DER computation (GROUND TRUTH)
│   │   └── compare_pipeline.py # Stage-by-stage validation
│   └── plda.bin                # Pre-computed PLDA model
│
├── ggml/                       # GGML submodule (tensor library)
└── .sisyphus/                  # Work tracking
    ├── plans/                  # Task plans (READ ONLY)
    └── notepads/               # Learnings, decisions, issues
```

## Build Commands

```bash
# Segmentation only
cd segmentation-ggml && cmake -B build && cmake --build build

# Embedding only  
cd embedding-ggml && cmake -B build && cmake --build build

# Full pipeline with CoreML
cd diarization-ggml
cmake -B build -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON
cmake --build build

# Generate CoreML models (run once)
cd segmentation-ggml && ../.venv/bin/python3 convert_coreml.py
cd embedding-ggml && ../.venv/bin/python3 convert_coreml.py
```

## Testing Protocol

**CRITICAL: Run BOTH tests after ANY code change that could affect numerical output.**

### Test 1: Segmentation Accuracy
```bash
cd /Users/andyye/dev/pyannote-audio
.venv/bin/python3 segmentation-ggml/tests/test_accuracy.py
```
Must pass: cosine > 0.995, max_err < 1.0

### Test 2: Full Pipeline DER (GROUND TRUTH)
```bash
cd /Users/andyye/dev/pyannote-audio/diarization-ggml

./build/bin/diarization-ggml \
  ../segmentation-ggml/segmentation.gguf \
  ../embedding-ggml/embedding.gguf \
  ../samples/sample.wav \
  --plda plda.bin \
  --coreml ../embedding-ggml/embedding.mlpackage \
  --seg-coreml ../segmentation-ggml/segmentation.mlpackage \
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

## Common Pitfalls

### 1. Per-element accuracy ≠ Pipeline accuracy
The segmentation accuracy test (cosine similarity) can pass while the full DER test fails. Small numerical differences compound through powerset→speaker_count→clustering. **Always run both tests.**

### 2. GGML tensor layout
GGML is column-major: `ne[0]` is the contiguous dimension. A tensor with shape `[589, 7, 1]` has 589 elements contiguous, then 7 rows. This is transposed from PyTorch's row-major layout.

### 3. CoreML output precision
CoreML may output F16 even when configured for F32 compute. The bridge code handles F16→F32 conversion. Check `output.dataType` before reading.

### 4. Segmentation model input
The segmentation model expects raw waveform `(1, 1, 160000)` for CoreML, but `(160000, 1, 1)` for GGML (column-major). The transpose happens in the bridge.

## Performance Targets

| Component | Target | Achieved |
|-----------|--------|----------|
| Segmentation (CoreML) | <25ms/chunk | 12ms ✓ |
| Embedding (CoreML) | <15ms/chunk | 13ms ✓ |
| AHC clustering | <1s for 3000 embeddings | 0.8s ✓ |
| Full pipeline | Real-time or better | 39x RT ✓ |

## Dependencies

- **GGML**: Tensor library (submodule)
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
