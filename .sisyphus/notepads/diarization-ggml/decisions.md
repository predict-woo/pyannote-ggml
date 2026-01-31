# Decisions — diarization-ggml

## Architecture
- Library extraction: segmentation-core and embedding-core STATIC libraries
- PLDA data: pre-convert .npz → flat binary via convert_plda.py (no cnpy/zlib in C++)
- Eigendecomposition pre-computed in Python converter (avoids LAPACK sign/order ambiguity)
- Embedding masking: zero out fbank frames where speaker is inactive

## Scope
- VBx full fidelity (no AHC-only fallback)
- CoreML for embedding inference
- RTTM output format
- No KMeans fallback, no streaming, no GPU VBx
