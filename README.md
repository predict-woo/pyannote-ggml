# diarization-ggml

Native C++ speaker diarization pipeline achieving **39x real-time** on Apple Silicon.

A complete port of [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) to C++ using GGML and CoreML.

## Performance

| Component | Time | Notes |
|-----------|------|-------|
| Segmentation (CoreML) | 12ms/chunk | SincNet + BiLSTM + Linear |
| Embedding (CoreML) | 13ms/chunk | WeSpeaker ResNet34 |
| AHC Clustering | 0.8s | fastcluster O(n²) for 3000 embeddings |
| **Full Pipeline** | **39x real-time** | 45 min audio in 70 seconds |

Tested on Apple M2 Pro. DER matches Python reference within 1%.

## Quick Start

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- CMake 3.20+
- Xcode Command Line Tools
- Python 3.10+ with `coremltools` (for model conversion)

### Build

```bash
# Clone with submodules
git clone --recursive https://github.com/your-repo/diarization-ggml.git
cd diarization-ggml

# Build full pipeline with CoreML
cd diarization-ggml
cmake -B build -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON
cmake --build build
```

### Convert Models (one-time)

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchaudio pyannote.audio coremltools

# Convert segmentation model
cd segmentation-ggml
python convert.py --output segmentation.gguf
python convert_coreml.py

# Convert embedding model
cd ../embedding-ggml
python convert_coreml.py
```

### Run Diarization

```bash
cd diarization-ggml

./build/bin/diarization-ggml \
  ../segmentation-ggml/segmentation.gguf \
  ../embedding-ggml/embedding.gguf \
  ../samples/sample.wav \
  --plda plda.bin \
  --coreml ../embedding-ggml/embedding.mlpackage \
  --seg-coreml ../segmentation-ggml/segmentation.mlpackage \
  -o output.rttm
```

Output is in [RTTM format](https://catalog.ldc.upenn.edu/docs/LDC2004T12/RTTM-format-v13.pdf):
```
SPEAKER sample 1 0.497 2.085 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER sample 1 2.667 1.950 <NA> <NA> SPEAKER_01 <NA> <NA>
...
```

## Project Structure

```
.
├── segmentation-ggml/     # Segmentation model (SincNet + BiLSTM)
│   ├── src/               # C++ implementation
│   ├── convert.py         # PyTorch → GGUF converter
│   └── convert_coreml.py  # PyTorch → CoreML converter
│
├── embedding-ggml/        # Speaker embedding model (ResNet34)
│   ├── src/               # C++ implementation
│   └── convert_coreml.py  # PyTorch → CoreML converter
│
├── diarization-ggml/      # Full pipeline orchestration
│   ├── src/
│   │   ├── diarization.cpp   # Pipeline orchestration
│   │   ├── powerset.cpp      # Powerset → multilabel
│   │   ├── plda.cpp          # PLDA scoring
│   │   ├── vbx.cpp           # VBx clustering
│   │   └── clustering.cpp    # fastcluster AHC
│   ├── plda.bin              # Pre-computed PLDA model
│   └── tests/
│
├── ggml/                  # GGML tensor library (submodule)
├── samples/               # Test audio files
└── AGENTS.md              # AI agent guidelines
```

## Architecture

### Pipeline Stages

1. **Segmentation**: Raw waveform → 7-class powerset logits
   - SincNet feature extraction (learnable sinc filters)
   - 4-layer bidirectional LSTM
   - Linear classifier with log-softmax

2. **Powerset Decoding**: Logits → per-frame speaker activity
   - Maps 7 powerset classes to 3 speakers
   - Handles overlapping speech

3. **Embedding Extraction**: Speech segments → 256-dim vectors
   - WeSpeaker ResNet34 architecture
   - Mel filterbank features (80 bins)

4. **Clustering**: Embeddings → speaker labels
   - PLDA scoring for pairwise similarity
   - VBx variational Bayes refinement
   - Agglomerative hierarchical clustering (fastcluster)

### CoreML Acceleration

Both neural network models run on Apple's Neural Engine via CoreML:
- Segmentation: 46ms → 12ms (3.8x speedup)
- Embedding: 25ms → 13ms (1.9x speedup)

## Testing

```bash
# Segmentation accuracy test
.venv/bin/python3 segmentation-ggml/tests/test_accuracy.py

# Full pipeline DER test
./diarization-ggml/build/bin/diarization-ggml \
  segmentation-ggml/segmentation.gguf \
  embedding-ggml/embedding.gguf \
  samples/sample.wav \
  --plda diarization-ggml/plda.bin \
  --coreml embedding-ggml/embedding.mlpackage \
  --seg-coreml segmentation-ggml/segmentation.mlpackage \
  -o /tmp/test.rttm

.venv/bin/python3 diarization-ggml/tests/compare_rttm.py \
  /tmp/test.rttm /tmp/py_reference.rttm --threshold 1.0
```

## Dependencies

- **GGML**: Tensor library (submodule)
- **kaldi-native-fbank**: Mel filterbank extraction
- **fastcluster**: O(n²) hierarchical clustering (vendored)
- **Accelerate.framework**: BLAS/LAPACK on macOS
- **CoreML.framework**: Neural Engine inference

## License

MIT License - see [LICENSE](LICENSE)

## Acknowledgments

- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - Original Python implementation
- [GGML](https://github.com/ggerganov/ggml) - Tensor library
- [WeSpeaker](https://github.com/wenet-e2e/wespeaker) - Speaker embedding model
- [fastcluster](https://danifold.net/fastcluster.html) - Efficient hierarchical clustering
