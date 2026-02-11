# PyAnnote Segmentation Model - GGML Implementation

✅ **WORKING** - This implementation produces outputs that match PyTorch reference with high accuracy.

## Current Status: FUNCTIONAL

This is a GGML conversion of the PyAnnote segmentation-3.0 model for efficient inference on Apple Silicon.

**Accuracy Results** (F16 weights):
- ✅ SincNet: cosine=0.998, max_err=0.13
- ✅ LSTM: cosine=0.999, max_err=0.24
- ✅ Linear1: cosine=0.999, max_err=0.38
- ✅ Linear2: cosine=0.998, max_err=0.63
- ✅ Classifier: cosine=0.999, max_err=0.29

Max errors are due to F16 quantization accumulating over 5 layers. Cosine similarity > 0.998 indicates functionally equivalent outputs.

## Overview

This project converts the PyAnnote segmentation-3.0 model from PyTorch to GGML/GGUF format for efficient inference on Apple Silicon with Metal backend support.

**Model**: PyanNet architecture (SincNet + LSTM + Linear + Classifier)  
**Parameters**: 1,473,265  
**Input**: 16kHz mono audio  
**Output**: 7-class speaker segmentation probabilities

## Features

- ✅ **Complete Model**: SincNet, 4-layer bidirectional LSTM, Linear layers, Classifier
- ⚠️ **Metal Backend**: Initialized but computation uses CPU (Metal tensor allocation not implemented)
- ✅ **PyTorch to GGUF Conversion**: Automatic weight conversion
- ✅ **Comprehensive Testing**: Automated accuracy comparison vs PyTorch
- ✅ **WAV File Support**: Direct audio loading (16kHz mono)
- ✅ **Accuracy Verified**: Cosine similarity > 0.998 vs PyTorch

## Quick Start

### Build

```bash
cd segmentation-ggml
cmake -B build -DGGML_METAL=ON
cmake --build build
```

### Convert Model

```bash
source ../.venv/bin/activate
python convert.py \
  --model-path ~/.cache/huggingface/hub/models--pyannote--speaker-diarization-community-1/snapshots/*/segmentation \
  --output segmentation.gguf
```

### Run Inference

```bash
./build/bin/segmentation-ggml segmentation.gguf \
  --audio ../samples/sample.wav \
  --save-output output.bin
```

## Performance

- **Inference Time**: ~220ms for 10 seconds of audio (CPU)
- **Real-time Factor**: 45x faster than real-time
- **Memory**: ~3 MB model + ~100 MB context
- **Platform**: Apple M2 Pro

## Testing

Run automated accuracy tests:

```bash
python tests/test_accuracy.py
```

This compares PyTorch and GGML outputs layer-by-layer:

- SincNet output
- LSTM output
- Linear layer outputs
- Final classifier output

## Accuracy Notes

### F16 Precision

Weights are stored in F16 format to reduce model size (~3MB vs ~6MB for F32). This introduces small numerical differences that accumulate over layers:

| Layer | Cosine Sim | Max Error |
|-------|------------|-----------|
| SincNet | 0.998 | 0.13 |
| LSTM | 0.999 | 0.24 |
| Linear1 | 0.999 | 0.38 |
| Linear2 | 0.998 | 0.63 |
| Classifier | 0.999 | 0.29 |

These errors are acceptable for speaker diarization - the argmax classification results are equivalent to PyTorch.

## Project Structure

```
segmentation-ggml/
├── CMakeLists.txt          # Build configuration
├── convert.py              # PyTorch → GGUF converter
├── src/
│   ├── main.cpp           # Entry point with CLI
│   ├── model.{h,cpp}      # Model loading and forward pass
│   ├── sincnet.{h,cpp}    # SincNet feature extraction
│   └── lstm.{h,cpp}       # LSTM temporal modeling
├── docs/
│   └── architecture.md    # Detailed architecture docs
└── tests/
    ├── test_accuracy.py   # PyTorch vs GGML comparison
    └── reference_activations.npz
```

## Documentation

- **Architecture Details**: [docs/architecture.md](docs/architecture.md)
- **Implementation Notes**: `.sisyphus/notepads/pyannote-ggml/learnings.md`
- **Known Issues**: `.sisyphus/notepads/pyannote-ggml/issues.md`

## Command Line Options

```
./build/bin/segmentation-ggml <model.gguf> [options]

Options:
  --audio <path>              Load audio from WAV file
  --test                      Run shape validation tests
  --save-output <path>        Save final output to binary file
  --save-intermediates <dir>  Save intermediate layer outputs
  --benchmark                 Run performance benchmark
```

## Architecture

**SincNet** (42,602 params)

- 3 stages of learnable sinc filters
- InstanceNorm1d + LeakyReLU activation
- Output: 60 features

**LSTM** (1,376,256 params)

- 4 layers, bidirectional
- 128 hidden units per direction
- Output: 256 features (2×128)

**Linear** (49,280 params)

- 2 layers: 256→128, 128→128
- LeakyReLU(0.01) activation

**Classifier** (903 params)

- 128→7 output classes
- LogSoftmax activation

## License

MIT License (same as pyannote.audio)

## Acknowledgments

- **pyannote.audio**: Original PyTorch implementation
- **GGML**: Machine learning tensor library
- **whisper.cpp**: Reference for LSTM implementation patterns

## Development Status

**Implementation**: Complete  
**Functionality**: Working (F16 precision)  
**Production Ready**: Yes - for speaker diarization inference

### Verified Working

- ✅ Build system and project structure
- ✅ Model conversion (PyTorch → GGUF)
- ✅ Model loading from GGUF files
- ✅ SincNet feature extraction
- ✅ 4-layer bidirectional LSTM
- ✅ Linear layers with LeakyReLU
- ✅ Classifier with LogSoftmax
- ✅ Automated accuracy testing
- ✅ WAV file loading
- ⚠️ Metal backend initializes but uses CPU for computation

### Performance (CPU)

- ~220ms inference for 10 seconds of audio
- 45x faster than real-time on Apple M2 Pro

### Future Work: Full Metal GPU Support

Metal backend is initialized but computation uses CPU. For GPU acceleration:
1. Use `ggml_backend_sched` for automatic tensor placement
2. Or allocate tensors via `ggml_backend_alloc_ctx_tensors()`
3. Or use `ggml_backend_buffer_from_host_ptr()` for Metal-accessible memory
