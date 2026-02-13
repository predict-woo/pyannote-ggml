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
cd models/segmentation-ggml
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
  ../models/segmentation-ggml/segmentation.gguf \
  ../models/embedding-ggml/embedding.gguf \
  ../samples/sample.wav \
  --plda plda.gguf \
  --coreml ../models/embedding-ggml/embedding.mlpackage \
  --seg-coreml ../models/segmentation-ggml/segmentation.mlpackage \
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
├── models/                    # Neural network model implementations
│   ├── segmentation-ggml/     # Segmentation model (SincNet + BiLSTM)
│   │   ├── src/               # C++ implementation
│   │   ├── convert.py         # PyTorch → GGUF converter
│   │   └── convert_coreml.py  # PyTorch → CoreML converter
│   │
│   └── embedding-ggml/        # Speaker embedding model (ResNet34)
│       ├── src/               # C++ implementation
│       └── convert_coreml.py  # PyTorch → CoreML converter
│
├── diarization-ggml/      # Full pipeline orchestration
│   ├── src/
│   │   ├── diarization.cpp   # Pipeline orchestration
│   │   ├── powerset.cpp      # Powerset → multilabel
│   │   ├── plda.cpp          # PLDA scoring
│   │   ├── vbx.cpp           # VBx clustering
│   │   └── clustering.cpp    # fastcluster AHC
│   ├── plda.gguf             # Pre-computed PLDA model (GGUF)
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

## Streaming API

The streaming API processes audio incrementally (typically 1 second at a time), producing VAD (voice activity detection) chunks in real-time and full diarization results on demand. When finalized, it produces byte-identical output to the offline pipeline.

### API Functions

**`streaming_init(config)`** → `StreamingState*`

Loads CoreML segmentation + embedding models and PLDA. Returns `nullptr` on failure.

**`streaming_push(state, samples, num_samples)`** → `vector<VADChunk>`

Appends audio to buffer, processes any new complete chunks, returns combined VAD activity (OR of 3 local speakers) for each newly processed chunk. First chunk requires 10s of audio; subsequent chunks are produced every 1s.

**`streaming_recluster(state)`** → `DiarizationResult`

User-triggered. Runs full clustering pipeline (filter → PLDA → AHC → VBx → centroid → assignment → reconstruction) on all accumulated data. Returns timeline segments with global speaker labels.

**`streaming_finalize(state)`** → `DiarizationResult`

Processes any remaining partial audio (zero-padded to match offline chunk count formula), then reclusters. Produces byte-identical output to the offline pipeline.

**`streaming_free(state)`**

Frees all resources.

### Data Structures

```cpp
struct VADChunk {
    int chunk_index;        // 0-based chunk number
    double start_time;      // chunk_index * 1.0 seconds
    double duration;        // always 10.0s
    int num_frames;         // always 589
    std::vector<float> vad; // [589] — 1.0 if ANY speaker active, 0.0 otherwise
};

struct DiarizationResult {
    struct Segment {
        double start;
        double duration;
        std::string speaker;  // "SPEAKER_00", "SPEAKER_01", etc.
    };
    std::vector<Segment> segments;  // sorted by start time
};
```

### Three Data Stores

| Store | Lifetime | Size per chunk | 1 hour | Purpose |
|---|---|---|---|---|
| `audio_buffer` | Sliding window (~10s) | trimmed to ~160K samples | ~640 KB | Raw audio for segmentation. Old samples discarded via `samples_trimmed` offset. |
| `embeddings` | Grows forever | 3 × 256 × 4B = 3 KB | ~11 MB | 3 speaker embeddings per chunk (NaN for silent). Needed for soft_clusters against ALL embeddings during recluster. |
| `binarized` | Grows forever | 589 × 3 × 4B = 7 KB | ~25 MB | Per-frame, per-speaker binary activity. Needed for: (1) filtering silent embeddings, (2) marking inactive speakers, (3) speaker_count aggregation, (4) reconstructing the global timeline from local speaker assignments. |

**Why embeddings and binarized must grow forever:** Reclustering computes `soft_clusters` against ALL embeddings (not just filtered) and rebuilds the full timeline from ALL binarized segmentations. Without the full history, you can't reassign earlier chunks when new speakers appear.

### Data Flow

```
Push 1s audio ──→ audio_buffer (sliding ~10s)
                         │
                    process_one_chunk()
                    ┌────┴────┐
                    ▼         ▼
              Seg CoreML   Emb CoreML (×3 speakers)
              logits→binarized   fbank→embedding
                    │         │
                    ▼         ▼
              ┌──────────┐ ┌──────────────┐
              │binarized │ │ embeddings   │
              │(forever) │ │ (forever)    │
              └────┬─────┘ └──────┬───────┘
                   │              │
                   └──────┬───────┘
                          │  recluster() / finalize()
                          ▼
                filter → PLDA → AHC → VBx → centroids
                → soft_clusters (ALL embeddings)
                → constrained_argmax → reconstruct
                          │
                          ▼
                   DiarizationResult
                   [{start, duration, speaker}, ...]
```

### Constants

| Constant | Value | Meaning |
|---|---|---|
| SAMPLE_RATE | 16000 | Hz |
| CHUNK_SAMPLES | 160000 | 10s window |
| STEP_SAMPLES | 16000 | 1s hop (9s overlap) |
| FRAMES_PER_CHUNK | 589 | Segmentation output frames per chunk |
| NUM_LOCAL_SPEAKERS | 3 | Max speakers per chunk (powerset) |
| EMBEDDING_DIM | 256 | Speaker embedding dimensionality |

### Usage Example

```bash
cd diarization-ggml
./build/bin/streaming_test \
  ../samples/sample.wav \
  --plda plda.gguf \
  --coreml ../models/embedding-ggml/embedding.mlpackage \
  --seg-coreml ../models/segmentation-ggml/segmentation.mlpackage \
  -o output.rttm
```

### Limitations

- Currently requires CoreML (no GGML-only fallback for streaming)
- Push→recluster→push→recluster cycles work correctly. Recluster uses local variables for filtering and preserves the unfiltered state for subsequent pushes.

## Transcription + Diarization

This integrated pipeline combines pyannote streaming diarization with Whisper transcription to produce speaker-labeled, word-level transcripts in real time.

### Build Instructions

```bash
cd diarization-ggml
cmake -B build -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON -DWHISPER_COREML=ON
cmake --build build
```

### Model Requirements

- Segmentation GGUF model (`segmentation.gguf`)
- Embedding GGUF model (`embedding.gguf`)
- PLDA model (`plda.gguf`)
- Whisper GGUF model (for example `ggml-large-v3-turbo.bin` or `ggml-base.en.bin`)
- CoreML models: segmentation `.mlpackage`, embedding `.mlpackage`
- Optional: Silero VAD model (`ggml-silero-v6.2.0.bin`)

### CLI Usage

```bash
./build/bin/transcribe ../samples/sample.wav \
  --seg-model ../models/segmentation-ggml/segmentation.gguf \
  --emb-model ../models/embedding-ggml/embedding.gguf \
  --whisper-model path/to/ggml-large-v3-turbo.bin \
  --plda plda.gguf \
  --seg-coreml ../models/segmentation-ggml/segmentation.mlpackage \
  --emb-coreml ../models/embedding-ggml/embedding.mlpackage \
  --language en
```

### JSON Output Format

```json
{
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.5,
      "duration": 2.1,
      "words": [
        {"text": "Hello", "start": 0.5, "end": 0.8},
        {"text": "world", "start": 0.9, "end": 1.2}
      ]
    }
  ]
}
```

### C++ API Usage Example

```cpp
#include "pipeline.h"

void on_result(const std::vector<AlignedSegment>& segments, void* user_data) {
    for (const auto& seg : segments) {
        printf("[%s] %.2f-%.2f:", seg.speaker.c_str(), seg.start, seg.start + seg.duration);
        for (const auto& w : seg.words) printf(" %s", w.text.c_str());
        printf("\n");
    }
}

PipelineConfig config{};
config.diarization.seg_model_path = "segmentation.gguf";
config.diarization.emb_model_path = "embedding.gguf";
config.diarization.plda_path = "plda.gguf";
config.diarization.seg_coreml_path = "segmentation.mlpackage";
config.diarization.coreml_path = "embedding.mlpackage";
config.transcriber.whisper_model_path = "ggml-large-v3-turbo.bin";
config.transcriber.language = "en";
config.vad_model_path = nullptr;

PipelineState* state = pipeline_init(config, on_result, nullptr);
// Push audio in 1-second chunks
pipeline_push(state, samples, 16000);
pipeline_finalize(state);
pipeline_free(state);
```

### Pipeline Architecture

1. **VAD Silence Filter**: Compresses long silence gaps to at most 2 seconds while preserving smooth speech transitions.
2. **Audio Buffer**: Maintains a FIFO of filtered audio with absolute frame tracking for timestamp-safe dequeue/range reads.
3. **Segmentation**: Uses pyannote streaming VAD to detect speech-to-silence boundaries and produce segment end timestamps.
4. **Transcription**: Sends buffered audio to Whisper (20s minimum, ~30s cap) on a worker thread and returns word timestamps.
5. **Alignment**: Assigns each Whisper word to a diarized speaker via maximum overlap (WhisperX-style word-speaker matching).
6. **Finalize**: Flushes silence filter, pyannote, and Whisper, then runs final recluster and alignment at end of stream.
7. **Callback**: Emits incremental speaker-labeled word output after each alignment update.

## Testing

```bash
# Segmentation accuracy test
.venv/bin/python3 models/segmentation-ggml/tests/test_accuracy.py

# Full pipeline DER test
./diarization-ggml/build/bin/diarization-ggml \
  models/segmentation-ggml/segmentation.gguf \
  models/embedding-ggml/embedding.gguf \
  samples/sample.wav \
  --plda diarization-ggml/plda.gguf \
  --coreml models/embedding-ggml/embedding.mlpackage \
  --seg-coreml models/segmentation-ggml/segmentation.mlpackage \
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
