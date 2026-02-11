# pyannote-cpp-node

Node.js native bindings for real-time speaker diarization

![Platform](https://img.shields.io/badge/platform-macOS-lightgrey)
![Node](https://img.shields.io/badge/node-%3E%3D18-brightgreen)

## Overview

`pyannote-cpp-node` provides Node.js bindings to a high-performance C++ port of the [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) pipeline. It achieves **39x real-time** performance on Apple Silicon by leveraging CoreML acceleration (Neural Engine + GPU) for neural network inference and optimized C++ implementations of clustering algorithms.

The library supports two modes:

- **Offline diarization**: Process an entire audio file at once and receive speaker-labeled segments
- **Streaming diarization**: Process audio incrementally in real-time, receive voice activity detection (VAD) as audio arrives, and trigger speaker clustering on demand

All heavy operations are asynchronous and run on libuv worker threads, ensuring the Node.js event loop remains responsive.

## Features

- **Offline diarization** — Process full audio files and get speaker-labeled segments
- **Streaming diarization** — Push audio incrementally, receive real-time VAD, recluster on demand
- **Async/await API** — All heavy operations return Promises and run on worker threads
- **CoreML acceleration** — Neural networks run on Apple's Neural Engine, GPU, and CPU
- **TypeScript-first** — Full type definitions included
- **Zero-copy audio input** — Direct `Float32Array` input for maximum efficiency
- **Byte-identical output** — Streaming finalize produces identical results to offline pipeline

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3/M4) or Intel x64
- **Node.js** >= 18
- **Model files**:
  - Segmentation GGUF model (`segmentation.gguf`)
  - Embedding GGUF model (`embedding.gguf`)
  - PLDA GGUF model (`plda.gguf`)
  - Segmentation CoreML model package (`segmentation.mlpackage/`)
  - Embedding CoreML model package (`embedding.mlpackage/`)

Model files can be obtained by converting the original PyTorch models using the conversion scripts in the parent repository.

## Installation

```bash
npm install pyannote-cpp-node
```

Or with pnpm:

```bash
pnpm add pyannote-cpp-node
```

The package uses `optionalDependencies` to automatically install the correct platform-specific native addon (`@pyannote-cpp-node/darwin-arm64` or `@pyannote-cpp-node/darwin-x64`).

## Quick Start

```typescript
import { Pyannote } from 'pyannote-cpp-node';
import { readFileSync } from 'node:fs';

// Load model (validates all paths exist)
const model = await Pyannote.load({
  segModelPath: './models/segmentation.gguf',
  embModelPath: './models/embedding.gguf',
  pldaPath: './models/plda.gguf',
  coremlPath: './models/embedding.mlpackage',
  segCoremlPath: './models/segmentation.mlpackage',
});

// Load audio (16kHz mono Float32Array - see "Audio Format Requirements")
const audio = loadWavFile('./audio.wav');

// Run diarization
const result = await model.diarize(audio);

// Print results
for (const segment of result.segments) {
  console.log(
    `[${segment.start.toFixed(2)}s - ${(segment.start + segment.duration).toFixed(2)}s] ${segment.speaker}`
  );
}

// Clean up
model.close();
```

## API Reference

### `Pyannote` Class

The main entry point for loading diarization models.

#### `static async load(config: ModelConfig): Promise<Pyannote>`

Factory method for loading a diarization model. Validates that all model paths exist before initializing. CoreML model compilation happens synchronously during initialization and is typically fast.

**Parameters:**
- `config: ModelConfig` — Configuration object with paths to all required model files

**Returns:** `Promise<Pyannote>` — Initialized model instance

**Throws:**
- `Error` if any model path does not exist or is invalid

**Example:**
```typescript
const model = await Pyannote.load({
  segModelPath: './models/segmentation.gguf',
  embModelPath: './models/embedding.gguf',
  pldaPath: './models/plda.gguf',
  coremlPath: './models/embedding.mlpackage',
  segCoremlPath: './models/segmentation.mlpackage',
});
```

#### `async diarize(audio: Float32Array): Promise<DiarizationResult>`

Performs offline diarization on the entire audio file. Audio must be 16kHz mono in `Float32Array` format with values in the range [-1.0, 1.0].

Internally, this method uses the streaming API: it initializes a streaming session, pushes all audio in 1-second chunks, calls finalize, and cleans up. The operation runs on a worker thread and is non-blocking.

**Parameters:**
- `audio: Float32Array` — Audio samples (16kHz mono, values in [-1.0, 1.0])

**Returns:** `Promise<DiarizationResult>` — Diarization result with speaker-labeled segments sorted by start time

**Throws:**
- `Error` if model is closed
- `TypeError` if audio is not a `Float32Array`
- `Error` if audio is empty

**Example:**
```typescript
const result = await model.diarize(audio);
console.log(`Detected ${result.segments.length} segments`);
```

#### `createStreamingSession(): StreamingSession`

Creates a new independent streaming session. Each session maintains its own internal state and can be used to process audio incrementally.

**Returns:** `StreamingSession` — New streaming session instance

**Throws:**
- `Error` if model is closed

**Example:**
```typescript
const session = model.createStreamingSession();
```

#### `close(): void`

Releases all native resources associated with the model. This method is idempotent and safe to call multiple times.

Once closed, the model cannot be used for diarization or creating new streaming sessions. Existing streaming sessions should be closed before closing the model.

**Example:**
```typescript
model.close();
console.log(model.isClosed); // true
```

#### `get isClosed: boolean`

Indicates whether the model has been closed.

**Returns:** `boolean` — `true` if the model is closed, `false` otherwise

### `StreamingSession` Class

Handles incremental audio processing for real-time diarization.

#### `async push(audio: Float32Array): Promise<VADChunk[]>`

Pushes audio samples to the streaming session. Audio must be 16kHz mono `Float32Array`. Typically, push 1 second of audio (16,000 samples) at a time.

The first chunk requires 10 seconds of accumulated audio to produce output (the segmentation model uses a 10-second window). After that, each subsequent push returns approximately one `VADChunk` (depending on the 1-second hop size).

The returned VAD chunks contain frame-level voice activity (OR of all speakers) for the newly processed 10-second windows.

**Parameters:**
- `audio: Float32Array` — Audio samples (16kHz mono, values in [-1.0, 1.0])

**Returns:** `Promise<VADChunk[]>` — Array of VAD chunks (empty until 10 seconds accumulated)

**Throws:**
- `Error` if session is closed
- `TypeError` if audio is not a `Float32Array`

**Example:**
```typescript
const vadChunks = await session.push(audioChunk);
for (const chunk of vadChunks) {
  console.log(`VAD chunk ${chunk.chunkIndex}: ${chunk.numFrames} frames`);
}
```

#### `async recluster(): Promise<DiarizationResult>`

Triggers full clustering on all accumulated audio data. This runs the complete diarization pipeline (embedding extraction → PLDA scoring → hierarchical clustering → VBx refinement → speaker assignment) and returns speaker-labeled segments with global speaker IDs.

**Warning:** This method mutates the internal session state. Specifically, it replaces the internal embedding and chunk index arrays with filtered versions (excluding silent speakers). Calling `push` after `recluster` may produce unexpected results. Use `recluster` sparingly (e.g., every 30 seconds for live progress updates) or only call `finalize` when the stream ends.

The operation runs on a worker thread and is non-blocking.

**Returns:** `Promise<DiarizationResult>` — Complete diarization result with global speaker labels

**Throws:**
- `Error` if session is closed

**Example:**
```typescript
// Trigger intermediate clustering after accumulating data
const intermediateResult = await session.recluster();
console.log(`Current speaker count: ${new Set(intermediateResult.segments.map(s => s.speaker)).size}`);
```

#### `async finalize(): Promise<DiarizationResult>`

Processes any remaining audio (zero-padding partial chunks to match the offline pipeline's chunk count formula), then performs final clustering. This method produces byte-identical output to the offline `diarize()` method when given the same input audio.

Call this method when the audio stream has ended to get the final diarization result.

The operation runs on a worker thread and is non-blocking.

**Returns:** `Promise<DiarizationResult>` — Final diarization result

**Throws:**
- `Error` if session is closed

**Example:**
```typescript
const finalResult = await session.finalize();
console.log(`Final result: ${finalResult.segments.length} segments`);
```

#### `close(): void`

Releases all native resources associated with the streaming session. This method is idempotent and safe to call multiple times.

**Example:**
```typescript
session.close();
```

#### `get isClosed: boolean`

Indicates whether the session has been closed.

**Returns:** `boolean` — `true` if the session is closed, `false` otherwise

### Types

#### `ModelConfig`

Configuration object for loading diarization models.

```typescript
interface ModelConfig {
  segModelPath: string;    // Path to segmentation GGUF model file
  embModelPath: string;    // Path to embedding GGUF model file
  pldaPath: string;        // Path to PLDA GGUF model file
  coremlPath: string;      // Path to embedding CoreML .mlpackage directory
  segCoremlPath: string;   // Path to segmentation CoreML .mlpackage directory
}
```

#### `VADChunk`

Voice activity detection result for a single 10-second audio chunk.

```typescript
interface VADChunk {
  chunkIndex: number;      // Zero-based chunk number (increments every 1 second)
  startTime: number;       // Absolute start time in seconds (chunkIndex * 1.0)
  duration: number;        // Always 10.0 (chunk window size)
  numFrames: number;       // Always 589 (segmentation model output frames)
  vad: Float32Array;       // [589] frame-level voice activity: 1.0 if any speaker active, 0.0 otherwise
}
```

The `vad` array contains 589 frames, each representing approximately 17ms of audio. A value of 1.0 indicates speech activity (any speaker), 0.0 indicates silence.

#### `Segment`

A contiguous speech segment with speaker label.

```typescript
interface Segment {
  start: number;           // Start time in seconds
  duration: number;        // Duration in seconds
  speaker: string;         // Speaker label (e.g., "SPEAKER_00", "SPEAKER_01", ...)
}
```

#### `DiarizationResult`

Complete diarization output with speaker-labeled segments.

```typescript
interface DiarizationResult {
  segments: Segment[];     // Array of segments, sorted by start time
}
```

## Usage Examples

### Example 1: Offline Diarization

Process an entire audio file and print a timeline of speaker segments.

```typescript
import { Pyannote } from 'pyannote-cpp-node';
import { readFileSync } from 'node:fs';

// Helper to load 16-bit PCM WAV and convert to Float32Array
function loadWavFile(filePath: string): Float32Array {
  const buffer = readFileSync(filePath);
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

  // Find data chunk
  let offset = 12; // Skip RIFF header
  while (offset < view.byteLength - 8) {
    const chunkId = String.fromCharCode(
      view.getUint8(offset),
      view.getUint8(offset + 1),
      view.getUint8(offset + 2),
      view.getUint8(offset + 3)
    );
    const chunkSize = view.getUint32(offset + 4, true);
    offset += 8;

    if (chunkId === 'data') {
      // Convert Int16 PCM to Float32 by dividing by 32768
      const numSamples = chunkSize / 2;
      const float32 = new Float32Array(numSamples);
      for (let i = 0; i < numSamples; i++) {
        float32[i] = view.getInt16(offset + i * 2, true) / 32768.0;
      }
      return float32;
    }

    offset += chunkSize;
    if (chunkSize % 2 !== 0) offset++; // Align to word boundary
  }

  throw new Error('No data chunk found in WAV file');
}

async function main() {
  // Load model
  const model = await Pyannote.load({
    segModelPath: './models/segmentation.gguf',
    embModelPath: './models/embedding.gguf',
    pldaPath: './models/plda.gguf',
    coremlPath: './models/embedding.mlpackage',
    segCoremlPath: './models/segmentation.mlpackage',
  });

  // Load audio
  const audio = loadWavFile('./audio.wav');
  console.log(`Loaded ${audio.length} samples (${(audio.length / 16000).toFixed(1)}s)`);

  // Diarize
  const result = await model.diarize(audio);

  // Print timeline
  console.log(`\nDetected ${result.segments.length} segments:`);
  for (const segment of result.segments) {
    const startTime = segment.start.toFixed(2);
    const endTime = (segment.start + segment.duration).toFixed(2);
    console.log(`[${startTime}s - ${endTime}s] ${segment.speaker}`);
  }

  // Count speakers
  const speakers = new Set(result.segments.map(s => s.speaker));
  console.log(`\nTotal speakers: ${speakers.size}`);

  model.close();
}

main();
```

### Example 2: Streaming Diarization

Process audio incrementally in 1-second chunks, displaying real-time VAD.

```typescript
import { Pyannote } from 'pyannote-cpp-node';

async function streamingDiarization() {
  const model = await Pyannote.load({
    segModelPath: './models/segmentation.gguf',
    embModelPath: './models/embedding.gguf',
    pldaPath: './models/plda.gguf',
    coremlPath: './models/embedding.mlpackage',
    segCoremlPath: './models/segmentation.mlpackage',
  });

  const session = model.createStreamingSession();

  // Load full audio file
  const audio = loadWavFile('./audio.wav');

  // Push audio in 1-second chunks (16,000 samples)
  const CHUNK_SIZE = 16000;
  let totalChunks = 0;

  for (let offset = 0; offset < audio.length; offset += CHUNK_SIZE) {
    const end = Math.min(offset + CHUNK_SIZE, audio.length);
    const chunk = audio.slice(offset, end);

    const vadChunks = await session.push(chunk);

    // VAD chunks are returned after first 10 seconds
    for (const vad of vadChunks) {
      // Count active frames (speech detected)
      const activeFrames = vad.vad.filter(v => v > 0.5).length;
      const speechRatio = (activeFrames / vad.numFrames * 100).toFixed(1);
      
      console.log(
        `Chunk ${vad.chunkIndex}: ${vad.startTime.toFixed(1)}s - ${(vad.startTime + vad.duration).toFixed(1)}s | ` +
        `Speech: ${speechRatio}%`
      );
      totalChunks++;
    }
  }

  console.log(`\nProcessed ${totalChunks} chunks`);

  // Get final diarization result
  console.log('\nFinalizing...');
  const result = await session.finalize();

  console.log(`\nFinal result: ${result.segments.length} segments`);
  for (const segment of result.segments) {
    console.log(
      `[${segment.start.toFixed(2)}s - ${(segment.start + segment.duration).toFixed(2)}s] ${segment.speaker}`
    );
  }

  session.close();
  model.close();
}

streamingDiarization();
```

### Example 3: On-Demand Reclustering

Push audio and trigger reclustering every 30 seconds to get intermediate results.

```typescript
import { Pyannote } from 'pyannote-cpp-node';

async function reclusteringExample() {
  const model = await Pyannote.load({
    segModelPath: './models/segmentation.gguf',
    embModelPath: './models/embedding.gguf',
    pldaPath: './models/plda.gguf',
    coremlPath: './models/embedding.mlpackage',
    segCoremlPath: './models/segmentation.mlpackage',
  });

  const session = model.createStreamingSession();
  const audio = loadWavFile('./audio.wav');

  const CHUNK_SIZE = 16000; // 1 second
  const RECLUSTER_INTERVAL = 30; // Recluster every 30 seconds

  let secondsProcessed = 0;

  for (let offset = 0; offset < audio.length; offset += CHUNK_SIZE) {
    const end = Math.min(offset + CHUNK_SIZE, audio.length);
    const chunk = audio.slice(offset, end);

    await session.push(chunk);
    secondsProcessed++;

    // Recluster every 30 seconds
    if (secondsProcessed % RECLUSTER_INTERVAL === 0) {
      console.log(`\n--- Reclustering at ${secondsProcessed}s ---`);
      const intermediateResult = await session.recluster();
      
      const speakers = new Set(intermediateResult.segments.map(s => s.speaker));
      console.log(`Current speakers detected: ${speakers.size}`);
      console.log(`Current segments: ${intermediateResult.segments.length}`);
    }
  }

  // Final result
  console.log('\n--- Final result ---');
  const finalResult = await session.finalize();
  const speakers = new Set(finalResult.segments.map(s => s.speaker));
  console.log(`Total speakers: ${speakers.size}`);
  console.log(`Total segments: ${finalResult.segments.length}`);

  session.close();
  model.close();
}

reclusteringExample();
```

### Example 4: Generating RTTM Output

Format diarization results into standard RTTM (Rich Transcription Time Marked) format.

```typescript
import { Pyannote, type DiarizationResult } from 'pyannote-cpp-node';
import { writeFileSync } from 'node:fs';

function toRTTM(result: DiarizationResult, filename: string = 'audio'): string {
  const lines = result.segments.map(segment => {
    // RTTM format: SPEAKER <file> <chnl> <tbeg> <tdur> <ortho> <stype> <name> <conf> <slat>
    return [
      'SPEAKER',
      filename,
      '1',
      segment.start.toFixed(3),
      segment.duration.toFixed(3),
      '<NA>',
      '<NA>',
      segment.speaker,
      '<NA>',
      '<NA>',
    ].join(' ');
  });

  return lines.join('\n') + '\n';
}

async function generateRTTM() {
  const model = await Pyannote.load({
    segModelPath: './models/segmentation.gguf',
    embModelPath: './models/embedding.gguf',
    pldaPath: './models/plda.gguf',
    coremlPath: './models/embedding.mlpackage',
    segCoremlPath: './models/segmentation.mlpackage',
  });

  const audio = loadWavFile('./audio.wav');
  const result = await model.diarize(audio);

  // Generate RTTM
  const rttm = toRTTM(result, 'audio');
  
  // Write to file
  writeFileSync('./output.rttm', rttm);
  console.log('RTTM file written to output.rttm');

  // Also print to console
  console.log('\nRTTM output:');
  console.log(rttm);

  model.close();
}

generateRTTM();
```

## Architecture

The diarization pipeline consists of four main stages:

### 1. Segmentation (SincNet + BiLSTM)

The segmentation model processes 10-second audio windows and outputs 7-class powerset logits for 589 frames (approximately one frame every 17ms). The model architecture:

- **SincNet**: Learnable sinc filter bank for feature extraction
- **4-layer BiLSTM**: Bidirectional long short-term memory layers
- **Linear classifier**: Projects to 7 powerset classes with log-softmax

The 7 powerset classes represent all possible combinations of up to 3 simultaneous speakers:
- Class 0: silence (no speakers)
- Classes 1-3: single speakers
- Classes 4-6: speaker overlaps

### 2. Powerset Decoding

Converts the 7-class powerset predictions into binary speaker activity for 3 local speakers per chunk. Each frame is decoded to indicate which of the 3 local speaker "slots" are active.

### 3. Embedding Extraction (WeSpeaker ResNet34)

For each active speaker in each chunk, the embedding model extracts a 256-dimensional speaker vector:

- **Mel filterbank**: 80-bin log-mel spectrogram features
- **ResNet34**: Deep residual network for speaker representation
- **Output**: 256-dimensional L2-normalized embedding

Silent speakers receive NaN embeddings, which are filtered before clustering.

### 4. Clustering (PLDA + AHC + VBx)

The final stage maps local speaker labels to global speaker identities:

- **PLDA transformation**: Probabilistic Linear Discriminant Analysis projects embeddings from 256 to 128 dimensions
- **Agglomerative Hierarchical Clustering (AHC)**: fastcluster implementation with O(n²) complexity, using centroid linkage and a distance threshold of 0.6
- **VBx refinement**: Variational Bayes diarization with parameters FA=0.07, FB=0.8, maximum 20 iterations

The clustering stage computes speaker centroids and assigns each embedding to the closest centroid while respecting the constraint that two local speakers in the same chunk cannot map to the same global speaker.

### CoreML Acceleration

Both neural networks run on Apple's CoreML framework, which automatically distributes computation across:

- **Neural Engine**: Dedicated ML accelerator on Apple Silicon
- **GPU**: Metal-accelerated operations
- **CPU**: Fallback for unsupported operations

CoreML models use Float16 computation for optimal performance while maintaining accuracy within acceptable bounds (cosine similarity > 0.999 vs Float32).

### Streaming Architecture

The streaming API uses a sliding 10-second window with a 1-second hop (9 seconds of overlap between consecutive chunks). Three data stores maintain the state:

- **`audio_buffer`**: Sliding window (~10s, ~640 KB for 1 hour) — old samples are discarded
- **`embeddings`**: Grows forever (~11 MB for 1 hour) — stores 3 × 256-dim vectors per chunk (NaN for silent speakers)
- **`binarized`**: Grows forever (~25 MB for 1 hour) — stores 589 × 3 binary activity masks per chunk

During reclustering, all accumulated embeddings are used to compute soft cluster assignments, and all binarized segmentations are used to reconstruct the global timeline. This is why the `embeddings` and `binarized` arrays must persist for the entire session.

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| SAMPLE_RATE | 16000 Hz | Audio sample rate |
| CHUNK_SAMPLES | 160000 | 10-second window size |
| STEP_SAMPLES | 16000 | 1-second hop between chunks |
| FRAMES_PER_CHUNK | 589 | Segmentation output frames |
| NUM_LOCAL_SPEAKERS | 3 | Maximum speakers per chunk |
| EMBEDDING_DIM | 256 | Speaker embedding dimension |
| FBANK_NUM_BINS | 80 | Mel filterbank bins |

## Audio Format Requirements

The library expects raw PCM audio in a specific format:

- **Sample rate**: 16000 Hz (16 kHz) — **required**
- **Channels**: Mono (single channel) — **required**
- **Format**: `Float32Array` with values in the range **[-1.0, 1.0]**

The library does **not** handle audio decoding. You must provide raw PCM samples.

### Loading Audio Files

For WAV files, you can use the `loadWavFile` function from Example 1, or use third-party libraries:

```bash
npm install node-wav
```

```typescript
import { read } from 'node-wav';
import { readFileSync } from 'node:fs';

const buffer = readFileSync('./audio.wav');
const wav = read(buffer);

// Convert to mono if stereo
const mono = wav.channelData.length > 1
  ? wav.channelData[0].map((v, i) => (v + wav.channelData[1][i]) / 2)
  : wav.channelData[0];

// Resample to 16kHz if needed (using a resampling library)
// ...

const audio = new Float32Array(mono);
```

For other audio formats (MP3, M4A, etc.), use ffmpeg to convert to 16kHz mono WAV first:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -f f32le -acodec pcm_f32le - | \
  node process.js
```

## Important Notes and Caveats

### Platform Limitations

- **macOS only**: The library requires CoreML for neural network inference. There is currently no fallback implementation for other platforms.
- **No Linux/Windows support**: CoreML is exclusive to Apple platforms.

### `recluster()` Mutates State

The `recluster()` method overwrites the internal session state, specifically replacing the `embeddings` and chunk index arrays with filtered versions (excluding NaN embeddings from silent speakers). This means:

- Calling `push()` after `recluster()` may produce incorrect results
- Subsequent `recluster()` calls may not work as expected
- The data structure assumes the original unfiltered layout (3 embeddings per chunk)

**Best practice**: Use `recluster()` sparingly for live progress updates (e.g., every 30 seconds), or avoid it entirely and only call `finalize()` when the stream ends.

### Operations Are Serialized

Operations on a streaming session are serialized internally. Do not call `push()` while another `push()`, `recluster()`, or `finalize()` is in progress. Wait for the Promise to resolve before making the next call.

### Resource Management

- **Close sessions before models**: Always close streaming sessions before closing the parent model
- **Idempotent close**: Both `model.close()` and `session.close()` are safe to call multiple times
- **No reuse after close**: Once closed, models and sessions cannot be reused

### Model Loading

- **Path validation**: `Pyannote.load()` validates that all paths exist using `fs.accessSync()` before initialization
- **CoreML compilation**: The CoreML framework compiles `.mlpackage` models internally on first load (typically fast, ~100ms)
- **No explicit loading step**: Model weights are loaded synchronously in the constructor

### Threading Model

All heavy operations (`diarize`, `push`, `recluster`, `finalize`) run on libuv worker threads and never block the Node.js event loop. However, the operations do hold native locks internally, so concurrent operations on the same session are serialized.

### Memory Usage

For a 1-hour audio file:
- `audio_buffer`: ~640 KB (sliding window)
- `embeddings`: ~11 MB (grows throughout session)
- `binarized`: ~25 MB (grows throughout session)
- CoreML models: ~50 MB (loaded once per model)

Total memory footprint: approximately 100 MB for a 1-hour streaming session.

## Performance

Measured on Apple M2 Pro with 16 GB RAM:

| Component | Time per Chunk | Notes |
|-----------|----------------|-------|
| Segmentation (CoreML) | ~12ms | 10-second audio window, 589 frames |
| Embedding (CoreML) | ~13ms | Per speaker per chunk (up to 3 speakers) |
| AHC Clustering | ~0.8s | 3000 embeddings (1000 chunks) |
| VBx Refinement | ~1.2s | 20 iterations, 3000 embeddings |
| **Full Pipeline (offline)** | **39x real-time** | 45-minute audio processed in 70 seconds |

### Streaming Performance

- **First chunk latency**: 10 seconds (requires full window)
- **Incremental latency**: ~30ms per 1-second push (after first chunk)
- **Recluster latency**: ~2 seconds for 30 minutes of audio (~1800 embeddings)

Streaming mode has higher per-chunk overhead due to the incremental nature but enables real-time applications.

## Supported Platforms

| Platform | Architecture | Status |
|----------|--------------|--------|
| macOS | arm64 (Apple Silicon) | ✅ Supported |
| macOS | x64 (Intel) | 🔜 Planned |
| Linux | any | ❌ Not supported (CoreML unavailable) |
| Windows | any | ❌ Not supported (CoreML unavailable) |

Intel macOS support is planned but not yet available. The CoreML dependency makes cross-platform support challenging without alternative inference backends.

## License

MIT

---

For issues, feature requests, or contributions, please visit the [GitHub repository](https://github.com/predict-woo/pyannote-ggml).
