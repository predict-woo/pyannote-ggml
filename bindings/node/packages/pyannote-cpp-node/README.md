# pyannote-cpp-node

![Platform](https://img.shields.io/badge/platform-macOS-lightgrey)
![Node](https://img.shields.io/badge/node-%3E%3D18-brightgreen)

Node.js native bindings for integrated Whisper transcription + speaker diarization with speaker-labeled segment output.

## Overview

`pyannote-cpp-node` exposes the integrated C++ pipeline that combines streaming diarization and Whisper transcription into a single API.

Given 16 kHz mono PCM audio (`Float32Array`), it produces cumulative and final transcript segments shaped as:

- speaker label (`SPEAKER_00`, `SPEAKER_01`, ...)
- segment start/duration in seconds
- segment text

The API supports both one-shot processing (`transcribe`) and incremental streaming (`createSession` + `push`/`finalize`). All heavy operations are asynchronous and run on libuv worker threads.

## Features

- Integrated transcription + diarization in one pipeline
- Speaker-labeled transcript segments with sentence-level text
- One-shot and streaming APIs with the same output schema
- Incremental `segments` events plus separate real-time `audio` chunk streaming
- Deterministic output for the same audio/models/config
- CoreML-accelerated inference on macOS
- TypeScript-first API with complete type definitions

## Requirements

- macOS (Apple Silicon or Intel)
- Node.js >= 18
- Model files:
  - Segmentation GGUF (`segModelPath`)
  - Embedding GGUF (`embModelPath`)
  - PLDA GGUF (`pldaPath`)
  - Embedding CoreML `.mlpackage` (`coremlPath`)
  - Segmentation CoreML `.mlpackage` (`segCoremlPath`)
  - Whisper GGUF (`whisperModelPath`)
  - Optional Silero VAD model (`vadModelPath`)

## Installation

```bash
npm install pyannote-cpp-node
```

```bash
pnpm add pyannote-cpp-node
```

The package installs a platform-specific native addon through `optionalDependencies`.

## Quick Start

```typescript
import { Pipeline } from 'pyannote-cpp-node';

const pipeline = await Pipeline.load({
  segModelPath: './models/segmentation.gguf',
  embModelPath: './models/embedding.gguf',
  pldaPath: './models/plda.gguf',
  coremlPath: './models/embedding.mlpackage',
  segCoremlPath: './models/segmentation.mlpackage',
  whisperModelPath: './models/ggml-large-v3-turbo-q5_0.bin',
  language: 'en',
});

const audio = loadAudioAsFloat32Array('./audio-16khz-mono.wav');
const result = await pipeline.transcribe(audio);

for (const segment of result.segments) {
  const end = segment.start + segment.duration;
  console.log(
    `[${segment.speaker}] ${segment.start.toFixed(2)}-${end.toFixed(2)} ${segment.text.trim()}`
  );
}

pipeline.close();
```

## API Reference

### `Pipeline`

```typescript
class Pipeline {
  static async load(config: ModelConfig): Promise<Pipeline>;
  async transcribe(audio: Float32Array): Promise<TranscriptionResult>;
  setLanguage(language: string): void;
  setDecodeOptions(options: DecodeOptions): void;
  createSession(): PipelineSession;
  close(): void;
  get isClosed(): boolean;
}
```

#### `static async load(config: ModelConfig): Promise<Pipeline>`

Validates model paths and initializes native pipeline resources.

#### `async transcribe(audio: Float32Array): Promise<TranscriptionResult>`

Runs one-shot transcription + diarization on the full audio buffer.

#### `setLanguage(language: string): void`

Updates the Whisper decode language for subsequent `transcribe()` calls. This is a convenience shorthand for `setDecodeOptions({ language })`.

#### `setDecodeOptions(options: DecodeOptions): void`

Updates one or more Whisper decode options for subsequent `transcribe()` calls. Only the fields you pass are changed; others retain their current values. See `DecodeOptions` for available fields.

#### `createSession(): PipelineSession`

Creates an independent streaming session for incremental processing.
This method takes no arguments; native segment/audio callbacks are wired internally.

#### `close(): void`

Releases native resources. Safe to call multiple times.

#### `get isClosed(): boolean`

Returns `true` after `close()`.

### `PipelineSession` (extends `EventEmitter`)

```typescript
class PipelineSession extends EventEmitter {
  async push(audio: Float32Array): Promise<boolean[]>;
  setLanguage(language: string): void;
  setDecodeOptions(options: DecodeOptions): void;
  async finalize(): Promise<TranscriptionResult>;
  close(): void;
  get isClosed(): boolean;
  on<K extends keyof PipelineSessionEvents>(
    event: K,
    listener: (...args: PipelineSessionEvents[K]) => void
  ): this;
}
```

```typescript
interface PipelineSessionEvents {
  segments: [segments: AlignedSegment[]];
  audio: [audio: Float32Array];
  error: [error: Error];
}
```

#### `async push(audio: Float32Array): Promise<boolean[]>`

Pushes an arbitrary number of samples into the streaming pipeline.

- Return value is per-frame VAD booleans (`true` = speech, `false` = silence)
- First 10 seconds return an empty array because the pipeline needs a full 10-second window
- Chunk size is flexible; not restricted to 16,000-sample pushes

#### `setLanguage(language: string): void`

Updates the Whisper decode language on the live streaming session. Takes effect on the next Whisper decode run. Thread-safe — the change is pushed to the C++ pipeline immediately.

#### `setDecodeOptions(options: DecodeOptions): void`

Updates one or more Whisper decode options on the live streaming session. Takes effect on the next Whisper decode run. Thread-safe — changes are pushed to the C++ pipeline immediately. Only the fields you pass are changed; others retain their current values.

#### `async finalize(): Promise<TranscriptionResult>`

Flushes all stages, runs final recluster + alignment, and returns the definitive result.

```typescript
type TranscriptionResult = {
  segments: AlignedSegment[];
};
```

#### `close(): void`

Releases native session resources. Safe to call multiple times.

#### `get isClosed(): boolean`

Returns `true` after `close()`.

#### Event: `'segments'`

Emitted after each Whisper transcription result with the latest cumulative aligned output.

```typescript
session.on('segments', (segments: AlignedSegment[]) => {
  // `segments` contains the latest cumulative speaker-labeled transcript
});
```

#### Event: `'audio'`

Emitted in real-time with silence-filtered PCM chunks (`Float32Array`) as the pipeline processes audio.

```typescript
session.on('audio', (chunk: Float32Array) => {
  // `chunk` is silence-filtered audio emitted for streaming consumers
});
```

### Types

```typescript
export interface ModelConfig {
  // === Required Model Paths ===
  /** Path to segmentation GGUF model */
  segModelPath: string;

  /** Path to embedding GGUF model */
  embModelPath: string;

  /** Path to PLDA GGUF model */
  pldaPath: string;

  /** Path to embedding CoreML .mlpackage directory */
  coremlPath: string;

  /** Path to segmentation CoreML .mlpackage directory */
  segCoremlPath: string;

  /** Path to Whisper GGUF model */
  whisperModelPath: string;

  // === Optional Model Paths ===
  /** Path to Silero VAD model (optional, enables silence compression) */
  vadModelPath?: string;

  // === Whisper Context Options (model loading) ===
  /** Enable GPU acceleration (default: true) */
  useGpu?: boolean;

  /** Enable Flash Attention (default: true) */
  flashAttn?: boolean;

  /** GPU device index (default: 0) */
  gpuDevice?: number;

  /**
   * Enable CoreML acceleration for Whisper encoder on macOS (default: false).
   * The CoreML model must be placed next to the GGUF model with naming convention:
   * e.g., ggml-base.en.bin -> ggml-base.en-encoder.mlmodelc/
   */
  useCoreml?: boolean;

  /** Suppress whisper.cpp log output (default: false) */
  noPrints?: boolean;

  // === Whisper Decode Options ===
  /** Number of threads for Whisper inference (default: 4) */
  nThreads?: number;

  /** Language code (e.g., 'en', 'zh'). Omit for auto-detect. (default: 'en') */
  language?: string;

  /** Translate non-English speech to English (default: false) */
  translate?: boolean;

  /** Auto-detect spoken language. Overrides 'language' when true. (default: false) */
  detectLanguage?: boolean;

  // === Sampling ===
  /** Sampling temperature. 0.0 = greedy deterministic. (default: 0.0) */
  temperature?: number;

  /** Temperature increment for fallback retries (default: 0.2) */
  temperatureInc?: number;

  /** Disable temperature fallback. If true, temperatureInc is ignored. (default: false) */
  noFallback?: boolean;

  /** Beam search size. -1 uses greedy decoding. >1 enables beam search. (default: -1) */
  beamSize?: number;

  /** Best-of-N sampling candidates for greedy decoding (default: 5) */
  bestOf?: number;

  // === Thresholds ===
  /** Entropy threshold for decoder fallback (default: 2.4) */
  entropyThold?: number;

  /** Log probability threshold for decoder fallback (default: -1.0) */
  logprobThold?: number;

  /** No-speech probability threshold (default: 0.6) */
  noSpeechThold?: number;

  // === Context ===
  /** Initial prompt text to condition the decoder (default: none) */
  prompt?: string;

  /** Don't use previous segment as context for next segment (default: true) */
  noContext?: boolean;

  /** Suppress blank outputs at the beginning of segments (default: true) */
  suppressBlank?: boolean;

  /** Suppress non-speech tokens (default: false) */
  suppressNst?: boolean;
}

export interface DecodeOptions {
  /** Language code (e.g., 'en', 'zh'). Omit for auto-detect. */
  language?: string;
  /** Translate non-English speech to English */
  translate?: boolean;
  /** Auto-detect spoken language. Overrides 'language' when true. */
  detectLanguage?: boolean;
  /** Number of threads for Whisper inference */
  nThreads?: number;
  /** Sampling temperature. 0.0 = greedy deterministic. */
  temperature?: number;
  /** Temperature increment for fallback retries */
  temperatureInc?: number;
  /** Disable temperature fallback. If true, temperatureInc is ignored. */
  noFallback?: boolean;
  /** Beam search size. -1 uses greedy decoding. >1 enables beam search. */
  beamSize?: number;
  /** Best-of-N sampling candidates for greedy decoding */
  bestOf?: number;
  /** Entropy threshold for decoder fallback */
  entropyThold?: number;
  /** Log probability threshold for decoder fallback */
  logprobThold?: number;
  /** No-speech probability threshold */
  noSpeechThold?: number;
  /** Initial prompt text to condition the decoder */
  prompt?: string;
  /** Don't use previous segment as context for next segment */
  noContext?: boolean;
  /** Suppress blank outputs at the beginning of segments */
  suppressBlank?: boolean;
  /** Suppress non-speech tokens */
  suppressNst?: boolean;
}

export interface AlignedSegment {
  /** Global speaker label (e.g., SPEAKER_00). */
  speaker: string;

  /** Segment start time in seconds. */
  start: number;

  /** Segment duration in seconds. */
  duration: number;

  /** Transcribed text for this segment. */
  text: string;
}

export interface TranscriptionResult {
  /** Full speaker-labeled transcript segments. */
  segments: AlignedSegment[];
}
```

## Usage Examples

### One-shot transcription

```typescript
import { Pipeline } from 'pyannote-cpp-node';

async function runOneShot(audio: Float32Array) {
  const pipeline = await Pipeline.load({
    segModelPath: './models/segmentation.gguf',
    embModelPath: './models/embedding.gguf',
    pldaPath: './models/plda.gguf',
    coremlPath: './models/embedding.mlpackage',
    segCoremlPath: './models/segmentation.mlpackage',
    whisperModelPath: './models/ggml-large-v3-turbo-q5_0.bin',
  });

  const result = await pipeline.transcribe(audio);

  for (const seg of result.segments) {
    const end = seg.start + seg.duration;
    console.log(`[${seg.speaker}] ${seg.start.toFixed(2)}-${end.toFixed(2)} ${seg.text.trim()}`);
  }

  pipeline.close();
}
```

### Streaming transcription

```typescript
import { Pipeline } from 'pyannote-cpp-node';

async function runStreaming(audio: Float32Array) {
  const pipeline = await Pipeline.load({
    segModelPath: './models/segmentation.gguf',
    embModelPath: './models/embedding.gguf',
    pldaPath: './models/plda.gguf',
    coremlPath: './models/embedding.mlpackage',
    segCoremlPath: './models/segmentation.mlpackage',
    whisperModelPath: './models/ggml-large-v3-turbo-q5_0.bin',
  });

  const session = pipeline.createSession();
  session.on('segments', (segments) => {
    const latest = segments[segments.length - 1];
    if (latest) {
      const end = latest.start + latest.duration;
      console.log(`[live][${latest.speaker}] ${latest.start.toFixed(2)}-${end.toFixed(2)} ${latest.text.trim()}`);
    }
  });

  session.on('audio', (chunk) => {
    console.log(`silence-filtered audio chunk: ${chunk.length} samples`);
  });

  const chunkSize = 16000;
  for (let i = 0; i < audio.length; i += chunkSize) {
    const chunk = audio.slice(i, Math.min(i + chunkSize, audio.length));
    const vad = await session.push(chunk);
    if (vad.length > 0) {
      const speechFrames = vad.filter(Boolean).length;
      console.log(`VAD frames: ${vad.length}, speech frames: ${speechFrames}`);
    }
  }

  const finalResult = await session.finalize();
  console.log(`Final segments: ${finalResult.segments.length}`);

  session.close();
  pipeline.close();
}
```

### Custom Whisper decode options

```typescript
import { Pipeline } from 'pyannote-cpp-node';

const pipeline = await Pipeline.load({
  segModelPath: './models/segmentation.gguf',
  embModelPath: './models/embedding.gguf',
  pldaPath: './models/plda.gguf',
  coremlPath: './models/embedding.mlpackage',
  segCoremlPath: './models/segmentation.mlpackage',
  whisperModelPath: './models/ggml-large-v3-turbo-q5_0.bin',

  // Whisper runtime options
  useGpu: true,
  flashAttn: true,
  gpuDevice: 0,
  useCoreml: false,

  // Decode strategy
  nThreads: 8,
  language: 'ko',
  translate: false,
  detectLanguage: false,
  temperature: 0.0,
  temperatureInc: 0.2,
  noFallback: false,
  beamSize: 5,
  bestOf: 5,

  // Thresholds and context
  entropyThold: 2.4,
  logprobThold: -1.0,
  noSpeechThold: 0.6,
  prompt: 'Meeting transcript with technical terminology.',
  noContext: true,
  suppressBlank: true,
  suppressNst: false,
});
```

### Changing language at runtime

```typescript
import { Pipeline } from 'pyannote-cpp-node';

const pipeline = await Pipeline.load({
  segModelPath: './models/segmentation.gguf',
  embModelPath: './models/embedding.gguf',
  pldaPath: './models/plda.gguf',
  coremlPath: './models/embedding.mlpackage',
  segCoremlPath: './models/segmentation.mlpackage',
  whisperModelPath: './models/ggml-large-v3-turbo-q5_0.bin',
  language: 'en',
});

// First transcription in English
const result1 = await pipeline.transcribe(englishAudio);

// Switch to Korean for the next transcription
pipeline.setLanguage('ko');
const result2 = await pipeline.transcribe(koreanAudio);

// Or update multiple decode options at once
pipeline.setDecodeOptions({
  language: 'zh',
  temperature: 0.2,
  beamSize: 5,
});
const result3 = await pipeline.transcribe(chineseAudio);

pipeline.close();
```

Streaming sessions also support runtime changes:

```typescript
const session = pipeline.createSession();

session.on('segments', (segments) => {
  console.log(segments);
});

// Push English audio
await session.push(englishChunk);

// Switch language mid-stream — takes effect on the next Whisper decode
session.setLanguage('ko');
await session.push(koreanChunk);

const result = await session.finalize();

session.close();
```

## JSON Output Format

The pipeline returns this JSON shape:

```json
{
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.497000,
      "duration": 2.085000,
      "text": "Hello world"
    }
  ]
}
```

## Audio Format Requirements

- Input must be `Float32Array`
- Sample rate must be `16000` Hz
- Audio must be mono
- Recommended amplitude range: `[-1.0, 1.0]`

All API methods expect decoded PCM samples; file decoding/resampling is handled by the caller.

## Architecture

The integrated pipeline runs in 7 stages:

1. VAD silence filter (optional compression of long silence)
2. Audio buffer (stream-safe FIFO with timestamp tracking)
3. Segmentation (speech activity over rolling windows)
4. Transcription (Whisper sentence-level segments)
5. Alignment (segment-level speaker assignment by overlap)
6. Finalize (flush + final recluster + final alignment)
7. Callback/event emission (`segments` updates + `audio` chunk streaming)

## Performance

- Diarization only: **39x real-time**
- Integrated transcription + diarization: **~14.6x real-time**
- 45-minute Korean meeting test (6 speakers): **2713s audio in 186s**
- Each Whisper segment maps 1:1 to a speaker-labeled segment (no merging)
- Speaker confusion rate: **2.55%**

## Platform Support

| Platform | Status |
| --- | --- |
| macOS arm64 (Apple Silicon) | Supported |
| macOS x64 (Intel) | Supported |
| Linux | Not supported |
| Windows | Not supported |

## License

MIT
