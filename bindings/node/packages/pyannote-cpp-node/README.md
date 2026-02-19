# pyannote-cpp-node

![Platform](https://img.shields.io/badge/platform-macOS-lightgrey)
![Node](https://img.shields.io/badge/node-%3E%3D18-brightgreen)

Node.js native bindings for integrated Whisper transcription + speaker diarization with speaker-labeled, word-level output.

## Overview

`pyannote-cpp-node` exposes the integrated C++ pipeline that combines streaming diarization and Whisper transcription into a single API.

Given 16 kHz mono PCM audio (`Float32Array`), it produces cumulative and final transcript segments shaped as:

- speaker label (`SPEAKER_00`, `SPEAKER_01`, ...)
- segment start/duration in seconds
- segment text
- per-word timestamps

The API supports both one-shot processing (`transcribe`) and incremental streaming (`createSession` + `push`/`finalize`). All heavy operations are asynchronous and run on libuv worker threads.

## Features

- Integrated transcription + diarization in one pipeline
- Speaker-labeled, word-level transcript output
- One-shot and streaming APIs with the same output schema
- Incremental `segments` events for live applications
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
  createSession(): PipelineSession;
  close(): void;
  get isClosed(): boolean;
}
```

#### `static async load(config: ModelConfig): Promise<Pipeline>`

Validates model paths and initializes native pipeline resources.

#### `async transcribe(audio: Float32Array): Promise<TranscriptionResult>`

Runs one-shot transcription + diarization on the full audio buffer.

#### `createSession(): PipelineSession`

Creates an independent streaming session for incremental processing.

#### `close(): void`

Releases native resources. Safe to call multiple times.

#### `get isClosed(): boolean`

Returns `true` after `close()`.

### `PipelineSession` (extends `EventEmitter`)

```typescript
class PipelineSession extends EventEmitter {
  async push(audio: Float32Array): Promise<boolean[]>;
  async finalize(): Promise<TranscriptionResult>;
  close(): void;
  get isClosed(): boolean;
  // Event: 'segments' -> (segments: AlignedSegment[], audio: Float32Array)
}
```

#### `async push(audio: Float32Array): Promise<boolean[]>`

Pushes an arbitrary number of samples into the streaming pipeline.

- Return value is per-frame VAD booleans (`true` = speech, `false` = silence)
- First 10 seconds return an empty array because the pipeline needs a full 10-second window
- Chunk size is flexible; not restricted to 16,000-sample pushes

#### `async finalize(): Promise<TranscriptionResult>`

Flushes all stages, runs final recluster + alignment, and returns the definitive result.

#### `close(): void`

Releases native session resources. Safe to call multiple times.

#### `get isClosed(): boolean`

Returns `true` after `close()`.

#### Event: `'segments'`

Emitted after each Whisper transcription result with the latest cumulative aligned output.

```typescript
session.on('segments', (segments: AlignedSegment[], audio: Float32Array) => {
  // `segments` contains the latest cumulative speaker-labeled transcript
  // `audio` contains the chunk submitted for this callback cycle
});
```

### Types

```typescript
export interface ModelConfig {
  /** Path to segmentation GGUF model file. */
  segModelPath: string;

  /** Path to embedding GGUF model file. */
  embModelPath: string;

  /** Path to PLDA GGUF model file. */
  pldaPath: string;

  /** Path to embedding CoreML .mlpackage directory. */
  coremlPath: string;

  /** Path to segmentation CoreML .mlpackage directory. */
  segCoremlPath: string;

  /** Path to Whisper GGUF model file. */
  whisperModelPath: string;

  /** Optional path to Silero VAD model file; enables silence compression. */
  vadModelPath?: string;

  /** Enable GPU for Whisper. Default: true. */
  useGpu?: boolean;

  /** Enable flash attention when supported. Default: true. */
  flashAttn?: boolean;

  /** GPU device index. Default: 0. */
  gpuDevice?: number;

  /**
   * Enable Whisper CoreML encoder.
   * Default: false.
   * Requires a matching `-encoder.mlmodelc` next to the GGUF model.
   */
  useCoreml?: boolean;

  /** Suppress Whisper native logs. Default: false. */
  noPrints?: boolean;

  /** Number of decode threads. Default: 4. */
  nThreads?: number;

  /** Language code for transcription. Default: 'en'. Omit for auto-detect behavior with model settings. */
  language?: string;

  /** Translate to English. Default: false. */
  translate?: boolean;

  /** Force language detection pass. Default: false. */
  detectLanguage?: boolean;

  /** Base sampling temperature. Default: 0.0 (greedy). */
  temperature?: number;

  /** Temperature increment for fallback sampling. Default: 0.2. */
  temperatureInc?: number;

  /** Disable temperature fallback ladder. Default: false. */
  noFallback?: boolean;

  /** Beam size. Default: -1 (greedy with best_of). */
  beamSize?: number;

  /** Number of candidates in best-of sampling. Default: 5. */
  bestOf?: number;

  /** Compression/entropy threshold. Default: 2.4. */
  entropyThold?: number;

  /** Average logprob threshold. Default: -1.0. */
  logprobThold?: number;

  /** No-speech probability threshold. Default: 0.6. */
  noSpeechThold?: number;

  /** Optional initial prompt text. Default: undefined. */
  prompt?: string;

  /** Disable context carry-over between decode windows. Default: true. */
  noContext?: boolean;

  /** Suppress blank tokens. Default: true. */
  suppressBlank?: boolean;

  /** Suppress non-speech tokens. Default: false. */
  suppressNst?: boolean;
}

export interface AlignedWord {
  /** Word text (may include leading space from Whisper tokenization). */
  text: string;

  /** Word start time in seconds. */
  start: number;

  /** Word end time in seconds. */
  end: number;
}

export interface AlignedSegment {
  /** Global speaker label (for example, SPEAKER_00). */
  speaker: string;

  /** Segment start time in seconds. */
  start: number;

  /** Segment duration in seconds. */
  duration: number;

  /** Segment text (concatenated from words). */
  text: string;

  /** Word-level timestamps for the segment. */
  words: AlignedWord[];
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

## JSON Output Format

The pipeline returns this JSON shape:

```json
{
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.497000,
      "duration": 2.085000,
      "text": "Hello world",
      "words": [
        {"text": " Hello", "start": 0.500000, "end": 0.800000},
        {"text": " world", "start": 0.900000, "end": 1.200000}
      ]
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
4. Transcription (Whisper sentence + word timestamps)
5. Alignment (segment-level speaker assignment by overlap)
6. Finalize (flush + final recluster + final alignment)
7. Callback/event emission (`segments` updates)

## Performance

- Diarization only: **39x real-time**
- Integrated transcription + diarization: **~14.6x real-time**
- 45-minute Korean meeting test (6 speakers): **2713s audio in 186s**
- Alignment reduction: **701 Whisper segments -> 186 aligned speaker segments**
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
