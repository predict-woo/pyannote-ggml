import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, it, expect, beforeAll, afterAll } from 'vitest';

import { Pipeline, type AlignedSegment } from '../src/index.js';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const PROJECT_ROOT = resolve(__dirname, '../../../../..');

const config = {
  segModelPath: resolve(PROJECT_ROOT, 'models/segmentation-ggml/segmentation.gguf'),
  embModelPath: resolve(PROJECT_ROOT, 'models/embedding-ggml/embedding.gguf'),
  pldaPath: resolve(PROJECT_ROOT, 'diarization-ggml/plda.gguf'),
  coremlPath: resolve(PROJECT_ROOT, 'models/embedding-ggml/embedding.mlpackage'),
  segCoremlPath: resolve(PROJECT_ROOT, 'models/segmentation-ggml/segmentation.mlpackage'),
  whisperModelPath: resolve(PROJECT_ROOT, 'whisper.cpp/models/ggml-base.en.bin'),
  vadModelPath: resolve(PROJECT_ROOT, 'whisper.cpp/models/ggml-silero-v6.2.0.bin'),
  language: 'en',
};

const SAMPLE_RATE = 16000;
const CHUNK_SIZE = 512;
const SILENCE_DURATION_SEC = 30;
const TOTAL_SAMPLES = SAMPLE_RATE * SILENCE_DURATION_SEC;

describe('Silence handling with VAD', () => {
  let pipeline: Pipeline;

  beforeAll(async () => {
    pipeline = await Pipeline.load(config);
  });

  afterAll(() => {
    pipeline.close();
  });

  it('streaming 30s of silence produces no audio events, no segments, and empty finalize', async () => {
    const session = pipeline.createSession();

    const audioChunks: Float32Array[] = [];
    const segmentEvents: AlignedSegment[][] = [];

    session.on('audio', (chunk: Float32Array) => {
      audioChunks.push(new Float32Array(chunk));
    });

    session.on('segments', (segments: AlignedSegment[]) => {
      segmentEvents.push(segments);
    });

    // Push 30 seconds of silence in 512-sample chunks
    const silentChunk = new Float32Array(CHUNK_SIZE); // all zeros
    const numChunks = Math.ceil(TOTAL_SAMPLES / CHUNK_SIZE);

    for (let i = 0; i < numChunks; i++) {
      await session.push(silentChunk);
    }

    const result = await session.finalize();

    // No audio should have been passed to Whisper
    expect(audioChunks.length).toBe(0);

    // No segment events should have fired
    expect(segmentEvents.length).toBe(0);

    // Finalize should return empty segments
    expect(result.segments).toBeDefined();
    expect(result.segments.length).toBe(0);

    session.close();
  }, 120000);
});
