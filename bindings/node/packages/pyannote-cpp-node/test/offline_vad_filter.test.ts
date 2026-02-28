import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, it, expect, beforeAll, afterAll } from 'vitest';

import { Pipeline } from '../src/index.js';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const PROJECT_ROOT = resolve(__dirname, '../../../../..');

const baseConfig = {
  segModelPath: resolve(PROJECT_ROOT, 'models/segmentation-ggml/segmentation.gguf'),
  embModelPath: resolve(PROJECT_ROOT, 'models/embedding-ggml/embedding.gguf'),
  pldaPath: resolve(PROJECT_ROOT, 'diarization-ggml/plda.gguf'),
  coremlPath: resolve(PROJECT_ROOT, 'models/embedding-ggml/embedding.mlpackage'),
  segCoremlPath: resolve(PROJECT_ROOT, 'models/segmentation-ggml/segmentation.mlpackage'),
  whisperModelPath: resolve(PROJECT_ROOT, 'whisper.cpp/models/ggml-base.en.bin'),
  language: 'en',
};

const vadConfig = {
  ...baseConfig,
  vadModelPath: resolve(PROJECT_ROOT, 'whisper.cpp/models/ggml-silero-v6.2.0.bin'),
};

function loadWav(filePath: string): Float32Array {
  const buffer = readFileSync(filePath);
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

  let offset = 0;
  const riff = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (riff !== 'RIFF') throw new Error('Not a RIFF file');
  offset = 12;

  while (offset < view.byteLength - 8) {
    const chunkId = String.fromCharCode(
      view.getUint8(offset), view.getUint8(offset + 1),
      view.getUint8(offset + 2), view.getUint8(offset + 3),
    );
    const chunkSize = view.getUint32(offset + 4, true);
    offset += 8;

    if (chunkId === 'data') {
      const numSamples = chunkSize / 2;
      const float32 = new Float32Array(numSamples);
      for (let i = 0; i < numSamples; i++) {
        float32[i] = view.getInt16(offset + i * 2, true) / 32768.0;
      }
      return float32;
    }

    offset += chunkSize;
    if (chunkSize % 2 !== 0) offset++;
  }

  throw new Error('No data chunk found in WAV file');
}

const SAMPLE_RATE = 16000;

describe('Offline VAD filter — with VAD model', () => {
  let pipeline: Pipeline;
  const audio = loadWav(resolve(PROJECT_ROOT, 'samples/sample.wav'));

  beforeAll(async () => {
    pipeline = await Pipeline.load(vadConfig);
  });

  afterAll(() => {
    pipeline.close();
  });

  it('returns filteredAudio as Float32Array', async () => {
    const result = await pipeline.transcribeOffline(audio);

    expect(result.filteredAudio).toBeDefined();
    expect(result.filteredAudio).toBeInstanceOf(Float32Array);
    expect(result.filteredAudio!.length).toBeGreaterThan(0);
  });

  it('filteredAudio length <= original audio length', async () => {
    const result = await pipeline.transcribeOffline(audio);

    expect(result.filteredAudio).toBeDefined();
    expect(result.filteredAudio!.length).toBeLessThanOrEqual(audio.length);

    const originalDuration = audio.length / SAMPLE_RATE;
    const filteredDuration = result.filteredAudio!.length / SAMPLE_RATE;
    console.log(
      `  Original: ${originalDuration.toFixed(2)}s (${audio.length} samples)` +
      ` → Filtered: ${filteredDuration.toFixed(2)}s (${result.filteredAudio!.length} samples)` +
      ` (${((1 - filteredDuration / originalDuration) * 100).toFixed(1)}% removed)`,
    );
  });

  it('segment timestamps fit within filtered audio duration', async () => {
    const result = await pipeline.transcribeOffline(audio);

    expect(result.filteredAudio).toBeDefined();
    const filteredDuration = result.filteredAudio!.length / SAMPLE_RATE;

    for (const seg of result.segments) {
      const segEnd = seg.start + seg.duration;
      // Allow 0.5s tolerance for rounding at the tail
      expect(segEnd).toBeLessThanOrEqual(filteredDuration + 0.5);
      expect(seg.start).toBeGreaterThanOrEqual(0);
    }
  });

  it('produces valid speaker-labeled segments', async () => {
    const result = await pipeline.transcribeOffline(audio);

    expect(result.segments.length).toBeGreaterThan(0);
    for (const seg of result.segments) {
      expect(seg.speaker).toMatch(/^SPEAKER_\d+$/);
      expect(typeof seg.start).toBe('number');
      expect(seg.duration).toBeGreaterThan(0);
      expect(seg.text.length).toBeGreaterThan(0);
    }
  });
});

describe('Offline VAD filter — without VAD model', () => {
  let pipeline: Pipeline;
  const audio = loadWav(resolve(PROJECT_ROOT, 'samples/sample.wav'));

  beforeAll(async () => {
    pipeline = await Pipeline.load(baseConfig);
  });

  afterAll(() => {
    pipeline.close();
  });

  it('filteredAudio is undefined when no VAD model loaded', async () => {
    const result = await pipeline.transcribeOffline(audio);

    expect(result.filteredAudio).toBeUndefined();
  });

  it('still produces valid segments without VAD', async () => {
    const result = await pipeline.transcribeOffline(audio);

    expect(result.segments.length).toBeGreaterThan(0);
    for (const seg of result.segments) {
      expect(seg.speaker).toMatch(/^SPEAKER_\d+$/);
      expect(typeof seg.start).toBe('number');
      expect(seg.duration).toBeGreaterThan(0);
      expect(seg.text.length).toBeGreaterThan(0);
    }
  });
});

describe('Offline VAD filter — silence compression', () => {
  let pipeline: Pipeline;

  beforeAll(async () => {
    pipeline = await Pipeline.load(vadConfig);
  });

  afterAll(() => {
    pipeline.close();
  });

  it('compresses long silence in synthetic audio', async () => {
    // Create synthetic audio: 5s speech-like noise + 10s silence + 5s speech-like noise
    const speechDuration = 5;
    const silenceDuration = 10;
    const totalSamples = (speechDuration * 2 + silenceDuration) * SAMPLE_RATE;
    const synthetic = new Float32Array(totalSamples);

    // Fill speech sections with pseudo-random noise (amplitude ~0.1)
    // Using a simple deterministic pattern so it's reproducible
    for (let i = 0; i < speechDuration * SAMPLE_RATE; i++) {
      synthetic[i] = 0.1 * Math.sin(i * 0.1) * Math.sin(i * 0.037);
    }
    // Middle is silence (zeros) — already initialized
    const secondSpeechStart = (speechDuration + silenceDuration) * SAMPLE_RATE;
    for (let i = 0; i < speechDuration * SAMPLE_RATE; i++) {
      synthetic[secondSpeechStart + i] = 0.1 * Math.sin(i * 0.13) * Math.sin(i * 0.041);
    }

    const result = await pipeline.transcribeOffline(synthetic);

    expect(result.filteredAudio).toBeDefined();

    // The 10s silence should be compressed to ~2s, so filtered should be
    // roughly (5 + 2 + 5) = 12s instead of 20s — at least 5s shorter
    const originalDuration = totalSamples / SAMPLE_RATE;
    const filteredDuration = result.filteredAudio!.length / SAMPLE_RATE;
    const savedDuration = originalDuration - filteredDuration;

    console.log(
      `  Synthetic: ${originalDuration.toFixed(1)}s → ${filteredDuration.toFixed(1)}s` +
      ` (saved ${savedDuration.toFixed(1)}s)`,
    );

    // Should save at least 5 seconds (10s silence → 2s = 8s saved, minus tolerance)
    expect(savedDuration).toBeGreaterThanOrEqual(5);
  });
});
