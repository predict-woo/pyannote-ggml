import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, it, expect, beforeAll, afterAll } from 'vitest';

import { Pipeline } from '../src/index.js';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const PROJECT_ROOT = resolve(__dirname, '../../../../..');

const config = {
  segModelPath: resolve(PROJECT_ROOT, 'models/segmentation-ggml/segmentation.gguf'),
  embModelPath: resolve(PROJECT_ROOT, 'models/embedding-ggml/embedding.gguf'),
  pldaPath: resolve(PROJECT_ROOT, 'diarization-ggml/plda.gguf'),
  coremlPath: resolve(PROJECT_ROOT, 'models/embedding-ggml/embedding.mlpackage'),
  segCoremlPath: resolve(PROJECT_ROOT, 'models/segmentation-ggml/segmentation.mlpackage'),
  whisperModelPath: resolve(PROJECT_ROOT, 'whisper.cpp/models/ggml-base.en.bin'),
  language: 'en',
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

describe('transcribeOffline new-segment callback', () => {
  let model: Pipeline;
  const audio = loadWav(resolve(PROJECT_ROOT, 'samples/sample.wav'));

  beforeAll(async () => {
    model = await Pipeline.load(config);
  });

  afterAll(() => {
    model.close();
  });

  it('receives segments with start, end, text as Whisper produces them', async () => {
    const segments: Array<{ start: number; end: number; text: string }> = [];

    const result = await model.transcribeOffline(
      audio,
      undefined, // no progress callback
      (start, end, text) => {
        segments.push({ start, end, text });
      },
    );

    // Must produce valid final result
    expect(result.segments).toBeDefined();
    expect(result.segments.length).toBeGreaterThan(0);

    // Must have received segments via callback
    expect(segments.length).toBeGreaterThan(0);

    // Each segment must have valid fields
    for (const seg of segments) {
      expect(typeof seg.start).toBe('number');
      expect(typeof seg.end).toBe('number');
      expect(typeof seg.text).toBe('string');
      expect(seg.start).toBeGreaterThanOrEqual(0);
      expect(seg.end).toBeGreaterThanOrEqual(seg.start);
    }

    // Segments should be in chronological order
    for (let i = 1; i < segments.length; i++) {
      expect(segments[i].start).toBeGreaterThanOrEqual(segments[i - 1].start);
    }

    // Start/end should be in seconds (reasonable range for sample.wav ~30s)
    const maxEnd = Math.max(...segments.map(s => s.end));
    expect(maxEnd).toBeLessThan(120); // sanity: not centiseconds
    expect(maxEnd).toBeGreaterThan(1); // at least some audio
  });

  it('works with both onProgress and onSegment simultaneously', async () => {
    const progressEvents: Array<{ phase: number; progress: number }> = [];
    const segmentEvents: Array<{ start: number; end: number; text: string }> = [];

    const result = await model.transcribeOffline(
      audio,
      (phase, progress) => {
        progressEvents.push({ phase, progress });
      },
      (start, end, text) => {
        segmentEvents.push({ start, end, text });
      },
    );

    expect(result.segments.length).toBeGreaterThan(0);
    expect(progressEvents.length).toBeGreaterThan(0);
    expect(segmentEvents.length).toBeGreaterThan(0);

    // Progress should include phase 0 (whisper)
    expect(progressEvents.some(e => e.phase === 0)).toBe(true);
  });

  it('works without onSegment (backward compat)', async () => {
    const result = await model.transcribeOffline(audio);
    expect(result.segments).toBeDefined();
    expect(result.segments.length).toBeGreaterThan(0);
  });

  it('works with onProgress but no onSegment', async () => {
    const events: Array<{ phase: number; progress: number }> = [];
    const result = await model.transcribeOffline(audio, (phase, progress) => {
      events.push({ phase, progress });
    });
    expect(result.segments.length).toBeGreaterThan(0);
    expect(events.length).toBeGreaterThan(0);
  });

  it('works with undefined onProgress and valid onSegment', async () => {
    const segments: Array<{ start: number; end: number; text: string }> = [];
    const result = await model.transcribeOffline(
      audio,
      undefined,
      (start, end, text) => {
        segments.push({ start, end, text });
      },
    );
    expect(result.segments.length).toBeGreaterThan(0);
    expect(segments.length).toBeGreaterThan(0);
  });
});
