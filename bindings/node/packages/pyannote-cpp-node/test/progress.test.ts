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

describe('transcribeOffline progress callback', () => {
  let model: Pipeline;
  const audio = loadWav(resolve(PROJECT_ROOT, 'samples/sample.wav'));

  beforeAll(async () => {
    model = await Pipeline.load(config);
  });

  afterAll(() => {
    model.close();
  });

  it('receives phase 0 (whisper), phase 1 (diarization), phase 2 (alignment) in order', async () => {
    const events: Array<{ phase: number; progress: number }> = [];

    const result = await model.transcribeOffline(audio, (phase, progress) => {
      events.push({ phase, progress });
    });

    // Must produce valid segments
    expect(result.segments).toBeDefined();
    expect(result.segments.length).toBeGreaterThan(0);

    // Must receive some events
    expect(events.length).toBeGreaterThan(0);

    // Count events per phase
    const phase0 = events.filter(e => e.phase === 0);
    const phase1 = events.filter(e => e.phase === 1);
    const phase2 = events.filter(e => e.phase === 2);

    // Phase 0 (whisper) should have multiple progress calls
    expect(phase0.length).toBeGreaterThan(0);

    // Phase 0 should reach 100
    expect(Math.max(...phase0.map(e => e.progress))).toBe(100);

    // Phase 1 (diarization) should be called exactly once with progress 0
    expect(phase1.length).toBe(1);
    expect(phase1[0].progress).toBe(0);

    // Phase 2 (alignment) should be called exactly once with progress 0
    expect(phase2.length).toBe(1);
    expect(phase2[0].progress).toBe(0);

    // Phases must arrive in order: all 0s before 1 before 2
    let lastPhase = -1;
    for (const e of events) {
      expect(e.phase).toBeGreaterThanOrEqual(lastPhase);
      lastPhase = e.phase;
    }

    // Phase 0 progress should be non-decreasing
    let lastProgress = -1;
    for (const e of phase0) {
      expect(e.progress).toBeGreaterThanOrEqual(lastProgress);
      lastProgress = e.progress;
    }
  });

  it('works without onProgress (backward compat)', async () => {
    // Call without callback — should not throw
    const result = await model.transcribeOffline(audio);
    expect(result.segments).toBeDefined();
    expect(result.segments.length).toBeGreaterThan(0);
  });

  it('works with undefined onProgress', async () => {
    const result = await model.transcribeOffline(audio, undefined);
    expect(result.segments).toBeDefined();
    expect(result.segments.length).toBeGreaterThan(0);
  });
});
