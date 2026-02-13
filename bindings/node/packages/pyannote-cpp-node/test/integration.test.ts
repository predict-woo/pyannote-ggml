import { readFileSync, writeFileSync } from 'node:fs';
import { execSync } from 'node:child_process';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, it, expect, beforeAll, afterAll } from 'vitest';

import { Pyannote } from '../src/index.js';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const PROJECT_ROOT = resolve(__dirname, '../../../../..');

const config = {
  segModelPath: resolve(PROJECT_ROOT, 'models/segmentation-ggml/segmentation.gguf'),
  embModelPath: resolve(PROJECT_ROOT, 'models/embedding-ggml/embedding.gguf'),
  pldaPath: resolve(PROJECT_ROOT, 'diarization-ggml/plda.gguf'),
  coremlPath: resolve(PROJECT_ROOT, 'models/embedding-ggml/embedding.mlpackage'),
  segCoremlPath: resolve(PROJECT_ROOT, 'models/segmentation-ggml/segmentation.mlpackage'),
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

describe('Model loading', () => {
  it('loads model with valid paths', async () => {
    const model = await Pyannote.load(config);
    expect(model).toBeDefined();
    expect(model.isClosed).toBe(false);
    model.close();
  });

  it('throws on invalid path', async () => {
    await expect(Pyannote.load({ ...config, segModelPath: '/nonexistent' }))
      .rejects.toThrow();
  });

  it('close is idempotent', async () => {
    const model = await Pyannote.load(config);
    model.close();
    expect(model.isClosed).toBe(true);
    model.close();
    expect(model.isClosed).toBe(true);
  });
});

describe('Offline diarization', () => {
  let model: Pyannote;
  const audio = loadWav(resolve(PROJECT_ROOT, 'samples/sample.wav'));

  beforeAll(async () => {
    model = await Pyannote.load(config);
  });

  afterAll(() => {
    model.close();
  });

  it('produces non-empty segments', async () => {
    const result = await model.diarize(audio);
    expect(result.segments).toBeDefined();
    expect(result.segments.length).toBeGreaterThan(0);
  });

  it('segments have correct shape', async () => {
    const result = await model.diarize(audio);
    for (const seg of result.segments) {
      expect(typeof seg.start).toBe('number');
      expect(typeof seg.duration).toBe('number');
      expect(typeof seg.speaker).toBe('string');
      expect(seg.start).toBeGreaterThanOrEqual(0);
      expect(seg.duration).toBeGreaterThan(0);
      expect(seg.speaker).toMatch(/^SPEAKER_\d+$/);
    }
  });

  it('detects at least 2 speakers', async () => {
    const result = await model.diarize(audio);
    const speakers = new Set(result.segments.map(s => s.speaker));
    expect(speakers.size).toBeGreaterThanOrEqual(2);
  });

  it('segments are sorted by start time', async () => {
    const result = await model.diarize(audio);
    for (let i = 1; i < result.segments.length; i++) {
      expect(result.segments[i].start).toBeGreaterThanOrEqual(result.segments[i - 1].start);
    }
  });
});

describe('DER validation', () => {
  it('matches C++ reference output (DER ≤ 1.0%)', async () => {
    const model = await Pyannote.load(config);
    const audio = loadWav(resolve(PROJECT_ROOT, 'samples/sample.wav'));
    const result = await model.diarize(audio);
    model.close();

    const rttmLines = result.segments.map(seg =>
      `SPEAKER sample 1 ${seg.start.toFixed(3)} ${seg.duration.toFixed(3)} <NA> <NA> ${seg.speaker} <NA> <NA>`,
    );
    const rttmPath = '/tmp/node_test.rttm';
    writeFileSync(rttmPath, rttmLines.join('\n') + '\n');

    const pythonPath = resolve(PROJECT_ROOT, '.venv/bin/python3');
    const comparePath = resolve(PROJECT_ROOT, 'diarization-ggml/tests/compare_rttm.py');
    const refPath = '/tmp/py_reference.rttm';

    const output = execSync(
      `${pythonPath} ${comparePath} ${rttmPath} ${refPath} --threshold 1.0`,
      { encoding: 'utf-8' },
    );
    console.log(output);

    expect(output).toContain('PASS');
  });
});

describe('Streaming basic flow', () => {
  it('push returns VADChunks after 10s of audio', async () => {
    const model = await Pyannote.load(config);
    const session = model.createStreamingSession();
    const audio = loadWav(resolve(PROJECT_ROOT, 'samples/sample.wav'));

    const allChunks: Awaited<ReturnType<typeof session.push>> = [];
    const CHUNK_SIZE = 16000;

    for (let i = 0; i < 15; i++) {
      const start = i * CHUNK_SIZE;
      const end = Math.min(start + CHUNK_SIZE, audio.length);
      const chunk = audio.slice(start, end);
      const vadChunks = await session.push(chunk);
      allChunks.push(...vadChunks);
    }

    expect(allChunks.length).toBeGreaterThan(0);

    for (const chunk of allChunks) {
      expect(typeof chunk.chunkIndex).toBe('number');
      expect(typeof chunk.startFrame).toBe('number');
      expect(typeof chunk.numFrames).toBe('number');
      expect(chunk.numFrames).toBeGreaterThan(0);
      expect(chunk.numFrames).toBeLessThanOrEqual(589);
      expect(chunk.vad).toBeInstanceOf(Float32Array);
      expect(chunk.vad.length).toBe(chunk.numFrames);
    }

    session.close();
    model.close();
  });
});

describe('Streaming zero-latency mode', () => {
  it('returns frames on first push with zero latency', async () => {
    const model = await Pyannote.load({ ...config, zeroLatency: true });
    const session = model.createStreamingSession();
    const audio = loadWav(resolve(PROJECT_ROOT, 'samples/sample.wav'));

    const chunk = audio.slice(0, 16000);
    const vadChunks = await session.push(chunk);

    expect(vadChunks.length).toBeGreaterThan(0);
    expect(vadChunks[0].startFrame).toBe(0);

    session.close();
    model.close();
  });
});

describe('Streaming finalize matches offline', () => {
  it('produces same segments as offline diarize', async () => {
    const model = await Pyannote.load(config);
    const audio = loadWav(resolve(PROJECT_ROOT, 'samples/sample.wav'));

    const offlineResult = await model.diarize(audio);

    const session = model.createStreamingSession();
    const CHUNK_SIZE = 16000;
    for (let offset = 0; offset < audio.length; offset += CHUNK_SIZE) {
      const end = Math.min(offset + CHUNK_SIZE, audio.length);
      await session.push(audio.slice(offset, end));
    }
    const streamResult = await session.finalize();
    session.close();
    model.close();

    expect(streamResult.segments.length).toBe(offlineResult.segments.length);
    for (let i = 0; i < offlineResult.segments.length; i++) {
      expect(streamResult.segments[i].speaker).toBe(offlineResult.segments[i].speaker);
      expect(streamResult.segments[i].start).toBeCloseTo(offlineResult.segments[i].start, 2);
      expect(streamResult.segments[i].duration).toBeCloseTo(offlineResult.segments[i].duration, 2);
    }
  });
});

describe('Resource cleanup', () => {
  it('close session then model without crash', async () => {
    const model = await Pyannote.load(config);
    const session = model.createStreamingSession();
    session.close();
    expect(session.isClosed).toBe(true);
    model.close();
    expect(model.isClosed).toBe(true);
  });

  it('push after close throws', async () => {
    const model = await Pyannote.load(config);
    const session = model.createStreamingSession();
    session.close();
    await expect(session.push(new Float32Array(16000))).rejects.toThrow('closed');
    model.close();
  });

  it('recluster after close throws', async () => {
    const model = await Pyannote.load(config);
    const session = model.createStreamingSession();
    session.close();
    await expect(session.recluster()).rejects.toThrow('closed');
    model.close();
  });

  it('diarize after close throws', async () => {
    const model = await Pyannote.load(config);
    model.close();
    await expect(model.diarize(new Float32Array(16000))).rejects.toThrow('closed');
  });
});
