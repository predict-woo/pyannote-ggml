import { readFileSync } from 'node:fs';
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

describe('Model loading', () => {
  it('loads model with valid paths', async () => {
    const model = await Pipeline.load(config);
    expect(model).toBeDefined();
    expect(model.isClosed).toBe(false);
    model.close();
  });

  it('throws on invalid path', async () => {
    await expect(Pipeline.load({ ...config, segModelPath: '/nonexistent' }))
      .rejects.toThrow();
  });

  it('close is idempotent', async () => {
    const model = await Pipeline.load(config);
    model.close();
    expect(model.isClosed).toBe(true);
    model.close();
    expect(model.isClosed).toBe(true);
  });
});

describe('One-shot transcribe', () => {
  let model: Pipeline;
  const audio = loadWav(resolve(PROJECT_ROOT, 'samples/sample.wav'));

  beforeAll(async () => {
    model = await Pipeline.load(config);
  });

  afterAll(() => {
    model.close();
  });

  it('produces non-empty segments with text', async () => {
    const result = await model.transcribe(audio);
    expect(result.segments).toBeDefined();
    expect(result.segments.length).toBeGreaterThan(0);

    const firstSeg = result.segments[0];
    expect(firstSeg.text).toBeDefined();
    expect(firstSeg.text.length).toBeGreaterThan(0);
  });

  it('segments have correct shape', async () => {
    const result = await model.transcribe(audio);
    for (const seg of result.segments) {
      expect(typeof seg.speaker).toBe('string');
      expect(typeof seg.start).toBe('number');
      expect(typeof seg.duration).toBe('number');
      expect(typeof seg.text).toBe('string');
      expect(seg.start).toBeGreaterThanOrEqual(0);
      expect(seg.duration).toBeGreaterThan(0);
      expect(seg.speaker).toMatch(/^SPEAKER_\d+$/);
      expect(seg.text.length).toBeGreaterThan(0);
    }
  });

  it('detects speakers', async () => {
    const result = await model.transcribe(audio);
    const speakers = new Set(result.segments.map(s => s.speaker));
    expect(speakers.size).toBeGreaterThanOrEqual(1);
  });

  it('segments are sorted by start time', async () => {
    const result = await model.transcribe(audio);
    for (let i = 1; i < result.segments.length; i++) {
      expect(result.segments[i].start).toBeGreaterThanOrEqual(result.segments[i - 1].start);
    }
  });
});

describe('Streaming session', () => {
  it('push returns boolean[] VAD predictions', async () => {
    const model = await Pipeline.load(config);
    const session = model.createSession();
    const audio = loadWav(resolve(PROJECT_ROOT, 'samples/sample.wav'));

    const CHUNK_SIZE = 16000; // 1 second
    const allVad: boolean[] = [];

    // Push 15 seconds of audio
    for (let i = 0; i < 15; i++) {
      const start = i * CHUNK_SIZE;
      const end = Math.min(start + CHUNK_SIZE, audio.length);
      const chunk = audio.slice(start, end);
      const vad = await session.push(chunk);
      allVad.push(...vad);
    }

    // VAD predictions should be boolean[]
    expect(allVad.length).toBeGreaterThan(0);
    for (const v of allVad) {
      expect(typeof v).toBe('boolean');
    }

    session.close();
    model.close();
  });

  it('emits segments events during streaming', async () => {
    const model = await Pipeline.load(config);
    const session = model.createSession();
    const audio = loadWav(resolve(PROJECT_ROOT, 'samples/sample.wav'));

    const receivedSegments: AlignedSegment[][] = [];
    session.on('segments', (segments: AlignedSegment[]) => {
      receivedSegments.push(segments);
    });

    const CHUNK_SIZE = 16000;
    for (let offset = 0; offset < audio.length; offset += CHUNK_SIZE) {
      const end = Math.min(offset + CHUNK_SIZE, audio.length);
      await session.push(audio.slice(offset, end));
    }

    const result = await session.finalize();
    expect(result.segments).toBeDefined();
    expect(result.segments.length).toBeGreaterThan(0);

    session.close();
    model.close();
  });

  it('emits audio events with Float32Array chunks', async () => {
    const model = await Pipeline.load(config);
    const session = model.createSession();
    const audio = loadWav(resolve(PROJECT_ROOT, 'samples/sample.wav'));

    const audioChunks: Float32Array[] = [];
    session.on('audio', (chunk: Float32Array) => {
      audioChunks.push(chunk);
    });

    const CHUNK_SIZE = 16000;
    for (let offset = 0; offset < audio.length; offset += CHUNK_SIZE) {
      const end = Math.min(offset + CHUNK_SIZE, audio.length);
      await session.push(audio.slice(offset, end));
    }

    await session.finalize();

    expect(audioChunks.length).toBeGreaterThan(0);
    for (const chunk of audioChunks) {
      expect(chunk).toBeInstanceOf(Float32Array);
      expect(chunk.length).toBeGreaterThan(0);
    }

    const totalSamples = audioChunks.reduce((sum, c) => sum + c.length, 0);
    expect(totalSamples).toBeGreaterThan(audio.length * 0.9);
    expect(totalSamples).toBeLessThanOrEqual(audio.length);

    session.close();
    model.close();
  });

  it('finalize returns TranscriptionResult', async () => {
    const model = await Pipeline.load(config);
    const session = model.createSession();
    const audio = loadWav(resolve(PROJECT_ROOT, 'samples/sample.wav'));

    const CHUNK_SIZE = 16000;
    for (let offset = 0; offset < audio.length; offset += CHUNK_SIZE) {
      const end = Math.min(offset + CHUNK_SIZE, audio.length);
      await session.push(audio.slice(offset, end));
    }

    const result = await session.finalize();
    expect(result.segments).toBeDefined();
    expect(result.segments.length).toBeGreaterThan(0);

    // Check segment shape
    for (const seg of result.segments) {
      expect(typeof seg.speaker).toBe('string');
      expect(typeof seg.start).toBe('number');
      expect(typeof seg.duration).toBe('number');
      expect(typeof seg.text).toBe('string');
    }

    session.close();
    model.close();
  });
});

describe('Resource cleanup', () => {
  it('close session then model without crash', async () => {
    const model = await Pipeline.load(config);
    const session = model.createSession();
    session.close();
    expect(session.isClosed).toBe(true);
    model.close();
    expect(model.isClosed).toBe(true);
  });

  it('push after close throws', async () => {
    const model = await Pipeline.load(config);
    const session = model.createSession();
    session.close();
    await expect(session.push(new Float32Array(16000))).rejects.toThrow();
    model.close();
  });

  it('finalize after close throws', async () => {
    const model = await Pipeline.load(config);
    const session = model.createSession();
    session.close();
    await expect(session.finalize()).rejects.toThrow();
    model.close();
  });

  it('transcribe after close throws', async () => {
    const model = await Pipeline.load(config);
    model.close();
    await expect(model.transcribe(new Float32Array(16000))).rejects.toThrow();
  });
});

describe('Shared model cache', () => {
  let model: Pipeline;
  const audio = loadWav(resolve(PROJECT_ROOT, 'samples/sample.wav'));

  beforeAll(async () => {
    model = await Pipeline.load(config);
  });

  afterAll(() => {
    model.close();
  });

  it('transcribeOffline produces valid segments', async () => {
    const result = await model.transcribeOffline(audio);
    expect(result.segments).toBeDefined();
    expect(result.segments.length).toBeGreaterThan(0);

    for (const seg of result.segments) {
      expect(typeof seg.speaker).toBe('string');
      expect(typeof seg.start).toBe('number');
      expect(typeof seg.duration).toBe('number');
      expect(typeof seg.text).toBe('string');
      expect(seg.start).toBeGreaterThanOrEqual(0);
      expect(seg.duration).toBeGreaterThan(0);
      expect(seg.speaker).toMatch(/^SPEAKER_\d+$/);
      expect(seg.text.length).toBeGreaterThan(0);
    }
  });

  it('switching between offline and streaming on same instance', async () => {
    const offlineFirst = await model.transcribeOffline(audio);
    expect(offlineFirst.segments.length).toBeGreaterThan(0);

    const streaming = await model.transcribe(audio);
    expect(streaming.segments.length).toBeGreaterThan(0);

    const offlineSecond = await model.transcribeOffline(audio);
    expect(offlineSecond.segments.length).toBeGreaterThan(0);
  });
});

describe('Shared model cache', () => {
  let model: Pipeline;
  const audio = loadWav(resolve(PROJECT_ROOT, 'samples/sample.wav'));

  beforeAll(async () => {
    model = await Pipeline.load(config);
  });

  afterAll(() => {
    model.close();
  });

  it('transcribeOffline produces valid segments', async () => {
    const result = await model.transcribeOffline(audio);
    expect(result.segments).toBeDefined();
    expect(result.segments.length).toBeGreaterThan(0);

    for (const seg of result.segments) {
      expect(typeof seg.speaker).toBe('string');
      expect(typeof seg.start).toBe('number');
      expect(typeof seg.duration).toBe('number');
      expect(typeof seg.text).toBe('string');
      expect(seg.speaker).toMatch(/^SPEAKER_\d+$/);
    }
  });

  it('switches between offline and streaming on same instance', async () => {
    const offlineResult1 = await model.transcribeOffline(audio);
    expect(offlineResult1.segments.length).toBeGreaterThan(0);

    const streamResult = await model.transcribe(audio);
    expect(streamResult.segments.length).toBeGreaterThan(0);

    const offlineResult2 = await model.transcribeOffline(audio);
    expect(offlineResult2.segments.length).toBeGreaterThan(0);
  });
});

describe('Whisper CoreML mode switching', () => {
  // Use turbo model which has a CoreML encoder (.mlmodelc) available
  const coremlConfig = {
    ...config,
    whisperModelPath: resolve(PROJECT_ROOT, 'whisper.cpp/models/ggml-large-v3-turbo-q5_0.bin'),
  };
  let model: Pipeline;
  const audio = loadWav(resolve(PROJECT_ROOT, 'samples/sample.wav'));

  beforeAll(async () => {
    model = await Pipeline.load(coremlConfig);
  });

  afterAll(() => {
    model.close();
  });

  it('no-op when already in requested mode', async () => {
    // Pipeline starts with useCoreml=false (default). Calling setUseCoreml(false) is a no-op.
    await expect(model.setUseCoreml(false)).resolves.toBeUndefined();
  });

  it('switches to CoreML mode and transcribes correctly', async () => {
    // Triggers a real whisper context reload (false → true).
    // Turbo model has ggml-large-v3-turbo-q5_0-encoder.mlmodelc next to it.
    await model.setUseCoreml(true);

    const result = await model.transcribeOffline(audio);
    expect(result.segments.length).toBeGreaterThan(0);
    for (const seg of result.segments) {
      expect(seg.speaker).toMatch(/^SPEAKER_\d+$/);
      expect(typeof seg.start).toBe('number');
      expect(seg.duration).toBeGreaterThan(0);
      expect(seg.text.length).toBeGreaterThan(0);
    }
  });

  it('switches back to GPU-only mode and transcribes correctly', async () => {
    // Switch back: true → false (another real reload)
    await model.setUseCoreml(false);

    const result = await model.transcribeOffline(audio);
    expect(result.segments.length).toBeGreaterThan(0);
    for (const seg of result.segments) {
      expect(seg.speaker).toMatch(/^SPEAKER_\d+$/);
      expect(seg.text.length).toBeGreaterThan(0);
    }
  });

  it('streaming transcription works after mode switch', async () => {
    await model.setUseCoreml(true);

    const result = await model.transcribe(audio);
    expect(result.segments.length).toBeGreaterThan(0);
    for (const seg of result.segments) {
      expect(seg.speaker).toMatch(/^SPEAKER_\d+$/);
      expect(seg.text.length).toBeGreaterThan(0);
    }

    // Switch back for subsequent tests
    await model.setUseCoreml(false);
  });

  it('setUseCoreml on closed pipeline throws', async () => {
    const m = await Pipeline.load(coremlConfig);
    m.close();
    await expect(m.setUseCoreml(false)).rejects.toThrow();
  });
});
