import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, it, expect, beforeAll, afterAll } from 'vitest';

import { Pipeline, type AlignedSegment } from '../src/index.js';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const PROJECT_ROOT = resolve(__dirname, '../../../../..');

// Transcription-only config - no embedding, PLDA, or embedding CoreML needed
const transcriptionOnlyConfig = {
  segModelPath: resolve(PROJECT_ROOT, 'models/segmentation-ggml/segmentation.gguf'),
  segCoremlPath: resolve(PROJECT_ROOT, 'models/segmentation-ggml/segmentation.mlpackage'),
  whisperModelPath: resolve(PROJECT_ROOT, 'whisper.cpp/models/ggml-base.en.bin'),
  language: 'en',
  transcriptionOnly: true,
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

describe('Transcription-only mode', () => {
  it('loads without embedding/PLDA/coreml paths', async () => {
    const pipeline = await Pipeline.load(transcriptionOnlyConfig);
    expect(pipeline).toBeDefined();
    expect(pipeline.isClosed).toBe(false);
    pipeline.close();
  });

  it('produces segments with empty speaker', async () => {
    const pipeline = await Pipeline.load(transcriptionOnlyConfig);
    const audio = loadWav(resolve(PROJECT_ROOT, 'samples/sample.wav'));

    const result = await pipeline.transcribe(audio);
    expect(result.segments.length).toBeGreaterThan(0);

    for (const seg of result.segments) {
      expect(seg.speaker).toBe('');
      expect(typeof seg.start).toBe('number');
      expect(seg.duration).toBeGreaterThan(0);
      expect(seg.text.length).toBeGreaterThan(0);
    }

    pipeline.close();
  });

  it('streaming session works in transcription-only mode', async () => {
    const pipeline = await Pipeline.load(transcriptionOnlyConfig);
    const audio = loadWav(resolve(PROJECT_ROOT, 'samples/sample.wav'));
    const session = pipeline.createSession();

    const CHUNK_SIZE = 16000;
    for (let offset = 0; offset < audio.length; offset += CHUNK_SIZE) {
      const end = Math.min(offset + CHUNK_SIZE, audio.length);
      await session.push(audio.slice(offset, end));
    }

    const result = await session.finalize();
    expect(result.segments.length).toBeGreaterThan(0);

    for (const seg of result.segments) {
      expect(seg.speaker).toBe('');
      expect(seg.text.length).toBeGreaterThan(0);
    }

    session.close();
    pipeline.close();
  });

  it('rejects missing embModelPath without transcriptionOnly', async () => {
    await expect(Pipeline.load({
      segModelPath: transcriptionOnlyConfig.segModelPath,
      segCoremlPath: transcriptionOnlyConfig.segCoremlPath,
      whisperModelPath: transcriptionOnlyConfig.whisperModelPath,
      language: 'en',
      // No embModelPath, pldaPath, coremlPath - and transcriptionOnly is NOT set
    })).rejects.toThrow();
  });
});
