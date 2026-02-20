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

describe('Audio passthrough without VAD', () => {
  let pipeline: Pipeline;
  const input = loadWav(resolve(PROJECT_ROOT, 'samples/sample.wav'));

  beforeAll(async () => {
    pipeline = await Pipeline.load(config);
  });

  afterAll(() => {
    pipeline.close();
  });

  it('collected audio from audio events is byte-identical to input when no VAD model is used', async () => {
    const session = pipeline.createSession();
    const audioChunks: Float32Array[] = [];

    session.on('audio', (chunk: Float32Array) => {
      audioChunks.push(new Float32Array(chunk));
    });

    const CHUNK_SIZE = 16000;
    for (let offset = 0; offset < input.length; offset += CHUNK_SIZE) {
      const end = Math.min(offset + CHUNK_SIZE, input.length);
      await session.push(input.slice(offset, end));
    }

    await session.finalize();

    const totalLength = audioChunks.reduce((sum, c) => sum + c.length, 0);
    const collected = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of audioChunks) {
      collected.set(chunk, offset);
      offset += chunk.length;
    }

    expect(collected.length).toBe(input.length);

    const collectedBytes = Buffer.from(collected.buffer);
    const inputBytes = Buffer.from(input.buffer);
    const bytesEqual = collectedBytes.equals(inputBytes);

    if (!bytesEqual) {
      let firstDiff = -1;
      for (let i = 0; i < input.length; i++) {
        if (collected[i] !== input[i]) {
          firstDiff = i;
          break;
        }
      }

      if (firstDiff >= 0) {
        throw new Error(
          `Byte mismatch at sample ${firstDiff}: collected=${collected[firstDiff]}, input=${input[firstDiff]}`,
        );
      }

      throw new Error('Byte mismatch detected but no differing sample was found');
    }

    expect(bytesEqual).toBe(true);
    session.close();
  }, 120000);
});
