import { readFileSync, writeFileSync } from 'node:fs';
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
function writeWav(filePath: string, samples: Float32Array, sampleRate = 16000): void {
  const numSamples = samples.length;
  const dataBytes = numSamples * 2; // 16-bit PCM
  const headerSize = 44;
  const buffer = Buffer.alloc(headerSize + dataBytes);

  // RIFF header
  buffer.write('RIFF', 0);
  buffer.writeUInt32LE(headerSize + dataBytes - 8, 4);
  buffer.write('WAVE', 8);

  // fmt chunk
  buffer.write('fmt ', 12);
  buffer.writeUInt32LE(16, 16);        // chunk size
  buffer.writeUInt16LE(1, 20);          // PCM format
  buffer.writeUInt16LE(1, 22);          // mono
  buffer.writeUInt32LE(sampleRate, 24); // sample rate
  buffer.writeUInt32LE(sampleRate * 2, 28); // byte rate
  buffer.writeUInt16LE(2, 32);          // block align
  buffer.writeUInt16LE(16, 34);         // bits per sample

  // data chunk
  buffer.write('data', 36);
  buffer.writeUInt32LE(dataBytes, 40);

  for (let i = 0; i < numSamples; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    buffer.writeInt16LE(Math.round(clamped * 32767), headerSize + i * 2);
  }

  writeFileSync(filePath, buffer);
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

    // Write collected audio to WAV for manual listening
    const outPath = resolve(PROJECT_ROOT, 'samples/audio_passthrough_output.wav');
    writeWav(outPath, collected);
    console.log(`\nWrote passthrough audio to: ${outPath}`);
    console.log(`  Samples: ${collected.length} (${(collected.length / 16000).toFixed(2)}s at 16kHz)`);

    session.close();
  }, 120000);
});

describe('Audio passthrough with VAD', () => {
  let pipeline: Pipeline;
  const input = loadWav(resolve(PROJECT_ROOT, 'samples/sample.wav'));

  beforeAll(async () => {
    pipeline = await Pipeline.load({
      ...config,
      vadModelPath: resolve(PROJECT_ROOT, 'whisper.cpp/models/ggml-silero-v6.2.0.bin'),
    });
  });

  afterAll(() => {
    pipeline.close();
  });

  it('saves VAD-filtered audio to WAV for manual listening', async () => {
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

    expect(collected.length).toBeGreaterThan(0);

    const outPath = resolve(PROJECT_ROOT, 'samples/audio_vad_output.wav');
    writeWav(outPath, collected);
    console.log(`\nWrote VAD-filtered audio to: ${outPath}`);
    console.log(`  Samples: ${collected.length} (${(collected.length / 16000).toFixed(2)}s at 16kHz)`);
    console.log(`  Compression: ${((1 - collected.length / input.length) * 100).toFixed(1)}% removed by VAD`);

    session.close();
  }, 120000);
});
