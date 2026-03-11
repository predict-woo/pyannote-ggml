import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { getCapabilities, type AlignedSegment, type TranscriptSegment } from '../src/index.js';

const repoRoot = path.resolve(fileURLToPath(new URL('.', import.meta.url)), '../../../../../');

export const capabilities = getCapabilities();
export const pipelineSupported = capabilities.pipeline;
export const lowLevelSupported = capabilities.whisper && capabilities.vad;

export const sampleRate = 16000;

export const whisperModelPath = path.join(repoRoot, 'whisper.cpp/models/ggml-base.en.bin');
export const vadModelPath = path.join(repoRoot, 'whisper.cpp/models/ggml-silero-v6.2.0.bin');
export const whisperAudioPath = path.join(repoRoot, 'whisper.cpp/samples/jfk.wav');
export const pipelineAudioPath = path.join(repoRoot, 'samples/sample.wav');
export const transcribeBinPath = path.join(repoRoot, 'diarization-ggml/build/bin/transcribe');

export const pipelineConfig = {
  segModelPath: path.join(repoRoot, 'models/segmentation-ggml/segmentation.gguf'),
  embModelPath: path.join(repoRoot, 'models/embedding-ggml/embedding.gguf'),
  pldaPath: path.join(repoRoot, 'diarization-ggml/plda.gguf'),
  coremlPath: path.join(repoRoot, 'models/embedding-ggml/embedding.mlpackage'),
  segCoremlPath: path.join(repoRoot, 'models/segmentation-ggml/segmentation.mlpackage'),
  whisperModelPath,
  language: 'en',
} as const;

export const pipelineVadConfig = {
  ...pipelineConfig,
  vadModelPath,
} as const;

export const transcriptionOnlyConfig = {
  segModelPath: path.join(repoRoot, 'models/segmentation-ggml/segmentation.gguf'),
  segCoremlPath: path.join(repoRoot, 'models/segmentation-ggml/segmentation.mlpackage'),
  whisperModelPath,
  vadModelPath,
  language: 'en',
  transcriptionOnly: true,
} as const;

export function hasRealLowLevelAssets(): boolean {
  return [
    whisperModelPath,
    vadModelPath,
    whisperAudioPath,
  ].every(existsSync);
}

export function hasRealPipelineAssets(): boolean {
  return [
    pipelineConfig.segModelPath,
    pipelineConfig.embModelPath,
    pipelineConfig.pldaPath,
    pipelineConfig.coremlPath,
    pipelineConfig.segCoremlPath,
    pipelineConfig.whisperModelPath,
    pipelineAudioPath,
  ].every(existsSync);
}

export function hasTranscribeBinary(): boolean {
  return existsSync(transcribeBinPath);
}

export function loadWav(filePath: string): Float32Array {
  const buffer = readFileSync(filePath);
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

  let offset = 12;
  const riff = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (riff !== 'RIFF') {
    throw new Error(`Not a RIFF file: ${filePath}`);
  }

  while (offset < view.byteLength - 8) {
    const chunkId = String.fromCharCode(
      view.getUint8(offset),
      view.getUint8(offset + 1),
      view.getUint8(offset + 2),
      view.getUint8(offset + 3),
    );
    const chunkSize = view.getUint32(offset + 4, true);
    offset += 8;

    if (chunkId === 'data') {
      const numSamples = chunkSize / 2;
      const samples = new Float32Array(numSamples);
      for (let i = 0; i < numSamples; i++) {
        samples[i] = view.getInt16(offset + i * 2, true) / 32768.0;
      }
      return samples;
    }

    offset += chunkSize;
    if (chunkSize % 2 !== 0) {
      offset++;
    }
  }

  throw new Error(`No data chunk found in WAV file: ${filePath}`);
}

export function normalizeText(text: string): string {
  return text
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase();
}

export function transcriptText(
  segments: TranscriptSegment[] | AlignedSegment[],
): string {
  return normalizeText(segments.map((segment) => segment.text).join(' '));
}

export function getHighestEnergyWindow(samples: Float32Array, windowSize: number): Float32Array {
  if (samples.length <= windowSize) {
    return samples;
  }

  let bestOffset = 0;
  let bestEnergy = -1;

  for (let offset = 0; offset + windowSize <= samples.length; offset += windowSize) {
    let energy = 0;
    for (let i = 0; i < windowSize; i++) {
      const value = samples[offset + i];
      energy += value * value;
    }
    if (energy > bestEnergy) {
      bestEnergy = energy;
      bestOffset = offset;
    }
  }

  return samples.slice(bestOffset, bestOffset + windowSize);
}
