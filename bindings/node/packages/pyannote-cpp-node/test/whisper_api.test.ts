import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import {
  WhisperContext,
  createVadContext,
  getCapabilities,
  getGpuDevices,
  transcribeAsync,
} from '../src/index.js';
import { hasRealLowLevelAssets, lowLevelSupported } from './real_assets.js';

const repoRoot = path.resolve(fileURLToPath(new URL('.', import.meta.url)), '../../../../../');
const whisperModelPath = path.join(repoRoot, 'whisper.cpp/models/ggml-base.en.bin');
const audioPath = path.join(repoRoot, 'whisper.cpp/samples/jfk.wav');
const vadModelPath = path.join(repoRoot, 'whisper.cpp/models/ggml-silero-v6.2.0.bin');

describe.runIf(lowLevelSupported && hasRealLowLevelAssets())(
  'low-level whisper API',
  () => {
    it('reports available capabilities and gpu enumeration', () => {
      const capabilities = getCapabilities();
      expect(capabilities.whisper).toBe(true);
      expect(capabilities.vad).toBe(true);
      expect(Array.isArray(getGpuDevices())).toBe(true);
    });

    it('transcribes audio through WhisperContext', async () => {
      const ctx = new WhisperContext({
        model: whisperModelPath,
        use_gpu: false,
        no_prints: true,
      });

      try {
        const result = await transcribeAsync(ctx, {
          fname_inp: audioPath,
          language: 'en',
          no_timestamps: false,
        });

        expect(result.segments.length).toBeGreaterThan(0);
        const text = result.segments.map((segment) => segment.text).join(' ').toLowerCase();
        expect(text).toContain('ask not');
      } finally {
        ctx.free();
      }
    });

    it('creates and runs a VadContext', () => {
      const vad = createVadContext({
        model: vadModelPath,
        no_prints: true,
      });

      try {
        expect(vad.getSampleRate()).toBe(16000);
        expect(vad.getWindowSamples()).toBeGreaterThan(0);

        const probability = vad.process(new Float32Array(vad.getWindowSamples()));
        expect(probability).toBeGreaterThanOrEqual(0);
        expect(probability).toBeLessThanOrEqual(1);
      } finally {
        vad.free();
      }
    });
  },
);
