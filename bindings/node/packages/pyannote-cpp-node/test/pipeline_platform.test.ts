import { afterEach, describe, expect, it, vi } from 'vitest';

describe('pipeline platform guard', () => {
  afterEach(() => {
    vi.resetModules();
    vi.clearAllMocks();
  });

  it('fails before path validation when the native addon reports pipeline support is unavailable', async () => {
    vi.doMock('../src/binding.js', () => ({
      getBinding: () => {
        throw new Error('getBinding should not be called when pipeline capability is false');
      },
      getCapabilities: () => ({
        whisper: true,
        vad: true,
        gpuDiscovery: true,
        pipeline: false,
        diarization: false,
      }),
    }));

    const { Pipeline } = await import('../src/Pipeline.js');

    await expect(
      Pipeline.load({
        segModelPath: '/missing/segmentation.gguf',
        segCoremlPath: '/missing/segmentation.mlpackage',
        whisperModelPath: '/missing/ggml-base.en.bin',
      }),
    ).rejects.toThrow(
      'Pipeline is only supported on macOS Apple Silicon. Low-level whisper/VAD APIs remain available on this platform.',
    );
  });
});
