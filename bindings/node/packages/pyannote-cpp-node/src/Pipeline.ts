import { accessSync } from 'node:fs';
import { getBinding, type NativePipelineModel } from './binding.js';
import { PipelineSession } from './PipelineSession.js';
import type { ModelConfig, TranscriptionResult, DecodeOptions } from './types.js';

export class Pipeline {
  private native: NativePipelineModel;

  private constructor(native: NativePipelineModel) {
    this.native = native;
  }

  static async load(config: ModelConfig): Promise<Pipeline> {
    const requiredPaths = [
      config.segModelPath,
      config.embModelPath,
      config.pldaPath,
      config.coremlPath,
      config.segCoremlPath,
      config.whisperModelPath,
    ];
    for (const path of requiredPaths) accessSync(path);
    if (config.vadModelPath) accessSync(config.vadModelPath);

    const binding = getBinding();
    const native = new binding.PipelineModel(config);
    await native.loadModels();
    return new Pipeline(native);
  }

  async transcribe(audio: Float32Array): Promise<TranscriptionResult> {
    if (this.native.isClosed) throw new Error('Pipeline is closed');
    if (!(audio instanceof Float32Array)) throw new TypeError('Expected Float32Array');
    if (audio.length === 0) throw new Error('Audio must not be empty');
    return this.native.transcribe(audio);
  }

  async transcribeOffline(audio: Float32Array): Promise<TranscriptionResult> {
    if (this.native.isClosed) throw new Error('Pipeline is closed');
    if (!(audio instanceof Float32Array)) throw new TypeError('Expected Float32Array');
    if (audio.length === 0) throw new Error('Audio must not be empty');
    return this.native.transcribeOffline(audio);
  }

  setLanguage(language: string): void {
    if (this.native.isClosed) throw new Error('Pipeline is closed');
    this.native.setLanguage(language);
  }

  setDecodeOptions(options: DecodeOptions): void {
    if (this.native.isClosed) throw new Error('Pipeline is closed');
    this.native.setDecodeOptions({ ...options });
  }

  createSession(): PipelineSession {
    if (this.native.isClosed) throw new Error('Pipeline is closed');
    const session = new PipelineSession();
    const nativeSession = this.native.createSession(
      (segments: any[]) => session._onSegmentsCallback(segments),
      (audio: Float32Array) => session._onAudioCallback(audio),
    );
    session._setNative(nativeSession);
    return session;
  }

  async setUseCoreml(useCoreml: boolean): Promise<void> {
    if (this.native.isClosed) throw new Error('Pipeline is closed');
    return this.native.switchWhisperMode(useCoreml);
  }

  close(): void { this.native.close(); }
  get isClosed(): boolean { return this.native.isClosed; }
}
