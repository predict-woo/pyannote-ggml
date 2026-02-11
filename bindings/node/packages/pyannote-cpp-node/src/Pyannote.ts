import { accessSync } from 'node:fs';

import { getBinding, type NativePyannoteModel } from './binding.js';
import { StreamingSession } from './StreamingSession.js';
import type { DiarizationResult, ModelConfig } from './types.js';

export class Pyannote {
  private native: NativePyannoteModel;

  private constructor(native: NativePyannoteModel) {
    this.native = native;
  }

  static async load(config: ModelConfig): Promise<Pyannote> {
    const paths = [
      config.segModelPath,
      config.embModelPath,
      config.pldaPath,
      config.coremlPath,
      config.segCoremlPath,
    ];

    for (const path of paths) {
      accessSync(path);
    }

    const binding = getBinding();
    const native = new binding.PyannoteModel(config);
    return new Pyannote(native);
  }

  async diarize(audio: Float32Array): Promise<DiarizationResult> {
    if (this.native.isClosed) {
      throw new Error('Model is closed');
    }
    if (!(audio instanceof Float32Array)) {
      throw new TypeError('Expected Float32Array');
    }
    if (audio.length === 0) {
      throw new Error('Audio must not be empty');
    }
    return this.native.diarize(audio);
  }

  createStreamingSession(): StreamingSession {
    if (this.native.isClosed) {
      throw new Error('Model is closed');
    }
    const nativeSession = this.native.createStreamingSession();
    return new StreamingSession(nativeSession);
  }

  close(): void {
    this.native.close();
  }

  get isClosed(): boolean {
    return this.native.isClosed;
  }
}
