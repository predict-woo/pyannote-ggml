import type { NativeStreamingSession } from './binding.js';
import type { DiarizationResult, VADChunk } from './types.js';

export class StreamingSession {
  private native: NativeStreamingSession;

  constructor(native: NativeStreamingSession) {
    this.native = native;
  }

  async push(audio: Float32Array): Promise<VADChunk[]> {
    if (this.native.isClosed) {
      throw new Error('Session is closed');
    }
    if (!(audio instanceof Float32Array)) {
      throw new TypeError('Expected Float32Array');
    }
    return this.native.push(audio);
  }

  async recluster(): Promise<DiarizationResult> {
    if (this.native.isClosed) {
      throw new Error('Session is closed');
    }
    return this.native.recluster();
  }

  async finalize(): Promise<DiarizationResult> {
    if (this.native.isClosed) {
      throw new Error('Session is closed');
    }
    return this.native.finalize();
  }

  close(): void {
    this.native.close();
  }

  get isClosed(): boolean {
    return this.native.isClosed;
  }
}
