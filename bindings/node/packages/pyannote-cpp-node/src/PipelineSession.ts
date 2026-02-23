import { EventEmitter } from 'node:events';

import type { NativePipelineSession } from './binding.js';
import type { AlignedSegment, TranscriptionResult, DecodeOptions } from './types.js';

export interface PipelineSessionEvents {
  segments: [segments: AlignedSegment[]];
  audio: [audio: Float32Array];
  error: [error: Error];
}

export class PipelineSession extends EventEmitter {
  private native: NativePipelineSession | null = null;

  constructor() {
    super();
  }

  _setNative(native: NativePipelineSession): void {
    this.native = native;
  }

  _onSegmentsCallback(segments: any[]): void {
    this.emit('segments', segments as AlignedSegment[]);
  }

  _onAudioCallback(audio: Float32Array): void {
    this.emit('audio', audio);
  }

  async push(audio: Float32Array): Promise<boolean[]> {
    if (!this.native || this.native.isClosed) {
      throw new Error('Session is closed');
    }
    if (!(audio instanceof Float32Array)) {
      throw new TypeError('Expected Float32Array');
    }

    return this.native.push(audio);
  }

  setLanguage(language: string): void {
    if (!this.native || this.native.isClosed) {
      throw new Error('Session is closed');
    }
    this.native.setLanguage(language);
  }

  setDecodeOptions(options: DecodeOptions): void {
    if (!this.native || this.native.isClosed) {
      throw new Error('Session is closed');
    }
    this.native.setDecodeOptions({ ...options });
  }

  async finalize(): Promise<TranscriptionResult> {
    if (!this.native || this.native.isClosed) {
      throw new Error('Session is closed');
    }

    return this.native.finalize();
  }

  close(): void {
    if (this.native) {
      this.native.close();
    }
  }

  get isClosed(): boolean {
    return !this.native || this.native.isClosed;
  }
}
