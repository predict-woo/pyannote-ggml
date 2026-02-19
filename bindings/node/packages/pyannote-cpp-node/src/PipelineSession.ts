import { EventEmitter } from 'node:events';

import type { NativePipelineSession } from './binding.js';
import type { AlignedSegment, TranscriptionResult } from './types.js';

export interface PipelineSessionEvents {
  segments: [segments: AlignedSegment[], audio: Float32Array];
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

  _onNativeCallback(segments: any[], audio: Float32Array): void {
    const typedSegments: AlignedSegment[] = segments as AlignedSegment[];
    this.emit('segments', typedSegments, audio);
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
