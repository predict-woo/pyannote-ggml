export interface ModelConfig {
  segModelPath: string;
  embModelPath: string;
  pldaPath: string;
  coremlPath: string;
  segCoremlPath: string;
  zeroLatency?: boolean;   // Pre-fill 10s silence so first push returns frames immediately (default: false)
}

export interface VADChunk {
  chunkIndex: number;
  startFrame: number;    // global frame index of first new frame; time = startFrame * 0.016875
  numFrames: number;     // ~59-60 per push, or 589 for first chunk in normal mode
  vad: Float32Array;     // [numFrames] voice activity (1.0 = speech, 0.0 = silence)
}

export interface Segment {
  start: number;
  duration: number;
  speaker: string;
}

export interface DiarizationResult {
  segments: Segment[];
}
