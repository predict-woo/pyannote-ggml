export interface ModelConfig {
  segModelPath: string;
  embModelPath: string;
  pldaPath: string;
  coremlPath: string;
  segCoremlPath: string;
}

export interface VADChunk {
  chunkIndex: number;
  startTime: number;
  duration: number;
  numFrames: number;
  vad: Float32Array;
}

export interface Segment {
  start: number;
  duration: number;
  speaker: string;
}

export interface DiarizationResult {
  segments: Segment[];
}
