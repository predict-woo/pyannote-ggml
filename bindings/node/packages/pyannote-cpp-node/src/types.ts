export interface Capabilities {
  whisper: boolean;
  vad: boolean;
  gpuDiscovery: boolean;
  pipeline: boolean;
  diarization: boolean;
}

export interface WhisperContextOptions {
  model: string;
  use_gpu?: boolean;
  flash_attn?: boolean;
  gpu_device?: number;
  use_coreml?: boolean;
  use_openvino?: boolean;
  openvino_model_path?: string;
  openvino_device?: string;
  openvino_cache_dir?: string;
  dtw?: string;
  dtw_norm_top_k?: number;
  no_prints?: boolean;
}

export interface TranscribeOptionsBase {
  language?: string;
  translate?: boolean;
  detect_language?: boolean;
  n_threads?: number;
  n_processors?: number;
  offset_ms?: number;
  duration_ms?: number;
  audio_ctx?: number;
  no_timestamps?: boolean;
  single_segment?: boolean;
  max_len?: number;
  max_tokens?: number;
  max_context?: number;
  split_on_word?: boolean;
  token_timestamps?: boolean;
  word_thold?: number;
  comma_in_time?: boolean;
  temperature?: number;
  temperature_inc?: number;
  best_of?: number;
  beam_size?: number;
  no_fallback?: boolean;
  entropy_thold?: number;
  logprob_thold?: number;
  no_speech_thold?: number;
  prompt?: string;
  no_context?: boolean;
  suppress_blank?: boolean;
  suppress_nst?: boolean;
  diarize?: boolean;
  tinydiarize?: boolean;
  print_special?: boolean;
  print_progress?: boolean;
  print_realtime?: boolean;
  print_timestamps?: boolean;
  vad?: boolean;
  vad_model?: string;
  vad_threshold?: number;
  vad_min_speech_duration_ms?: number;
  vad_min_silence_duration_ms?: number;
  vad_max_speech_duration_s?: number;
  vad_speech_pad_ms?: number;
  vad_samples_overlap?: number;
  progress_callback?: (progress: number) => void;
  on_new_segment?: (segment: StreamingSegment) => void;
}

export interface TranscribeOptionsFile extends TranscribeOptionsBase {
  fname_inp: string;
  pcmf32?: never;
}

export interface TranscribeOptionsBuffer extends TranscribeOptionsBase {
  pcmf32: Float32Array;
  fname_inp?: never;
}

export type TranscribeOptions = TranscribeOptionsFile | TranscribeOptionsBuffer;

export interface StreamingToken {
  text: string;
  probability: number;
  t0: number;
  t1: number;
  t_dtw: number;
}

export interface TranscriptSegment {
  start: string;
  end: string;
  text: string;
  tokens?: StreamingToken[];
}

export interface StreamingSegment {
  start: string;
  end: string;
  text: string;
  segment_index: number;
  is_partial: boolean;
  tokens?: StreamingToken[];
}

export interface TranscribeResult {
  segments: TranscriptSegment[];
  language?: string;
}

export interface GpuDevice {
  index: number;
  name: string;
  description: string;
  type: 'gpu' | 'igpu';
  memory_free: number;
  memory_total: number;
}

export interface VadContextOptions {
  model: string;
  threshold?: number;
  n_threads?: number;
  no_prints?: boolean;
}

export interface WhisperContext {
  getSystemInfo(): string;
  isMultilingual(): boolean;
  free(): void;
}

export interface WhisperContextConstructor {
  new (options: WhisperContextOptions): WhisperContext;
}

export interface VadContext {
  getWindowSamples(): number;
  getSampleRate(): number;
  process(samples: Float32Array): number;
  reset(): void;
  free(): void;
}

export interface VadContextConstructor {
  new (options: VadContextOptions): VadContext;
}

export type TranscribeCallback = (
  error: Error | null,
  result?: TranscribeResult
) => void;

export interface ModelConfig {
  segModelPath: string;
  embModelPath?: string;
  pldaPath?: string;
  coremlPath?: string;
  segCoremlPath: string;
  whisperModelPath: string;
  vadModelPath?: string;
  transcriptionOnly?: boolean;
  useGpu?: boolean;
  flashAttn?: boolean;
  gpuDevice?: number;
  useCoreml?: boolean;
  noPrints?: boolean;
  nThreads?: number;
  language?: string;
  translate?: boolean;
  detectLanguage?: boolean;
  temperature?: number;
  temperatureInc?: number;
  noFallback?: boolean;
  beamSize?: number;
  bestOf?: number;
  entropyThold?: number;
  logprobThold?: number;
  noSpeechThold?: number;
  prompt?: string;
  noContext?: boolean;
  suppressBlank?: boolean;
  suppressNst?: boolean;
}

export interface DecodeOptions {
  language?: string;
  translate?: boolean;
  detectLanguage?: boolean;
  nThreads?: number;
  temperature?: number;
  temperatureInc?: number;
  noFallback?: boolean;
  beamSize?: number;
  bestOf?: number;
  entropyThold?: number;
  logprobThold?: number;
  noSpeechThold?: number;
  prompt?: string;
  noContext?: boolean;
  suppressBlank?: boolean;
  suppressNst?: boolean;
}

export interface AlignedSegment {
  speaker: string;
  start: number;
  duration: number;
  text: string;
}

export interface TranscriptionResult {
  segments: AlignedSegment[];
  filteredAudio?: Float32Array;
}
