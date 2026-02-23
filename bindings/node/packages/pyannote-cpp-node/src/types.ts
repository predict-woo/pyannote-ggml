export interface ModelConfig {
  // === Required Model Paths ===
  /** Path to segmentation GGUF model */
  segModelPath: string;
  /** Path to embedding GGUF model */
  embModelPath: string;
  /** Path to PLDA GGUF model */
  pldaPath: string;
  /** Path to embedding CoreML .mlpackage directory */
  coremlPath: string;
  /** Path to segmentation CoreML .mlpackage directory */
  segCoremlPath: string;
  /** Path to Whisper GGUF model */
  whisperModelPath: string;

  // === Optional Model Paths ===
  /** Path to Silero VAD model (optional, enables silence compression) */
  vadModelPath?: string;

  // === Whisper Context Options (model loading) ===
  /** Enable GPU acceleration (default: true) */
  useGpu?: boolean;
  /** Enable Flash Attention (default: true) */
  flashAttn?: boolean;
  /** GPU device index (default: 0) */
  gpuDevice?: number;
  /**
   * Enable CoreML acceleration for Whisper encoder on macOS (default: false).
   * The CoreML model must be placed next to the GGUF model with naming convention:
   * e.g., ggml-base.en.bin -> ggml-base.en-encoder.mlmodelc/
   */
  useCoreml?: boolean;
  /** Suppress whisper.cpp log output (default: false) */
  noPrints?: boolean;

  // === Whisper Decode Options ===
  /** Number of threads for Whisper inference (default: 4) */
  nThreads?: number;
  /** Language code (e.g., 'en', 'zh'). Omit for auto-detect. (default: 'en') */
  language?: string;
  /** Translate non-English speech to English (default: false) */
  translate?: boolean;
  /** Auto-detect spoken language. Overrides 'language' when true. (default: false) */
  detectLanguage?: boolean;

  // === Sampling ===
  /** Sampling temperature. 0.0 = greedy deterministic. (default: 0.0) */
  temperature?: number;
  /** Temperature increment for fallback retries (default: 0.2) */
  temperatureInc?: number;
  /** Disable temperature fallback. If true, temperatureInc is ignored. (default: false) */
  noFallback?: boolean;
  /** Beam search size. -1 uses greedy decoding. >1 enables beam search. (default: -1) */
  beamSize?: number;
  /** Best-of-N sampling candidates for greedy decoding (default: 5) */
  bestOf?: number;

  // === Thresholds ===
  /** Entropy threshold for decoder fallback (default: 2.4) */
  entropyThold?: number;
  /** Log probability threshold for decoder fallback (default: -1.0) */
  logprobThold?: number;
  /** No-speech probability threshold (default: 0.6) */
  noSpeechThold?: number;

  // === Context ===
  /** Initial prompt text to condition the decoder (default: none) */
  prompt?: string;
  /** Don't use previous segment as context for next segment (default: true) */
  noContext?: boolean;
  /** Suppress blank outputs at the beginning of segments (default: true) */
  suppressBlank?: boolean;
  /** Suppress non-speech tokens (default: false) */
  suppressNst?: boolean;
}

export interface DecodeOptions {
  /** Language code (e.g., 'en', 'zh'). Omit for auto-detect. */
  language?: string;
  /** Translate non-English speech to English */
  translate?: boolean;
  /** Auto-detect spoken language. Overrides 'language' when true. */
  detectLanguage?: boolean;
  /** Number of threads for Whisper inference */
  nThreads?: number;
  /** Sampling temperature. 0.0 = greedy deterministic. */
  temperature?: number;
  /** Temperature increment for fallback retries */
  temperatureInc?: number;
  /** Disable temperature fallback. If true, temperatureInc is ignored. */
  noFallback?: boolean;
  /** Beam search size. -1 uses greedy decoding. >1 enables beam search. */
  beamSize?: number;
  /** Best-of-N sampling candidates for greedy decoding */
  bestOf?: number;
  /** Entropy threshold for decoder fallback */
  entropyThold?: number;
  /** Log probability threshold for decoder fallback */
  logprobThold?: number;
  /** No-speech probability threshold */
  noSpeechThold?: number;
  /** Initial prompt text to condition the decoder */
  prompt?: string;
  /** Don't use previous segment as context for next segment */
  noContext?: boolean;
  /** Suppress blank outputs at the beginning of segments */
  suppressBlank?: boolean;
  /** Suppress non-speech tokens */
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
}
