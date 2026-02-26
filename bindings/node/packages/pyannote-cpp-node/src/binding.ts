import { createRequire } from 'module';

import type { ModelConfig, TranscriptionResult } from './types.js';

const require = createRequire(import.meta.url);

export interface NativePipelineModel {
  transcribe(audio: Float32Array): Promise<TranscriptionResult>;
  transcribeOffline(audio: Float32Array): Promise<TranscriptionResult>;
  setLanguage(language: string): void;
  setDecodeOptions(options: Record<string, unknown>): void;
  createSession(
    segmentsCb: (segments: any[]) => void,
    audioCb: (audio: Float32Array) => void,
  ): NativePipelineSession;
  close(): void;
  isClosed: boolean;
  loadModels(): Promise<void>;
  isLoaded: boolean;
}

export interface NativePipelineSession {
  push(audio: Float32Array): Promise<boolean[]>;
  setLanguage(language: string): void;
  setDecodeOptions(options: Record<string, unknown>): void;
  finalize(): Promise<TranscriptionResult>;
  close(): void;
  isClosed: boolean;
}

export interface NativeBinding {
  PipelineModel: new (config: ModelConfig) => NativePipelineModel;
  PipelineSession: new (...args: unknown[]) => NativePipelineSession;
}

let cachedBinding: NativeBinding | null = null;

function getPackageName(): string {
  if (process.platform !== 'darwin') {
    throw new Error(
      `Unsupported platform: ${process.platform}. pyannote-cpp-node currently supports macOS only.`,
    );
  }

  if (process.arch === 'arm64') {
    return '@pyannote-cpp-node/darwin-arm64';
  }

  if (process.arch === 'x64') {
    return '@pyannote-cpp-node/darwin-x64';
  }

  throw new Error(
    `Unsupported architecture on macOS: ${process.arch}. Supported architectures are arm64 and x64.`,
  );
}

function isNativeBinding(value: unknown): value is NativeBinding {
  if (typeof value !== 'object' || value === null) {
    return false;
  }

  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.PipelineModel === 'function' &&
    typeof candidate.PipelineSession === 'function'
  );
}

export function getBinding(): NativeBinding {
  if (cachedBinding !== null) {
    return cachedBinding;
  }

  const packageName = getPackageName();

  let loaded: unknown;
  try {
    loaded = require(packageName);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Failed to load native module '${packageName}'. Ensure the platform package is installed. Original error: ${message}`,
    );
  }

  if (!isNativeBinding(loaded)) {
    throw new Error(
      `Invalid native module export from '${packageName}'. Expected PipelineModel and PipelineSession constructors.`,
    );
  }

  cachedBinding = loaded;
  return cachedBinding;
}
