import { createRequire } from 'module';

import type { DiarizationResult, ModelConfig, VADChunk } from './types.js';

const require = createRequire(import.meta.url);

export interface NativePyannoteModel {
  diarize(audio: Float32Array): Promise<DiarizationResult>;
  createStreamingSession(): NativeStreamingSession;
  close(): void;
  isClosed: boolean;
}

export interface NativeStreamingSession {
  push(audio: Float32Array): Promise<VADChunk[]>;
  recluster(): Promise<DiarizationResult>;
  finalize(): Promise<DiarizationResult>;
  close(): void;
  isClosed: boolean;
}

export interface NativeBinding {
  PyannoteModel: new (config: ModelConfig) => NativePyannoteModel;
  StreamingSession: new (...args: unknown[]) => NativeStreamingSession;
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
    typeof candidate.PyannoteModel === 'function' &&
    typeof candidate.StreamingSession === 'function'
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
      `Invalid native module export from '${packageName}'. Expected PyannoteModel and StreamingSession constructors.`,
    );
  }

  cachedBinding = loaded;
  return cachedBinding;
}
