import { existsSync } from 'node:fs';
import { createRequire } from 'node:module';
import { arch as osArch, platform as osPlatform } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import type {
  Capabilities,
  GpuDevice,
  ModelConfig,
  TranscribeCallback,
  TranscribeOptions,
  TranscriptionResult,
  WhisperContext,
  VadContextConstructor,
  WhisperContextConstructor,
} from './types.js';

const require = createRequire(import.meta.url);
const thisDir = fileURLToPath(new URL('.', import.meta.url));

const SUPPORTED_PLATFORMS = {
  'darwin-arm64': '@pyannote-cpp-node/darwin-arm64',
  'win32-x64': '@pyannote-cpp-node/win32-x64',
} as const;

export interface NativePipelineModel {
  transcribe(audio: Float32Array): Promise<TranscriptionResult>;
  transcribeOffline(
    audio: Float32Array,
    onProgress?: (phase: number, progress: number) => void,
    onSegment?: (start: number, end: number, text: string) => void,
  ): Promise<TranscriptionResult>;
  setLanguage(language: string): void;
  setDecodeOptions(options: Record<string, unknown>): void;
  createSession(
    segmentsCb: (segments: unknown[]) => void,
    audioCb: (audio: Float32Array) => void,
  ): NativePipelineSession;
  close(): void;
  isClosed: boolean;
  loadModels(): Promise<void>;
  isLoaded: boolean;
  switchWhisperMode(useCoreml: boolean): Promise<void>;
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
  WhisperContext: WhisperContextConstructor;
  VadContext: VadContextConstructor;
  transcribe: (
    context: WhisperContext,
    options: TranscribeOptions,
    callback: TranscribeCallback,
  ) => void;
  getGpuDevices: () => GpuDevice[];
  PipelineModel: new (config: ModelConfig) => NativePipelineModel;
  PipelineSession: new (...args: unknown[]) => NativePipelineSession;
  getCapabilities: () => Capabilities;
}

let cachedBinding: NativeBinding | null = null;

export function getPlatformKey(
  platform: string = osPlatform(),
  arch: string = osArch(),
): string {
  return `${platform}-${arch}`;
}

export function getNativePackageNameForPlatform(
  platform: string = osPlatform(),
  arch: string = osArch(),
): string {
  const platformKey = getPlatformKey(platform, arch);
  const packageName = SUPPORTED_PLATFORMS[platformKey as keyof typeof SUPPORTED_PLATFORMS];
  if (packageName) {
    return packageName;
  }

  throw new Error(
    `Unsupported platform: ${platformKey}. pyannote-cpp-node supports macOS Apple Silicon and Windows x64 only.`,
  );
}

function tryWorkspacePath(platformKey: string): string | null {
  const possiblePaths = [
    join(thisDir, '..', '..', platformKey),
    join(thisDir, '..', '..', '..', platformKey),
  ];

  for (const candidate of possiblePaths) {
    if (existsSync(join(candidate, 'package.json'))) {
      return candidate;
    }
  }

  return null;
}

function isNativeBinding(value: unknown): value is NativeBinding {
  if (typeof value !== 'object' || value === null) {
    return false;
  }

  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.WhisperContext === 'function' &&
    typeof candidate.VadContext === 'function' &&
    typeof candidate.transcribe === 'function' &&
    typeof candidate.getGpuDevices === 'function' &&
    typeof candidate.PipelineModel === 'function' &&
    typeof candidate.PipelineSession === 'function' &&
    typeof candidate.getCapabilities === 'function'
  );
}

export function getBinding(): NativeBinding {
  if (cachedBinding !== null) {
    return cachedBinding;
  }

  const platformKey = getPlatformKey();
  const packageName = getNativePackageNameForPlatform();

  const workspacePath = tryWorkspacePath(platformKey);
  if (workspacePath) {
    const loaded = require(workspacePath) as unknown;
    if (!isNativeBinding(loaded)) {
      throw new Error(`Invalid native module export from workspace package '${workspacePath}'.`);
    }
    cachedBinding = loaded;
    return cachedBinding;
  }

  try {
    const loaded = require(packageName) as unknown;
    if (!isNativeBinding(loaded)) {
      throw new Error(`Invalid native module export from '${packageName}'.`);
    }
    cachedBinding = loaded;
    return cachedBinding;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const moduleNotFound =
      error &&
      typeof error === 'object' &&
      'code' in error &&
      (error as NodeJS.ErrnoException).code === 'MODULE_NOT_FOUND';

    if (moduleNotFound) {
      throw new Error(
        `Native binary not found. Please ensure ${packageName} is installed. Original error: ${message}`,
      );
    }

    const windowsHint =
      process.platform === 'win32' && message.includes('The specified module could not be found')
        ? '\nThis usually means a required DLL dependency is missing. If running inside Electron, ensure native addon DLLs are excluded from asar packaging (use asarUnpack in your build config).'
        : '';

    throw new Error(`Failed to load native module '${packageName}'. ${message}${windowsHint}`);
  }
}

export function getCapabilities(): Capabilities {
  return getBinding().getCapabilities();
}
