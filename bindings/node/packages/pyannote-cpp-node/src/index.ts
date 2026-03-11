import { promisify } from 'node:util';

import { Pipeline } from './Pipeline.js';
import { PipelineSession } from './PipelineSession.js';
import { getBinding, getCapabilities, getNativePackageNameForPlatform, getPlatformKey } from './binding.js';
import type {
  AlignedSegment,
  Capabilities,
  DecodeOptions,
  GpuDevice,
  ModelConfig,
  StreamingSegment,
  StreamingToken,
  TranscriptSegment,
  TranscribeOptions,
  TranscribeOptionsBase,
  TranscribeOptionsBuffer,
  TranscribeOptionsFile,
  TranscribeResult,
  TranscriptionResult,
  VadContext as VadContextInstance,
  VadContextConstructor,
  VadContextOptions,
  WhisperContext as WhisperContextInstance,
  WhisperContextConstructor,
  WhisperContextOptions,
} from './types.js';

const binding = getBinding();

export const WhisperContextClass = binding.WhisperContext;
export const VadContextClass = binding.VadContext;
export const WhisperContext = binding.WhisperContext;
export const VadContext = binding.VadContext;
export const transcribe = binding.transcribe;
export const transcribeAsync = promisify(binding.transcribe) as (
  context: WhisperContextInstance,
  options: TranscribeOptions,
) => Promise<TranscribeResult>;

export function createWhisperContext(options: WhisperContextOptions): WhisperContextInstance {
  return new binding.WhisperContext(options);
}

export function createVadContext(options: VadContextOptions): VadContextInstance {
  return new binding.VadContext(options);
}

export function getGpuDevices(): GpuDevice[] {
  return binding.getGpuDevices();
}

export {
  Pipeline,
  PipelineSession,
  getBinding,
  getCapabilities,
  getNativePackageNameForPlatform,
  getPlatformKey,
};

export type {
  AlignedSegment,
  Capabilities,
  DecodeOptions,
  GpuDevice,
  ModelConfig,
  StreamingSegment,
  StreamingToken,
  TranscriptSegment,
  TranscribeOptions,
  TranscribeOptionsBase,
  TranscribeOptionsBuffer,
  TranscribeOptionsFile,
  TranscribeResult,
  TranscriptionResult,
  VadContextConstructor,
  VadContextOptions,
  WhisperContextConstructor,
  WhisperContextOptions,
};

export type {
  NativeBinding,
  NativePipelineModel,
  NativePipelineSession,
} from './binding.js';

export default {
  WhisperContext,
  WhisperContextClass,
  VadContext,
  VadContextClass,
  transcribe,
  transcribeAsync,
  createWhisperContext,
  createVadContext,
  getGpuDevices,
  getCapabilities,
  Pipeline,
  PipelineSession,
};
