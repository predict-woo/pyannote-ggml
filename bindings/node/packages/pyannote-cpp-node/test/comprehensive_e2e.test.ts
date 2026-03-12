import { execFileSync } from 'node:child_process';

import { beforeAll, afterAll, describe, expect, it } from 'vitest';

import {
  Pipeline,
  WhisperContext,
  createVadContext,
  transcribe,
  transcribeAsync,
  type AlignedSegment,
  type TranscriptSegment,
} from '../src/index.js';
import {
  getHighestEnergyWindow,
  hasRealLowLevelAssets,
  hasRealPipelineAssets,
  hasTranscribeBinary,
  loadWav,
  lowLevelSupported,
  pipelineAudioPath,
  pipelineConfig,
  pipelineVadConfig,
  pipelineSupported,
  sampleRate,
  transcriptText,
  transcribeBinPath,
  vadModelPath,
  whisperAudioPath,
  whisperModelPath,
} from './real_assets.js';

function transcribeWithCallback(
  context: WhisperContext,
  options: Parameters<typeof transcribe>[1],
): Promise<Awaited<ReturnType<typeof transcribeAsync>>> {
  return new Promise((resolve, reject) => {
    transcribe(context, options, (error, result) => {
      if (error) {
        reject(error);
        return;
      }

      resolve(result!);
    });
  });
}

function collectSpeakerSet(segments: AlignedSegment[]): Set<string> {
  return new Set(segments.map((segment) => segment.speaker));
}

function assertSortedByTime(segments: Array<TranscriptSegment | AlignedSegment>): void {
  for (let i = 1; i < segments.length; i++) {
    const previousStart = 'start' in segments[i - 1] ? segments[i - 1].start : '';
    const currentStart = 'start' in segments[i] ? segments[i].start : '';
    expect(String(currentStart) >= String(previousStart)).toBe(true);
  }
}

function jsonEscape(value: string): string {
  return value
    .replace(/\\/g, '\\\\')
    .replace(/"/g, '\\"')
    .replace(/\n/g, '\\n')
    .replace(/\r/g, '\\r')
    .replace(/\t/g, '\\t');
}

function trimAsciiWhitespace(value: string): string {
  let start = 0;
  while (start < value.length && value.charCodeAt(start) <= 0x20) {
    start++;
  }

  let end = value.length;
  while (end > start && value.charCodeAt(end - 1) <= 0x20) {
    end--;
  }

  return value.slice(start, end);
}

function writeSegmentsJson(segments: AlignedSegment[]): string {
  let output = '{\n  "segments": [\n';

  for (let i = 0; i < segments.length; i++) {
    const segment = segments[i];
    output += `    {"speaker": "${jsonEscape(segment.speaker)}", "start": ${segment.start.toFixed(6)}, "duration": ${segment.duration.toFixed(6)}, "text": "${jsonEscape(trimAsciiWhitespace(segment.text))}"}`;
    if (i + 1 < segments.length) {
      output += ',';
    }
    output += '\n';
  }

  output += '  ]\n}\n';
  return output;
}

function wordAgreementRatio(left: string, right: string): number {
  const leftWords = left.split(/\s+/).filter(Boolean);
  const rightWords = right.split(/\s+/).filter(Boolean);
  const maxLength = Math.max(leftWords.length, rightWords.length);

  if (maxLength === 0) {
    return 1;
  }

  let matches = 0;
  const limit = Math.min(leftWords.length, rightWords.length);
  for (let i = 0; i < limit; i++) {
    if (leftWords[i] === rightWords[i]) {
      matches++;
    }
  }

  return matches / maxLength;
}

describe.runIf(lowLevelSupported && hasRealLowLevelAssets())('merged low-level whisper/VAD E2E', () => {
  const audio = loadWav(whisperAudioPath);

  it('keeps callback, promise, file, and PCM entrypoints aligned on a real utterance', async () => {
    const context = new WhisperContext({
      model: whisperModelPath,
      use_gpu: false,
      no_prints: true,
    });

    try {
      const callbackResult = await transcribeWithCallback(context, {
        fname_inp: whisperAudioPath,
        language: 'en',
        no_timestamps: false,
      });
      const fileResult = await transcribeAsync(context, {
        fname_inp: whisperAudioPath,
        language: 'en',
        no_timestamps: false,
      });
      const pcmResult = await transcribeAsync(context, {
        pcmf32: audio,
        language: 'en',
        no_timestamps: false,
      });

      const callbackText = transcriptText(callbackResult.segments);
      const fileText = transcriptText(fileResult.segments);
      const pcmText = transcriptText(pcmResult.segments);

      expect(callbackText).toContain('ask not what your country can do for you');
      expect(callbackText).toBe(fileText);
      expect(fileText).toBe(pcmText);
      expect(callbackResult.segments.length).toBe(fileResult.segments.length);
      expect(fileResult.segments.length).toBe(pcmResult.segments.length);
      assertSortedByTime(fileResult.segments);
    } finally {
      context.free();
    }
  }, 60_000);

  it('streams progress and segment callbacks coherently on real audio', async () => {
    const context = new WhisperContext({
      model: whisperModelPath,
      use_gpu: false,
      no_prints: true,
    });
    const progressValues: number[] = [];
    const streamedSegments: TranscriptSegment[] = [];

    try {
      const result = await transcribeAsync(context, {
        fname_inp: whisperAudioPath,
        language: 'en',
        no_timestamps: false,
        progress_callback: (progress) => {
          progressValues.push(progress);
        },
        on_new_segment: (segment) => {
          streamedSegments.push({
            start: segment.start,
            end: segment.end,
            text: segment.text,
            tokens: segment.tokens,
          });
        },
      });

      expect(progressValues.length).toBeGreaterThan(0);
      for (let i = 1; i < progressValues.length; i++) {
        expect(progressValues[i]).toBeGreaterThanOrEqual(progressValues[i - 1]);
      }
      expect(Math.min(...progressValues)).toBeGreaterThanOrEqual(0);
      expect(Math.max(...progressValues)).toBeLessThanOrEqual(100);

      expect(streamedSegments.length).toBe(result.segments.length);
      expect(transcriptText(streamedSegments)).toBe(transcriptText(result.segments));
    } finally {
      context.free();
    }
  }, 60_000);

  it('uses the real VAD model to separate speech from silence', () => {
    const vad = createVadContext({
      model: vadModelPath,
      no_prints: true,
    });

    try {
      const speechWindow = getHighestEnergyWindow(audio, vad.getWindowSamples());
      const silenceWindow = new Float32Array(vad.getWindowSamples());

      const speechProbability = vad.process(speechWindow);
      vad.reset();
      const silenceProbability = vad.process(silenceWindow);

      expect(speechProbability).toBeGreaterThan(0.01);
      expect(silenceProbability).toBeLessThan(0.01);
    } finally {
      vad.free();
    }
  });
});

describe.runIf(pipelineSupported && hasRealPipelineAssets())('pipeline E2E with real models and audio', () => {
  let pipeline: Pipeline;
  const audio = loadWav(pipelineAudioPath);

  beforeAll(async () => {
    pipeline = await Pipeline.load(pipelineConfig);
  }, 60_000);

  afterAll(() => {
    pipeline.close();
  });

  it('keeps one-shot, offline, and streaming transcripts aligned on the sample audio', async () => {
    const oneShot = await pipeline.transcribe(audio);
    const offline = await pipeline.transcribeOffline(audio);

    const session = pipeline.createSession();
    try {
      const chunkSize = sampleRate;
      for (let offset = 0; offset < audio.length; offset += chunkSize) {
        await session.push(audio.slice(offset, offset + chunkSize));
      }

      const streaming = await session.finalize();

      expect(oneShot.segments.length).toBeGreaterThan(0);
      expect(offline.segments.length).toBeGreaterThan(0);
      expect(streaming.segments.length).toBeGreaterThan(0);

      const oneShotText = transcriptText(oneShot.segments);
      const offlineText = transcriptText(offline.segments);
      const streamingText = transcriptText(streaming.segments);

      expect(wordAgreementRatio(oneShotText, offlineText)).toBeGreaterThan(0.8);
      expect(wordAgreementRatio(streamingText, offlineText)).toBeGreaterThan(0.8);
      expect(oneShotText).toContain('this is diane in new jersey');
      expect(offlineText).toContain('this is diane in new jersey');
      expect(streamingText).toContain('this is diane in new jersey');
      expect(collectSpeakerSet(oneShot.segments)).toEqual(collectSpeakerSet(offline.segments));
      expect(collectSpeakerSet(streaming.segments)).toEqual(collectSpeakerSet(offline.segments));
    } finally {
      session.close();
    }
  }, 120_000);

  it.skipIf(!hasTranscribeBinary())('matches the C++ transcribe CLI exactly for streaming finalize output', async () => {
    const cppJson = execFileSync(transcribeBinPath, [
      pipelineAudioPath,
      '--seg-model', pipelineVadConfig.segModelPath,
      '--emb-model', pipelineVadConfig.embModelPath,
      '--whisper-model', pipelineVadConfig.whisperModelPath,
      '--plda', pipelineVadConfig.pldaPath,
      '--seg-coreml', pipelineVadConfig.segCoremlPath,
      '--emb-coreml', pipelineVadConfig.coremlPath,
      '--language', pipelineVadConfig.language,
    ], {
      encoding: 'utf-8',
      maxBuffer: 50 * 1024 * 1024,
    });

    const session = pipeline.createSession();
    try {
      const chunkSize = sampleRate;
      for (let offset = 0; offset < audio.length; offset += chunkSize) {
        await session.push(audio.slice(offset, offset + chunkSize));
      }
      const result = await session.finalize();
      expect(writeSegmentsJson(result.segments)).toBe(cppJson);
    } finally {
      session.close();
    }
  }, 120_000);
});
