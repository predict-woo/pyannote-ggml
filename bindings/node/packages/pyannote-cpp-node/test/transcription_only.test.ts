import { existsSync, readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, it, expect, beforeAll, afterAll } from 'vitest';

import { Pipeline, type AlignedSegment } from '../src/index.js';
import { hasRealPipelineAssets, pipelineSupported } from './real_assets.js';

const describePipeline = describe.runIf(pipelineSupported && hasRealPipelineAssets());

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const PROJECT_ROOT = resolve(__dirname, '../../../../..');

const transcriptionOnlyConfig = {
  segModelPath: resolve(PROJECT_ROOT, 'models/segmentation-ggml/segmentation.gguf'),
  segCoremlPath: resolve(PROJECT_ROOT, 'models/segmentation-ggml/segmentation.mlpackage'),
  whisperModelPath: resolve(PROJECT_ROOT, 'whisper.cpp/models/ggml-base.en.bin'),
  language: 'en',
  transcriptionOnly: true,
};

const TRANSCRIBE_BIN = resolve(PROJECT_ROOT, 'diarization-ggml/build/bin/transcribe');
const AUDIO_PATH = resolve(PROJECT_ROOT, 'samples/sample.wav');

function loadWav(filePath: string): Float32Array {
  const buffer = readFileSync(filePath);
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

  let offset = 0;
  const riff = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (riff !== 'RIFF') throw new Error('Not a RIFF file');
  offset = 12;

  while (offset < view.byteLength - 8) {
    const chunkId = String.fromCharCode(
      view.getUint8(offset), view.getUint8(offset + 1),
      view.getUint8(offset + 2), view.getUint8(offset + 3),
    );
    const chunkSize = view.getUint32(offset + 4, true);
    offset += 8;

    if (chunkId === 'data') {
      const numSamples = chunkSize / 2;
      const float32 = new Float32Array(numSamples);
      for (let i = 0; i < numSamples; i++) {
        float32[i] = view.getInt16(offset + i * 2, true) / 32768.0;
      }
      return float32;
    }

    offset += chunkSize;
    if (chunkSize % 2 !== 0) offset++;
  }

  throw new Error('No data chunk found in WAV file');
}

function jsonEscape(s: string): string {
  let out = '';
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (c === '\\') {
      out += '\\\\';
    } else if (c === '"') {
      out += '\\"';
    } else if (c === '\n') {
      out += '\\n';
    } else if (c === '\r') {
      out += '\\r';
    } else if (c === '\t') {
      out += '\\t';
    } else {
      out += c;
    }
  }
  return out;
}

function trimAsciiWhitespace(s: string): string {
  let start = 0;
  while (start < s.length && s.charCodeAt(start) <= 0x20) {
    start++;
  }
  let end = s.length;
  while (end > start && s.charCodeAt(end - 1) <= 0x20) {
    end--;
  }
  return s.slice(start, end);
}

function writeSegmentsJson(segments: AlignedSegment[]): string {
  let out = '{\n  "segments": [\n';

  for (let s = 0; s < segments.length; s++) {
    const seg = segments[s];
    const text = trimAsciiWhitespace(seg.text);

    if (seg.speaker === '') {
      out += `    {"start": ${seg.start.toFixed(6)}, "duration": ${seg.duration.toFixed(6)}, "text": "${jsonEscape(text)}"}`;
    } else {
      out += `    {"speaker": "${jsonEscape(seg.speaker)}", "start": ${seg.start.toFixed(6)}, "duration": ${seg.duration.toFixed(6)}, "text": "${jsonEscape(text)}"}`;
    }
    if (s + 1 < segments.length) {
      out += ',';
    }
    out += '\n';
  }

  out += '  ]\n}\n';
  return out;
}

function firstDifference(a: string, b: string): string {
  const maxLen = Math.max(a.length, b.length);
  for (let i = 0; i < maxLen; i++) {
    if (a[i] !== b[i]) {
      const start = Math.max(0, i - 40);
      const end = Math.min(maxLen, i + 40);
      return [
        `First difference at byte ${i}:`,
        `  C++: ...${JSON.stringify(a.slice(start, end))}...`,
        `  TS:  ...${JSON.stringify(b.slice(start, end))}...`,
        `  C++ char: ${a[i] ? JSON.stringify(a[i]) : '<EOF>'} (${a.charCodeAt(i) || 'N/A'})`,
        `  TS  char: ${b[i] ? JSON.stringify(b[i]) : '<EOF>'} (${b.charCodeAt(i) || 'N/A'})`,
      ].join('\n');
    }
  }
  return 'Strings are identical';
}

function assertTranscriptionOnlySegmentShape(segments: AlignedSegment[]): void {
  for (const seg of segments) {
    expect(seg.speaker).toBe('');
    expect(typeof seg.start).toBe('number');
    expect(seg.start).toBeGreaterThanOrEqual(0);
    expect(typeof seg.duration).toBe('number');
    expect(seg.duration).toBeGreaterThan(0);
    expect(typeof seg.text).toBe('string');
    expect(seg.text.length).toBeGreaterThan(0);
  }
}

describePipeline('Loading', () => {
  it('loads without embedding/PLDA/coreml paths', async () => {
    const pipeline = await Pipeline.load(transcriptionOnlyConfig);
    expect(pipeline).toBeDefined();
    expect(pipeline.isClosed).toBe(false);
    pipeline.close();
  });

  it('close is idempotent', async () => {
    const pipeline = await Pipeline.load(transcriptionOnlyConfig);
    pipeline.close();
    expect(pipeline.isClosed).toBe(true);
    pipeline.close();
    expect(pipeline.isClosed).toBe(true);
  });

  it('rejects missing segModelPath', async () => {
    const { segModelPath: _removed, ...invalidConfig } = transcriptionOnlyConfig;
    await expect(Pipeline.load(invalidConfig as unknown as Parameters<typeof Pipeline.load>[0])).rejects.toThrow();
  });

  it('rejects missing segCoremlPath', async () => {
    const { segCoremlPath: _removed, ...invalidConfig } = transcriptionOnlyConfig;
    await expect(Pipeline.load(invalidConfig as unknown as Parameters<typeof Pipeline.load>[0])).rejects.toThrow();
  });

  it('rejects missing whisperModelPath', async () => {
    const { whisperModelPath: _removed, ...invalidConfig } = transcriptionOnlyConfig;
    await expect(Pipeline.load(invalidConfig as unknown as Parameters<typeof Pipeline.load>[0])).rejects.toThrow();
  });

  it('rejects when embModelPath missing without transcriptionOnly', async () => {
    const invalidConfig = {
      segModelPath: transcriptionOnlyConfig.segModelPath,
      segCoremlPath: transcriptionOnlyConfig.segCoremlPath,
      whisperModelPath: transcriptionOnlyConfig.whisperModelPath,
      language: 'en',
    };
    await expect(Pipeline.load(invalidConfig as unknown as Parameters<typeof Pipeline.load>[0])).rejects.toThrow();
  });
});

describePipeline('One-shot transcribe', () => {
  let pipeline: Pipeline;
  const audio = loadWav(AUDIO_PATH);

  beforeAll(async () => {
    pipeline = await Pipeline.load(transcriptionOnlyConfig);
  });

  afterAll(() => {
    pipeline.close();
  });

  it('produces non-empty segments', async () => {
    const result = await pipeline.transcribe(audio);
    expect(result.segments.length).toBeGreaterThan(0);
  });

  it('segments have correct shape with empty speaker', async () => {
    const result = await pipeline.transcribe(audio);
    assertTranscriptionOnlySegmentShape(result.segments);
  });

  it('segments are sorted by start time', async () => {
    const result = await pipeline.transcribe(audio);
    for (let i = 1; i < result.segments.length; i++) {
      expect(result.segments[i].start).toBeGreaterThanOrEqual(result.segments[i - 1].start);
    }
  });

  it('segments cover the audio timeline', async () => {
    const result = await pipeline.transcribe(audio);
    expect(result.segments.length).toBeGreaterThan(0);

    const first = result.segments[0];
    const last = result.segments[result.segments.length - 1];
    const audioDurationSec = audio.length / 16000;

    expect(first.start).toBeLessThan(2.0);
    expect(last.start + last.duration).toBeGreaterThan(audioDurationSec - 2.0);
    expect(last.start + last.duration).toBeLessThanOrEqual(audioDurationSec + 2.0);
  });
});

describePipeline('Offline transcribeOffline', () => {
  let pipeline: Pipeline;
  const audio = loadWav(AUDIO_PATH);

  beforeAll(async () => {
    pipeline = await Pipeline.load(transcriptionOnlyConfig);
  });

  afterAll(() => {
    pipeline.close();
  });

  it('produces valid segments in transcription-only mode', async () => {
    const result = await pipeline.transcribeOffline(audio);
    expect(result.segments.length).toBeGreaterThan(0);
    assertTranscriptionOnlySegmentShape(result.segments);
  });

  it('segments have text content', async () => {
    const result = await pipeline.transcribeOffline(audio);
    for (const seg of result.segments) {
      expect(seg.text.length).toBeGreaterThan(0);
    }
  });
});

describePipeline('Streaming session', () => {
  let pipeline: Pipeline;
  const audio = loadWav(AUDIO_PATH);

  beforeAll(async () => {
    pipeline = await Pipeline.load(transcriptionOnlyConfig);
  });

  afterAll(() => {
    pipeline.close();
  });

  it('push returns boolean[] VAD predictions', async () => {
    const session = pipeline.createSession();

    try {
      const CHUNK_SIZE = 16000;
      const allVad: boolean[] = [];

      for (let i = 0; i < 15; i++) {
        const start = i * CHUNK_SIZE;
        const end = Math.min(start + CHUNK_SIZE, audio.length);
        const chunk = audio.slice(start, end);
        const vad = await session.push(chunk);
        allVad.push(...vad);
      }

      expect(allVad.length).toBeGreaterThan(0);
      for (const v of allVad) {
        expect(typeof v).toBe('boolean');
      }
    } finally {
      session.close();
    }
  });

  it('finalize returns segments with empty speaker', async () => {
    const session = pipeline.createSession();

    try {
      const CHUNK_SIZE = 16000;
      for (let offset = 0; offset < audio.length; offset += CHUNK_SIZE) {
        const end = Math.min(offset + CHUNK_SIZE, audio.length);
        await session.push(audio.slice(offset, end));
      }

      const result = await session.finalize();
      expect(result.segments.length).toBeGreaterThan(0);
      assertTranscriptionOnlySegmentShape(result.segments);
    } finally {
      session.close();
    }
  });

  it('emits segments events with empty speaker', async () => {
    const session = pipeline.createSession();

    try {
      const received: AlignedSegment[][] = [];
      session.on('segments', (segments: AlignedSegment[]) => {
        received.push(segments);
      });

      const CHUNK_SIZE = 16000;
      for (let offset = 0; offset < audio.length; offset += CHUNK_SIZE) {
        const end = Math.min(offset + CHUNK_SIZE, audio.length);
        await session.push(audio.slice(offset, end));
      }

      const finalResult = await session.finalize();
      expect(finalResult.segments.length).toBeGreaterThan(0);

      // Allow event loop to process TSFN callback for segments event
      await new Promise(resolve => setTimeout(resolve, 200));

      const observedSegments = received.flat();
      expect(observedSegments.length).toBeGreaterThan(0);
      for (const seg of observedSegments) {
        expect(seg.speaker).toBe('');
      }
    } finally {
      session.close();
    }
  });

  it('emits audio events', async () => {
    const session = pipeline.createSession();

    try {
      const chunks: Float32Array[] = [];
      session.on('audio', (chunk: Float32Array) => {
        chunks.push(chunk);
      });

      const CHUNK_SIZE = 16000;
      for (let offset = 0; offset < audio.length; offset += CHUNK_SIZE) {
        const end = Math.min(offset + CHUNK_SIZE, audio.length);
        await session.push(audio.slice(offset, end));
      }

      await session.finalize();
      expect(chunks.length).toBeGreaterThan(0);

      const totalSamples = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
      expect(totalSamples).toBeGreaterThan(0);
      for (const chunk of chunks) {
        expect(chunk).toBeInstanceOf(Float32Array);
      }
    } finally {
      session.close();
    }
  });
});

describePipeline('Deterministic output', () => {
  let pipeline: Pipeline;
  const audio = loadWav(AUDIO_PATH);

  beforeAll(async () => {
    pipeline = await Pipeline.load(transcriptionOnlyConfig);
  });

  afterAll(() => {
    pipeline.close();
  });

  it('two transcribe calls produce identical results', async () => {
    const first = await pipeline.transcribe(audio);
    const second = await pipeline.transcribe(audio);

    expect(first.segments.length).toBe(second.segments.length);
    for (let i = 0; i < first.segments.length; i++) {
      expect(first.segments[i].speaker).toBe(second.segments[i].speaker);
      expect(first.segments[i].start).toBe(second.segments[i].start);
      expect(first.segments[i].duration).toBe(second.segments[i].duration);
      expect(first.segments[i].text).toBe(second.segments[i].text);
    }
  });
});

describePipeline('E2E byte-identical JSON output vs C++ CLI', () => {
  it.skipIf(!existsSync(TRANSCRIBE_BIN))(
    'C++ transcribe --no-diarize and TS Pipeline streaming session produce identical JSON',
    async () => {
      const cppJson = execFileSync(TRANSCRIBE_BIN, [
        AUDIO_PATH,
        '--seg-model', transcriptionOnlyConfig.segModelPath,
        '--whisper-model', transcriptionOnlyConfig.whisperModelPath,
        '--seg-coreml', transcriptionOnlyConfig.segCoremlPath,
        '--language', 'en',
        '--no-diarize',
      ], {
        encoding: 'utf-8',
        maxBuffer: 50 * 1024 * 1024,
      });

      const audio = loadWav(AUDIO_PATH);
      const pipeline = await Pipeline.load(transcriptionOnlyConfig);

      try {
        const session = pipeline.createSession();

        try {
          const CHUNK_SIZE = 16000;
          for (let offset = 0; offset < audio.length; offset += CHUNK_SIZE) {
            const end = Math.min(offset + CHUNK_SIZE, audio.length);
            await session.push(audio.slice(offset, end));
          }

          const result = await session.finalize();
          const tsJson = writeSegmentsJson(result.segments);

          if (tsJson !== cppJson) {
            console.error(firstDifference(cppJson, tsJson));
          }
          expect(tsJson).toBe(cppJson);
        } finally {
          session.close();
        }
      } finally {
        pipeline.close();
      }
    },
    120_000,
  );
});

describePipeline('Resource cleanup', () => {
  it('push after session close throws', async () => {
    const pipeline = await Pipeline.load(transcriptionOnlyConfig);
    const session = pipeline.createSession();
    session.close();
    await expect(session.push(new Float32Array(16000))).rejects.toThrow();
    pipeline.close();
  });

  it('finalize after session close throws', async () => {
    const pipeline = await Pipeline.load(transcriptionOnlyConfig);
    const session = pipeline.createSession();
    session.close();
    await expect(session.finalize()).rejects.toThrow();
    pipeline.close();
  });

  it('transcribe after pipeline close throws', async () => {
    const pipeline = await Pipeline.load(transcriptionOnlyConfig);
    pipeline.close();
    await expect(pipeline.transcribe(new Float32Array(16000))).rejects.toThrow();
  });
});
