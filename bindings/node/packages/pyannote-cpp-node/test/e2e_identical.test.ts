import { existsSync, readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, it, expect } from 'vitest';

import { Pipeline, type AlignedSegment } from '../src/index.js';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const PROJECT_ROOT = resolve(__dirname, '../../../../..');

const TRANSCRIBE_BIN = resolve(PROJECT_ROOT, 'diarization-ggml/build/bin/transcribe');

const config = {
  segModelPath: resolve(PROJECT_ROOT, 'models/segmentation-ggml/segmentation.gguf'),
  embModelPath: resolve(PROJECT_ROOT, 'models/embedding-ggml/embedding.gguf'),
  pldaPath: resolve(PROJECT_ROOT, 'diarization-ggml/plda.gguf'),
  coremlPath: resolve(PROJECT_ROOT, 'models/embedding-ggml/embedding.mlpackage'),
  segCoremlPath: resolve(PROJECT_ROOT, 'models/segmentation-ggml/segmentation.mlpackage'),
  whisperModelPath: resolve(PROJECT_ROOT, 'whisper.cpp/models/ggml-base.en.bin'),
  language: 'en',
};

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

function segmentText(seg: AlignedSegment): string {
  let raw = '';
  for (const word of seg.words) {
    raw += word.text;
  }
  return trimAsciiWhitespace(raw);
}

function writeSegmentsJson(segments: AlignedSegment[]): string {
  let out = '{\n  "segments": [\n';

  for (let s = 0; s < segments.length; s++) {
    const seg = segments[s];
    const combinedText = segmentText(seg);

    out += `    {"speaker": "${jsonEscape(seg.speaker)}", "start": ${seg.start.toFixed(6)}, "duration": ${seg.duration.toFixed(6)}, "text": "${jsonEscape(combinedText)}", "words": [`;

    for (let w = 0; w < seg.words.length; w++) {
      const word = seg.words[w];
      if (w > 0) {
        out += ', ';
      }
      out += `{"text": "${jsonEscape(word.text)}", "start": ${word.start.toFixed(6)}, "end": ${word.end.toFixed(6)}}`;
    }

    out += ']}';
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

describe('E2E byte-identical JSON output', () => {
  it.skipIf(!existsSync(TRANSCRIBE_BIN))(
    'C++ transcribe binary and TS Pipeline streaming session produce identical JSON',
    async () => {
      const cppJson = execFileSync(TRANSCRIBE_BIN, [
        AUDIO_PATH,
        '--seg-model', config.segModelPath,
        '--emb-model', config.embModelPath,
        '--whisper-model', config.whisperModelPath,
        '--plda', config.pldaPath,
        '--seg-coreml', config.segCoremlPath,
        '--emb-coreml', config.coremlPath,
        '--language', config.language,
      ], {
        encoding: 'utf-8',
        maxBuffer: 50 * 1024 * 1024,
      });

      const audio = loadWav(AUDIO_PATH);
      const pipeline = await Pipeline.load(config);

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
