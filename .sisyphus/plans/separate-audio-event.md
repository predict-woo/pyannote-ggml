# Plan: Separate Audio Event from Segments Callback

## Goal
Split the pipeline callback into two distinct events:
- `segments` — fires with only `AlignedSegment[]` (no audio)
- `audio` — fires with `Float32Array` chunks of silence-filtered audio as they're produced

This gives consumers clean separation: collect `audio` events for the complete VAD-filtered stream (in sync with segment timestamps), and `segments` events for transcript updates.

## Current State

### C++ pipeline (`pipeline.h` / `pipeline.cpp`)
- Single callback: `typedef void (*pipeline_callback)(const std::vector<AlignedSegment>& segments, const std::vector<float>& audio, void* user_data);`
- `pipeline_init(config, callback, user_data)` — takes one callback + one user_data
- `handle_whisper_result` fires callback with `(all_segments, last_submitted_audio)`
- `pipeline_finalize` fires callback with `(all_segments, all_audio)` — full accumulated audio
- `PipelineState` has `last_submitted_audio` and `all_audio` fields

### C++ native addon
- `PipelineSession.cpp`: `pipeline_cb` static method matches the single callback signature
  - Marshals segments + audio into JS, fires TSFN
  - Also stores `last_segments_` for finalize result
- `PipelineSession.h`: One TSFN (`tsfn_`), one JS callback ref (`js_callback_`)
- `TranscribeWorker.cpp`: `OnPipelineCallback` matches same signature, ignores audio (`/*audio*/`)
- `PipelineFinalizeWorker.cpp`: Gets segments from `session_->GetLastSegments()`, builds `{segments: [...]}`

### TypeScript
- `PipelineSession.ts`:
  - `_onNativeCallback(segments, audio)` → emits `'segments'` event with both
  - Event signature: `segments: [segments: AlignedSegment[], audio: Float32Array]`
- `Pipeline.ts`: `createSession()` passes `(segments, audio) => session._onNativeCallback(segments, audio)`
- `types.ts`: `TranscriptionResult = { segments: AlignedSegment[] }` — no audio in return type

### Tests
- `integration.test.ts` line 159-162: Tests `segments` event with `(segments, audioData)` signature
- `e2e_identical.test.ts`: Only uses `finalize()` return value, doesn't test events

### CLI (`main_transcribe.cpp`)
- Line 425: Callback ignores audio param (`/*audio*/`)

## Changes Required

### Task 1: C++ pipeline — add audio callback, remove audio from segments callback

**File: `diarization-ggml/src/pipeline.h`**
- Change `pipeline_callback` to remove audio param:
  ```cpp
  typedef void (*pipeline_callback)(const std::vector<AlignedSegment>& segments, void* user_data);
  ```
- Add new audio callback type:
  ```cpp
  typedef void (*pipeline_audio_callback)(const float* samples, int n_samples, void* user_data);
  ```
- Update `pipeline_init` signature to accept both:
  ```cpp
  PipelineState* pipeline_init(const PipelineConfig& config,
                                pipeline_callback cb,
                                pipeline_audio_callback audio_cb,
                                void* user_data);
  ```

**File: `diarization-ggml/src/pipeline.cpp`**
- Add `pipeline_audio_callback audio_callback;` to `PipelineState`
- Remove `last_submitted_audio` and `all_audio` from `PipelineState`
- In `pipeline_init`: store `audio_cb` in state
- In `pipeline_push`: after `silence_filter_push`, if `sf_result.audio` is non-empty, fire `audio_callback(sf_result.audio.data(), sf_result.audio.size(), user_data)`
- In `pipeline_finalize`: after `silence_filter_flush`, if `sf_flush.audio` is non-empty, fire `audio_callback(sf_flush.audio.data(), sf_flush.audio.size(), user_data)`
- In `handle_whisper_result`: change callback call from `callback(segments, last_submitted_audio, user_data)` to `callback(segments, user_data)`
- In `pipeline_finalize`: change callback call from `callback(segments, all_audio, user_data)` to `callback(segments, user_data)`
- Remove `enqueue_audio_chunk`'s `all_audio.insert(...)` line
- Remove `try_submit_next`'s `last_submitted_audio = sub.audio` line

### Task 2: CLI — update callback signature

**File: `diarization-ggml/src/main_transcribe.cpp`**
- Line 425: Update callback lambda signature to remove `audio` param:
  ```cpp
  auto callback = [](const std::vector<AlignedSegment>& segments, void* user_data) { ... };
  ```
- Line 459: Update `pipeline_init` call to pass `nullptr` for audio_cb:
  ```cpp
  PipelineState* state = pipeline_init(config, callback, nullptr, &callback_ctx);
  ```

### Task 3: C++ tests — update any pipeline callback signatures

Check these files for pipeline_callback usage and update signatures:
- `diarization-ggml/tests/test_pipeline.cpp`
- `diarization-ggml/tests/test_aligner.cpp` (probably doesn't use pipeline directly)
- `diarization-ggml/tests/test_transcriber.cpp` (probably doesn't use pipeline directly)

### Task 4: Native addon — split into two TSFNs

**File: `bindings/node/packages/darwin-arm64/src/PipelineSession.h`**
- Add `Napi::ThreadSafeFunction audio_tsfn_;` member
- Add `bool audio_tsfn_released_ = false;`
- Update `pipeline_cb` signature to match new `pipeline_callback` (no audio param)
- Add `static void audio_cb(const float* samples, int n_samples, void* user_data);`

**File: `bindings/node/packages/darwin-arm64/src/PipelineSession.cpp`**
- Remove audio marshalling from `pipeline_cb`:
  - TSFNCallbackData: remove `std::vector<float> audio` field
  - TSFN lambda: remove Float32Array creation, just call `jsCallback.Call({segmentsArr})`
- Add new `audio_cb` static method:
  - Create a `std::vector<float>*` copy of the audio data
  - Call `audio_tsfn_.NonBlockingCall(data, lambda)` where lambda marshals to Float32Array
- In constructor:
  - Create second TSFN for audio events (needs a second JS callback)
  - Pass both callbacks to `pipeline_init(config, pipeline_cb, audio_cb, this)`
- In `Cleanup()`: release `audio_tsfn_` too

**IMPORTANT DESIGN DECISION**: The constructor currently takes ONE JS callback function. We now need TWO (one for segments, one for audio). Options:
  - **Option A**: Constructor takes two callback functions → `model.createSession(segmentsCb, audioCb)`
  - **Option B**: Constructor takes one callback, add separate method → session emits different events from TS side
  - **Option C**: Constructor takes an object `{ onSegments, onAudio }` → cleaner

Recommendation: **Option A** — simplest C++ change. The TS wrapper will split them back into separate events anyway.

So `PipelineModel::CreateSession` (in PipelineModel.cpp) needs to pass two callbacks, and `PipelineSession` constructor expects two Function args.

**File: `bindings/node/packages/darwin-arm64/src/PipelineModel.cpp`**
- Update `CreateSession` to expect two callback args from JS

**File: `bindings/node/packages/darwin-arm64/src/TranscribeWorker.h`**
- Update `OnPipelineCallback` signature to remove audio param
- `pipeline_init` call needs `nullptr` for audio_cb (one-shot doesn't need audio events)

**File: `bindings/node/packages/darwin-arm64/src/TranscribeWorker.cpp`**
- Update `OnPipelineCallback` signature
- Update `pipeline_init` call in `Execute()`

### Task 5: TypeScript wrapper — separate events

**File: `bindings/node/packages/pyannote-cpp-node/src/PipelineSession.ts`**
- Update event interface:
  ```typescript
  export interface PipelineSessionEvents {
    segments: [segments: AlignedSegment[]];
    audio: [audio: Float32Array];
    error: [error: Error];
  }
  ```
- Split `_onNativeCallback` into two methods:
  ```typescript
  _onSegmentsCallback(segments: any[]): void {
    this.emit('segments', segments as AlignedSegment[]);
  }
  _onAudioCallback(audio: Float32Array): void {
    this.emit('audio', audio);
  }
  ```

**File: `bindings/node/packages/pyannote-cpp-node/src/Pipeline.ts`**
- Update `createSession()` to pass two callbacks:
  ```typescript
  const nativeSession = this.native.createSession(
    (segments: any[]) => session._onSegmentsCallback(segments),
    (audio: Float32Array) => session._onAudioCallback(audio),
  );
  ```

**File: `bindings/node/packages/pyannote-cpp-node/src/types.ts`**
- `TranscriptionResult` stays the same (no audio in return type — it's event-only)

### Task 6: Update tests

**File: `bindings/node/packages/pyannote-cpp-node/test/integration.test.ts`**
- Line 159-162: Update segments event listener to new signature (no audio param):
  ```typescript
  const receivedEvents: { segments: AlignedSegment[] }[] = [];
  session.on('segments', (segments: AlignedSegment[]) => {
    receivedEvents.push({ segments });
  });
  ```
- Add test for `audio` event:
  ```typescript
  it('emits audio events with Float32Array chunks', async () => {
    const model = await Pipeline.load(config);
    const session = model.createSession();
    const audio = loadWav(resolve(PROJECT_ROOT, 'samples/sample.wav'));
    const audioChunks: Float32Array[] = [];
    session.on('audio', (chunk: Float32Array) => {
      audioChunks.push(chunk);
    });
    const CHUNK_SIZE = 16000;
    for (let offset = 0; offset < audio.length; offset += CHUNK_SIZE) {
      const end = Math.min(offset + CHUNK_SIZE, audio.length);
      await session.push(audio.slice(offset, end));
    }
    await session.finalize();
    expect(audioChunks.length).toBeGreaterThan(0);
    for (const chunk of audioChunks) {
      expect(chunk).toBeInstanceOf(Float32Array);
      expect(chunk.length).toBeGreaterThan(0);
    }
    session.close();
    model.close();
  });
  ```

**File: `bindings/node/packages/pyannote-cpp-node/test/e2e_identical.test.ts`**
- No changes needed — it only uses `finalize()` return value, not events

### Task 7: Update PipelineModel.cpp (createSession)

**File: `bindings/node/packages/darwin-arm64/src/PipelineModel.cpp`**
- `CreateSession` method currently passes one callback to PipelineSession constructor
- Update to pass two callbacks (segments + audio)

### Task 8: Rebuild and test
- Run `bindings/node/rebuild.sh`
- Run C++ tests
- Run TS integration tests
- Run TS e2e test

### Task 9: Update README
- Update `segments` event documentation (no audio param)
- Add `audio` event documentation
- Update examples

## File Change Summary

| File | Change |
|------|--------|
| `diarization-ggml/src/pipeline.h` | New callback types, updated `pipeline_init` |
| `diarization-ggml/src/pipeline.cpp` | Audio callback, remove `all_audio`/`last_submitted_audio` |
| `diarization-ggml/src/main_transcribe.cpp` | Update callback signature |
| `diarization-ggml/tests/test_pipeline.cpp` | Update if it uses pipeline_callback |
| `bindings/node/packages/darwin-arm64/src/PipelineSession.h` | Add audio TSFN, update signatures |
| `bindings/node/packages/darwin-arm64/src/PipelineSession.cpp` | Split callbacks, two TSFNs |
| `bindings/node/packages/darwin-arm64/src/PipelineModel.cpp` | Pass two callbacks |
| `bindings/node/packages/darwin-arm64/src/TranscribeWorker.h` | Update callback signature |
| `bindings/node/packages/darwin-arm64/src/TranscribeWorker.cpp` | Update callback + pipeline_init |
| `bindings/node/packages/darwin-arm64/src/PipelineFinalizeWorker.cpp` | No change needed |
| `bindings/node/packages/pyannote-cpp-node/src/PipelineSession.ts` | Separate events |
| `bindings/node/packages/pyannote-cpp-node/src/Pipeline.ts` | Two callbacks |
| `bindings/node/packages/pyannote-cpp-node/src/types.ts` | No change |
| `bindings/node/packages/pyannote-cpp-node/test/integration.test.ts` | Update event tests |
| `bindings/node/packages/pyannote-cpp-node/README.md` | Update event docs |

## Execution Order

1. C++ pipeline changes (pipeline.h → pipeline.cpp) — the foundation
2. CLI update (main_transcribe.cpp) — uses pipeline directly
3. C++ tests update (test_pipeline.cpp)
4. Build C++ (`cmake --build build && cmake --build build-static`)
5. Native addon changes (PipelineSession, TranscribeWorker, PipelineModel)
6. TypeScript changes (PipelineSession.ts, Pipeline.ts)
7. Test updates (integration.test.ts)
8. Rebuild with `bindings/node/rebuild.sh`
9. Run all tests
10. Update README
11. Commit and push

## Key Design Decisions

1. **Two separate C++ callbacks** — cleaner than modifying the existing one
2. **`nullptr` for audio_cb** — one-shot transcribe and CLI don't need audio events
3. **Audio fires from silence filter output** — not from Whisper chunks. This is the source of truth for the filtered audio timeline.
4. **No accumulation in C++** — audio is emitted immediately, never stored. Consumer decides whether to collect.
5. **Pipeline.ts passes two callbacks** to native createSession
6. **TranscriptionResult unchanged** — audio is event-only, not in the return type
