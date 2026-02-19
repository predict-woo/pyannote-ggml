# Learnings

## 2026-02-19 Session Start
- Plan: node-bindings-pipeline (10 tasks, 5 waves)
- Starting from Task 1: Add pipeline-lib STATIC target + rebuild build-static

## Task 1: pipeline-lib STATIC target
- Added `pipeline-lib` STATIC library between `transcription-lib` INTERFACE and `transcribe` executable (CMakeLists.txt line ~119)
- 6 source files: pipeline.cpp, silence_filter.cpp, audio_buffer.cpp, segment_detector.cpp, transcriber.cpp, aligner.cpp
- Links `transcription-lib` (PUBLIC) which transitively provides diarization-lib + whisper
- Include dirs: `src/` and `include/` (PUBLIC)
- Had to `rm -rf build-static` due to stale CMakeCache.txt pointing to old source path `/Users/andyye/dev/pyannote-audio/` vs current `/Users/andyye/dev/whisper-diarization.cpp/pyannote.cpp/`
- All 5 required `.a` files produced: libpipeline-lib.a, libdiarization-lib.a, libwhisper.a, libsegmentation-core.a, libembedding-core.a
- Existing targets (transcribe, diarization-ggml) unaffected

## Task 6: Node bindings pipeline type migration
- Replaced `src/types.ts` diarization-only types with pipeline types: `ModelConfig`, `AlignedWord`, `AlignedSegment`, `TranscriptionResult`.
- Updated `ModelConfig` to include whisper/vad/transcriber fields and removed `zeroLatency` from TS surface for this task.
- Replaced native interfaces in `src/binding.ts`: `NativePipelineModel`, `NativePipelineSession`, `NativeBinding` with `PipelineModel`/`PipelineSession` constructor checks.
- Kept existing binding loader behavior unchanged (`createRequire`, package resolution, runtime cache, `getBinding()`).
- Updated `src/index.ts` to temporary exports: types + native binding types + `getBinding`; left `Pipeline`/`PipelineSession` exports commented until wrapper files exist.
- Verification: `npx tsc --noEmit` currently fails because existing `src/Pyannote.ts` and `src/StreamingSession.ts` still import removed legacy symbols (`NativePyannoteModel`, `NativeStreamingSession`, `DiarizationResult`, `VADChunk`) pending Tasks 7/8 wrapper migration.

## Task 2: Update darwin-arm64 CMakeLists.txt for pipeline+whisper
- Updated `bindings/node/packages/darwin-arm64/CMakeLists.txt` (97→107 lines)
- Added `libpipeline-lib.a` as first link target (before diarization-lib)
- Added `libwhisper.a` and `libwhisper.coreml.a` from `whisper.cpp/src/`
- Changed ALL 5 ggml paths from `${DIARIZATION_BUILD}/ggml/src/` → `${DIARIZATION_BUILD}/whisper.cpp/ggml/src/`
- Added 3 whisper include dirs: `whisper.cpp/include`, `whisper.cpp/src`, `whisper.cpp/ggml/include`
- Changed ggml include from `${DIARIZATION_ROOT}/ggml/include` → `${DIARIZATION_ROOT}/whisper.cpp/ggml/include`
- Added `WHISPER_USE_COREML` compile definition alongside existing ones
- `pnpm build` succeeds — compiles all 7 existing source files and links to `.node` with no duplicate symbols
- Had to `rm -rf build` due to stale CMakeCache.txt (same old-path issue as Task 1)
- Link order matters: pipeline-lib first, then diarization-lib, then whisper, then component libs, then ggml

## Task 3: Replace addon.cpp registration with pipeline classes
- Replaced `bindings/node/packages/darwin-arm64/src/addon.cpp` includes: `PyannoteModel.h`/`StreamingSession.h` → `PipelineModel.h`/`PipelineSession.h`.
- Updated `Init` registrations to `PipelineModel::Init(env, exports)` and `PipelineSession::Init(env, exports)` only.
- Kept `NODE_API_MODULE(pyannote, Init)` unchanged for downstream module loading compatibility.

## Task 4: PipelineModel + TranscribeWorker
- Created 4 files: `PipelineModel.h`, `PipelineModel.cpp`, `TranscribeWorker.h`, `TranscribeWorker.cpp`
- PipelineModel stores all path/language fields as `std::string` members; `BuildConfig()` assembles `PipelineConfig` with `.c_str()` pointers safe as long as PipelineModel outlives the PipelineState
- TranscribeWorker creates its OWN `PipelineState` per transcription (via `pipeline_init` in `Execute()`), so one-shot transcribe doesn't conflict with streaming sessions
- Static callback `OnPipelineCallback` captures results into `TranscribeCallbackData` — the callback fires synchronously on the worker thread within `pipeline_push`/`pipeline_finalize`
- `CreateSession()` is a stub returning `undefined` — Task 5 will implement PipelineSession
- `busy_` flag prevents concurrent operations on the same model
- Text field in `AlignedSegment` marshaling: manually concatenated from `words[]` since `AlignedSegment` has no `text` field
- Cannot build-test in isolation because `file(GLOB ADDON_SOURCES "src/*.cpp")` picks up all .cpp files including old PyannoteModel — Task 6 rewrites addon.cpp

## Task 5: PipelineSession + PipelinePushWorker + PipelineFinalizeWorker
- Created 6 files in `bindings/node/packages/darwin-arm64/src/`:
  - `PipelineSession.h` / `PipelineSession.cpp` — ObjectWrap with TSFN for callback marshaling
  - `PipelinePushWorker.h` / `PipelinePushWorker.cpp` — calls `pipeline_push`, returns `vector<bool>` as JS `boolean[]`
  - `PipelineFinalizeWorker.h` / `PipelineFinalizeWorker.cpp` — calls `pipeline_finalize`, resolves with final `AlignedSegment[]`
- PipelineSession creates its OWN `PipelineState` via `pipeline_init` in constructor — each session is independent
- All string config copied from PipelineModel into own `std::string` members for lifetime safety (`TranscriberConfig` uses `const char*`)
- TSFN created with JS callback, `NonBlockingCall` queues `TSFNCallbackData{segments, audio}` to JS thread
- Static `pipeline_cb` stores `last_segments_` (mutex-guarded) for PipelineFinalizeWorker to read after finalize completes
- `MarshalSegments()` / `MarshalSegment()` are free functions in PipelineSession.cpp, shared by TSFN lambda and PipelineFinalizeWorker (via extern declaration)
- `GetLastSegments()` public getter returns mutex-guarded copy for PipelineFinalizeWorker
- Busy flag prevents concurrent push/finalize; Close releases TSFN + pipeline_free
- `tsfn_released_` bool prevents double-release of TSFN in error + destructor paths
- `PipelineModel::CreateSession()` is still a stub returning undefined — needs wiring when `PipelineSession::constructor` is registered via `PipelineSession::Init()` in addon.cpp (Task 6 integration)

## Task 7: Node Pipeline wrapper + binding signature correction
- Added `bindings/node/packages/pyannote-cpp-node/src/Pipeline.ts` with `load`, `transcribe`, `createSession`, `close`, and `isClosed` around `NativePipelineModel`.
- `Pipeline.load()` validates required model paths (`segModelPath`, `embModelPath`, `pldaPath`, `coremlPath`, `segCoremlPath`, `whisperModelPath`) and optional paths (`whisperCoremlPath`, `vadModelPath`) via `accessSync`.
- Updated `bindings/node/packages/pyannote-cpp-node/src/binding.ts` interfaces to match native C++ API shape: `PipelineModel` constructor now takes only `config`, and `NativePipelineModel.transcribe` now takes only `audio`.
- `Pipeline.createSession()` wires native callback into `session._onNativeCallback(...)` and injects native handle with `session._setNative(...)`.
- `bindings/node/packages/pyannote-cpp-node/src/index.ts` already exported `Pipeline` in current tree state; no additional edit required for that export line.

## Task 8: TypeScript PipelineSession wrapper
- Created `bindings/node/packages/pyannote-cpp-node/src/PipelineSession.ts` as an `EventEmitter` wrapper around `NativePipelineSession` with two-phase init via `_setNative()`.
- `_onNativeCallback(segments, audio)` casts native callback payload to `AlignedSegment[]` and emits `'segments'` with `(segments, audio)`.
- Implemented required API only: `push(audio): Promise<boolean[]>`, `finalize(): Promise<TranscriptionResult>`, `close()`, and `isClosed` getter; no `recluster()` exposed.
- `push()` validates `Float32Array` input and closed-state guard, returning native `vector<bool>` mapped as JS `boolean[]`.
- Updated `bindings/node/packages/pyannote-cpp-node/src/index.ts` to export `Pipeline` and `PipelineSession` (uncommented exports).

## Task 10: Rewrite integration tests for Pipeline API
- Rewrote `bindings/node/packages/pyannote-cpp-node/test/integration.test.ts` from old `Pyannote`/diarization API to new `Pipeline`/transcription API.
- 15 tests across 4 describe blocks: Model loading (3), One-shot transcribe (5), Streaming session (3), Resource cleanup (4).
- Removed old tests: DER validation (requires python), streaming zero-latency, streaming finalize-matches-offline, recluster-after-close.
- Config adds `whisperModelPath` (ggml-base.en.bin) and `language: 'en'` for fast deterministic tests.
- Key API changes tested: `Pipeline.load()`, `model.transcribe()`, `model.createSession()`, `session.push()` returns `boolean[]`, `session.finalize()` returns `TranscriptionResult`, session EventEmitter `'segments'` event.
- Segment shape validation: `speaker`, `start`, `duration`, `text`, `words[]`; word shape: `text`, `start`, `end`.
- Had to copy fresh `pyannote-addon.node` (2.8MB with pipeline support) from `packages/darwin-arm64/build/Release/` to `node_modules/@pyannote-cpp-node/darwin-arm64/build/Release/` — stale 894KB copy in node_modules still exported old `PyannoteModel`/`StreamingSession`.
- All 15 tests pass in 25s (model loading ~800ms each, transcribe ~1.8s each, streaming finalize ~1.8s).
