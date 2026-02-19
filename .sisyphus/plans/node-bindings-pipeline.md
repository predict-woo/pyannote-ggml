# Node.js Bindings: Transcription+Diarization Pipeline Rewrite

## TL;DR

> **Quick Summary**: Complete rewrite of `pyannote-cpp-node` bindings to replace the deprecated diarization-only API with the new integrated transcription+diarization pipeline (`pipeline.h`). The new API exposes both a one-shot `transcribe()` method and a streaming `PipelineSession` with EventEmitter for incremental results.
> 
> **Deliverables**:
> - Updated `diarization-ggml/CMakeLists.txt` with `pipeline-lib` static target
> - Rebuilt `build-static` with whisper.cpp + pipeline libraries
> - Rewritten C++ native addon (N-API) with PipelineModel + PipelineSession + ThreadSafeFunction
> - Rewritten TypeScript wrapper with Pipeline class + PipelineSession EventEmitter
> - Updated types (AlignedSegment, AlignedWord, TranscriptionResult)
> - Complete integration test suite for both modes
> 
> **Estimated Effort**: Large
> **Parallel Execution**: YES - 2 waves after build system
> **Critical Path**: Task 1 (build system) → Task 2 (CMakeLists) → Tasks 3-5 (C++ addon) → Tasks 6-8 (TS layer) → Task 9 (tests)

---

## Context

### Original Request
Update the Node.js bindings in `./bindings/node` to support the new integrated transcription+diarization pipeline (`pipeline.h/pipeline.cpp`). The old diarization-only bindings are deprecated and should be entirely replaced.

### Interview Summary
**Key Discussions**:
- **API Scope**: Replace old diarization-only API entirely — no backward compat
- **Both modes**: One-shot `transcribe(Float32Array)` + streaming `PipelineSession` with EventEmitter
- **Config rewrite**: New `ModelConfig` with whisper paths, VAD model, language, nThreads
- **Streaming events**: `session.on('segments', cb)` emits updated segments + audio after each Whisper result
- **VAD exposed**: `push()` returns `Promise<boolean[]>` at Silero window granularity (~31 per 1s push)
- **Singleton state**: `Pipeline.load()` creates ONE `PipelineState` (Whisper loaded once)
- **Audio in events**: Include transcribed audio `Float32Array` in segment events

**Research Findings**:
- Current `build-static` missing whisper.cpp and pipeline source files
- Pipeline sources (`pipeline.cpp`, `transcriber.cpp`, `aligner.cpp`, etc.) not in any static library — compiled per-executable
- C++ pipeline uses callback pattern requiring N-API `ThreadSafeFunction` (not just AsyncWorker)
- `TranscriberConfig` uses raw `const char*` requiring string lifetime management

### Metis Review
**Identified Gaps** (addressed):
- **ThreadSafeFunction lifecycle**: TSFN created at session init, released at close. Guard flag prevents segfault if callback fires after release.
- **build-static stale**: Must rebuild with whisper.cpp, update ggml paths (whisper provides ggml now)
- **Pipeline sources not in library**: Create `pipeline-lib` STATIC target as prerequisite
- **String lifetime**: Store `std::string` in C++ class, point `const char*` at `.c_str()`
- **Callback during finalize**: FinalizeWorker holds TSFN reference; TSFN release only in `close()`
- **Duplicate ggml symbols**: Only link ggml from whisper.cpp subtree
- **VAD granularity correction**: 512-sample windows, not per-sample
- **Close during in-flight**: Deferred cleanup — flag + post-operation free
- **Language auto-detect**: undefined = Whisper auto-detect
- **Error propagation**: `'error'` event on PipelineSession

---

## Work Objectives

### Core Objective
Replace the deprecated diarization-only Node.js bindings with a complete rewrite wrapping the new integrated transcription+diarization C++ pipeline, exposing both one-shot and streaming APIs with full TypeScript types.

### Concrete Deliverables
- `diarization-ggml/CMakeLists.txt` modified with `pipeline-lib` STATIC target
- `build-static/` rebuilt with all required `.a` files including whisper
- 8 new C++ source files in `darwin-arm64/src/` (PipelineModel, PipelineSession, 3 workers)
- 5 rewritten TypeScript files in `pyannote-cpp-node/src/`
- Updated `CMakeLists.txt` in `darwin-arm64/`
- Complete integration test suite
- Updated `package.json` files if needed

### Definition of Done
- [ ] `cd bindings/node/packages/darwin-arm64 && pnpm build` succeeds
- [ ] `cd bindings/node/packages/pyannote-cpp-node && pnpm build` succeeds (TS compiles)
- [ ] `cd bindings/node && pnpm test` passes all tests
- [ ] One-shot transcribe returns AlignedSegments with words for sample.wav
- [ ] Streaming session emits 'segments' events during push
- [ ] Finalize returns complete TranscriptionResult

### Must Have
- One-shot `transcribe(Float32Array)` → `Promise<TranscriptionResult>`
- Streaming `PipelineSession` with `push()`, `finalize()`, `close()`
- EventEmitter `'segments'` event with incremental results + audio
- `'error'` event for pipeline failures
- VAD predictions returned from `push()` as `boolean[]`
- Proper resource cleanup (TSFN release, pipeline_free, close guards)
- All model paths validated at load time
- `ModelConfig` with whisper, VAD, language, nThreads fields

### Must NOT Have (Guardrails)
- NO old API remnants (streaming_init, recluster, DiarizeWorker, PyannoteModel)
- NO per-sample VAD interpolation — window granularity only
- NO file path audio input — Float32Array only
- NO diarization-only fallback mode
- NO stream cancellation / abort mechanism
- NO model sharing between separate Pipeline instances (sessions borrow parent Pipeline's state)
- NO audio format conversion (resampling, stereo-to-mono)
- NO modifications to C++ pipeline source files (pipeline.cpp, transcriber.cpp, etc.)
- NO progress events, latency metrics, or timing callbacks beyond 'segments'
- NO standalone ggml linking (whisper.cpp provides ggml)

---

## Verification Strategy (MANDATORY)

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL tasks in this plan MUST be verifiable WITHOUT any human action.

### Test Decision
- **Infrastructure exists**: YES (vitest configured in workspace root)
- **Automated tests**: YES (Tests-after — rewrite integration.test.ts after implementation)
- **Framework**: vitest (existing setup in `bindings/node/vitest.config.ts`)

### Agent-Executed QA Scenarios (MANDATORY — ALL tasks)

Verification tools by component:
| Type | Tool | How Agent Verifies |
|------|------|-------------------|
| Build system | Bash | cmake commands, check file existence |
| Native addon build | Bash | cmake-js build, check .node file |
| TypeScript compilation | Bash | tsc, check dist/ output |
| Integration tests | Bash | vitest run, assert all pass |

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
└── Task 1: Add pipeline-lib STATIC target + rebuild build-static

Wave 2 (After Wave 1):
├── Task 2: Update addon CMakeLists.txt (link whisper + pipeline)
└── Task 3: Rewrite TypeScript types + binding.ts (no native dependency)

Wave 3 (After Task 2):
├── Task 4: C++ PipelineModel + TranscribeWorker (one-shot path)
├── Task 5: C++ PipelineSession + TSFN + PushWorker + FinalizeWorker (streaming path)
└── Task 6: Rewrite addon.cpp (register new classes)

Wave 4 (After Tasks 3-6):
├── Task 7: TypeScript Pipeline.ts wrapper
├── Task 8: TypeScript PipelineSession.ts EventEmitter wrapper
└── Task 9: Delete old source files

Wave 5 (After Wave 4):
└── Task 10: Integration tests + build verification
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2, 3 | None (must be first) |
| 2 | 1 | 4, 5, 6 | 3 |
| 3 | 1 | 7, 8 | 2 |
| 4 | 2 | 7, 10 | 5, 6 |
| 5 | 2 | 8, 10 | 4, 6 |
| 6 | 2 | 10 | 4, 5 |
| 7 | 3, 4 | 10 | 8, 9 |
| 8 | 3, 5 | 10 | 7, 9 |
| 9 | 4, 5, 6 | 10 | 7, 8 |
| 10 | 7, 8, 9 | None | None (final) |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Agents |
|------|-------|-------------------|
| 1 | 1 | deep (build system changes need careful cmake understanding) |
| 2 | 2, 3 | parallel: deep for CMakeLists, quick for TS types |
| 3 | 4, 5, 6 | parallel: 3x deep (C++ N-API is complex) |
| 4 | 7, 8, 9 | parallel: quick for TS wrappers and file deletion |
| 5 | 10 | deep (integration testing with native addon) |

---

## TODOs

- [ ] 1. Add `pipeline-lib` STATIC Target + Rebuild build-static

  **What to do**:
  - Edit `diarization-ggml/CMakeLists.txt` to add a new `pipeline-lib` STATIC library target containing: `src/pipeline.cpp`, `src/silence_filter.cpp`, `src/audio_buffer.cpp`, `src/segment_detector.cpp`, `src/transcriber.cpp`, `src/aligner.cpp`
  - This target should link `transcription-lib` (the INTERFACE target that links `diarization-lib` + `whisper`)
  - Add proper include directories: `${CMAKE_CURRENT_SOURCE_DIR}/src` and `${CMAKE_CURRENT_SOURCE_DIR}/include`
  - Rebuild build-static: `cmake -B build-static -DDIARIZATION_STATIC=ON -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON -DWHISPER_COREML=ON && cmake --build build-static`
  - Verify output includes: `libpipeline-lib.a`, `libwhisper.a`, and all existing `.a` files

  **Must NOT do**:
  - Do NOT modify existing library targets or executable targets
  - Do NOT remove the per-executable source file compilations (those still work for the standalone binaries)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: CMake build system changes require understanding dependency chains and static library linking
  - **Skills**: []
    - No specialized skills needed — this is pure cmake + shell

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1 (solo)
  - **Blocks**: Tasks 2, 3, 4, 5, 6, 7, 8, 9, 10
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `diarization-ggml/CMakeLists.txt:45-66` — Existing `diarization-lib` STATIC target pattern — follow this exact structure for `pipeline-lib`
  - `diarization-ggml/CMakeLists.txt:116-117` — Existing `transcription-lib` INTERFACE target — `pipeline-lib` should link this

  **API/Type References**:
  - `diarization-ggml/CMakeLists.txt:119-135` — Current `transcribe` executable lists all pipeline source files — these are the exact sources that go into `pipeline-lib`

  **Documentation References**:
  - `AGENTS.md:Build Commands` section — build-static cmake flags

  **WHY Each Reference Matters**:
  - Line 45-66: Shows the exact pattern for creating a static library with include dirs and link dependencies — copy this structure
  - Line 116-117: `transcription-lib` is the INTERFACE that bundles `diarization-lib + whisper` — `pipeline-lib` must link this
  - Line 119-135: Lists ALL 6 source files that constitute the pipeline — these are exactly what goes into the new static library

  **Acceptance Criteria**:
  - [ ] `diarization-ggml/CMakeLists.txt` contains `add_library(pipeline-lib STATIC ...)` with all 6 pipeline source files
  - [ ] `pipeline-lib` links `transcription-lib`
  - [ ] Build completes: `cd diarization-ggml && cmake -B build-static -DDIARIZATION_STATIC=ON -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON -DWHISPER_COREML=ON && cmake --build build-static`

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: build-static produces all required static libraries
    Tool: Bash
    Preconditions: diarization-ggml/CMakeLists.txt modified with pipeline-lib target
    Steps:
      1. cd diarization-ggml && cmake -B build-static -DDIARIZATION_STATIC=ON -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON -DWHISPER_COREML=ON
      2. cmake --build build-static
      3. test -f build-static/libpipeline-lib.a && echo "pipeline-lib: PASS"
      4. test -f build-static/libdiarization-lib.a && echo "diarization-lib: PASS"
      5. find build-static -name "libwhisper.a" | head -1 && echo "whisper: PASS"
      6. test -f build-static/segmentation-ggml/libsegmentation-core.a && echo "seg-core: PASS"
      7. test -f build-static/embedding-ggml/libembedding-core.a && echo "emb-core: PASS"
    Expected Result: All 5 checks print PASS
    Failure Indicators: cmake error, missing .a file, link errors
    Evidence: Terminal output captured

  Scenario: Existing build targets still work after CMakeLists modification
    Tool: Bash
    Preconditions: CMakeLists.txt modified
    Steps:
      1. cd diarization-ggml && cmake -B build -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON -DWHISPER_COREML=ON
      2. cmake --build build --target transcribe
      3. cmake --build build --target diarization-ggml
      4. echo "Existing targets: PASS"
    Expected Result: Both existing executables build without error
    Evidence: Terminal output captured
  ```

  **Commit**: YES
  - Message: `build(diarization): add pipeline-lib static target for Node.js addon linking`
  - Files: `diarization-ggml/CMakeLists.txt`
  - Pre-commit: `cd diarization-ggml && cmake -B build-static -DDIARIZATION_STATIC=ON -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON -DWHISPER_COREML=ON && cmake --build build-static`

---

- [ ] 2. Update Addon CMakeLists.txt for Pipeline + Whisper Linking

  **What to do**:
  - Rewrite `bindings/node/packages/darwin-arm64/CMakeLists.txt` to:
    - Add `libpipeline-lib.a` to the link list
    - Add whisper static libraries: find `libwhisper.a` path in build-static (likely under `whisper.cpp/src/`)
    - Update ggml paths: whisper.cpp now provides ggml, so change from `${DIARIZATION_BUILD}/ggml/src/libggml*.a` to `${DIARIZATION_BUILD}/whisper.cpp/ggml/src/libggml*.a` (or wherever they end up in build-static)
    - Add whisper include directories: `${DIARIZATION_ROOT}/whisper.cpp/include`, `${DIARIZATION_ROOT}/whisper.cpp/src`
    - Add pipeline include directories: already covered by existing `${DIARIZATION_ROOT}/diarization-ggml/src`
    - Add compile definitions: `WHISPER_USE_COREML` (if Whisper CoreML is enabled)
    - Remove `EMBEDDING_USE_COREML` and `SEGMENTATION_USE_COREML` only if they're now transitively defined by pipeline-lib — but KEEP them if needed for the static library symbols

  **Must NOT do**:
  - Do NOT use `add_subdirectory` to build anything — link pre-built static libs only
  - Do NOT link standalone ggml `.a` files alongside whisper's ggml — duplicate symbols
  - Do NOT change the cmake-js build flow (keep `cmake-js build` as the entry point)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Static library linking paths need investigation — exact .a paths depend on Task 1's build output
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 3)
  - **Blocks**: Tasks 4, 5, 6
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `bindings/node/packages/darwin-arm64/CMakeLists.txt:24-97` — Current complete CMakeLists.txt — this is the base to modify. Preserve the cmake-js integration, NAPI definitions, and framework linking.
  - `bindings/node/packages/darwin-arm64/CMakeLists.txt:46-71` — Current link list — replace/extend with pipeline-lib and whisper

  **API/Type References**:
  - `diarization-ggml/CMakeLists.txt:116-135` — `transcription-lib` INTERFACE and `transcribe` target — shows what pipeline-lib depends on and what include dirs it needs
  - `diarization-ggml/src/pipeline.h:1-4` — Pipeline headers include `streaming_state.h`, `transcriber.h`, `aligner.h` — addon needs these in include path

  **WHY Each Reference Matters**:
  - Current CMakeLists line 46-71: This is the exact block to modify — add pipeline-lib and whisper to the existing link list, update ggml paths
  - Current CMakeLists line 36-43: Include directories — add whisper.cpp headers here
  - The .a file paths MUST match exactly what Task 1's build-static produces — run `find build-static -name "*.a"` to verify

  **Acceptance Criteria**:
  - [ ] CMakeLists.txt links `libpipeline-lib.a`
  - [ ] CMakeLists.txt links whisper static library (path verified against build-static output)
  - [ ] CMakeLists.txt includes whisper.cpp header directories
  - [ ] No duplicate ggml linking (only whisper's ggml, not standalone)
  - [ ] `cd bindings/node/packages/darwin-arm64 && pnpm build` succeeds (even with placeholder addon.cpp)

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Addon compiles with new link configuration
    Tool: Bash
    Preconditions: Task 1 complete (build-static exists), CMakeLists.txt updated
    Steps:
      1. cd bindings/node/packages/darwin-arm64
      2. pnpm build 2>&1
      3. test -f build/Release/pyannote-addon.node && echo "ADDON BUILD: PASS"
    Expected Result: .node file produced without link errors
    Failure Indicators: Undefined symbols, duplicate symbols, missing .a files
    Evidence: Terminal output captured

  Scenario: No duplicate symbol errors from ggml
    Tool: Bash
    Preconditions: CMakeLists.txt updated with whisper ggml paths only
    Steps:
      1. cd bindings/node/packages/darwin-arm64 && pnpm build 2>&1 | grep -i "duplicate symbol" | wc -l
      2. Assert: count is 0
    Expected Result: Zero duplicate symbol warnings/errors
    Evidence: grep output captured
  ```

  **Commit**: YES
  - Message: `build(node): update addon CMakeLists to link pipeline-lib and whisper.cpp`
  - Files: `bindings/node/packages/darwin-arm64/CMakeLists.txt`
  - Pre-commit: `cd bindings/node/packages/darwin-arm64 && pnpm build`

---

- [ ] 3. Rewrite TypeScript Types + Binding Interface

  **What to do**:
  - Rewrite `bindings/node/packages/pyannote-cpp-node/src/types.ts`:
    ```typescript
    export interface ModelConfig {
      segModelPath: string;
      embModelPath: string;
      pldaPath: string;
      coremlPath: string;
      segCoremlPath: string;
      whisperModelPath: string;
      whisperCoremlPath?: string;
      vadModelPath?: string;
      language?: string;    // undefined = auto-detect
      nThreads?: number;    // default: 4
    }

    export interface AlignedWord {
      text: string;
      start: number;  // seconds
      end: number;    // seconds
      // NOTE: C++ AlignedWord has a 'speaker' field but main_transcribe.cpp doesn't output it per-word.
      // Omit from TS type — speaker is on AlignedSegment only.
    }

    export interface AlignedSegment {
      speaker: string;
      start: number;
      duration: number;
      text: string;
      words: AlignedWord[];
    }

    export interface TranscriptionResult {
      segments: AlignedSegment[];
    }
    ```
  - Rewrite `bindings/node/packages/pyannote-cpp-node/src/binding.ts`:
    - Replace `NativePyannoteModel` with `NativePipelineModel` interface
    - Replace `NativeStreamingSession` with `NativePipelineSession` interface
    - `NativePipelineModel`: `transcribe(audio, callback)`, `createSession(callback)`, `close()`, `isClosed`
    - `NativePipelineSession`: `push(audio)`, `finalize()`, `close()`, `isClosed`
    - Update `NativeBinding` type and `isNativeBinding` guard
  - Rewrite `bindings/node/packages/pyannote-cpp-node/src/index.ts` with new exports

  **Must NOT do**:
  - Do NOT add any runtime code — this is pure type definitions and binding interface
  - Do NOT add VADChunk type — VAD is just `boolean[]` now
  - Do NOT keep old types (Segment, DiarizationResult, VADChunk)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Straightforward TypeScript type rewrite, no complex logic
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 2)
  - **Blocks**: Tasks 7, 8
  - **Blocked By**: Task 1 (needs to understand final API shape)

  **References**:

  **Pattern References**:
  - `bindings/node/packages/pyannote-cpp-node/src/types.ts:1-26` — Current types file — replace entirely with new types
  - `bindings/node/packages/pyannote-cpp-node/src/binding.ts:1-87` — Current binding interface — replace native interface contracts
  - `bindings/node/packages/pyannote-cpp-node/src/index.ts:1-9` — Current exports — update to new class/type names

  **API/Type References**:
  - `diarization-ggml/src/aligner.h:9-21` — C++ AlignedWord and AlignedSegment definitions — TS types must mirror these exactly
  - `diarization-ggml/src/pipeline.h:9-17` — PipelineConfig and pipeline_callback — informs native interface shape
  - `diarization-ggml/src/transcriber.h:5-10` — TranscriberConfig fields — maps to ModelConfig whisper fields

  **WHY Each Reference Matters**:
  - `aligner.h:9-21`: The TS `AlignedSegment` and `AlignedWord` must exactly match these C++ structs — field names, types, semantics
  - `pipeline.h:9-13`: PipelineConfig shows what the native layer needs — maps to ModelConfig fields
  - `transcriber.h:5-10`: Shows `whisper_model_path`, `whisper_coreml_path`, `n_threads`, `language` — must all appear in ModelConfig

  **Acceptance Criteria**:
  - [ ] `types.ts` exports `ModelConfig`, `AlignedWord`, `AlignedSegment`, `TranscriptionResult`
  - [ ] `binding.ts` exports `NativePipelineModel`, `NativePipelineSession`, `NativeBinding`
  - [ ] `index.ts` exports Pipeline, PipelineSession, and all types
  - [ ] `cd bindings/node/packages/pyannote-cpp-node && pnpm build` succeeds (tsc compiles)

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: TypeScript compiles without errors
    Tool: Bash
    Preconditions: types.ts, binding.ts, index.ts rewritten
    Steps:
      1. cd bindings/node/packages/pyannote-cpp-node
      2. npx tsc --noEmit 2>&1
      3. Assert: exit code 0
    Expected Result: Zero type errors
    Evidence: tsc output captured
  ```

  **Commit**: YES (groups with Task 7, 8)
  - Message: `feat(node): rewrite TypeScript types and binding interface for pipeline API`
  - Files: `types.ts`, `binding.ts`, `index.ts`

---

- [ ] 4. C++ PipelineModel + TranscribeWorker (One-Shot Path)

  **What to do**:
  - Create `bindings/node/packages/darwin-arm64/src/PipelineModel.h` — N-API ObjectWrap class:
    - Store `PipelineConfig` assembled from JS config object
    - Store `std::string` members for ALL path/language fields (string lifetime safety)
    - Methods: `Transcribe()`, `CreateSession()`, `Close()`, `GetIsClosed()`
    - Singleton pattern: `pipeline_init` called in constructor, `pipeline_free` in Close/destructor
    - ThreadSafeFunction is NOT needed here — Transcribe uses one-shot worker with internal callback capture
  - Create `bindings/node/packages/darwin-arm64/src/PipelineModel.cpp`:
    - Constructor: parse JS config object → populate `std::string` members → build `PipelineConfig` with `.c_str()` pointers → call `pipeline_init`
    - Handle optional fields: `whisperCoremlPath` (nullptr if absent), `vadModelPath` (nullptr if absent), `language` (nullptr for auto-detect), `nThreads` (default 4)
    - `Transcribe()`: validate input, create `TranscribeWorker`, queue it
    - `CreateSession()`: create PipelineSession via constructor ref (pass PipelineModel ref)
    - `Close()`: call `pipeline_free`, set closed flag
  - Create `bindings/node/packages/darwin-arm64/src/TranscribeWorker.h` and `TranscribeWorker.cpp`:
    - AsyncWorker that borrows the existing PipelineState from PipelineModel
    - In `Execute()`: push all audio in 16000-sample chunks via `pipeline_push`, then call `pipeline_finalize`
    - Capture results via C++ callback (set on PipelineState) — store last `vector<AlignedSegment>` in worker member
    - In `OnOK()`: convert `vector<AlignedSegment>` to JS objects (with words array), resolve promise
    - IMPORTANT: Since PipelineModel is singleton, must prevent concurrent transcribe/session operations (busy flag)

  **Must NOT do**:
  - Do NOT create ThreadSafeFunction here — one-shot path captures results in C++ and returns via promise
  - Do NOT modify pipeline.h or pipeline.cpp
  - Do NOT handle streaming events — that's Task 5

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: N-API ObjectWrap + AsyncWorker with callback capture is complex. Must handle string lifetime, singleton state, and busy-flag serialization correctly.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 5, 6)
  - **Blocks**: Tasks 7, 10
  - **Blocked By**: Task 2

  **References**:

  **Pattern References**:
  - `bindings/node/packages/darwin-arm64/src/PyannoteModel.cpp:22-66` — Current config parsing pattern — follow `getStringProp` helper, null checks, early returns on exception. Extend for optional fields.
  - `bindings/node/packages/darwin-arm64/src/DiarizeWorker.cpp:14-31` — Current one-shot worker pattern — `Execute()` does init/push-loop/finalize/free. Adapt: instead of creating fresh state, use shared PipelineState. Capture callback results in member variable.
  - `bindings/node/packages/darwin-arm64/src/DiarizeWorker.cpp:33-49` — Result marshaling pattern — convert C++ segments to JS objects. Extend to include `text` and `words` array.

  **API/Type References**:
  - `diarization-ggml/src/pipeline.h:9-24` — Complete pipeline C API: `PipelineConfig`, `pipeline_callback`, `pipeline_init/push/finalize/free`
  - `diarization-ggml/src/pipeline.cpp:24-43` — `PipelineState` struct — shows what state is managed internally
  - `diarization-ggml/src/aligner.h:9-21` — `AlignedWord` and `AlignedSegment` structs — must marshal these to JS
  - `diarization-ggml/src/transcriber.h:5-10` — `TranscriberConfig` fields with `const char*` — string lifetime concern
  - `diarization-ggml/src/streaming_state.h:9-16` — `StreamingConfig` fields — part of `PipelineConfig.diarization`

  **External References**:
  - N-API ThreadSafeFunction docs: https://github.com/nodejs/node-addon-api/blob/main/doc/threadsafe.md
  - N-API AsyncWorker docs: https://github.com/nodejs/node-addon-api/blob/main/doc/async_worker.md

  **WHY Each Reference Matters**:
  - `PyannoteModel.cpp:22-66`: Exact pattern to copy for config parsing — add whisper-specific optional fields
  - `DiarizeWorker.cpp:14-31`: Shows how to do push-loop in AsyncWorker — adapt for singleton state
  - `pipeline.h:15-17`: callback signature — determines how results are captured in TranscribeWorker
  - `transcriber.h:5-10`: `const char*` fields mean we MUST store `std::string` and use `.c_str()` pointers

  **Acceptance Criteria**:
  - [ ] PipelineModel.h/cpp created with Init, constructor, Transcribe, CreateSession, Close, GetIsClosed
  - [ ] TranscribeWorker.h/cpp created with Execute (push loop + finalize), OnOK (result marshaling)
  - [ ] Constructor parses all ModelConfig fields including optional ones
  - [ ] `std::string` members store all path/language strings (no dangling `const char*`)
  - [ ] Result marshaling includes `speaker`, `start`, `duration`, `text`, and `words[]` array
  - [ ] Each word has `text`, `start`, `end` fields
  - [ ] Busy flag prevents concurrent operations

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: PipelineModel compiles and loads
    Tool: Bash
    Preconditions: Tasks 1-2 complete, PipelineModel files created
    Steps:
      1. cd bindings/node/packages/darwin-arm64 && pnpm build 2>&1
      2. Assert: build succeeds, pyannote-addon.node produced
      3. node -e "const m = require('./build/Release/pyannote-addon.node'); console.log(typeof m.PipelineModel)"
      4. Assert: output is "function"
    Expected Result: Native module loads and exports PipelineModel constructor
    Evidence: Terminal output captured
  ```

  **Commit**: YES (groups with Tasks 5, 6)
  - Message: `feat(node): add PipelineModel and TranscribeWorker for one-shot transcription`
  - Files: `PipelineModel.h`, `PipelineModel.cpp`, `TranscribeWorker.h`, `TranscribeWorker.cpp`

---

- [ ] 5. C++ PipelineSession + TSFN + PushWorker + FinalizeWorker (Streaming Path)

  **What to do**:
  - Create `bindings/node/packages/darwin-arm64/src/PipelineSession.h` — N-API ObjectWrap:
    - Stores reference to parent `PipelineModel` (borrows its PipelineState)
    - Owns a `Napi::ThreadSafeFunction` for marshaling callbacks from C++ worker threads to JS
    - Methods: `Push()`, `Finalize()`, `Close()`, `GetIsClosed()`
    - The TSFN callback receives `vector<AlignedSegment>` + `vector<float>` audio and emits 'segments' event on the JS EventEmitter
  - Create `PipelineSession.cpp`:
    - Constructor: receive PipelineModel ref via External, store reference, create TSFN
    - The TSFN is the bridge: C++ callback → TSFN → JS event emission
    - `Push()`: create PipelinePushWorker, queue it. Worker calls `pipeline_push`, captures `vector<bool>` VAD result
    - `Finalize()`: create PipelineFinalizeWorker, queue it. Worker calls `pipeline_finalize` (blocking). TSFN fires for each intermediate callback during drain.
    - `Close()`: release TSFN, set closed flag. If in-flight operation, defer cleanup.
    - IMPORTANT: pipeline_push may trigger the callback synchronously (when Whisper result is ready). The TSFN must be set up BEFORE any push calls.
  - Create `PipelinePushWorker.h/.cpp`:
    - AsyncWorker that calls `pipeline_push(state, samples, n)` and stores the returned `vector<bool>`
    - `OnOK`: convert `vector<bool>` to JS `Array<boolean>`, resolve promise
    - NOTE: pipeline_push may internally fire the callback (via TSFN), but that happens on the worker thread and TSFN safely marshals it
  - Create `PipelineFinalizeWorker.h/.cpp`:
    - AsyncWorker that calls `pipeline_finalize(state)` (blocks until all Whisper results drained)
    - During finalize, the callback fires multiple times via TSFN
    - `OnOK`: the final result has already been emitted via TSFN callback. Resolve promise with the last emitted segments.
  - TSFN callback implementation:
    - Receives a struct with `vector<AlignedSegment>` segments + `vector<float>` audio
    - On JS main thread: construct JS segment objects, construct Float32Array from audio, emit 'segments' event via `session.Emit("segments", segmentsArray, audioArray)`
    - Use `Napi::ObjectWrap::Value()` to get the JS `this` and call `emit` on it

  **Must NOT do**:
  - Do NOT expose `recluster()` — pipeline manages reclustering internally
  - Do NOT allow concurrent push/finalize — busy flag serialization
  - Do NOT release TSFN while operations are in-flight — use deferred cleanup

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: ThreadSafeFunction lifecycle management is the hardest part of this entire plan. Must handle: TSFN creation, callback data marshaling, TSFN release timing, and interaction with AsyncWorker.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 4, 6)
  - **Blocks**: Tasks 8, 10
  - **Blocked By**: Task 2

  **References**:

  **Pattern References**:
  - `bindings/node/packages/darwin-arm64/src/StreamingSession.cpp:24-43` — Current session constructor pattern — receiving config via External, init state. Adapt: receive PipelineModel ref, set up TSFN.
  - `bindings/node/packages/darwin-arm64/src/StreamingSession.cpp:57-88` — Current Push method pattern — busy flag, copy audio, create worker. Same pattern but different worker.
  - `bindings/node/packages/darwin-arm64/src/PushWorker.cpp:20-43` — Result marshaling for push — adapt for `vector<bool>` instead of VADChunks.
  - `bindings/node/packages/darwin-arm64/src/FinalizeWorker.cpp:13-39` — Current finalize worker — simplify (no result marshaling needed, TSFN handles it)

  **API/Type References**:
  - `diarization-ggml/src/pipeline.h:15-17` — `pipeline_callback` typedef — this is what TSFN must bridge to JS
  - `diarization-ggml/src/pipeline.cpp:84-101` — `handle_whisper_result` — shows when/how callback fires: after each Whisper result with ALL accumulated segments
  - `diarization-ggml/src/pipeline.cpp:207-258` — `pipeline_finalize` — shows callback fires multiple times during drain loop

  **External References**:
  - N-API ThreadSafeFunction: https://github.com/nodejs/node-addon-api/blob/main/doc/threadsafe.md — MUST READ before implementing
  - N-API ThreadSafeFunction examples: search for `Napi::ThreadSafeFunction::New` usage patterns

  **WHY Each Reference Matters**:
  - `pipeline.h:15-17`: callback signature `(segments, audio, user_data)` determines the TSFN data struct
  - `pipeline.cpp:84-101`: callback fires with ALL segments (not incremental delta) — JS event emits complete array each time
  - `pipeline.cpp:207-258`: finalize fires callback N times — TSFN must handle multiple invocations before worker completes
  - TSFN docs: lifecycle rules (Acquire/Release) are critical to avoid segfaults and process hangs

  **Acceptance Criteria**:
  - [ ] PipelineSession.h/cpp with TSFN setup in constructor
  - [ ] PipelinePushWorker calls pipeline_push, returns vector<bool> as JS boolean[]
  - [ ] PipelineFinalizeWorker calls pipeline_finalize (blocking), TSFN fires for intermediate callbacks
  - [ ] TSFN callback marshals AlignedSegment[] + Float32Array audio to JS event
  - [ ] Busy flag prevents concurrent push/finalize
  - [ ] Close releases TSFN safely (deferred if in-flight)
  - [ ] 'segments' event emits on the PipelineSession JS object

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: PipelineSession push returns VAD predictions
    Tool: Bash
    Preconditions: Addon built, models available
    Steps:
      1. node -e "
         const addon = require('./build/Release/pyannote-addon.node');
         // (create model, create session, push 1s of audio)
         // Assert push returns boolean array of length ~31
         "
    Expected Result: push resolves with boolean[] of correct length
    Evidence: Terminal output captured

  Scenario: TSFN fires segments event during streaming
    Tool: Bash
    Preconditions: Full addon built with TSFN support
    Steps:
      1. Write test script that creates session, adds 'segments' listener, pushes 30s of audio, calls finalize
      2. Assert: segments event fired at least once before finalize resolved
      3. Assert: segments contain speaker, start, duration, text, words fields
    Expected Result: Events fire with correct data shape
    Evidence: Terminal output captured
  ```

  **Commit**: YES (groups with Tasks 4, 6)
  - Message: `feat(node): add PipelineSession with ThreadSafeFunction for streaming transcription`
  - Files: `PipelineSession.h`, `PipelineSession.cpp`, `PipelinePushWorker.h`, `PipelinePushWorker.cpp`, `PipelineFinalizeWorker.h`, `PipelineFinalizeWorker.cpp`

---

- [ ] 6. Rewrite addon.cpp (Register New Classes)

  **What to do**:
  - Replace `bindings/node/packages/darwin-arm64/src/addon.cpp` contents:
    ```cpp
    #include <napi.h>
    #include "PipelineModel.h"
    #include "PipelineSession.h"

    Napi::Object Init(Napi::Env env, Napi::Object exports) {
        PipelineModel::Init(env, exports);
        PipelineSession::Init(env, exports);
        return exports;
    }

    NODE_API_MODULE(pyannote, Init)
    ```
  - This is a minimal file — just registers the two new classes

  **Must NOT do**:
  - Do NOT reference old PyannoteModel or StreamingSession
  - Do NOT change module name from `pyannote` (downstream loading depends on it)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Trivial 12-line file replacement
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 4, 5)
  - **Blocks**: Task 10
  - **Blocked By**: Task 2

  **References**:
  - `bindings/node/packages/darwin-arm64/src/addon.cpp:1-12` — Current file — same structure, new includes

  **Acceptance Criteria**:
  - [ ] addon.cpp includes PipelineModel.h and PipelineSession.h
  - [ ] Init registers PipelineModel and PipelineSession
  - [ ] Module name remains `pyannote`

  **Commit**: YES (groups with Tasks 4, 5)
  - Message: (grouped)

---

- [ ] 7. TypeScript Pipeline.ts Wrapper

  **What to do**:
  - Create `bindings/node/packages/pyannote-cpp-node/src/Pipeline.ts` (replacing Pyannote.ts):
    ```typescript
    import { accessSync } from 'node:fs';
    import { getBinding, type NativePipelineModel } from './binding.js';
    import { PipelineSession } from './PipelineSession.js';
    import type { ModelConfig, TranscriptionResult } from './types.js';

    export class Pipeline {
      private native: NativePipelineModel;

      private constructor(native: NativePipelineModel) {
        this.native = native;
      }

      static async load(config: ModelConfig): Promise<Pipeline> {
        // Validate required paths exist
        const requiredPaths = [
          config.segModelPath, config.embModelPath, config.pldaPath,
          config.coremlPath, config.segCoremlPath, config.whisperModelPath,
        ];
        for (const path of requiredPaths) accessSync(path);

        // Validate optional paths if provided
        if (config.whisperCoremlPath) accessSync(config.whisperCoremlPath);
        if (config.vadModelPath) accessSync(config.vadModelPath);

        const binding = getBinding();
        const native = new binding.PipelineModel(config);
        return new Pipeline(native);
      }

      async transcribe(audio: Float32Array): Promise<TranscriptionResult> {
        if (this.native.isClosed) throw new Error('Pipeline is closed');
        if (!(audio instanceof Float32Array)) throw new TypeError('Expected Float32Array');
        if (audio.length === 0) throw new Error('Audio must not be empty');
        return this.native.transcribe(audio);
      }

      createSession(): PipelineSession {
        if (this.native.isClosed) throw new Error('Pipeline is closed');
        const nativeSession = this.native.createSession();
        return new PipelineSession(nativeSession);
      }

      close(): void { this.native.close(); }
      get isClosed(): boolean { return this.native.isClosed; }
    }
    ```
  - Delete `Pyannote.ts`

  **Must NOT do**:
  - Do NOT add file path audio loading
  - Do NOT add diarize() method — only transcribe()

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Straightforward TS class, follows existing Pyannote.ts pattern closely
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 8, 9)
  - **Blocks**: Task 10
  - **Blocked By**: Tasks 3, 4

  **References**:
  - `bindings/node/packages/pyannote-cpp-node/src/Pyannote.ts:1-61` — Current wrapper — same structure, new method names and types

  **Acceptance Criteria**:
  - [ ] Pipeline.ts exports Pipeline class with load(), transcribe(), createSession(), close()
  - [ ] Path validation includes required + optional paths
  - [ ] TypeScript compiles cleanly

  **Commit**: YES (groups with Tasks 3, 8)
  - Message: `feat(node): rewrite TypeScript wrapper for pipeline API`
  - Files: `Pipeline.ts`, `PipelineSession.ts`, `types.ts`, `binding.ts`, `index.ts`

---

- [ ] 8. TypeScript PipelineSession.ts EventEmitter Wrapper

  **What to do**:
  - Create `bindings/node/packages/pyannote-cpp-node/src/PipelineSession.ts` (replacing StreamingSession.ts):
    ```typescript
    import { EventEmitter } from 'node:events';
    import type { NativePipelineSession } from './binding.js';
    import type { AlignedSegment, TranscriptionResult } from './types.js';

    export interface PipelineSessionEvents {
      segments: [segments: AlignedSegment[], audio: Float32Array];
      error: [error: Error];
    }

    export class PipelineSession extends EventEmitter {
      private native: NativePipelineSession;

      constructor(native: NativePipelineSession) {
        super();
        this.native = native;
        // Native session will call this.emit('segments', ...) via TSFN
      }

      async push(audio: Float32Array): Promise<boolean[]> {
        if (this.native.isClosed) throw new Error('Session is closed');
        if (!(audio instanceof Float32Array)) throw new TypeError('Expected Float32Array');
        return this.native.push(audio);
      }

      async finalize(): Promise<TranscriptionResult> {
        if (this.native.isClosed) throw new Error('Session is closed');
        return this.native.finalize();
      }

      close(): void { this.native.close(); }
      get isClosed(): boolean { return this.native.isClosed; }
    }
    ```
  - Delete `StreamingSession.ts`
  - NOTE: The native TSFN calls back with segments data. The native PipelineSession must have a reference to the JS PipelineSession object to call `emit()` on it. This is handled in the C++ constructor where the JS `this` is captured for TSFN callback.

  **Must NOT do**:
  - Do NOT add recluster() method
  - Do NOT expose raw VADChunk type — push returns boolean[] directly

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple EventEmitter subclass
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 7, 9)
  - **Blocks**: Task 10
  - **Blocked By**: Tasks 3, 5

  **References**:
  - `bindings/node/packages/pyannote-cpp-node/src/StreamingSession.ts:1-43` — Current session wrapper — similar structure but extends EventEmitter instead of plain class

  **Acceptance Criteria**:
  - [ ] PipelineSession extends EventEmitter
  - [ ] Has push(), finalize(), close(), isClosed
  - [ ] TypeScript compiles with proper event typing

  **Commit**: YES (groups with Task 7)

---

- [ ] 9. Delete Old Source Files

  **What to do**:
  - Delete from `bindings/node/packages/darwin-arm64/src/`:
    - `PyannoteModel.h`, `PyannoteModel.cpp`
    - `StreamingSession.h`, `StreamingSession.cpp`
    - `DiarizeWorker.h`, `DiarizeWorker.cpp`
    - `PushWorker.h`, `PushWorker.cpp`
    - `ReclusterWorker.h`, `ReclusterWorker.cpp`
    - `FinalizeWorker.h`, `FinalizeWorker.cpp`
  - Delete from `bindings/node/packages/pyannote-cpp-node/src/`:
    - `Pyannote.ts`
    - `StreamingSession.ts`

  **Must NOT do**:
  - Do NOT delete `addon.cpp` (already rewritten in Task 6)
  - Do NOT delete `binding.ts`, `types.ts`, `index.ts` (already rewritten in Task 3)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: File deletion only
  - **Skills**: [`git-master`]
    - `git-master`: Clean git rm for tracked files

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 7, 8)
  - **Blocks**: Task 10
  - **Blocked By**: Tasks 4, 5, 6

  **References**:
  - Full file list in `bindings/node/packages/darwin-arm64/src/` — 13 files currently, 6 new + addon.cpp remain after deletion

  **Acceptance Criteria**:
  - [ ] All 12 old files deleted (6 C++ header/source pairs + 2 TS files)
  - [ ] Only new files remain in both src/ directories
  - [ ] Build still compiles after deletion

  **Commit**: YES
  - Message: `refactor(node): remove deprecated diarization-only binding files`
  - Files: all deleted files

---

- [ ] 10. Integration Tests + Build Verification

  **What to do**:
  - Rewrite `bindings/node/packages/pyannote-cpp-node/test/integration.test.ts` with:

    **Test groups:**
    1. **Model loading**: load with valid config, throw on invalid path, close is idempotent
    2. **One-shot transcribe**: returns AlignedSegment[] with words, detects speakers, segments sorted
    3. **Streaming push + events**: push audio, receive 'segments' events with correct shape
    4. **Streaming finalize**: returns complete TranscriptionResult
    5. **VAD predictions**: push returns boolean[] of correct length (~31 for 1s)
    6. **Event data shape**: segments have speaker/start/duration/text/words, words have text/start/end, audio is Float32Array
    7. **Resource cleanup**: close session then pipeline, push after close throws, finalize after close throws
    8. **JSON output comparison**: transcribe sample.wav, compare output format matches CLI `transcribe` binary output

  - Update `loadWav` helper (keep existing)
  - Config must include whisper model path + all new fields
  - For streaming tests: push 30+s of audio to trigger at least one Whisper result

  **Must NOT do**:
  - Do NOT test diarization-only API (deleted)
  - Do NOT test recluster (not exposed)
  - Do NOT require human verification of speaker accuracy

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Integration tests with native addon require careful setup, model paths, and async event handling
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 5 (final)
  - **Blocks**: None
  - **Blocked By**: Tasks 7, 8, 9

  **References**:

  **Pattern References**:
  - `bindings/node/packages/pyannote-cpp-node/test/integration.test.ts:1-255` — Current test file — same vitest structure, same `loadWav` helper, same model paths pattern. Extend config with whisper paths.

  **API/Type References**:
  - `diarization-ggml/src/main_transcribe.cpp:260-283` — JSON output format from CLI — test should verify JS output matches this structure
  - `diarization-ggml/src/aligner.h:9-21` — AlignedSegment/AlignedWord fields — test each field exists and has correct type

  **WHY Each Reference Matters**:
  - `integration.test.ts`: Copy test structure, loadWav helper, model path resolution. Add whisper model path.
  - `main_transcribe.cpp:260-283`: The JSON format ({segments: [{speaker, start, duration, text, words: [{text, start, end}]}]}) — JS output must match this shape exactly

  **Acceptance Criteria**:
  - [ ] All test groups pass: `cd bindings/node && pnpm test`
  - [ ] One-shot transcribe produces non-empty segments with words
  - [ ] Streaming events fire with correct data shape
  - [ ] VAD predictions are boolean[] of expected length
  - [ ] Resource cleanup tests pass (no crashes, proper error messages)

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Full test suite passes
    Tool: Bash
    Preconditions: All previous tasks complete, addon built, TS compiled
    Steps:
      1. cd bindings/node
      2. pnpm test 2>&1
      3. Assert: all tests pass, exit code 0
    Expected Result: vitest reports all tests passing
    Failure Indicators: Test failures, segfaults, timeout
    Evidence: .sisyphus/evidence/task-10-test-output.txt

  Scenario: One-shot transcribe returns words
    Tool: Bash
    Preconditions: Addon built, models available
    Steps:
      1. node -e "
         const { Pipeline } = require('pyannote-cpp-node');
         (async () => {
           const p = await Pipeline.load({...config});
           const r = await p.transcribe(loadWav('samples/sample.wav'));
           console.log(JSON.stringify(r.segments[0], null, 2));
           p.close();
         })();
         "
      2. Assert: output contains "speaker", "text", "words" fields
      3. Assert: words array is non-empty with text/start/end
    Expected Result: Complete AlignedSegment with words
    Evidence: Terminal output captured

  Scenario: Streaming session emits segments events
    Tool: Bash
    Preconditions: Full pipeline working
    Steps:
      1. Write script: create session, listen for 'segments', push 30s audio, finalize
      2. Count segment events received
      3. Assert: at least 1 event before finalize
      4. Assert: events contain AlignedSegment[] with words
    Expected Result: Incremental events fire during streaming
    Evidence: Terminal output captured
  ```

  **Commit**: YES
  - Message: `test(node): rewrite integration tests for pipeline API`
  - Files: `integration.test.ts`
  - Pre-commit: `cd bindings/node && pnpm test`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `build(diarization): add pipeline-lib static target for Node.js addon linking` | CMakeLists.txt | cmake build succeeds |
| 2 | `build(node): update addon CMakeLists to link pipeline-lib and whisper.cpp` | darwin-arm64/CMakeLists.txt | pnpm build |
| 4+5+6 | `feat(node): rewrite native addon for transcription+diarization pipeline` | 8 new C++ files + addon.cpp | pnpm build |
| 3+7+8 | `feat(node): rewrite TypeScript layer for pipeline API` | types.ts, binding.ts, index.ts, Pipeline.ts, PipelineSession.ts | tsc compiles |
| 9 | `refactor(node): remove deprecated diarization-only binding files` | 12 deleted files | pnpm build |
| 10 | `test(node): rewrite integration tests for pipeline API` | integration.test.ts | pnpm test |

---

## Success Criteria

### Verification Commands
```bash
# Build native addon
cd bindings/node/packages/darwin-arm64 && pnpm build  # Expected: .node file produced

# Build TypeScript
cd bindings/node/packages/pyannote-cpp-node && pnpm build  # Expected: dist/ with .js and .d.ts

# Run tests
cd bindings/node && pnpm test  # Expected: all tests pass

# Quick smoke test
node -e "const {Pipeline} = require('./bindings/node/packages/pyannote-cpp-node/dist/index.js'); console.log(typeof Pipeline)"
# Expected: function
```

### Final Checklist
- [ ] All "Must Have" present (transcribe, streaming, events, VAD, cleanup)
- [ ] All "Must NOT Have" absent (no old API, no recluster, no file path input)
- [ ] All tests pass
- [ ] No dangling `const char*` in native code
- [ ] TSFN properly released on close
- [ ] No duplicate ggml symbols at link time
