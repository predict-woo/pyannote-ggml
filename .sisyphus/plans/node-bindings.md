# Node.js Native Addon Bindings for pyannote-diarization-ggml

## TL;DR

> **Quick Summary**: Build a Node.js native addon (`pyannote-cpp-node`) wrapping the existing C++ diarization pipeline, following streaming-sortformer-ggml's binding patterns. Provides both offline diarization and streaming (push/recluster/finalize) APIs via async Promise-based interface.
> 
> **Deliverables**:
> - pnpm monorepo at `bindings/node/` with platform-specific native packages
> - C++ N-API binding layer (ObjectWrap + AsyncWorker)
> - TypeScript wrapper with full type definitions
> - Integration tests comparing DER against Python reference
> 
> **Estimated Effort**: Large
> **Parallel Execution**: YES — 3 waves
> **Critical Path**: Task 1 (static build) → Task 2 (scaffolding) → Task 3 (C++ binding) → Task 5 (TS wrapper) → Task 7 (tests)

---

## Context

### Original Request
Build a native addon npm binding for the pyannote-audio C++ diarization pipeline. Follow the high-level philosophy of streaming-sortformer-ggml's Node.js bindings as a loose reference, adapted to pyannote's different architecture (no low/mid/high level API — only VAD from push, diarization from recluster).

### Interview Summary
**Key Discussions**:
- Package name: `pyannote-cpp-node`
- Offline input: Float32Array only (user decodes audio)
- Platform: macOS-only (CoreML required for streaming)
- Scope: Core binding + tests only, no example apps
- All heavy operations async (push, recluster, finalize, diarize)
- Tests after implementation

**Research Findings**:
- pyannote streaming API: `init → push → recluster/finalize → free`
- push returns VADChunks (589 frames × binary activity per 10s chunk)
- recluster/finalize return DiarizationResult (sorted segments with speaker labels)
- Reference project (streaming-sortformer-ggml) uses: ObjectWrap, AsyncWorker, cmake-js, pnpm monorepo with platform packages, TS wrapper layer
- C API header (`diarization_stream.h`) exists but is NOT implemented — bind C++ API directly

### Metis Review
**Identified Gaps** (addressed):
- **diarize_buffer() vs streaming path**: Instead of modifying `diarization.cpp`, use `streaming_init → push(all_audio) → finalize` for offline path. AGENTS.md confirms finalize produces "byte-identical output to the offline pipeline." This avoids touching core C++.
- **Dynamic library linking**: GGML and CoreML bridges build as .dylib. Need static rebuild first or the .node won't be portable.
- **Thread safety**: streaming_recluster mutates state. Must serialize all operations per session.
- **Model loading may block event loop**: Need async factory method if loading takes >1s.
- **CoreML bridge ObjC++ flags**: addon CMake must handle `-fobjc-arc` for CoreML bridge compilation.

---

## Work Objectives

### Core Objective
Create a production-quality Node.js native addon that exposes the pyannote C++ diarization pipeline through an async, type-safe TypeScript API, following the architectural patterns established by streaming-sortformer-ggml.

### Concrete Deliverables
- `bindings/node/` — pnpm monorepo root
- `bindings/node/packages/pyannote-cpp-node/` — main TypeScript package
- `bindings/node/packages/darwin-arm64/` — native addon for macOS ARM64
- `bindings/node/packages/darwin-x64/` — native addon stub for macOS x64
- Modified `diarization-ggml/CMakeLists.txt` — static build option
- Integration tests comparing output to Python reference

### Definition of Done
- [x] `npx cmake-js build` in `packages/darwin-arm64/` produces `pyannote-addon.node`
- [x] `npx tsc` in `packages/pyannote-cpp-node/` compiles without errors
- [x] Integration test: offline diarize on sample.wav → DER ≤ 1.0% vs Python reference
- [x] Integration test: streaming push → finalize → same result as offline
- [x] `model.close()` and `session.close()` free all resources without crash

### Must Have
- Async (Promise-based) push, recluster, finalize, and diarize operations
- Float32Array input for audio (16kHz mono)
- DiarizationResult with typed segments: `{ start: number, duration: number, speaker: string }`
- VADChunk with typed vad array: `{ chunkIndex: number, startTime: number, vad: Float32Array }`
- Explicit `close()` + GC-based cleanup (destructor calls free)
- Clear error messages for: invalid model paths, closed model/session, invalid audio input
- Serialized access to StreamingSession (no concurrent push/recluster)

### Must NOT Have (Guardrails)
- **No ActivityStream/DiarizeStream wrappers** — pyannote has different semantics than sortformer
- **No WAV file loading** — Float32Array only, user decodes audio
- **No RTTM string generation** — return structured segments, consumer formats
- **No Linux/Windows support** — macOS + CoreML only this phase
- **No example apps** — core library + tests only
- **No provisional clustering** — standard recluster/finalize only
- **No npm publish/CI pipeline** — build and test locally
- **No modification to diarization.cpp** — use streaming API for offline path
- **No C API implementation** — bind C++ API directly via node-addon-api

---

## Verification Strategy

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
> ALL verification is executed by the agent using tools. No manual testing.

### Test Decision
- **Infrastructure exists**: NO (new project)
- **Automated tests**: YES (tests after implementation)
- **Framework**: vitest (matches reference project)

### Agent-Executed QA Scenarios (MANDATORY — ALL tasks)

All verification uses Bash commands to build, run, and assert. Evidence captured as terminal output.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Rebuild diarization-ggml with static linking
└── Task 2: Scaffold monorepo structure (package.json, tsconfig, etc.)

Wave 2 (After Wave 1):
├── Task 3: C++ N-API binding layer (ObjectWrap + AsyncWorkers)
└── Task 4: CMakeLists.txt for darwin-arm64 addon

Wave 3 (After Wave 2):
├── Task 5: TypeScript wrapper + type definitions
├── Task 6: Build verification (compile addon + TS)
└── Task 7: Integration tests + DER validation
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 3, 4 | 2 |
| 2 | None | 3, 4, 5 | 1 |
| 3 | 1, 2 | 5, 6, 7 | 4 |
| 4 | 1, 2 | 6 | 3 |
| 5 | 2, 3 | 7 | 6 |
| 6 | 3, 4 | 7 | 5 |
| 7 | 5, 6 | None | None (final) |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Agents |
|------|-------|-------------------|
| 1 | 1, 2 | task(category="quick") for each |
| 2 | 3, 4 | task(category="deep") for C++ binding, task(category="quick") for CMake |
| 3 | 5, 6, 7 | task(category="medium") for TS, task(category="quick") for build, task(category="deep") for tests |

---

## TODOs

- [x] 1. Rebuild diarization-ggml with Static Linking

  **What to do**:
  - Modify `diarization-ggml/CMakeLists.txt` to support static linking of all dependencies
  - Add option `DIARIZATION_STATIC` (default OFF) that forces static build of GGML, kaldi-native-fbank
  - Set `BUILD_SHARED_LIBS=OFF` for GGML submodule when static mode enabled
  - Build with: `cmake -B build-static -DDIARIZATION_STATIC=ON -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON && cmake --build build-static`
  - Verify all output libraries are `.a` files (not `.dylib`)
  - Verify the CLI executable still works with static build: run diarization on sample.wav

  **Must NOT do**:
  - Do NOT change the default build behavior (dynamic stays default)
  - Do NOT break the existing non-static build
  - Do NOT modify any pipeline logic — only build system changes

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Straightforward CMake modification with clear verification
  - **Skills**: []
    - No special skills needed — CMake is standard tooling

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: Tasks 3, 4
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `diarization-ggml/CMakeLists.txt` — Current build config, all targets and link deps defined here
  - `/tmp/streaming-sortformer-ggml/bindings/node/packages/darwin-arm64/CMakeLists.txt:17-23` — Reference shows linking against `.a` static libs

  **API/Type References**:
  - `diarization-ggml/CMakeLists.txt:6-7` — `EMBEDDING_COREML` and `SEGMENTATION_COREML` options
  - `diarization-ggml/CMakeLists.txt:28-39` — `diarization-lib` target sources
  - `diarization-ggml/CMakeLists.txt:46-49` — Link dependencies (segmentation-core, embedding-core)

  **Documentation References**:
  - `AGENTS.md` — Build commands section shows current cmake invocation pattern

  **WHY Each Reference Matters**:
  - `CMakeLists.txt` — You need to understand the existing target structure to add static build option without breaking dynamic
  - Reference CMake — Shows the expected output: linking `.a` files directly

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Static build produces .a libraries
    Tool: Bash
    Preconditions: diarization-ggml source exists
    Steps:
      1. cd diarization-ggml && cmake -B build-static -DDIARIZATION_STATIC=ON -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON
      2. cmake --build build-static
      3. find build-static -name "*.a" | sort
      4. Assert: libdiarization-lib.a exists
      5. Assert: libsegmentation-core.a exists
      6. Assert: libembedding-core.a exists
      7. Assert: libggml.a or libggml-base.a exists
    Expected Result: All core libraries built as static .a files
    Evidence: Terminal output of find command

  Scenario: Static-built CLI produces correct output
    Tool: Bash
    Preconditions: Static build complete, models exist
    Steps:
      1. Run: ./build-static/bin/diarization-ggml with sample.wav and all model paths
      2. Compare output RTTM with Python reference
      3. Assert: DER ≤ 1.0%
    Expected Result: Identical pipeline behavior with static linking
    Evidence: compare_rttm.py output

  Scenario: Default (non-static) build still works
    Tool: Bash
    Preconditions: No build-static directory conflicts
    Steps:
      1. cd diarization-ggml && cmake -B build -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON
      2. cmake --build build
      3. Assert: exit code 0
    Expected Result: Default build unaffected by new option
    Evidence: Build output
  ```

  **Commit**: YES
  - Message: `build(diarization): add DIARIZATION_STATIC option for static linking`
  - Files: `diarization-ggml/CMakeLists.txt`
  - Pre-commit: `cmake --build diarization-ggml/build-static`

---

- [x] 2. Scaffold Monorepo Structure

  **What to do**:
  - Create `bindings/node/` directory structure:
    ```
    bindings/node/
    ├── package.json             (workspace root, private)
    ├── pnpm-workspace.yaml      (packages: ['packages/*'])
    ├── tsconfig.json             (base TS config)
    ├── vitest.config.ts          (test config)
    ├── packages/
    │   ├── pyannote-cpp-node/   (main TS package)
    │   │   ├── package.json      (type: module, optionalDeps on platform pkgs)
    │   │   ├── tsconfig.json
    │   │   └── src/
    │   │       └── (placeholder index.ts)
    │   ├── darwin-arm64/         (native addon package)
    │   │   ├── package.json      (os: darwin, cpu: arm64)
    │   │   ├── index.js          (require('./build/Release/pyannote-addon.node'))
    │   │   └── src/
    │   │       └── (placeholder addon.cpp)
    │   └── darwin-x64/           (placeholder for x64)
    │       └── package.json      (os: darwin, cpu: x64)
    ```
  - All package.json files should have: name, version (0.1.0), engines (node >= 18)
  - Main package: `"type": "module"`, `"main": "./dist/index.js"`, `"types": "./dist/index.d.ts"`
  - Platform packages: `"main": "./index.js"`, os/cpu fields

  **Must NOT do**:
  - Do NOT implement any C++ or TypeScript logic — just scaffolding
  - Do NOT install dependencies yet (Task 3/5 will do that)
  - Do NOT create CMakeLists.txt for native addon (that's Task 4)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: File creation only, no logic
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: Tasks 3, 4, 5
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `/tmp/streaming-sortformer-ggml/bindings/node/package.json` — Workspace root structure (private, workspaces field)
  - `/tmp/streaming-sortformer-ggml/bindings/node/pnpm-workspace.yaml` — Workspace config (`packages: ['packages/*']`)
  - `/tmp/streaming-sortformer-ggml/bindings/node/packages/darwin-arm64/package.json` — Platform package with os/cpu fields, cmake-js/node-addon-api deps
  - `/tmp/streaming-sortformer-ggml/bindings/node/packages/darwin-arm64/index.js` — Single-line require of .node file
  - `/tmp/streaming-sortformer-ggml/bindings/node/packages/streaming-sortformer-node/package.json` — Main package with type:module, optionalDeps on platform packages

  **WHY Each Reference Matters**:
  - These files define the exact monorepo structure pattern. Copy the structure, change names/versions.

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: All package.json files are valid JSON with correct names
    Tool: Bash
    Preconditions: Files created
    Steps:
      1. node -e "JSON.parse(require('fs').readFileSync('bindings/node/package.json'))"
      2. node -e "const p = JSON.parse(require('fs').readFileSync('bindings/node/packages/pyannote-cpp-node/package.json')); assert(p.name === 'pyannote-cpp-node')"
      3. node -e "const p = JSON.parse(require('fs').readFileSync('bindings/node/packages/darwin-arm64/package.json')); assert(p.name === '@pyannote-cpp-node/darwin-arm64'); assert(p.os[0] === 'darwin'); assert(p.cpu[0] === 'arm64')"
    Expected Result: All JSON valid, names correct, platform fields correct
    Evidence: No error output from node commands
  ```

  **Commit**: YES
  - Message: `feat(bindings): scaffold Node.js monorepo structure`
  - Files: `bindings/node/**`
  - Pre-commit: `node -e "JSON.parse(require('fs').readFileSync('bindings/node/package.json'))"`

---

- [x] 3. Implement C++ N-API Binding Layer

  **What to do**:
  - Create `bindings/node/packages/darwin-arm64/src/` with these files:

  **addon.cpp** — Module entry point:
  - Register `PyannoteModel` and `StreamingSession` classes
  - `NODE_API_MODULE(pyannote, Init)`

  **PyannoteModel.h/.cpp** — Model wrapper (ObjectWrap):
  - Constructor: Takes config object `{ segModelPath, embModelPath, pldaPath, coremlPath, segCoremlPath }`
  - Calls `streaming_init()` to load models (validates all paths)
  - Stores `StreamingState*` (used for offline diarize-via-streaming and as model context source)
  - `diarize(Float32Array)` → creates DiarizeWorker → returns Promise
    - Under the hood: init new StreamingState → push all audio → finalize → free temp state → return segments
  - `createStreamingSession()` → returns new StreamingSession wrapping a fresh StreamingState
  - `close()` → calls streaming_free, sets ctx to null
  - `~PyannoteModel()` → calls Cleanup()

  **StreamingSession.h/.cpp** — Streaming session wrapper (ObjectWrap):
  - Constructor: Takes PyannoteModel + creates StreamingState via streaming_init with model's config
  - `push(Float32Array)` → creates PushWorker → returns Promise<VADChunk[]>
  - `recluster()` → creates ReclusterWorker → returns Promise<DiarizationResult>
  - `finalize()` → creates FinalizeWorker → returns Promise<DiarizationResult>
  - `close()` → calls streaming_free
  - Private member: `bool busy_` flag to prevent concurrent async operations
  - If `busy_` is true when push/recluster/finalize called → throw error

  **PushWorker.h/.cpp** — AsyncWorker for streaming_push:
  - Execute(): calls `streaming_push(state, samples, num_samples)`
  - OnOK(): marshals `vector<VADChunk>` → JS array of objects with Float32Array vad fields
  - OnError(): rejects promise

  **ReclusterWorker.h/.cpp** — AsyncWorker for streaming_recluster:
  - Execute(): calls `streaming_recluster(state)`
  - OnOK(): marshals `DiarizationResult.segments` → JS array of {start, duration, speaker}

  **FinalizeWorker.h/.cpp** — AsyncWorker for streaming_finalize:
  - Execute(): calls `streaming_finalize(state)`
  - OnOK(): same marshalling as ReclusterWorker

  **DiarizeWorker.h/.cpp** — AsyncWorker for offline diarization:
  - Execute(): creates temp StreamingState, pushes all audio in 16000-sample chunks, calls finalize, frees state
  - OnOK(): marshals DiarizationResult → JS array of segments
  - This encapsulates the entire offline pipeline in one async operation

  **Must NOT do**:
  - Do NOT use the C API (`diarization_stream.h`) — it's not implemented
  - Do NOT modify any existing C++ pipeline files
  - Do NOT add ActivityStream or DiarizeStream equivalent wrappers
  - Do NOT generate RTTM strings — return structured data only
  - Do NOT try to share StreamingState across threads

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Complex C++ binding with multiple interacting classes, async patterns, careful memory management
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 4)
  - **Blocks**: Tasks 5, 6, 7
  - **Blocked By**: Tasks 1, 2

  **References**:

  **Pattern References**:
  - `/tmp/streaming-sortformer-ggml/bindings/node/packages/darwin-arm64/src/addon.cpp` — Entry point pattern (Init, NODE_API_MODULE)
  - `/tmp/streaming-sortformer-ggml/bindings/node/packages/darwin-arm64/src/SortformerModel.h` — ObjectWrap class pattern (Init, constructor, destructor, Close, Cleanup)
  - `/tmp/streaming-sortformer-ggml/bindings/node/packages/darwin-arm64/src/SortformerModel.cpp` — Constructor with string arg, DefineClass, static FunctionReference
  - `/tmp/streaming-sortformer-ggml/bindings/node/packages/darwin-arm64/src/StreamingSession.h` — Session ObjectWrap with Feed/Flush/Reset/Close methods
  - `/tmp/streaming-sortformer-ggml/bindings/node/packages/darwin-arm64/src/StreamingSession.cpp:27-73` — Constructor unwrapping model ObjectWrap to get context pointer
  - `/tmp/streaming-sortformer-ggml/bindings/node/packages/darwin-arm64/src/StreamFeedWorker.h` — AsyncWorker pattern (Execute/OnOK/OnError, vector results, deferred promise)
  - `/tmp/streaming-sortformer-ggml/bindings/node/packages/darwin-arm64/src/StreamFeedWorker.cpp:19-74` — Execute on worker thread (no Napi calls!), OnOK marshals to JS types, memcpy to ArrayBuffer

  **API/Type References**:
  - `diarization-ggml/src/streaming.h` — Full streaming C++ API: streaming_init, streaming_push, streaming_recluster, streaming_finalize, streaming_free, VADChunk struct
  - `diarization-ggml/src/streaming_state.h` — StreamingConfig struct (5 model path strings), StreamingState struct
  - `diarization-ggml/include/diarization.h` — DiarizationResult struct (Segment with start, duration, speaker)

  **Documentation References**:
  - `AGENTS.md:Streaming Architecture` — Data flow, constants, lifecycle, critical gotchas
  - `AGENTS.md:Common Pitfalls #5` — streaming_recluster mutates state (chunk_idx, local_speaker_idx, embeddings overwritten)
  - `AGENTS.md:Common Pitfalls #6` — Streaming requires CoreML, no GGML fallback

  **WHY Each Reference Matters**:
  - sortformer addon.cpp — exact pattern to copy for module registration
  - SortformerModel — shows ObjectWrap lifecycle (constructor loads, destructor frees, Close is idempotent)
  - StreamingSession — shows how to unwrap another ObjectWrap's context pointer (model → session relationship)
  - StreamFeedWorker — shows AsyncWorker threading rules (no Napi in Execute, marshalling in OnOK)
  - streaming.h — the actual C++ functions we're calling
  - AGENTS.md — critical constraints (CoreML, state mutation, thread safety)

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: All C++ source files compile without errors
    Tool: Bash
    Preconditions: Task 4 CMakeLists.txt created, Task 1 static build complete
    Steps:
      1. cd bindings/node/packages/darwin-arm64
      2. npx cmake-js build
      3. Assert: exit code 0
      4. ls build/Release/pyannote-addon.node
      5. Assert: file exists and size > 0
    Expected Result: Native addon builds successfully
    Evidence: Build output + file listing

  Scenario: Addon loads in Node.js without crash
    Tool: Bash
    Preconditions: Build complete
    Steps:
      1. node -e "const m = require('./bindings/node/packages/darwin-arm64/build/Release/pyannote-addon.node'); console.log(Object.keys(m));"
      2. Assert: output contains 'PyannoteModel' and 'StreamingSession'
    Expected Result: Module exports both classes
    Evidence: Terminal output showing exported class names
  ```

  **Commit**: YES
  - Message: `feat(bindings): implement C++ N-API binding layer`
  - Files: `bindings/node/packages/darwin-arm64/src/*.cpp`, `bindings/node/packages/darwin-arm64/src/*.h`
  - Pre-commit: `cd bindings/node/packages/darwin-arm64 && npx cmake-js build`

---

- [x] 4. Create CMakeLists.txt for Native Addon

  **What to do**:
  - Create `bindings/node/packages/darwin-arm64/CMakeLists.txt`
  - Target: `pyannote-addon` (SHARED library, outputs `.node`)
  - Set `NAPI_VERSION=7`
  - Find `node-addon-api` include path via `node -p "require('node-addon-api').include"`
  - Include dirs: CMAKE_JS_INC, node-addon-api, diarization-ggml/include, diarization-ggml/src, streaming.h location
  - Link against static libraries from Task 1's `build-static/` output:
    - `libdiarization-lib.a`
    - `libsegmentation-core.a`, `libembedding-core.a`
    - `libsegmentation-coreml.a` (or .dylib if static not possible for ObjC++)
    - `libembedding-coreml.a` (or .dylib)
    - GGML static libs: `libggml.a`, `libggml-base.a`, `libggml-cpu.a`
    - `libkaldi-native-fbank-core.a`
  - Link Apple frameworks: Accelerate, Foundation, CoreML, Metal
  - Set compile definitions: `EMBEDDING_USE_COREML`, `SEGMENTATION_USE_COREML`, `ACCELERATE_NEW_LAPACK`
  - C++17 standard
  - ObjC++ ARC flags for CoreML bridge
  - Output: `PREFIX ""`, `SUFFIX ".node"`

  **Must NOT do**:
  - Do NOT hardcode absolute paths — use relative paths from `CMAKE_CURRENT_SOURCE_DIR`
  - Do NOT link dynamic libraries unless absolutely needed (CoreML bridge may require it)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single CMake file, patterns well-established from reference
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 3)
  - **Blocks**: Task 6
  - **Blocked By**: Tasks 1, 2

  **References**:

  **Pattern References**:
  - `/tmp/streaming-sortformer-ggml/bindings/node/packages/darwin-arm64/CMakeLists.txt` — Complete reference CMake for Node addon (NAPI_VERSION, node-addon-api discovery, static lib linking, CoreML framework linking, .node suffix)

  **API/Type References**:
  - `diarization-ggml/CMakeLists.txt:28-66` — All library targets and their link deps
  - `diarization-ggml/CMakeLists.txt:41-44` — Include directory structure

  **WHY Each Reference Matters**:
  - Reference CMakeLists.txt — nearly identical structure to copy, just change library paths and names
  - diarization CMakeLists.txt — shows exact link chain needed

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: CMakeLists.txt is valid and configurable
    Tool: Bash
    Preconditions: Task 1 static build complete, node-addon-api installed
    Steps:
      1. cd bindings/node/packages/darwin-arm64
      2. npm install node-addon-api cmake-js
      3. npx cmake-js configure
      4. Assert: exit code 0, no CMake errors
    Expected Result: CMake configuration succeeds
    Evidence: CMake output showing "Configuration done"
  ```

  **Commit**: YES (groups with Task 3)
  - Message: `build(bindings): add CMakeLists.txt for darwin-arm64 native addon`
  - Files: `bindings/node/packages/darwin-arm64/CMakeLists.txt`
  - Pre-commit: `cd bindings/node/packages/darwin-arm64 && npx cmake-js configure`

---

- [x] 5. Implement TypeScript Wrapper + Type Definitions

  **What to do**:
  - Create `bindings/node/packages/pyannote-cpp-node/src/` with these files:

  **types.ts** — Type definitions:
  ```typescript
  export interface ModelConfig {
    segModelPath: string;
    embModelPath: string;
    pldaPath: string;
    coremlPath: string;      // embedding .mlpackage
    segCoremlPath: string;   // segmentation .mlpackage
  }

  export interface VADChunk {
    chunkIndex: number;
    startTime: number;
    duration: number;
    numFrames: number;
    vad: Float32Array;        // [589] binary activity
  }

  export interface Segment {
    start: number;
    duration: number;
    speaker: string;          // "SPEAKER_00", etc.
  }

  export interface DiarizationResult {
    segments: Segment[];
  }
  ```

  **binding.ts** — Platform-specific binding loader:
  - Detect `process.platform` + `process.arch`
  - Load `@pyannote-cpp-node/darwin-arm64` or `@pyannote-cpp-node/darwin-x64`
  - Cache binding, throw clear error for unsupported platforms

  **Pyannote.ts** — Main API class:
  - `static async load(config: ModelConfig): Promise<Pyannote>` — factory method
    - Validates all 5 paths exist (fs.accessSync)
    - Creates native PyannoteModel
    - Returns wrapped instance
  - `async diarize(audio: Float32Array): Promise<DiarizationResult>` — offline path
    - Validates: not closed, instanceof Float32Array, length > 0
    - Calls native diarize, returns typed result
  - `createStreamingSession(): StreamingSession` — streaming factory
    - Validates: not closed
    - Creates native session, wraps in TS class
  - `close(): void` — idempotent cleanup
  - `get isClosed(): boolean`

  **StreamingSession.ts** — Streaming wrapper:
  - `async push(audio: Float32Array): Promise<VADChunk[]>` — feed audio
  - `async recluster(): Promise<DiarizationResult>` — on-demand clustering
  - `async finalize(): Promise<DiarizationResult>` — final clustering
  - `close(): void`
  - `get isClosed(): boolean`

  **index.ts** — Exports:
  - Export `Pyannote`, `StreamingSession`
  - Export all types

  **Must NOT do**:
  - Do NOT add ActivityStream, DiarizeStream, or any high-level wrappers
  - Do NOT add RTTM formatting utilities
  - Do NOT add WAV file loading helpers
  - Do NOT type native binding as `any` — use `unknown` with runtime checks or interface assertions

  **Recommended Agent Profile**:
  - **Category**: `medium`
    - Reason: Multiple TS files with validation logic, but patterns well-established from reference
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 6, 7)
  - **Blocks**: Task 7
  - **Blocked By**: Tasks 2, 3

  **References**:

  **Pattern References**:
  - `/tmp/streaming-sortformer-ggml/bindings/node/packages/streaming-sortformer-node/src/Sortformer.ts` — Main class pattern (private constructor, static load(), diarize(), createStreamingSession(), close())
  - `/tmp/streaming-sortformer-ggml/bindings/node/packages/streaming-sortformer-node/src/binding.ts` — Platform detection and caching pattern
  - `/tmp/streaming-sortformer-ggml/bindings/node/packages/streaming-sortformer-node/src/StreamingSession.ts` — Session wrapper with feed/flush/reset/close and totalFrames getter
  - `/tmp/streaming-sortformer-ggml/bindings/node/packages/streaming-sortformer-node/src/types.ts` — Type definitions pattern (interfaces for options, results, callbacks)
  - `/tmp/streaming-sortformer-ggml/bindings/node/packages/streaming-sortformer-node/src/index.ts` — Export pattern

  **API/Type References**:
  - `diarization-ggml/src/streaming.h:7-13` — VADChunk struct (C++) to mirror in TS
  - `diarization-ggml/include/diarization.h:16-23` — DiarizationResult and Segment structs to mirror in TS
  - `diarization-ggml/src/streaming_state.h:9-17` — StreamingConfig struct (maps to ModelConfig)

  **WHY Each Reference Matters**:
  - Sortformer.ts — exact class structure to adapt (load/diarize/createStreamingSession/close pattern)
  - binding.ts — platform detection logic to copy (change package names)
  - types.ts — type definition organization to follow
  - C++ headers — source of truth for what the native side returns

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: TypeScript compiles without errors
    Tool: Bash
    Preconditions: TS files created, tsconfig.json configured
    Steps:
      1. cd bindings/node/packages/pyannote-cpp-node
      2. npm install typescript
      3. npx tsc --noEmit
      4. Assert: exit code 0
    Expected Result: No type errors
    Evidence: tsc output (empty = success)

  Scenario: All types are properly exported
    Tool: Bash
    Preconditions: TS compiled to dist/
    Steps:
      1. npx tsc
      2. node -e "import('file://' + process.cwd() + '/bindings/node/packages/pyannote-cpp-node/dist/index.js').then(m => console.log(Object.keys(m)))"
      3. Assert: output contains 'Pyannote' and 'StreamingSession'
    Expected Result: Both classes exported
    Evidence: Terminal output
  ```

  **Commit**: YES
  - Message: `feat(bindings): implement TypeScript wrapper and type definitions`
  - Files: `bindings/node/packages/pyannote-cpp-node/src/*.ts`
  - Pre-commit: `cd bindings/node/packages/pyannote-cpp-node && npx tsc --noEmit`

---

- [x] 6. Build Verification (End-to-End Compile)

  **What to do**:
  - Install all dependencies: `cd bindings/node && pnpm install`
  - Build native addon: `cd packages/darwin-arm64 && npx cmake-js build`
  - Build TypeScript: `cd packages/pyannote-cpp-node && npx tsc`
  - Verify the .node file loads in Node.js
  - Verify TypeScript output in dist/ has correct exports
  - Fix any build issues (link errors, missing includes, type errors)

  **Must NOT do**:
  - Do NOT skip any build step
  - Do NOT manually copy .node files — build system should handle paths

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Running build commands and fixing errors
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 5, 7) — but depends on Tasks 3, 4
  - **Blocks**: Task 7
  - **Blocked By**: Tasks 3, 4

  **References**:

  **Pattern References**:
  - `/tmp/streaming-sortformer-ggml/bindings/node/packages/darwin-arm64/package.json:17-20` — Build scripts (cmake-js build/clean)
  - `/tmp/streaming-sortformer-ggml/bindings/node/package.json:11-16` — Workspace build scripts

  **WHY Each Reference Matters**:
  - Shows the expected build workflow for each package in the monorepo

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Full build pipeline succeeds
    Tool: Bash
    Preconditions: All source files from Tasks 1-5 exist
    Steps:
      1. cd bindings/node && pnpm install
      2. cd packages/darwin-arm64 && npx cmake-js build
      3. Assert: build/Release/pyannote-addon.node exists
      4. cd ../pyannote-cpp-node && npx tsc
      5. Assert: dist/index.js exists
      6. node -e "require('../darwin-arm64/build/Release/pyannote-addon.node')"
      7. Assert: no crash
    Expected Result: All steps pass, addon loads
    Evidence: Build outputs + file existence checks

  Scenario: Module loads and exports correct API
    Tool: Bash
    Preconditions: Build complete
    Steps:
      1. node -e "
         const addon = require('./bindings/node/packages/darwin-arm64/build/Release/pyannote-addon.node');
         console.log('Exports:', Object.keys(addon));
         console.log('Has PyannoteModel:', typeof addon.PyannoteModel === 'function');
         console.log('Has StreamingSession:', typeof addon.StreamingSession === 'function');
         "
      2. Assert: PyannoteModel and StreamingSession are functions (constructors)
    Expected Result: Both classes available as constructors
    Evidence: Terminal output showing function types
  ```

  **Commit**: YES (if fixes needed)
  - Message: `fix(bindings): resolve build issues`
  - Files: `bindings/node/**`
  - Pre-commit: full build pipeline

---

- [x] 7. Integration Tests + DER Validation

  **What to do**:
  - Create `bindings/node/packages/pyannote-cpp-node/test/integration.test.ts`
  - Install vitest as devDependency in workspace root
  - Write test helper to load sample.wav as Float32Array (read WAV file, parse header, extract PCM, convert to float32)
  - Tests:

  **Test 1: Model loading**
  - Load model with valid paths → succeeds
  - Load model with invalid path → throws with clear error message
  - close() → no crash, isClosed is true
  - close() twice → no crash (idempotent)

  **Test 2: Offline diarization**
  - Load model, diarize sample.wav Float32Array
  - Assert: result.segments is non-empty array
  - Assert: each segment has start (number), duration (number), speaker (string)
  - Assert: segments are sorted by start time
  - Assert: at least 2 unique speakers detected
  - Assert: total speech duration > 0

  **Test 3: DER validation (gold standard)**
  - Diarize sample.wav
  - Write result to RTTM format string
  - Save to temp file, compare with Python reference using compare_rttm.py
  - Assert: DER ≤ 1.0%

  **Test 4: Streaming basic flow**
  - Create streaming session
  - Push 16000 samples × 15 times (15 seconds)
  - After 10 pushes: should start getting VADChunks back
  - Each VADChunk: chunkIndex is number, vad is Float32Array of length 589
  - Call finalize → DiarizationResult with segments

  **Test 5: Streaming finalize matches offline**
  - Load same audio, run offline diarize and streaming finalize
  - Compare: same number of segments, same speaker labels, timestamps within 0.001s

  **Test 6: Resource cleanup**
  - Create model, create session, close session, close model → no crash
  - Push after close → throws error
  - Recluster after close → throws error

  **Must NOT do**:
  - Do NOT use external WAV parsing libraries — write minimal parser (16-bit PCM only)
  - Do NOT test on audio other than samples/sample.wav
  - Do NOT test performance/timing — just correctness

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Integration tests require careful audio handling, DER comparison, and multiple async patterns
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (final task)
  - **Blocks**: None (final)
  - **Blocked By**: Tasks 5, 6

  **References**:

  **Pattern References**:
  - `/tmp/streaming-sortformer-ggml/bindings/node/packages/streaming-sortformer-node/test/integration.test.ts` — Integration test patterns (if it exists)
  - `/tmp/streaming-sortformer-ggml/bindings/node/test/basic.test.ts` — Basic test patterns

  **Test References**:
  - `diarization-ggml/tests/compare_rttm.py` — DER comparison script (ground truth validation)
  - `diarization-ggml/tests/test_streaming.cpp` — C++ streaming test patterns (push flow, timing)

  **Documentation References**:
  - `AGENTS.md:Testing Protocol` — Full DER test command sequence
  - `AGENTS.md:Streaming Architecture` — Constants (SAMPLE_RATE=16000, CHUNK_SAMPLES=160000, FRAMES_PER_CHUNK=589)

  **External References**:
  - `samples/sample.wav` — Test audio file (must exist)
  - `/tmp/py_reference.rttm` — Python reference RTTM (generated via AGENTS.md instructions)

  **WHY Each Reference Matters**:
  - compare_rttm.py — THE ground truth validation tool. DER ≤ 1.0% is the acceptance bar.
  - AGENTS.md — specifies expected test outputs (2 speakers, 13 segments)
  - streaming test — shows the push flow and timing expectations

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: All integration tests pass
    Tool: Bash
    Preconditions: Full build complete, sample.wav exists, Python reference RTTM exists
    Steps:
      1. cd bindings/node
      2. npx vitest run --reporter=verbose
      3. Assert: all tests pass (0 failures)
      4. Assert: output shows test names for model loading, offline, streaming, DER, cleanup
    Expected Result: All 6 test groups pass
    Evidence: vitest output with pass/fail counts

  Scenario: DER validation against Python reference
    Tool: Bash
    Preconditions: Offline diarize test writes /tmp/node_test.rttm
    Steps:
      1. .venv/bin/python3 diarization-ggml/tests/compare_rttm.py /tmp/node_test.rttm /tmp/py_reference.rttm --threshold 1.0
      2. Assert: output shows DER ≤ 1.0%
      3. Assert: 2 speakers detected
    Expected Result: Node.js binding produces identical output to Python pipeline
    Evidence: compare_rttm.py output showing DER percentage
  ```

  **Commit**: YES
  - Message: `test(bindings): add integration tests with DER validation`
  - Files: `bindings/node/packages/pyannote-cpp-node/test/*.ts`, `bindings/node/vitest.config.ts`
  - Pre-commit: `cd bindings/node && npx vitest run`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `build(diarization): add DIARIZATION_STATIC option for static linking` | `diarization-ggml/CMakeLists.txt` | cmake --build build-static |
| 2 | `feat(bindings): scaffold Node.js monorepo structure` | `bindings/node/**` | valid JSON in all package.json |
| 3+4 | `feat(bindings): implement C++ N-API binding layer with CMake` | `bindings/node/packages/darwin-arm64/src/*`, `CMakeLists.txt` | cmake-js build |
| 5 | `feat(bindings): implement TypeScript wrapper and type definitions` | `bindings/node/packages/pyannote-cpp-node/src/*` | tsc --noEmit |
| 6 | `fix(bindings): resolve build issues` (if needed) | `bindings/node/**` | full build |
| 7 | `test(bindings): add integration tests with DER validation` | `bindings/node/packages/pyannote-cpp-node/test/*` | vitest run |

---

## Success Criteria

### Verification Commands
```bash
# Build native addon
cd bindings/node/packages/darwin-arm64 && npx cmake-js build
# Expected: pyannote-addon.node exists in build/Release/

# Build TypeScript
cd bindings/node/packages/pyannote-cpp-node && npx tsc
# Expected: dist/ populated with .js and .d.ts files

# Run tests
cd bindings/node && npx vitest run
# Expected: all tests pass

# DER validation
.venv/bin/python3 diarization-ggml/tests/compare_rttm.py /tmp/node_test.rttm /tmp/py_reference.rttm --threshold 1.0
# Expected: DER ≤ 1.0%
```

### Final Checklist
- [x] All "Must Have" present (async API, Float32Array input, typed results, close+GC cleanup, error messages, serialized access)
- [x] All "Must NOT Have" absent (no ActivityStream, no WAV loading, no RTTM gen, no Linux, no examples, no provisional, no diarization.cpp changes)
- [x] Native addon loads without crash on macOS ARM64
- [x] TypeScript compiles without errors
- [x] All integration tests pass
- [x] DER ≤ 1.0% vs Python reference
