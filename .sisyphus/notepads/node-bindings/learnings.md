# Learnings — node-bindings plan


## Task: Add DIARIZATION_STATIC CMake Option (Wave 1)

### Architecture & Pattern

**CMake BUILD_SHARED_LIBS propagation:**
- GGML respects `BUILD_SHARED_LIBS` option (line 82 of ggml/CMakeLists.txt)
- kaldi-native-fbank respects `BUILD_SHARED_LIBS` (lines 67-78 of its CMakeLists.txt)
- CoreML bridge libraries (segmentation-coreml, embedding-coreml) inherit from BUILD_SHARED_LIBS if not explicitly typed
- Solution: Set `BUILD_SHARED_LIBS=OFF` BEFORE `add_subdirectory()` calls to force static linking

### Implementation Details

1. **Added DIARIZATION_STATIC option** (line 8 of diarization-ggml/CMakeLists.txt):
   - Defaults to OFF (preserves dynamic linking by default)
   - When ON: forces all dependencies to build as static (.a)

2. **Conditional BUILD_SHARED_LIBS override:**
   - Added before GGML add_subdirectory (lines 12-14)
   - Added before kaldi-native-fbank add_subdirectory (lines 22-24)
   - Uses `set(...CACHE BOOL "" FORCE)` to override cached values
   - Must be INSIDE the `if(NOT TARGET ...)` guard to avoid re-setting on multi-project builds

3. **Build artifacts:**
   - **Dynamic (default):** libggml*.dylib, libkaldi-native-fbank-core.dylib, lib*-coreml.dylib
   - **Static (DIARIZATION_STATIC=ON):** libggml*.a, libkaldi-native-fbank-core.a, lib*-coreml.a

### Verification

✅ **Static build test:**
```bash
cmake -B build-static -DDIARIZATION_STATIC=ON -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON
cmake --build build-static
./build-static/bin/diarization-ggml ... (produces identical RTTM output)
find build-static -name "*.dylib" -o -name "*.a" | grep -c dylib  # → 0 dylibs
```

✅ **Default behavior preserved:**
```bash
cmake -B build-default -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON
cmake --build build-default
# Produces: libggml*.0.9.5.dylib, libkaldi-native-fbank-core.dylib, etc.
```

### Key Insights

1. **Cache variable handling:** Using `CACHE BOOL "" FORCE` is necessary to override CMake's built-in BUILD_SHARED_LIBS cache entry, especially when subprojects have already cached it with different values.

2. **Guard placement:** The `if(DIARIZATION_STATIC)` block must be INSIDE `if(NOT TARGET ...)` to avoid re-setting BUILD_SHARED_LIBS on subsequent CMake runs in multi-project builds.

3. **CoreML bridges automatically follow:** Since segmentation-coreml and embedding-coreml libraries don't specify SHARED/STATIC explicitly in their add_library() calls, they automatically inherit the BUILD_SHARED_LIBS setting.

4. **Pipeline correctness:** Static and dynamic builds produce byte-identical diarization results (same RTTM output). The linking approach doesn't affect numerical accuracy.

### Usage

```bash
# Default dynamic build (existing behavior)
cmake -B build -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON

# Static build (new)
cmake -B build-static -DDIARIZATION_STATIC=ON -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON

# Query the option
cmake -B build && cmake -B build -LA | grep DIARIZATION_STATIC
```

### Testing Protocol

For any future changes to build system, verify:
1. Default build produces .dylib/.so files for dependencies
2. Static build produces .a files for all dependencies
3. Both builds pass the pipeline test (sample.wav → RTTM) with identical output
4. No missing symbols or linker errors in either build mode


## Task: Create Node.js CMakeLists.txt for darwin-arm64 (Wave 2)

### CMakeLists.txt Architecture

**File:** `bindings/node/packages/darwin-arm64/CMakeLists.txt`

**Key Design Decisions:**

1. **Path Resolution:**
   - DIARIZATION_ROOT uses relative path: `${CMAKE_CURRENT_SOURCE_DIR}/../../../..` (4 levels up from darwin-arm64/)
   - Points to monorepo root
   - DIARIZATION_BUILD points to `${DIARIZATION_ROOT}/diarization-ggml/build-static`
   - Avoids hardcoded absolute paths

2. **Node.js Integration:**
   - Uses `cmake-js` variables: `${CMAKE_JS_INC}`, `${CMAKE_JS_SRC}`, `${CMAKE_JS_LIB}`
   - Discovers `node-addon-api` path via `execute_process(node -p "require(...).include")`
   - Removes quotes from node-addon-api path (CMake quirk)
   - Sets NAPI_VERSION=7 and NAPI_DISABLE_CPP_EXCEPTIONS

3. **Static Library Linking (from build-static):**
   - All 12 static libraries linked in dependency order:
     - Core pipeline: libdiarization-lib.a
     - Segmentation (2): libsegmentation-core.a + libsegmentation-coreml.a
     - Embedding (2): libembedding-core.a + libembedding-coreml.a
     - GGML (5): libggml.a, libggml-base.a, libggml-cpu.a, libggml-metal.a, libggml-blas.a
     - Fbank (2): libkaldi-native-fbank-core.a + libkissfft-float.a

4. **Apple Frameworks:**
   - Accelerate: BLAS/LAPACK for numeric operations
   - Foundation: Basic ObjC runtime
   - CoreML: Neural network inference on Neural Engine
   - Metal: GPU compute
   - MetalPerformanceShaders: Optimized ML ops on Metal

5. **Include Paths:**
   - Diarization headers: include/, src/
   - Model headers: models/segmentation-ggml/src, models/embedding-ggml/src
   - GGML headers: ggml/include/
   - kaldi-native-fbank headers: build-static/kaldi-native-fbank/include

6. **Output Target:**
   - Creates `pyannote-addon` shared library (.so / .dylib)
   - Renamed to .node suffix via set_target_properties
   - This is the actual Node.js native addon

### Node-Addon-API Integration

The CMakeLists.txt integrates with Node.js addon build pipeline via cmake-js:
- cmake-js calls CMake with CMAKE_JS_INC, CMAKE_JS_SRC, CMAKE_JS_LIB
- Node-addon-api provides C++ N-API bindings without exceptions
- Task 3 will create .cpp files that wrap diarization API using N-API macros

### Static Library Assumption

This CMakeLists.txt assumes build-static/ exists with all 12 libraries present.
**Verification needed in Task 6 (build & test).**

### File Structure

✅ **Created:** `bindings/node/packages/darwin-arm64/CMakeLists.txt` (96 lines)

Key sections:
- Lines 1-5: Project setup
- Lines 7-20: cmake-js and node-addon-api integration
- Lines 22-25: NAPI configuration
- Lines 27-30: Path resolution (DIARIZATION_ROOT, DIARIZATION_BUILD)
- Lines 32-38: Source collection and addon library creation
- Lines 40-48: Include directories
- Lines 50-65: Static library linking
- Lines 67-73: Apple frameworks
- Lines 75-80: Compile definitions (CoreML flags)
- Lines 83-87: .node suffix output configuration


## Task 3: N-API Binding Layer Implementation

### Architecture

**ObjectWrap pattern:**
- `PyannoteModel`: Holds `StreamingConfig` (5 paths). Does NOT call `streaming_init` — that's deferred to StreamingSession constructor or DiarizeWorker::Execute.
- `StreamingSession`: Holds `StreamingState*`. Created via `model.createStreamingSession()` which passes config as `Napi::External`.
- Config is passed from PyannoteModel→StreamingSession via `Napi::External<StreamingConfig>` pointer, avoiding JS-visible intermediate.

**AsyncWorker pattern:**
- All 4 workers (Push, Recluster, Finalize, Diarize) use `Napi::Promise::Deferred` for async returns.
- `Execute()` runs on libuv worker thread — NO Napi:: calls allowed.
- `OnOK()`/`OnError()` run on main thread — safe to create JS objects.
- Float32Array data MUST be copied to `std::vector<float>` in constructor (main thread) before `Execute` runs, because JS GC may collect the ArrayBuffer.

**Busy flag serialization:**
- `StreamingSession::busy_` set to `true` before creating worker (main thread), set to `false` in `OnOK`/`OnError` (main thread).
- Prevents concurrent operations on same StreamingState (not thread-safe).

**DiarizeWorker (offline):**
- Creates temporary `StreamingState` in `Execute()`, pushes audio in 16000-sample (1s) chunks, calls `streaming_finalize`, then `streaming_free`.
- Completely self-contained — no shared state with caller.

### Key Implementation Details

1. VADChunk vad field → Float32Array: Created via `Napi::ArrayBuffer::New` + `memcpy` + `Napi::Float32Array::New` wrapping the buffer.
2. DiarizationResult → JS: `{ segments: [{ start, duration, speaker }] }` — structured data, no RTTM strings.
3. StreamingSession destructor calls `Cleanup()` → `streaming_free()` to prevent leaks if JS doesn't call `close()`.
4. Module exports both `PyannoteModel` and `StreamingSession` classes, but users should only construct `PyannoteModel` directly.

### Files Created (13 total)
- `addon.cpp` — NODE_API_MODULE entry point
- `PyannoteModel.h/cpp` — ObjectWrap, config holder
- `StreamingSession.h/cpp` — ObjectWrap, state owner
- `PushWorker.h/cpp` — AsyncWorker for streaming_push
- `ReclusterWorker.h/cpp` — AsyncWorker for streaming_recluster
- `FinalizeWorker.h/cpp` — AsyncWorker for streaming_finalize
- `DiarizeWorker.h/cpp` — AsyncWorker for offline diarization

### LSP Note
All LSP errors in these files are `napi.h not found` — node-addon-api isn't installed (that's Task 6). The C++ is structurally correct per the N-API patterns.


## Task 6: End-to-End Build Verification

### Build Results — ALL CLEAN ON FIRST TRY

1. **pnpm install** — succeeded, 132 packages installed (8.5s)
   - darwin-x64 correctly skipped (unsupported platform on arm64)
   - cmake-js auto-ran as install script for darwin-arm64

2. **Native addon build (cmake-js)** — clean compile, no errors
   - All 7 .cpp files compiled without warnings
   - Linked against 12 static .a libraries + 5 Apple frameworks
   - Output: `build/Release/pyannote-addon.node` (894KB)

3. **TypeScript build (tsc)** — clean compile, no errors
   - Output: `dist/index.js` (133B) + declarations

4. **Addon load test** — exports verified:
   - `PyannoteModel: function` ✓
   - `StreamingSession: function` ✓

### No Fixes Required
The entire build chain (CMakeLists.txt, C++ sources, TypeScript sources) worked correctly on the first attempt. No source modifications needed.

### Generated artifacts (not committed):
- `bindings/node/node_modules/` — pnpm install output
- `bindings/node/packages/darwin-arm64/build/Release/pyannote-addon.node` — native addon
- `bindings/node/packages/pyannote-cpp-node/dist/` — TypeScript output
- `bindings/node/pnpm-lock.yaml` — lockfile

### Build environment:
- Node.js v22.11.0
- pnpm 10.18.3
- AppleClang 17.0.0.17000319
- cmake-js 7.4.0
- TypeScript 5.7.x
