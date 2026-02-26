# Model Cache — Learnings

## Ownership Pattern
- The "borrowed models" pattern uses raw `bool owns_*` flags (not smart pointers) to match existing codebase style
- Each subsystem has its own flag: `owns_models` (StreamingState), `owns_ctx` (Transcriber), `owns_vad` (PipelineState)
- `*_free()` functions check ownership before freeing model handles

## PLDA Ownership
- PLDAModel is a value type (struct with vectors), so it can be safely copied by value
- When borrowed by `streaming_init_with_models`, PLDA is COPIED into StreamingState (avoids lifetime issues)
- Same for `diarize_from_samples_with_models` — PLDA passed by const ref

## CoreML Guards
- `diarize_from_samples_with_models` is guarded by `#if defined(SEGMENTATION_USE_COREML) && defined(EMBEDDING_USE_COREML)` since it accepts CoreML context pointers directly
- `streaming_init_with_models` works regardless of ifdef guards since the pointers are stored but only used inside `#ifdef` blocks in process_one_chunk
- The `model_cache_load/free` functions use the same `#ifdef` pattern as the rest of the codebase

## CMakeLists.txt Pattern
- Sources must be explicitly listed in 3 separate targets: `pipeline-lib`, `transcribe`, `test_pipeline`
- The `diarization-lib` target does NOT need model_cache.cpp — it's only used by the pipeline/transcribe layer
- `transcription-lib` is an INTERFACE library linking diarization-lib + whisper

## Build
- Full build command: `cmake -B build -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON -DWHISPER_COREML=ON && cmake --build build`
- Build from: `/Users/andyye/dev/whisper-diarization.cpp/pyannote.cpp/diarization-ggml`

# Model Cache — Node.js Addon Integration Learnings

## Shared Model Cache Pattern (Node addon)

- `LoadModelsWorker` is a `Napi::AsyncWorker` that calls `model_cache_load()` on a background thread.
- The `ModelCacheConfig` is stored as a member of the worker. Its `std::string` fields are self-contained after move. The `TranscriberConfig` `const char*` pointers reference `PipelineModel`'s member strings, which stay alive because JS holds a reference to the PipelineModel object during the worker's execution.
- `vad_model_path` (a `const char*` in `ModelCacheConfig`) is additionally backed by a `std::string vad_path_` member in the worker for extra safety.
- `PipelineModel` owns the `ModelCache*` — only `Close()` and the destructor call `model_cache_free()`.
- Workers (`TranscribeWorker`, `OfflineTranscribeWorker`) and `PipelineSession` borrow the cache pointer via `model->GetCache()` and NEVER free it.
- Backward compatibility preserved: when `cache_ == nullptr`, all workers/sessions fall back to the original path-based initialization (`pipeline_init()`, `offline_transcribe()`).

## CMake Auto-Discovery

- `file(GLOB ADDON_SOURCES "src/*.cpp")` in darwin-arm64's CMakeLists.txt auto-discovers `LoadModelsWorker.cpp` — no cmake changes needed.
- The existing include paths already cover `diarization-ggml/src/` where `model_cache.h` lives.

## TypeScript Integration

- `Pipeline.load()` now awaits `native.loadModels()` after construction — models are loaded asynchronously on a worker thread before the Pipeline instance is returned to the caller.
- `NativePipelineModel` interface extended with `loadModels(): Promise<void>` and `isLoaded: boolean`.

## Build Verification

- `rebuild.sh` passes all 3 stages: C++ static libs (build + build-static), cmake-js addon, TypeScript compilation.
- No additional npm dependencies required.