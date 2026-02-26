# Whisper CoreML Switch — Learnings

## Pattern: Runtime model reload via dedicated model_cache function
- `model_cache_reload_whisper()` follows the same param-building pattern as step 4 of `model_cache_load()` but only touches `whisper_ctx`
- Frees existing context first, then loads new one — no second context needed
- Returns bool for success/failure; on failure sets whisper_ctx to nullptr

## Pattern: AsyncWorker for N-API Promise
- SwitchWhisperModeWorker follows the exact same pattern as LoadModelsWorker
- Store backing string (`whisper_path_`) locally to keep `const char*` valid during Execute()
- Set `busy_=true` before Queue(), reset in both OnOK and OnError
- If mode already matches, skip worker entirely — return immediately-resolving promise

## CMake auto-discovery
- darwin-arm64 uses `file(GLOB ADDON_SOURCES "src/*.cpp")` — new SwitchWhisperModeWorker.cpp auto-discovered, no cmake changes needed

## Test integration
- Added test block AFTER the duplicate Shared model cache blocks (line 348)
- The `setUseCoreml(false)` no-op case (already false) tests the early-return path
- The `transcription works after mode switch` test verifies the full reload→transcribe cycle