
- 2026-02-13: Updated AGENTS.md Pitfall #5 and README streaming limitations to document that push->recluster->push->recluster is safe after commit 7ed6b23, since recluster preserves unfiltered state and uses local filtered variables.

## Task 1: CMake Integration + GGML Reconciliation

- **whisper.cpp's ggml is vendored** (not a submodule). The `ggml/` inside `whisper.cpp/` has remote `predict-woo/whisper.cpp.git`, not `ggml-org/ggml`. Cannot update pyannote's ggml submodule to match by commit hash since they're different repos.
- **GGML version reconciliation**: whisper.cpp's vendored ggml (v0.9.6, commit 3a472ad3 within whisper repo) is used as the primary ggml build. Pyannote's ggml submodule (a8db410a from ggml-org/ggml) is kept as fallback but skipped when whisper provides ggml first.
- **whisper.cpp BUILD_SHARED_LIBS**: Defaults to ON for non-MINGW. Must set `BUILD_SHARED_LIBS OFF` before `add_subdirectory(whisper.cpp)` to get static library.
- **whisper.cpp standalone detection**: Sets `WHISPER_STANDALONE OFF` when `CMAKE_SOURCE_DIR != CMAKE_CURRENT_SOURCE_DIR`, which auto-disables tests/examples. We additionally force `WHISPER_BUILD_TESTS/EXAMPLES/SERVER OFF` for safety.
- **WHISPER_COREML option**: Must be declared BEFORE `add_subdirectory(whisper.cpp)` so the option value propagates. Declared as `option()` at diarization project level.
- **Pipeline output unchanged**: 2 speakers, 13 segments — matches AGENTS.md expectations exactly.
- **.venv is broken**: `libpython3.10.dylib` not found. Python-based tests (segmentation accuracy, DER comparison) cannot run. Pre-existing issue.
- **transcription-lib as INTERFACE**: Since no source files exist yet, INTERFACE library is the cleanest approach. Converts to STATIC when sources are added in later tasks.

## Task 4: Segment End Detector

- Implemented `segment_detector` state with integer `current_prediction_frame` and `last_frame_was_speech` for exact pyannote frame-time mapping (`frame * 0.016875`).
- `segment_detector_push` emits at most one within-chunk speech->silence transition and separately handles cross-boundary speech->silence at chunk start.
- Added `test_segment_detector` covering all-speech, within-chunk transition, cross-boundary transition, all-silence, multi-chunk frame accumulation, and flush signal behavior.

## Task 3: Audio Buffer

- Added `AudioBuffer` with FIFO sample storage and integer absolute-frame offset tracking via `dequeued_frames_`.
- `dequeue_up_to(abs)` clamps removal to available buffered frames and preserves consistent `total_frames() == dequeued_frames() + size()`.
- `read_range(abs_start, abs_end)` converts absolute to local indices, clamps to buffered bounds, and returns exact contiguous samples for Whisper chunk extraction.
- Added standalone `test_audio_buffer` target and tests for enqueue/dequeue consistency, exact range reads, multi-cycle accounting, over-dequeue clamping, and empty-buffer reads.

## Task 2: VAD Silence Filter

- Implemented `silence_filter` with internal 512-sample batching, integer sample counters, 1s passthrough silence, 1s rolling `endSilence` FIFO, and 5s discarded-frame flush signaling.
- Added test-only deterministic VAD fallback in `silence_filter.cpp` when `vad_ctx == nullptr`: all-zero frames are treated as silence and non-zero frames as speech, so unit tests run without a VAD model file.
- Added `test_silence_filter` covering all-speech passthrough, 10s+8s+5s compression to about 17s output, long-silence flush trigger timing, and alternating 0.5s speech/silence passthrough.

## Task 6: Word-Speaker Aligner

- Added `src/aligner.h` and `src/aligner.cpp` implementing WhisperX-style token speaker assignment with per-speaker maximum overlap accumulation and nearest-midpoint fallback for gap tokens.
- Added `tests/test_aligner.cpp` with synthetic coverage for clear separation, straddling words, no-overlap gap assignment, overlapping diarization segments, and empty inputs.
- Wired `test_aligner` into `diarization-ggml/CMakeLists.txt`; target builds and `./build/bin/test_aligner` exits successfully.

## Task 5: Whisper Transcriber

- `whisper_context_params` has `use_gpu` (bool) and `use_coreml` (bool) fields — CoreML is controlled via `use_coreml`, not a separate path parameter.
- `whisper_token_data.t0` and `.t1` are `int64_t` (not int), representing centiseconds (10ms ticks). Multiply by 0.01 to convert to seconds.
- `whisper_full()` takes `whisper_full_params` by value, not pointer.
- Worker thread uses `std::condition_variable` pair: `cv_submit` (worker waits for work), `cv_result` (caller waits for result).
- 30s hard cap truncates at `30 * 16000 = 480000` samples. <1s audio skipped with empty valid result.
- Test with `ggml-base.en.bin` on `sample.wav` (30s) produces 109 tokens — well above the ≥10 threshold.
- `transcription-lib` INTERFACE library provides both diarization-lib and whisper link dependencies.
