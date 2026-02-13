# Whisper + Pyannote Streaming Diarization Integration

## TL;DR

> **Quick Summary**: Build a streaming diarization+transcription C++ pipeline that combines pyannote.cpp speaker diarization with whisper.cpp transcription. Audio is pushed incrementally; VAD filters silence, pyannote detects segment boundaries, Whisper transcribes on a worker thread, and a WhisperX-style aligner assigns speaker labels to each word.
> 
> **Deliverables**:
> - Library API: `transcriber_init/push/finalize/free` with callback for speaker-labeled word-level results
> - CLI executable: WAV input → JSON output with speakers + word timestamps
> - Per-stage unit tests for all 7 pipeline stages
> - Updated AGENTS.md and README.md
> 
> **Estimated Effort**: Large
> **Parallel Execution**: YES — 3 waves
> **Critical Path**: Task 1 (CMake) → Task 2 (Silence Filter) → Task 5 (Transcriber) → Task 7 (Pipeline) → Task 8 (CLI)

---

## Context

### Original Request
Combine the existing pyannote C++ diarization pipeline (streaming mode) with whisper.cpp (as a dependency, not modified) to create an integrated diarization+transcription library and CLI. The user has a detailed 7-stage pipeline design in INTEGRATION_PLAN.md.

### Interview Summary
**Key Discussions**:
- Code lives in `diarization-ggml/src/` as new files
- Whisper.cpp linked via `add_subdirectory`, sharing a single GGML build
- Whisper's built-in Silero VAD for silence filtering (512-sample frames)
- Whisper runs on a separate worker thread (mutex+CV for results)
- Recluster on every Whisper return (fix verified: state is kept unfiltered)
- JSON output format with speaker labels + word-level timestamps
- CoreML for all neural models (segmentation, embedding, Whisper encoder)
- large-v3-turbo Whisper model as primary target
- Opaque handle API pattern matching existing `streaming_init/push/finalize/free`

**Research Findings**:
- WhisperX alignment: per-word maximum intersection against diarization segments, nearest-segment fallback for gaps
- Whisper token timestamps: `whisper_full_get_token_data().t0/.t1` in 10ms ticks, enable with `token_timestamps=true`
- GGML version mismatch between repos (pyannote: `a8db410a`, whisper.cpp: `3a472ad3`) — must reconcile
- Whisper has hard 30s context window — buffer must not exceed this
- Interval tree: sorted array + binary search sufficient; ekg/intervaltree as fallback option

### Metis Review
**Identified Gaps** (addressed):
- GGML version conflict → Task 1 reconciles upfront, validates both test suites
- Whisper 30s buffer limit → Task 5 adds hard 30s cap to transcriber
- Timeline domain confusion risk → All stages document "filtered timeline" invariant
- Thread safety of StreamingState → Main thread owns all pyannote access, Whisper worker isolated
- VAD model path needed → CLI accepts `--vad-model` flag
- Missing language flag → CLI accepts `--language` (default: "en")

---

## Work Objectives

### Core Objective
Build a streaming C++ pipeline that produces speaker-labeled, word-level transcriptions by combining pyannote diarization with Whisper transcription, exposed as both a library API and CLI tool.

### Concrete Deliverables
- `diarization-ggml/src/silence_filter.{h,cpp}` — VAD-based silence compression
- `diarization-ggml/src/audio_buffer.{h,cpp}` — FIFO with absolute frame tracking
- `diarization-ggml/src/segment_detector.{h,cpp}` — speech→silence transition detector
- `diarization-ggml/src/transcriber.{h,cpp}` — Whisper wrapper with worker thread
- `diarization-ggml/src/aligner.{h,cpp}` — WhisperX word→speaker assignment
- `diarization-ggml/src/pipeline.{h,cpp}` — orchestrator connecting all stages
- `diarization-ggml/src/main_transcribe.cpp` — CLI executable
- `diarization-ggml/include/transcriber_types.h` — public API types and callback definition
- Updated `diarization-ggml/CMakeLists.txt` — whisper.cpp integration + new targets
- Per-stage test executables in `diarization-ggml/tests/`
- Updated `AGENTS.md` and `README.md`

### Definition of Done
- [x] `cmake --build build` compiles all targets without warnings
- [x] Each stage test executable passes independently
- [x] CLI produces valid JSON for `samples/sample.wav` with ≥2 speakers and word timestamps
- [x] Existing `diarization-ggml` tests still pass (DER ≤ 1.0%) — DER=0.14%, PASS
- [x] Existing segmentation accuracy test still passes — cosine=0.9999, PASS

### Must Have
- All 7 pipeline stages from INTEGRATION_PLAN.md implemented
- Word-level timestamps with speaker labels
- Filtered-timeline consistency (all timestamps in same domain)
- Thread-safe Whisper worker with mutex+CV
- Hard 30s buffer cap for Whisper
- 5s silence flush to prevent indefinite audio holding
- CoreML acceleration for all neural models

### Must NOT Have (Guardrails)
- No modifications to whisper.cpp source code
- No modifications to existing `streaming.cpp`, `streaming.h`, `streaming_state.h`
- No modifications to existing `main.cpp` (diarization-only CLI)
- No microphone/real-time audio capture (WAV file only)
- No network/API server
- No speaker identification (only diarization)
- No audio preprocessing (noise reduction, normalization)
- No DTW token timestamps (standard `token_timestamps` only)
- No GGML-only fallback (CoreML required, matching existing streaming constraint)
- No new dependencies beyond whisper.cpp
- No Whisper internal VAD flag (`params.vad`) — use explicit `whisper_vad_detect_speech_single_frame` only
- No global/static state in any pipeline component

---

## Verification Strategy

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL verification is executed by the agent using tools (Bash commands, file inspection).
> No acceptance criteria require human action.

### Test Decision
- **Infrastructure exists**: YES (existing test executables + Python test scripts)
- **Automated tests**: Stage-by-stage unit tests (tests-after approach)
- **Framework**: Standalone C++ test executables with assertions + existing Python comparison scripts

### Agent-Executed QA Scenarios (MANDATORY — ALL tasks)

Each task includes specific test scenarios. The executing agent runs the built test executable and asserts on stdout/exit code.

**Verification Tool by Deliverable Type:**

| Type | Tool | How Agent Verifies |
|------|------|-------------------|
| C++ library code | Bash (build + run test binary) | Compile, run test, assert exit code 0 |
| CMake integration | Bash (cmake configure + build) | Configure succeeds, build succeeds |
| CLI executable | Bash (run with sample audio) | Parse JSON output, verify structure |
| Documentation | Read tool | Verify updated sections exist |

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1:  CMake + GGML reconciliation
└── Task 10: Commit recluster fix + update AGENTS.md

Wave 2 (After Wave 1):
├── Task 2:  Silence Filter
├── Task 3:  Audio Buffer
├── Task 4:  Segment Detector
└── Task 6:  Aligner

Wave 3 (After Wave 2):
├── Task 5:  Transcriber (depends: 1 for whisper build)
├── Task 7:  Pipeline Orchestrator (depends: 2,3,4,5,6)
├── Task 8:  CLI Executable (depends: 7)
├── Task 9:  Integration Tests (depends: 8)
└── Task 11: README.md update (depends: 8)

Critical Path: Task 1 → Task 5 → Task 7 → Task 8 → Task 9
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2,3,4,5,6,7 | 10 |
| 2 | 1 | 7 | 3,4,6 |
| 3 | 1 | 7 | 2,4,6 |
| 4 | 1 | 7 | 2,3,6 |
| 5 | 1 | 7 | 2,3,4,6 |
| 6 | 1 | 7 | 2,3,4,5 |
| 7 | 2,3,4,5,6 | 8 | None |
| 8 | 7 | 9,11 | None |
| 9 | 8 | None | 11 |
| 10 | None | None | 1 |
| 11 | 8 | None | 9 |

---

## TODOs

- [x] 1. CMake Integration + GGML Reconciliation

  **What to do**:
  - Update pyannote's GGML submodule to match whisper.cpp's version (`3a472ad3`)
  - Modify `diarization-ggml/CMakeLists.txt` to `add_subdirectory(../../whisper.cpp ...)` BEFORE the existing ggml guard, so whisper's GGML wins the `if(NOT TARGET ggml)` check
  - Add `WHISPER_COREML=ON` option passthrough
  - Add whisper VAD model download instruction
  - Add new library target `transcription-lib` linking `diarization-lib` + `whisper`
  - Add test targets for each stage
  - Add `transcribe` CLI executable target
  - Verify both whisper.cpp and diarization-ggml compile and link together
  - Run existing segmentation accuracy test to verify GGML version change is safe
  - Run existing full pipeline DER test to verify regression-free

  **Must NOT do**:
  - Do NOT modify whisper.cpp's CMakeLists.txt
  - Do NOT rename any existing targets
  - Do NOT change the existing `diarization-ggml` or `streaming_test` targets

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: GGML version reconciliation requires careful investigation of potential ABI/API differences between versions and impact on numerical accuracy
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 10)
  - **Blocks**: Tasks 2, 3, 4, 5, 6, 7
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `diarization-ggml/CMakeLists.txt:11-16` — Existing GGML guard pattern: `if(NOT TARGET ggml) add_subdirectory(../ggml ...)`. Whisper must be added BEFORE this so its ggml target exists first.
  - `diarization-ggml/CMakeLists.txt:35-46` — Pattern for adding static library with source files. New `transcription-lib` follows this.
  - `diarization-ggml/CMakeLists.txt:76-87` — Pattern for adding executable + linking to library. New `transcribe` CLI follows this.
  - `whisper.cpp/CMakeLists.txt:91-93` — Whisper CoreML option: `option(WHISPER_COREML "whisper: enable Core ML framework" OFF)`. Must be set ON before `add_subdirectory`.
  - `whisper.cpp/src/CMakeLists.txt:106-110` — The `whisper` target definition with public includes at `.` and `../include`. This is what we link against.

  **API/Type References**:
  - `whisper.cpp/include/whisper.h:1-762` — Full whisper C API. Our code includes this header.

  **Test References**:
  - `models/segmentation-ggml/tests/test_accuracy.py` — Must still pass after GGML version change
  - `diarization-ggml/tests/compare_rttm.py` — DER must remain ≤ 1.0% after integration

  **Documentation References**:
  - `AGENTS.md:Build Commands` — Existing build commands to verify compatibility
  - `INTEGRATION_PLAN.md` — Pipeline design reference

  **Acceptance Criteria**:

  - [x] `cd diarization-ggml && cmake -B build -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON -DWHISPER_COREML=ON` succeeds
  - [x] `cmake --build build` compiles all targets without errors
  - [x] `./build/bin/diarization-ggml` (existing CLI) still works: runs on sample.wav, produces RTTM
  - [x] `python3 models/segmentation-ggml/tests/test_accuracy.py` passes (cosine > 0.995) — cosine=0.9999, PASS
  - [x] `python3 diarization-ggml/tests/compare_rttm.py /tmp/test.rttm /tmp/py_reference.rttm --threshold 1.0` passes (DER ≤ 1.0%) — DER=0.14%, PASS

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Full CMake configure succeeds
    Tool: Bash
    Preconditions: GGML submodule updated, whisper.cpp submodule present
    Steps:
      1. cd diarization-ggml
      2. cmake -B build -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON -DWHISPER_COREML=ON
      3. Assert: exit code 0
      4. Assert: stdout contains "whisper.cpp" or "whisper" target
    Expected Result: CMake configures without errors
    Evidence: Terminal output captured

  Scenario: Build produces all targets
    Tool: Bash
    Preconditions: CMake configured
    Steps:
      1. cmake --build build 2>&1
      2. Assert: exit code 0
      3. Assert: build/bin/diarization-ggml exists
      4. Assert: build/bin/streaming_test exists
    Expected Result: All existing + new targets compile
    Evidence: Terminal output captured

  Scenario: Existing DER test passes after GGML update
    Tool: Bash
    Preconditions: Build complete, sample.wav and reference RTTM available
    Steps:
      1. Run diarization-ggml on sample.wav with CoreML models
      2. Run compare_rttm.py on output vs reference
      3. Assert: DER ≤ 1.0%
    Expected Result: No regression from GGML version change
    Evidence: compare_rttm.py output captured
  ```

  **Commit**: YES
  - Message: `build: integrate whisper.cpp as subdirectory, reconcile shared GGML`
  - Files: `diarization-ggml/CMakeLists.txt`, `.gitmodules` (if ggml submodule updated)
  - Pre-commit: existing tests pass

---

- [x] 2. VAD Silence Filter

  **What to do**:
  - Create `diarization-ggml/src/silence_filter.h` — public interface:
    - `SilenceFilter* silence_filter_init(whisper_vad_context* vad_ctx)` 
    - `SilenceFilterResult silence_filter_push(SilenceFilter* sf, const float* samples, int n)` — returns filtered audio + flush signal
    - `SilenceFilterResult silence_filter_flush(SilenceFilter* sf)` — flush endSilence buffer
    - `void silence_filter_free(SilenceFilter* sf)`
  - Create `diarization-ggml/src/silence_filter.cpp` implementing INTEGRATION_PLAN.md §1:
    - Internal 512-frame processing (matching whisper VAD window size from `whisper_vad_n_window`)
    - `silence_started` timestamp tracking
    - `endSilence` circular buffer (1 second = 16000 samples)
    - Speech onset: prepend endSilence contents
    - 1s trailing silence passthrough, then buffer in endSilence
    - `consecutive_discarded_frames` counter → flush signal at 5s (80,000 frames)
    - Call `whisper_vad_detect_speech_single_frame()` per 512-frame batch
    - Use configurable threshold (default 0.5) for speech/silence classification
  - Create `diarization-ggml/tests/test_silence_filter.cpp` — unit test:
    - Test 1: All-speech input passes through unchanged
    - Test 2: 10s speech + 8s silence + 5s speech → output ≈ 17s (silence compressed to ~2s)
    - Test 3: 10s silence → flush triggered after 5s of discarded frames
    - Test 4: Alternating 0.5s speech / 0.5s silence → all passes through (silence ≤ 1s)

  **Must NOT do**:
  - Must NOT modify sample values (only pass or discard frames)
  - Must NOT handle non-16kHz audio
  - Must NOT implement its own VAD model (use whisper's)
  - Must NOT track time in floating point (use integer sample counts)

  **Recommended Agent Profile**:
  - **Category**: `medium`
    - Reason: Well-specified logic from INTEGRATION_PLAN.md, moderate complexity with circular buffer + state machine
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3, 4, 6)
  - **Blocks**: Task 7
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `diarization-ggml/src/streaming.h:18-36` — Opaque handle API pattern (init/push/free) to follow
  - `diarization-ggml/src/streaming.cpp:246-331` — `streaming_push` pattern for processing buffered audio in a while loop

  **API/Type References**:
  - `whisper.cpp/include/whisper.h:698-717` — Silero VAD streaming API: `whisper_vad_detect_speech_single_frame()` returns float [0.0, 1.0], `whisper_vad_n_window()` returns 512, `whisper_vad_reset_state()`
  - `whisper.cpp/include/whisper.h:688-694` — `whisper_vad_context_params` and `whisper_vad_default_context_params()`

  **Documentation References**:
  - `INTEGRATION_PLAN.md:33-66` — Complete silence filter specification including buffer sizes, timing thresholds, and flush logic

  **Acceptance Criteria**:

  - [x] `test_silence_filter` binary builds and exits 0
  - [x] All-speech test: output sample count equals input sample count
  - [x] Silence compression test: 8s silence compressed to ~2s (output ≈ 17s × 16000 = 272000 ± 5%)
  - [x] Flush test: flush signal emitted after 5s of discarded silence

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Silence compression works correctly
    Tool: Bash
    Preconditions: test_silence_filter built, VAD model available
    Steps:
      1. ./build/bin/test_silence_filter --vad-model path/to/ggml-silero-v6.2.0.bin
      2. Assert: exit code 0
      3. Assert: stdout contains "PASS" for all test cases
      4. Assert: silence compression test shows output within 5% of expected
    Expected Result: All silence filter tests pass
    Evidence: Terminal output captured

  Scenario: Flush triggers on prolonged silence
    Tool: Bash
    Steps:
      1. Run test with 10s of silence input
      2. Assert: flush signal reported at ~6s mark (1s passthrough + 5s discarded)
    Expected Result: Flush fires at correct threshold
    Evidence: Terminal output captured
  ```

  **Commit**: YES
  - Message: `feat(integration): add VAD silence filter stage`
  - Files: `diarization-ggml/src/silence_filter.{h,cpp}`, `diarization-ggml/tests/test_silence_filter.cpp`
  - Pre-commit: `test_silence_filter` passes

---

- [x] 3. Audio Buffer

  **What to do**:
  - Create `diarization-ggml/src/audio_buffer.h` — public interface:
    - `AudioBuffer` class or opaque handle with:
    - `enqueue(const float* samples, int n)` — append frames
    - `dequeue_up_to(int64_t absolute_sample)` — remove frames from front, update offset
    - `read_range(int64_t abs_start, int64_t abs_end, std::vector<float>& out)` — extract audio between absolute positions
    - `int64_t dequeued_frames()` — total frames ever dequeued (offset for timestamp calc)
    - `int64_t total_frames()` — dequeued + currently buffered
    - `const float* data()` + `int size()` — direct buffer access
  - Create `diarization-ggml/src/audio_buffer.cpp` implementing INTEGRATION_PLAN.md §2
  - Create `diarization-ggml/tests/test_audio_buffer.cpp`:
    - Test 1: Enqueue 48000 samples, verify total_frames == 48000
    - Test 2: Dequeue 16000, verify dequeued_frames == 16000, remaining == 32000
    - Test 3: Read range [16000, 32000] returns correct 16000-sample slice
    - Test 4: Timestamp calculation: frame at position i → time = (dequeued + i) / 16000.0

  **Must NOT do**:
  - Must NOT copy data on range reads (provide view/pointer where possible)
  - Must NOT track timestamps in floating point (integer sample counts only, convert at output)
  - Must NOT allocate per-read (pre-allocate or use vector capacity)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple data structure, well-defined interface, no external dependencies
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 2, 4, 6)
  - **Blocks**: Task 7
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `diarization-ggml/src/streaming_state.h:23` — `audio_buffer` vector + `samples_trimmed` offset pattern. Same concept but cleaner encapsulated API.

  **Documentation References**:
  - `INTEGRATION_PLAN.md:69-94` — Audio buffer spec with operations and timestamp calculation

  **Acceptance Criteria**:
  - [x] `test_audio_buffer` builds and exits 0
  - [x] Enqueue/dequeue cycle: dequeued_frames tracks correctly
  - [x] Read range returns exact expected samples
  - [x] No memory leaks (valgrind clean or ASAN clean)

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Buffer operations are correct
    Tool: Bash
    Steps:
      1. ./build/bin/test_audio_buffer
      2. Assert: exit code 0
      3. Assert: stdout contains "PASS" for all tests
    Expected Result: All audio buffer tests pass
    Evidence: Terminal output
  ```

  **Commit**: YES
  - Message: `feat(integration): add audio buffer with absolute frame tracking`
  - Files: `diarization-ggml/src/audio_buffer.{h,cpp}`, `diarization-ggml/tests/test_audio_buffer.cpp`

---

- [x] 4. Segment End Detector

  **What to do**:
  - Create `diarization-ggml/src/segment_detector.h` — public interface:
    - `SegmentDetector* segment_detector_init()`
    - `std::vector<double> segment_detector_push(SegmentDetector* sd, const VADChunk& chunk)` — returns timestamps of detected segment ends
    - `void segment_detector_free(SegmentDetector* sd)`
  - Create `diarization-ggml/src/segment_detector.cpp` implementing INTEGRATION_PLAN.md §3:
    - Maintain `current_prediction_frame` counter (cumulative)
    - Maintain `last_frame_was_speech` bool
    - Cross-boundary detection: last_frame_was_speech && vad[0] == 0.0
    - Within-chunk detection: first speech→non-speech transition
    - Only detect first transition per chunk (per plan)
    - Timestamp = frame_number × 0.016875
    - Expose a `flush()` method for the 5s silence flush from VAD filter
  - Create `diarization-ggml/tests/test_segment_detector.cpp`:
    - Test 1: VADChunk with all-speech → no segment end
    - Test 2: VADChunk with speech at frames 0-29, silence at 30+ → segment end at (current_frame + 30) × 0.016875
    - Test 3: Cross-boundary: previous chunk ended with speech, new chunk starts with silence → segment end at start of new chunk
    - Test 4: All-silence chunk → no segment end (already silent)

  **Must NOT do**:
  - Must NOT scan more than the first speech→silence transition per chunk
  - Must NOT introduce its own VAD logic (only reads VADChunk.vad array)
  - Must NOT modify VADChunk data

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple state machine with well-specified logic
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 2, 3, 6)
  - **Blocks**: Task 7
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `diarization-ggml/src/streaming.h:11-16` — VADChunk struct definition (chunk_index, start_frame, num_frames, vad vector)

  **API/Type References**:
  - `diarization-ggml/src/streaming.h:11-16` — VADChunk used as input to segment detector

  **Documentation References**:
  - `INTEGRATION_PLAN.md:96-141` — Complete segment end detection spec with frame counting, cross-boundary, and flush logic

  **Acceptance Criteria**:
  - [x] `test_segment_detector` builds and exits 0
  - [x] Within-chunk transition detected at correct timestamp
  - [x] Cross-boundary transition detected
  - [x] Frame counting accumulates correctly across multiple pushes

  **Commit**: YES
  - Message: `feat(integration): add segment end detector using pyannote VAD frames`
  - Files: `diarization-ggml/src/segment_detector.{h,cpp}`, `diarization-ggml/tests/test_segment_detector.cpp`

---

- [x] 5. Whisper Transcriber

  **What to do**:
  - Create `diarization-ggml/include/transcriber_types.h` — shared types:
    - `struct TranscribeToken { std::string text; double start; double end; }` — word with absolute filtered-timeline timestamps
    - `struct TranscribeResult { std::vector<TranscribeToken> tokens; }` — output of one Whisper call
  - Create `diarization-ggml/src/transcriber.h` — public interface:
    - `struct TranscriberConfig { const char* whisper_model_path; const char* whisper_coreml_path; int n_threads; const char* language; }` 
    - `Transcriber* transcriber_init(const TranscriberConfig& config)`
    - `void transcriber_submit(Transcriber* t, const float* audio, int n_samples, double buffer_start_time)` — submit audio for transcription on worker thread
    - `bool transcriber_try_get_result(Transcriber* t, TranscribeResult& result)` — non-blocking check for results (mutex+CV)
    - `TranscribeResult transcriber_wait_result(Transcriber* t)` — blocking wait
    - `void transcriber_free(Transcriber* t)`
  - Create `diarization-ggml/src/transcriber.cpp`:
    - Initialize `whisper_context` from model path (with CoreML if path provided)
    - Worker thread: wait for audio submission, run `whisper_full()` with `token_timestamps=true`, `max_len=1`, `split_on_word=true`
    - Extract words: iterate segments → tokens → `whisper_full_get_token_data()`, convert t0/t1 from 10ms ticks to seconds, add `buffer_start_time` offset
    - Filter out non-text tokens (timestamps, special tokens) — check if `whisper_token_to_str` returns printable text
    - Signal completion via mutex+CV
    - Handle 30s max: if audio exceeds 30s × 16000 samples, truncate to 30s with warning
  - Create `diarization-ggml/tests/test_transcriber.cpp`:
    - Test: transcribe `samples/jfk.wav` (11s, exists in repo), verify word output

  **Must NOT do**:
  - Must NOT call `whisper_full` on <1s of audio (guard against tiny flush chunks)
  - Must NOT exceed 30s per Whisper call (hard truncate with warning)
  - Must NOT use `params.vad = true` (external VAD only)
  - Must NOT use DTW timestamps
  - Must NOT carry prompt tokens between calls (simplicity first)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Threading, whisper API integration, timestamp conversion all require careful correctness
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 2, 3, 4, 6) — can start as soon as Task 1 completes
  - **Blocks**: Task 7
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `diarization-ggml/src/streaming.cpp:169-244` — `streaming_init` pattern for model loading and error cleanup
  - `whisper.cpp/examples/cli/cli.cpp:1185-1188` — Enabling token timestamps in whisper params
  - `whisper.cpp/examples/cli/cli.cpp:742-748` — Reading token t0/t1 from results

  **API/Type References**:
  - `whisper.cpp/include/whisper.h:137-157` — `whisper_token_data` struct with t0, t1, p fields
  - `whisper.cpp/include/whisper.h:493-597` — `whisper_full_params` struct (token_timestamps, max_len, split_on_word, language, etc.)
  - `whisper.cpp/include/whisper.h:609-620` — `whisper_full()` and `whisper_full_with_state()` signatures
  - `whisper.cpp/include/whisper.h:661-674` — Token extraction: `whisper_full_n_tokens`, `whisper_full_get_token_data`
  - `whisper.cpp/include/whisper.h:117-135` — `whisper_context_params` with `use_coreml`

  **External References**:
  - whisper.cpp README: CoreML setup requires `.mlmodelc` (compiled model, NOT `.mlpackage`)

  **Acceptance Criteria**:
  - [x] `test_transcriber` builds and exits 0
  - [x] Transcribing jfk.wav returns ≥10 word tokens
  - [x] All tokens have t0 < t1 (no inverted timestamps)
  - [x] All tokens have non-empty text
  - [x] Concatenated text resembles "And so my fellow Americans..."
  - [x] Worker thread completes and resources are freed cleanly

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Word-level transcription of known audio
    Tool: Bash
    Preconditions: Whisper model downloaded, test built
    Steps:
      1. ./build/bin/test_transcriber --model path/to/ggml-large-v3-turbo.bin ../samples/jfk.wav
      2. Assert: exit code 0
      3. Assert: output contains at least 10 words
      4. Assert: all word timestamps are valid (start < end)
      5. Assert: concatenated output contains "Americans"
    Expected Result: Word-level transcription with valid timestamps
    Evidence: Terminal output captured
  ```

  **Commit**: YES
  - Message: `feat(integration): add Whisper transcriber with worker thread and word timestamps`
  - Files: `diarization-ggml/include/transcriber_types.h`, `diarization-ggml/src/transcriber.{h,cpp}`, `diarization-ggml/tests/test_transcriber.cpp`

---

- [x] 6. Word-Speaker Aligner

  **What to do**:
  - Create `diarization-ggml/src/aligner.h` — public interface:
    - `struct AlignedWord { std::string text; double start; double end; std::string speaker; }` 
    - `struct AlignedSegment { std::string speaker; double start; double duration; std::vector<AlignedWord> words; }`
    - `std::vector<AlignedSegment> align_words(const std::vector<TranscribeToken>& tokens, const DiarizationResult& diarization)`
  - Create `diarization-ggml/src/aligner.cpp` implementing INTEGRATION_PLAN.md §5:
    - Build sorted interval index from diarization segments (sorted by start time)
    - For each word token:
      - Find all overlapping diarization segments (binary search on sorted array)
      - Sum intersection duration per speaker
      - Assign word to speaker with maximum total intersection
      - If no overlap: assign to nearest segment by midpoint distance
    - Group consecutive words with same speaker into AlignedSegments
    - Handle edge cases: empty tokens, empty diarization, words with zero duration
  - Create `diarization-ggml/tests/test_aligner.cpp`:
    - Test 1: 2 speakers, 4 words clearly in different segments → correct assignment
    - Test 2: Word straddling two speaker segments → assigned to speaker with more overlap
    - Test 3: Word in gap between segments → assigned to nearest segment
    - Test 4: Empty inputs → empty output (no crash)

  **Must NOT do**:
  - Must NOT modify diarization segment boundaries
  - Must NOT re-run Whisper or pyannote
  - Must NOT use anything fancier than sorted-array overlap search (no external interval tree library needed)
  - Must NOT assume segments are non-overlapping (pyannote CAN produce overlapping segments)

  **Recommended Agent Profile**:
  - **Category**: `medium`
    - Reason: Algorithm implementation with edge cases, but well-specified by WhisperX reference
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 2, 3, 4, 5)
  - **Blocks**: Task 7
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - WhisperX `diarize.py:167-245` — Maximum intersection algorithm (see librarian research)
  - WhisperX `diarize.py:41-70` — Overlap query with sorted intervals
  - WhisperX `diarize.py:72-88` — Nearest fallback for gap words

  **API/Type References**:
  - `diarization-ggml/include/diarization.h:16-23` — `DiarizationResult::Segment` with start, duration, speaker
  - `diarization-ggml/include/transcriber_types.h` — `TranscribeToken` with text, start, end (created in Task 5)

  **Documentation References**:
  - `INTEGRATION_PLAN.md:195-248` — Alignment algorithm specification

  **Acceptance Criteria**:
  - [x] `test_aligner` builds and exits 0
  - [x] All test words assigned to correct speakers
  - [x] Gap words assigned to nearest segment (not left unassigned)
  - [x] Overlapping diarization segments handled correctly (max intersection wins)
  - [x] Empty inputs produce empty output without crash

  **Commit**: YES
  - Message: `feat(integration): add WhisperX-style word-speaker alignment`
  - Files: `diarization-ggml/src/aligner.{h,cpp}`, `diarization-ggml/tests/test_aligner.cpp`

---

- [x] 7. Pipeline Orchestrator

  **What to do**:
  - Create `diarization-ggml/src/pipeline.h` — public API:
    - `struct PipelineConfig { StreamingConfig diarization; TranscriberConfig transcriber; const char* vad_model_path; }` 
    - `typedef void (*pipeline_callback)(const std::vector<AlignedSegment>& segments, void* user_data)`
    - `PipelineState* pipeline_init(const PipelineConfig& config, pipeline_callback cb, void* user_data)`
    - `void pipeline_push(PipelineState* state, const float* samples, int n_samples)` — push raw audio
    - `void pipeline_finalize(PipelineState* state)` — flush all stages
    - `void pipeline_free(PipelineState* state)`
  - Create `diarization-ggml/src/pipeline.cpp` wiring all stages per INTEGRATION_PLAN.md:
    - `pipeline_init`: create SilenceFilter + AudioBuffer + StreamingState (zero_latency) + SegmentDetector + Transcriber + Aligner
    - `pipeline_push`:
      1. Push audio into SilenceFilter
      2. Filtered audio → enqueue into AudioBuffer AND push into pyannote streaming
      3. Process returned VADChunks through SegmentDetector
      4. On segment end: check if buffer_start → segment_end ≥ 20s
         - If yes: extract audio from AudioBuffer, submit to Transcriber
         - Also apply 30s hard cap
      5. On silence flush signal: force-send all buffered audio to Transcriber
      6. Check for Whisper results (non-blocking)
      7. On result: recluster pyannote, run alignment, invoke callback
    - `pipeline_finalize` per INTEGRATION_PLAN.md §6:
      1. Flush SilenceFilter endSilence → push to AudioBuffer + pyannote
      2. Call `streaming_finalize()` on pyannote
      3. Send remaining audio buffer to Whisper
      4. Wait for Whisper result
      5. Final alignment against finalize's diarization result
      6. Invoke callback with complete state
    - Document the "filtered timeline" invariant in a comment at the top
    - Thread model: main thread owns SilenceFilter + AudioBuffer + pyannote StreamingState + SegmentDetector + Aligner. Only Transcriber's `whisper_full()` runs on worker thread.
  - Create `diarization-ggml/tests/test_pipeline.cpp`:
    - Integration test: push `samples/sample.wav` through pipeline, verify callback fires with speakers + words

  **Must NOT do**:
  - Must NOT expose internal stage state through public API
  - Must NOT have global or static state
  - Must NOT call pyannote functions from the Whisper worker thread
  - Must NOT modify existing streaming.cpp/h

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Complex orchestration connecting 6 sub-systems with threading, state management, and finalization ordering
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Wave 3)
  - **Blocks**: Task 8
  - **Blocked By**: Tasks 2, 3, 4, 5, 6

  **References**:

  **Pattern References**:
  - `diarization-ggml/src/streaming.cpp:169-244` — Init pattern with cascade error cleanup
  - `diarization-ggml/src/streaming.cpp:246-331` — Push loop pattern (buffer audio, process while enough available)
  - `diarization-ggml/src/streaming.cpp:639-661` — Finalize pattern (process remaining, recluster, set flag)

  **API/Type References**:
  - All types from Tasks 2-6 headers
  - `diarization-ggml/src/streaming.h:19-36` — streaming API used internally
  - `diarization-ggml/src/streaming_state.h:9-16` — StreamingConfig for pyannote init

  **Documentation References**:
  - `INTEGRATION_PLAN.md:1-297` — The complete pipeline specification (primary reference for this task)

  **Acceptance Criteria**:
  - [x] `test_pipeline` builds and exits 0
  - [x] Callback fires at least once during pipeline processing
  - [x] Final callback output contains ≥2 distinct speakers for sample.wav
  - [x] All words in output have non-empty text and valid timestamps
  - [x] All words have speaker labels
  - [x] No memory leaks or thread issues (clean shutdown)

  **Commit**: YES
  - Message: `feat(integration): add pipeline orchestrator connecting all stages`
  - Files: `diarization-ggml/src/pipeline.{h,cpp}`, `diarization-ggml/tests/test_pipeline.cpp`

---

- [x] 8. CLI Executable

  **What to do**:
  - Create `diarization-ggml/src/main_transcribe.cpp`:
    - Parse CLI args: `--seg-model`, `--emb-model`, `--whisper-model`, `--plda`, `--seg-coreml`, `--emb-coreml`, `--vad-model`, `--language` (default "en"), `-o` output path, input WAV path
    - Load WAV file (reuse existing `load_wav_file` pattern from diarization.cpp)
    - Init pipeline with callback that accumulates results
    - Push audio in 1-second chunks (16000 samples)
    - Call `pipeline_finalize`
    - Output JSON to stdout or file:
    ```json
    {
      "segments": [
        {
          "speaker": "SPEAKER_00",
          "start": 0.5,
          "duration": 2.1,
          "words": [
            {"text": "Hello", "start": 0.5, "end": 0.8},
            {"text": "world", "start": 0.9, "end": 1.2}
          ]
        }
      ]
    }
    ```
    - Print timing summary to stderr
  - Add `transcribe` target to CMakeLists.txt

  **Must NOT do**:
  - Must NOT modify existing `main.cpp`
  - Must NOT accept non-WAV input
  - Must NOT require all model paths (sensible defaults or error messages)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Thin wrapper over pipeline API, mostly CLI parsing + JSON formatting
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Wave 3, after Task 7)
  - **Blocks**: Task 9
  - **Blocked By**: Task 7

  **References**:

  **Pattern References**:
  - `diarization-ggml/src/main.cpp` — Existing CLI arg parsing pattern (not read in this session but follows standard getopt pattern)
  - `diarization-ggml/src/diarization.cpp:82-141` — WAV loading pattern to reuse

  **Acceptance Criteria**:
  - [x] `build/bin/transcribe --help` shows usage
  - [x] `build/bin/transcribe [all model args] samples/sample.wav` produces JSON to stdout
  - [x] JSON is valid (parseable by Python `json.loads`)
  - [x] JSON has `segments` array with `speaker`, `start`, `duration`, `words` fields
  - [x] Each word has `text`, `start`, `end` fields

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: CLI produces valid JSON output
    Tool: Bash
    Preconditions: All models available, sample.wav exists
    Steps:
      1. ./build/bin/transcribe [model args] ../samples/sample.wav > /tmp/output.json
      2. python3 -c "import json; d=json.load(open('/tmp/output.json')); assert 'segments' in d; assert len(d['segments']) > 0; assert all('speaker' in s and 'words' in s for s in d['segments']); print('PASS')"
      3. Assert: exit code 0, stdout contains "PASS"
    Expected Result: Valid JSON with expected structure
    Evidence: /tmp/output.json saved
  ```

  **Commit**: YES
  - Message: `feat(integration): add transcribe CLI with JSON output`
  - Files: `diarization-ggml/src/main_transcribe.cpp`, `diarization-ggml/CMakeLists.txt` (add target)

---

- [x] 9. Integration Tests

  **What to do**:
  - Create `diarization-ggml/tests/test_integration.py`:
    - Run CLI on `samples/sample.wav`
    - Parse JSON output
    - Verify ≥2 speakers detected
    - Verify all words have timestamps
    - Verify timestamps are monotonically non-decreasing within each segment
    - Verify every word has a speaker label
    - Verify diarization segments roughly match existing RTTM reference (speaker boundaries within ±2s)
  - Verify existing diarization-only pipeline still works:
    - Run `diarization-ggml` on sample.wav → RTTM
    - Run `compare_rttm.py` → DER ≤ 1.0%
  - Add a timeline consistency test:
    - Verify that word timestamps from Whisper and segment boundaries from pyannote are in the same domain (filtered timeline)

  **Must NOT do**:
  - Must NOT require additional audio files beyond what's in samples/
  - Must NOT require manual inspection

  **Recommended Agent Profile**:
  - **Category**: `medium`
    - Reason: Python test script writing + running CLI + parsing JSON, moderate complexity
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 11)
  - **Blocks**: None
  - **Blocked By**: Task 8

  **References**:
  - `diarization-ggml/tests/compare_rttm.py` — Existing DER comparison test pattern
  - `diarization-ggml/tests/test_streaming.cpp` — Existing streaming test pattern

  **Acceptance Criteria**:
  - [x] `python3 tests/test_integration.py` exits 0
  - [x] ≥2 speakers in output
  - [x] All words have valid timestamps
  - [x] Existing DER test still passes (≤ 1.0%) — DER=0.14%, PASS

  **Commit**: YES
  - Message: `test(integration): add integration tests for transcription pipeline`
  - Files: `diarization-ggml/tests/test_integration.py`

---

- [x] 10. Commit Recluster Fix + Update AGENTS.md

  **What to do**:
  - Commit the existing uncommitted changes to `streaming.cpp` (the recluster fix that keeps embeddings unfiltered)
  - Update `AGENTS.md` Pitfall #5 to mark the issue as **FIXED**:
    - Change from "recluster overwrites state" warning to "FIXED: recluster now uses local variables, state->embeddings kept unfiltered"
  - Update `README.md` streaming limitations section: remove the warning about push→recluster→push instability

  **Must NOT do**:
  - Must NOT change any logic in streaming.cpp (only commit existing changes)
  - Must NOT remove the pitfall entirely (keep as historical note with FIXED label)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple commit + doc edits
  - **Skills**: [`git-master`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `diarization-ggml/src/streaming.cpp:509-510` — The fix comment to reference
  - `AGENTS.md:Common Pitfalls:5` — Pitfall #5 about recluster state mutation

  **Acceptance Criteria**:
  - [x] `git log -1` shows commit with recluster fix
  - [x] AGENTS.md Pitfall #5 says "FIXED"
  - [x] README.md streaming limitations updated

  **Commit**: YES
  - Message: `fix(streaming): commit recluster fix, update docs to reflect safe push→recluster cycles`
  - Files: `diarization-ggml/src/streaming.cpp`, `AGENTS.md`, `README.md`

---

- [x] 11. README.md Integration Documentation

  **What to do**:
  - Add "Transcription + Diarization" section to README.md:
    - Build instructions with all CMake flags
    - Model download instructions (Whisper model + Silero VAD + CoreML models)
    - CLI usage example
    - JSON output format documentation
    - API usage example (C++ code snippet showing init/push/finalize)
  - Add to AGENTS.md:
    - New "Integration Pipeline" section describing the 7-stage architecture
    - New build commands for the integrated pipeline
    - New test commands

  **Must NOT do**:
  - Must NOT remove existing documentation sections
  - Must NOT document features that aren't implemented

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Documentation writing task
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 9)
  - **Blocks**: None
  - **Blocked By**: Task 8

  **References**:
  - `README.md` — Existing documentation structure to follow
  - `AGENTS.md` — Existing agent guidelines structure

  **Acceptance Criteria**:
  - [x] README.md has "Transcription + Diarization" section
  - [x] Build commands in docs actually work (tested by Task 9)
  - [x] AGENTS.md has integration pipeline section

  **Commit**: YES
  - Message: `docs: add transcription+diarization integration documentation`
  - Files: `README.md`, `AGENTS.md`

---

## Commit Strategy

| After Task | Message | Key Files | Verification |
|------------|---------|-----------|--------------|
| 1 | `build: integrate whisper.cpp, reconcile shared GGML` | CMakeLists.txt | cmake build + existing tests |
| 2 | `feat(integration): add VAD silence filter` | silence_filter.{h,cpp} | test_silence_filter |
| 3 | `feat(integration): add audio buffer` | audio_buffer.{h,cpp} | test_audio_buffer |
| 4 | `feat(integration): add segment end detector` | segment_detector.{h,cpp} | test_segment_detector |
| 5 | `feat(integration): add Whisper transcriber` | transcriber.{h,cpp} | test_transcriber |
| 6 | `feat(integration): add word-speaker aligner` | aligner.{h,cpp} | test_aligner |
| 7 | `feat(integration): add pipeline orchestrator` | pipeline.{h,cpp} | test_pipeline |
| 8 | `feat(integration): add transcribe CLI` | main_transcribe.cpp | CLI runs on sample.wav |
| 9 | `test(integration): add integration tests` | test_integration.py | python3 test_integration.py |
| 10 | `fix(streaming): commit recluster fix, update docs` | streaming.cpp, AGENTS.md | git log |
| 11 | `docs: add integration documentation` | README.md, AGENTS.md | read docs |

---

## Success Criteria

### Verification Commands
```bash
# Build everything
cd diarization-ggml && cmake -B build -DEMBEDDING_COREML=ON -DSEGMENTATION_COREML=ON -DWHISPER_COREML=ON && cmake --build build

# Run all unit tests
./build/bin/test_silence_filter --vad-model path/to/silero.bin
./build/bin/test_audio_buffer
./build/bin/test_segment_detector
./build/bin/test_transcriber --model path/to/whisper.bin ../samples/jfk.wav
./build/bin/test_aligner
./build/bin/test_pipeline [model args] ../samples/sample.wav

# Run CLI
./build/bin/transcribe [model args] ../samples/sample.wav | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{len(d[\"segments\"])} segments, {sum(len(s[\"words\"]) for s in d[\"segments\"])} words')"

# Regression: existing diarization still works
./build/bin/diarization-ggml [existing args] -o /tmp/test.rttm
python3 tests/compare_rttm.py /tmp/test.rttm /tmp/py_reference.rttm --threshold 1.0

# Integration tests
python3 tests/test_integration.py
```

### Final Checklist
- [x] All "Must Have" present (7 pipeline stages, word timestamps, speaker labels)
- [x] All "Must NOT Have" absent (no whisper.cpp modifications, no global state, no GGML-only fallback)
- [x] All unit tests pass
- [x] Integration test passes
- [x] Existing diarization pipeline unaffected (DER ≤ 1.0%) — DER=0.14%, PASS
- [x] JSON output is valid and complete
- [x] No memory leaks on clean shutdown
