# Draft: Whisper + Pyannote Streaming Diarization Integration

## Requirements (confirmed)

- **Code location**: Extend diarization-ggml (add new files to diarization-ggml/src/)
- **Usage mode**: Both library API + CLI executable simultaneously
- **Whisper dependency**: add_subdirectory(whisper.cpp) in CMake, shared GGML
- **VAD source**: Whisper's built-in Silero VAD (`whisper_vad_detect_speech_single_frame`)
- **Recluster strategy**: Recluster on every Whisper return (user confirms the state mutation bug is fixed)

## Technical Decisions

- **Recluster safety**: CONFIRMED FIXED — streaming.cpp line 509 comment: "Keep state->embeddings/chunk_idx/local_speaker_idx unfiltered (3 per chunk) so subsequent push() calls can safely append to them." Recluster now uses local `filtered_emb` variables. AGENTS.md pitfall #5 needs updating.
- **Timestamp units**: Whisper token t0/t1 are in 10ms ticks (centiseconds). Convert: `seconds = t0 * 0.01`
- **Token extraction**: Use `whisper_full_get_token_data(ctx, seg, tok).t0/.t1`, NOT deprecated `whisper_full_get_token_t0/t1`
- **Word-level**: Enable with `params.token_timestamps = true`, `params.max_len = 1`, `params.split_on_word = true`
- **Alignment algorithm**: WhisperX maximum intersection (per-word overlap scoring against diarization segments)
- **Interval tree**: Simple sorted array + binary search sufficient (segments rebuilt per recluster). Consider ekg/intervaltree if perf needed.

## Research Findings

- **WhisperX algorithm**: Iterates words, queries overlapping diarization segments, sums intersection duration per speaker, picks argmax. Fallback: nearest segment by midpoint if no overlap (fill_nearest=true).
- **whisper.cpp streaming pattern**: Rolling buffer + `whisper_full()` per chunk. Prompt token carry for context continuity.
- **Whisper CMake target**: `whisper` links `ggml` + `Threads::Threads`. GGML conflict with pyannote's GGML submodule needs resolution.
- **CoreML**: Whisper supports `-DWHISPER_COREML=ON` for encoder ANE acceleration

## Open Questions

- ALL RESOLVED

## Additional Decisions (Round 2)

- **GGML conflict**: Shared single GGML — add once at top level, both whisper and diarization-lib use it
- **Whisper model**: large-v3-turbo (~809MB), configurable path at runtime
- **Output format**: JSON (segments + words). Callback struct for library API.
- **Threading**: Whisper on separate worker thread. Mutex + condition variable for results back to main thread. Main thread runs recluster + alignment.
- **Documentation**: Include AGENTS.md + README.md updates in the plan
- **Thread comms**: Mutex + CV. Whisper thread produces results, main thread consumes and runs recluster/alignment.
- **CLI input**: WAV file only (first implementation)
- **Whisper CoreML**: YES — enable CoreML for Whisper encoder (ANE acceleration). Include model conversion step.
- **API design**: Opaque handle + push/finalize pattern (transcriber_init/push/finalize/free) with callback for results

## Scope Boundaries

- INCLUDE: Full 7-stage pipeline from INTEGRATION_PLAN.md
- INCLUDE: CLI executable for testing
- INCLUDE: C++ library API
- EXCLUDE: Modifying whisper.cpp source code
- EXCLUDE: Non-macOS platforms (CoreML required for pyannote streaming)
- EXCLUDE: Real-time microphone input (file-based first)

## Pipeline Architecture (from INTEGRATION_PLAN.md)

1. VAD Silence Filter (whisper Silero VAD, 512-frame chunks)
2. Audio Buffer (FIFO + absolute frame tracking)
3. Segmentation / Segment End Detection (pyannote streaming VAD output)
4. Transcription (whisper_full with word-level timestamps)
5. Alignment (WhisperX maximum intersection)
6. Finalize (flush all stages)
7. Callback (labeled segments + words)
