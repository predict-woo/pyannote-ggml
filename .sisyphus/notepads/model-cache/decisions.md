# Model Cache — Decisions

## Decision: Separate function vs refactoring existing `diarize_from_samples`
**Choice**: Created `diarize_from_samples_with_models` as a standalone CoreML-only function rather than refactoring the existing function to delegate to it.
**Rationale**: The existing `diarize_from_samples` has interleaved model loading/freeing with pipeline steps (frees seg model after segmentation, frees emb model after embedding). It also handles both CoreML and GGML code paths with `#ifdef` guards. Refactoring would have changed the model freeing order (affecting peak memory) and added complexity for the non-CoreML path. The standalone approach preserves existing behavior with zero risk.
**Tradeoff**: ~350 lines of duplicated pipeline logic, but the code is straightforward and self-contained.

## Decision: `offline_transcribe_with_cache` fallback
**Choice**: When CoreML is not defined, `offline_transcribe_with_cache` falls back to `diarize_from_samples` (path-based loading) for the diarization step, while still using the cached whisper_context.
**Rationale**: This gives partial cache benefits (Whisper context reuse) even without CoreML, without requiring GGML model handles in the cache.

## Decision: PLDA copied by value, CoreML pointers borrowed
**Choice**: PLDA is copied into each subsystem's state, while CoreML context pointers are borrowed (shared).
**Rationale**: PLDA is a small struct (~100KB of vectors) that's cheap to copy, and copying avoids lifetime management issues. CoreML contexts are opaque pointers that can't be meaningfully copied — they must be shared.

## Decision: model_cache_load ordering
**Choice**: Load order: seg CoreML → emb CoreML → PLDA → Whisper → VAD. On any failure, free already-loaded models in reverse order and return nullptr.
**Rationale**: This matches the dependency order (diarization models first, then Whisper, then optional VAD) and ensures clean rollback on partial failures.

## Decision: `streaming_init_with_models` handles zero_latency
**Choice**: The new init function still processes the silence chunk in zero_latency mode.
**Rationale**: The silence chunk processing is essential for correct streaming behavior and only uses the provided model handles (doesn't load anything). Skipping it would break the pipeline.
