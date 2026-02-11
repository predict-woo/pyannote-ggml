# Decisions - Streaming Diarization

## Architecture Decisions
- Zero-pad at stream start (matches offline)
- Unbounded memory (guarantees offline-identical finalization)
- Incremental cosine clustering for provisional labels (threshold 0.6)
- 60s audio time recluster interval
- Models owned by StreamingState, freed in streaming_free()
