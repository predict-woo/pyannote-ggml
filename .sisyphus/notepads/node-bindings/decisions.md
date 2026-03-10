# Decisions — node-bindings plan

- 2026-03-10: Exposed `transcriptionOnly` in Node bindings without changing native interface shape; `BuildConfig()` now sets `config.transcription_only`, and empty embedding/PLDA/coreml strings are allowed so `model_cache_load()` can skip unavailable diarization models.
