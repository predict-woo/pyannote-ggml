# Decisions — node-bindings plan

- 2026-03-10: Exposed `transcriptionOnly` in Node bindings without changing native interface shape; `BuildConfig()` now sets `config.transcription_only`, and empty embedding/PLDA/coreml strings are allowed so `model_cache_load()` can skip unavailable diarization models.
- 2026-03-10: Rewrote `test/transcription_only.test.ts` as a full seven-group suite using shared `beforeAll`/`afterAll` pipelines and CLI byte-identical comparison, to lock behavior for all `transcriptionOnly: true` execution modes (`transcribe`, `transcribeOffline`, streaming session).
