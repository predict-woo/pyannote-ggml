# Decisions

## 2026-02-19 Planning Phase
- Replace old diarization-only API entirely (no backward compat)
- Both modes: one-shot transcribe() + streaming PipelineSession with EventEmitter
- Singleton state: Pipeline.load() creates ONE PipelineState
- TSFN for callbacks from C++ worker threads to JS
- VAD at Silero window granularity (~31 bools per 1s push)
- Include audio Float32Array in segments event
