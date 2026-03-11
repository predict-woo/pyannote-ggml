#include "whisper_addon.h"

// Reuse the existing whisper addon implementation while routing its exports
// through the unified pyannote addon entrypoint.
#define WhisperContext WhisperAddonWhisperContext
#define VadContext WhisperAddonVadContext
#define transcribe_params whisper_addon_transcribe_params
#define token_result whisper_addon_token_result
#define segment_result whisper_addon_segment_result
#define transcribe_result whisper_addon_transcribe_result
#define TranscribeWorker WhisperAddonTranscribeWorker
#define legacy_whisper_params whisper_addon_legacy_whisper_params
#define LegacyProgressWorker WhisperAddonLegacyProgressWorker
#define Transcribe WhisperAddonTranscribe
#define LegacyWhisper WhisperAddonLegacyWhisper
#define GetGpuDevices WhisperAddonGetGpuDevices
#undef NODE_API_MODULE
#define NODE_API_MODULE(name, initfn)
#include "../../../../whisper.cpp/examples/addon.node/addon.cpp"
#undef NODE_API_MODULE
#undef GetGpuDevices
#undef LegacyWhisper
#undef Transcribe
#undef LegacyProgressWorker
#undef legacy_whisper_params
#undef TranscribeWorker
#undef transcribe_result
#undef segment_result
#undef token_result
#undef transcribe_params
#undef VadContext
#undef WhisperContext

Napi::Object InitWhisperBindings(Napi::Env env, Napi::Object exports) {
    return Init(env, exports);
}
