#include <napi.h>
#include "../../../native/whisper/whisper_addon.h"
#include "../../../native/pipeline/win_pipeline_stubs.h"

Napi::Object InitPyannote(Napi::Env env, Napi::Object exports) {
    InitWhisperBindings(env, exports);
    InitPipelineStubs(env, exports);
    return exports;
}

NODE_API_MODULE(pyannote, InitPyannote)
