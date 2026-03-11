#include <napi.h>
#include "../../../native/whisper/whisper_addon.h"
#include "PipelineModel.h"
#include "PipelineSession.h"

namespace {

Napi::Value GetCapabilities(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Object capabilities = Napi::Object::New(env);
    capabilities.Set("whisper", Napi::Boolean::New(env, true));
    capabilities.Set("vad", Napi::Boolean::New(env, true));
    capabilities.Set("gpuDiscovery", Napi::Boolean::New(env, true));
    capabilities.Set("pipeline", Napi::Boolean::New(env, true));
    capabilities.Set("diarization", Napi::Boolean::New(env, true));
    return capabilities;
}

} // namespace

Napi::Object InitPyannote(Napi::Env env, Napi::Object exports) {
    InitWhisperBindings(env, exports);
    PipelineModel::Init(env, exports);
    PipelineSession::Init(env, exports);
    exports.Set("getCapabilities", Napi::Function::New(env, GetCapabilities));
    return exports;
}

NODE_API_MODULE(pyannote, InitPyannote)
