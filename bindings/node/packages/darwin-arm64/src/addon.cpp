#include <napi.h>
#include "PipelineModel.h"
#include "PipelineSession.h"

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    PipelineModel::Init(env, exports);
    PipelineSession::Init(env, exports);
    return exports;
}

NODE_API_MODULE(pyannote, Init)
