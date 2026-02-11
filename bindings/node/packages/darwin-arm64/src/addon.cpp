#include <napi.h>
#include "PyannoteModel.h"
#include "StreamingSession.h"

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    PyannoteModel::Init(env, exports);
    StreamingSession::Init(env, exports);
    return exports;
}

NODE_API_MODULE(pyannote, Init)
