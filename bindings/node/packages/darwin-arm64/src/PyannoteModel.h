#pragma once
#include <napi.h>
#include "streaming_state.h"

class StreamingSession;

class PyannoteModel : public Napi::ObjectWrap<PyannoteModel> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    explicit PyannoteModel(const Napi::CallbackInfo& info);

    const StreamingConfig& GetConfig() const { return config_; }
    bool IsClosed() const { return closed_; }

    static Napi::FunctionReference constructor;

private:
    Napi::Value Diarize(const Napi::CallbackInfo& info);
    Napi::Value CreateStreamingSession(const Napi::CallbackInfo& info);
    Napi::Value Close(const Napi::CallbackInfo& info);
    Napi::Value GetIsClosed(const Napi::CallbackInfo& info);

    StreamingConfig config_;
    bool closed_ = false;
};
