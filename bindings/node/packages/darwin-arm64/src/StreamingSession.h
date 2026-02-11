#pragma once
#include <napi.h>
#include "streaming.h"

class StreamingSession : public Napi::ObjectWrap<StreamingSession> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    explicit StreamingSession(const Napi::CallbackInfo& info);
    ~StreamingSession();

    StreamingState* GetState() const { return state_; }
    void SetBusy(bool busy) { busy_ = busy; }

    static Napi::FunctionReference constructor;

private:
    Napi::Value Push(const Napi::CallbackInfo& info);
    Napi::Value Recluster(const Napi::CallbackInfo& info);
    Napi::Value Finalize(const Napi::CallbackInfo& info);
    Napi::Value Close(const Napi::CallbackInfo& info);
    Napi::Value GetIsClosed(const Napi::CallbackInfo& info);

    void Cleanup();

    StreamingState* state_ = nullptr;
    bool closed_ = false;
    bool busy_ = false;
};
