#pragma once
#include <napi.h>
#include "streaming.h"
#include "diarization.h"

class StreamingSession;

class FinalizeWorker : public Napi::AsyncWorker {
public:
    FinalizeWorker(Napi::Env env,
                   StreamingSession* session,
                   StreamingState* state,
                   Napi::Promise::Deferred deferred);

    void Execute() override;
    void OnOK() override;
    void OnError(const Napi::Error& error) override;

private:
    StreamingSession* session_;
    StreamingState* state_;
    Napi::Promise::Deferred deferred_;
    DiarizationResult result_;
};
