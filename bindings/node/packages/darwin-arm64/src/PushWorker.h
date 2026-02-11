#pragma once
#include <napi.h>
#include <vector>
#include "streaming.h"

class StreamingSession;

class PushWorker : public Napi::AsyncWorker {
public:
    PushWorker(Napi::Env env,
               StreamingSession* session,
               StreamingState* state,
               std::vector<float>&& samples,
               Napi::Promise::Deferred deferred);

    void Execute() override;
    void OnOK() override;
    void OnError(const Napi::Error& error) override;

private:
    StreamingSession* session_;
    StreamingState* state_;
    std::vector<float> samples_;
    Napi::Promise::Deferred deferred_;
    std::vector<VADChunk> chunks_;
};
