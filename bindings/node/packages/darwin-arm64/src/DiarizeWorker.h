#pragma once
#include <napi.h>
#include <vector>
#include "streaming.h"
#include "streaming_state.h"
#include "diarization.h"

class DiarizeWorker : public Napi::AsyncWorker {
public:
    DiarizeWorker(Napi::Env env,
                  const StreamingConfig& config,
                  std::vector<float>&& audio,
                  Napi::Promise::Deferred deferred);

    void Execute() override;
    void OnOK() override;
    void OnError(const Napi::Error& error) override;

private:
    StreamingConfig config_;
    std::vector<float> audio_;
    Napi::Promise::Deferred deferred_;
    DiarizationResult result_;
};
