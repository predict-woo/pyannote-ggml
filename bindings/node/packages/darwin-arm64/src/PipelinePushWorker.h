#pragma once
#include <napi.h>
#include <vector>
#include "pipeline.h"

class PipelineSession;

class PipelinePushWorker : public Napi::AsyncWorker {
public:
    PipelinePushWorker(Napi::Env env,
                       PipelineSession* session,
                       PipelineState* state,
                       std::vector<float>&& samples,
                       Napi::Promise::Deferred deferred);

    void Execute() override;
    void OnOK() override;
    void OnError(const Napi::Error& error) override;

private:
    PipelineSession* session_;
    PipelineState* state_;
    std::vector<float> samples_;
    Napi::Promise::Deferred deferred_;
    std::vector<bool> vad_predictions_;
};
