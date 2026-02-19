#pragma once
#include <napi.h>
#include "pipeline.h"
#include "aligner.h"

class PipelineSession;

class PipelineFinalizeWorker : public Napi::AsyncWorker {
public:
    PipelineFinalizeWorker(Napi::Env env,
                           PipelineSession* session,
                           PipelineState* state,
                           Napi::Promise::Deferred deferred);

    void Execute() override;
    void OnOK() override;
    void OnError(const Napi::Error& error) override;

private:
    PipelineSession* session_;
    PipelineState* state_;
    Napi::Promise::Deferred deferred_;
};
