#pragma once
#include <napi.h>
#include <string>
#include "model_cache.h"

class PipelineModel;

class SwitchWhisperModeWorker : public Napi::AsyncWorker {
public:
    SwitchWhisperModeWorker(Napi::Env env,
                            PipelineModel* model,
                            bool use_coreml,
                            Napi::Promise::Deferred deferred);

    void Execute() override;
    void OnOK() override;
    void OnError(const Napi::Error& error) override;

private:
    PipelineModel* model_;
    bool use_coreml_;
    std::string whisper_path_;
    Napi::Promise::Deferred deferred_;
};
