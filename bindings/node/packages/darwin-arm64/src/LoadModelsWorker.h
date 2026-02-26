#pragma once
#include <napi.h>
#include <string>
#include "model_cache.h"

class PipelineModel;

class LoadModelsWorker : public Napi::AsyncWorker {
public:
    LoadModelsWorker(Napi::Env env,
                     PipelineModel* model,
                     ModelCacheConfig config,
                     Napi::Promise::Deferred deferred);

    void Execute() override;
    void OnOK() override;
    void OnError(const Napi::Error& error) override;

private:
    PipelineModel* model_;
    ModelCacheConfig config_;
    std::string vad_path_;
    Napi::Promise::Deferred deferred_;
    ModelCache* cache_ = nullptr;
};
