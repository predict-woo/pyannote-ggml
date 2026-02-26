#include "LoadModelsWorker.h"
#include "PipelineModel.h"

LoadModelsWorker::LoadModelsWorker(Napi::Env env,
                                   PipelineModel* model,
                                   ModelCacheConfig config,
                                   Napi::Promise::Deferred deferred)
    : Napi::AsyncWorker(env),
      model_(model),
      config_(std::move(config)),
      deferred_(deferred)
{
    // Store vad_model_path backing string locally so the const char* pointer
    // remains valid during Execute() regardless of the caller's lifetime.
    if (config_.vad_model_path) {
        vad_path_ = config_.vad_model_path;
        config_.vad_model_path = vad_path_.c_str();
    }
}

void LoadModelsWorker::Execute() {
    cache_ = model_cache_load(config_);
    if (!cache_) {
        SetError("Failed to load models into cache");
    }
}

void LoadModelsWorker::OnOK() {
    model_->SetCache(cache_);
    deferred_.Resolve(Env().Undefined());
}

void LoadModelsWorker::OnError(const Napi::Error& error) {
    deferred_.Reject(error.Value());
}
