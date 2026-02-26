#include "SwitchWhisperModeWorker.h"
#include "PipelineModel.h"

SwitchWhisperModeWorker::SwitchWhisperModeWorker(Napi::Env env,
                                                 PipelineModel* model,
                                                 bool use_coreml,
                                                 Napi::Promise::Deferred deferred)
    : Napi::AsyncWorker(env),
      model_(model),
      use_coreml_(use_coreml),
      deferred_(deferred)
{
    // Store whisper_model_path backing string locally so the const char* pointer
    // remains valid during Execute() regardless of the caller's lifetime.
    PipelineConfig pc = model_->BuildConfig();
    if (pc.transcriber.whisper_model_path) {
        whisper_path_ = pc.transcriber.whisper_model_path;
    }
}

void SwitchWhisperModeWorker::Execute() {
    ModelCache* cache = model_->GetCache();
    if (!cache) {
        SetError("No model cache available");
        return;
    }

    // Build TranscriberConfig from the model's current config, overriding use_coreml
    PipelineConfig pc = model_->BuildConfig();
    TranscriberConfig config = pc.transcriber;
    config.use_coreml = use_coreml_;
    config.whisper_model_path = whisper_path_.c_str();

    if (!model_cache_reload_whisper(cache, config)) {
        SetError("Failed to reload Whisper model");
    }
}

void SwitchWhisperModeWorker::OnOK() {
    model_->SetUseCoreml(use_coreml_);
    model_->SetBusy(false);
    deferred_.Resolve(Env().Undefined());
}

void SwitchWhisperModeWorker::OnError(const Napi::Error& error) {
    model_->SetBusy(false);
    deferred_.Reject(error.Value());
}
