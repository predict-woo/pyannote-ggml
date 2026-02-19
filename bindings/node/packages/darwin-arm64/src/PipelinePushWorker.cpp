#include "PipelinePushWorker.h"
#include "PipelineSession.h"

PipelinePushWorker::PipelinePushWorker(Napi::Env env,
                                       PipelineSession* session,
                                       PipelineState* state,
                                       std::vector<float>&& samples,
                                       Napi::Promise::Deferred deferred)
    : Napi::AsyncWorker(env),
      session_(session),
      state_(state),
      samples_(std::move(samples)),
      deferred_(deferred) {}

void PipelinePushWorker::Execute() {
    vad_predictions_ = pipeline_push(state_, samples_.data(),
                                     static_cast<int>(samples_.size()));
}

void PipelinePushWorker::OnOK() {
    Napi::Env env = Env();
    session_->SetBusy(false);

    Napi::Array result = Napi::Array::New(env, vad_predictions_.size());
    for (size_t i = 0; i < vad_predictions_.size(); i++) {
        result.Set(static_cast<uint32_t>(i),
                   Napi::Boolean::New(env, vad_predictions_[i]));
    }
    deferred_.Resolve(result);
}

void PipelinePushWorker::OnError(const Napi::Error& error) {
    session_->SetBusy(false);
    deferred_.Reject(error.Value());
}
