#include "ReclusterWorker.h"
#include "StreamingSession.h"

ReclusterWorker::ReclusterWorker(Napi::Env env,
                                 StreamingSession* session,
                                 StreamingState* state,
                                 Napi::Promise::Deferred deferred)
    : Napi::AsyncWorker(env),
      session_(session),
      state_(state),
      deferred_(deferred) {}

void ReclusterWorker::Execute() {
    result_ = streaming_recluster(state_);
}

void ReclusterWorker::OnOK() {
    Napi::Env env = Env();
    session_->SetBusy(false);

    Napi::Array segments = Napi::Array::New(env, result_.segments.size());
    for (size_t i = 0; i < result_.segments.size(); i++) {
        const auto& seg = result_.segments[i];
        Napi::Object obj = Napi::Object::New(env);
        obj.Set("start", Napi::Number::New(env, seg.start));
        obj.Set("duration", Napi::Number::New(env, seg.duration));
        obj.Set("speaker", Napi::String::New(env, seg.speaker));
        segments.Set(static_cast<uint32_t>(i), obj);
    }

    Napi::Object result = Napi::Object::New(env);
    result.Set("segments", segments);
    deferred_.Resolve(result);
}

void ReclusterWorker::OnError(const Napi::Error& error) {
    session_->SetBusy(false);
    deferred_.Reject(error.Value());
}
