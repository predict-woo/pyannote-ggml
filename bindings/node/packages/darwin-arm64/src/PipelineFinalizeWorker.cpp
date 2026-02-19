#include "PipelineFinalizeWorker.h"
#include "PipelineSession.h"

// Defined in PipelineSession.cpp
Napi::Array MarshalSegments(Napi::Env env, const std::vector<AlignedSegment>& segments);

PipelineFinalizeWorker::PipelineFinalizeWorker(Napi::Env env,
                                               PipelineSession* session,
                                               PipelineState* state,
                                               Napi::Promise::Deferred deferred)
    : Napi::AsyncWorker(env),
      session_(session),
      state_(state),
      deferred_(deferred) {}

void PipelineFinalizeWorker::Execute() {
    pipeline_finalize(state_);
}

void PipelineFinalizeWorker::OnOK() {
    Napi::Env env = Env();
    session_->SetBusy(false);

    std::vector<AlignedSegment> segments = session_->GetLastSegments();
    Napi::Array segmentsArr = MarshalSegments(env, segments);

    Napi::Object result = Napi::Object::New(env);
    result.Set("segments", segmentsArr);
    deferred_.Resolve(result);
}

void PipelineFinalizeWorker::OnError(const Napi::Error& error) {
    session_->SetBusy(false);
    deferred_.Reject(error.Value());
}
