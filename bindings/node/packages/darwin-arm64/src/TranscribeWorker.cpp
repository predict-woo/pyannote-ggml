#include "TranscribeWorker.h"
#include "PipelineModel.h"

static constexpr int SAMPLES_PER_SECOND = 16000;

TranscribeWorker::TranscribeWorker(Napi::Env env,
                                   PipelineModel* model,
                                   std::vector<float>&& audio,
                                   Napi::Promise::Deferred deferred)
    : Napi::AsyncWorker(env),
      model_(model),
      config_(model->BuildConfig()),
      audio_(std::move(audio)),
      deferred_(deferred) {}

void TranscribeWorker::OnPipelineCallback(
    const std::vector<AlignedSegment>& segments,
    void* user_data) {
    auto* data = static_cast<TranscribeCallbackData*>(user_data);
    data->segments = segments;
}

void TranscribeWorker::Execute() {
    PipelineState* state = pipeline_init(config_, OnPipelineCallback, nullptr, &cb_data_);
    if (!state) {
        SetError("Failed to initialize pipeline state");
        return;
    }

    const int total = static_cast<int>(audio_.size());
    int offset = 0;
    while (offset < total) {
        int chunk_size = std::min(SAMPLES_PER_SECOND, total - offset);
        pipeline_push(state, audio_.data() + offset, chunk_size);
        offset += chunk_size;
    }

    pipeline_finalize(state);
    pipeline_free(state);
}

void TranscribeWorker::OnOK() {
    Napi::Env env = Env();

    const auto& segments = cb_data_.segments;
    Napi::Array jsSegments = Napi::Array::New(env, segments.size());

    for (size_t i = 0; i < segments.size(); i++) {
        const auto& seg = segments[i];
        Napi::Object obj = Napi::Object::New(env);
        obj.Set("speaker", Napi::String::New(env, seg.speaker));
        obj.Set("start", Napi::Number::New(env, seg.start));
        obj.Set("duration", Napi::Number::New(env, seg.duration));
        obj.Set("text", Napi::String::New(env, seg.text));

        jsSegments.Set(static_cast<uint32_t>(i), obj);
    }

    Napi::Object result = Napi::Object::New(env);
    result.Set("segments", jsSegments);

    model_->SetBusy(false);
    deferred_.Resolve(result);
}

void TranscribeWorker::OnError(const Napi::Error& error) {
    model_->SetBusy(false);
    deferred_.Reject(error.Value());
}
