#include "DiarizeWorker.h"

static constexpr int SAMPLES_PER_SECOND = 16000;

DiarizeWorker::DiarizeWorker(Napi::Env env,
                             const StreamingConfig& config,
                             std::vector<float>&& audio,
                             Napi::Promise::Deferred deferred)
    : Napi::AsyncWorker(env),
      config_(config),
      audio_(std::move(audio)),
      deferred_(deferred) {}

void DiarizeWorker::Execute() {
    StreamingState* state = streaming_init(config_);
    if (!state) {
        SetError("Failed to initialize streaming state for diarization");
        return;
    }

    const int total = static_cast<int>(audio_.size());
    int offset = 0;
    while (offset < total) {
        int chunk_size = std::min(SAMPLES_PER_SECOND, total - offset);
        streaming_push(state, audio_.data() + offset, chunk_size);
        offset += chunk_size;
    }

    result_ = streaming_finalize(state);
    streaming_free(state);
}

void DiarizeWorker::OnOK() {
    Napi::Env env = Env();

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

void DiarizeWorker::OnError(const Napi::Error& error) {
    deferred_.Reject(error.Value());
}
