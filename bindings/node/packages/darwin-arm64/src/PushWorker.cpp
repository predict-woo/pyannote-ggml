#include "PushWorker.h"
#include "StreamingSession.h"
#include <cstring>

PushWorker::PushWorker(Napi::Env env,
                       StreamingSession* session,
                       StreamingState* state,
                       std::vector<float>&& samples,
                       Napi::Promise::Deferred deferred)
    : Napi::AsyncWorker(env),
      session_(session),
      state_(state),
      samples_(std::move(samples)),
      deferred_(deferred) {}

void PushWorker::Execute() {
    chunks_ = streaming_push(state_, samples_.data(), static_cast<int>(samples_.size()));
}

void PushWorker::OnOK() {
    Napi::Env env = Env();
    session_->SetBusy(false);

    Napi::Array result = Napi::Array::New(env, chunks_.size());

    for (size_t i = 0; i < chunks_.size(); i++) {
        const VADChunk& chunk = chunks_[i];
        Napi::Object obj = Napi::Object::New(env);

        obj.Set("chunkIndex", Napi::Number::New(env, chunk.chunk_index));
        obj.Set("startTime", Napi::Number::New(env, chunk.start_time));
        obj.Set("duration", Napi::Number::New(env, chunk.duration));
        obj.Set("numFrames", Napi::Number::New(env, chunk.num_frames));

        Napi::ArrayBuffer buf = Napi::ArrayBuffer::New(env, chunk.vad.size() * sizeof(float));
        std::memcpy(buf.Data(), chunk.vad.data(), chunk.vad.size() * sizeof(float));
        Napi::Float32Array vad = Napi::Float32Array::New(env, chunk.vad.size(), buf, 0);
        obj.Set("vad", vad);

        result.Set(static_cast<uint32_t>(i), obj);
    }

    deferred_.Resolve(result);
}

void PushWorker::OnError(const Napi::Error& error) {
    session_->SetBusy(false);
    deferred_.Reject(error.Value());
}
