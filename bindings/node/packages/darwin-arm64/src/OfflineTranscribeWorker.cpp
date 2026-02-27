#include "OfflineTranscribeWorker.h"
#include "PipelineModel.h"

OfflineTranscribeWorker::OfflineTranscribeWorker(Napi::Env env,
                                                   PipelineModel* model,
                                                   std::vector<float>&& audio,
                                                   Napi::Promise::Deferred deferred,
                                                   Napi::Function progress_callback)
    : Napi::AsyncProgressQueueWorker<ProgressData>(env),
      model_(model),
      audio_(std::move(audio)),
      deferred_(deferred),
      cache_(model->GetCache())
{
    // Build OfflinePipelineConfig from model's PipelineConfig.
    // String fields (seg_model_path, etc.) are std::string copies — safe.
    // TranscriberConfig const char* pointers reference model_ member strings,
    // which outlive this worker (busy_ flag prevents concurrent use).
    PipelineConfig pc = model_->BuildConfig();
    config_.seg_model_path  = pc.diarization.seg_model_path;
    config_.emb_model_path  = pc.diarization.emb_model_path;
    config_.plda_path       = pc.diarization.plda_path;
    config_.coreml_path     = pc.diarization.coreml_path;
    config_.seg_coreml_path = pc.diarization.seg_coreml_path;
    config_.transcriber     = pc.transcriber;

    // Store progress callback reference (if provided)
    if (!progress_callback.IsEmpty() && progress_callback.IsFunction()) {
        progress_callback_ = Napi::Persistent(progress_callback);
    }
}

void OfflineTranscribeWorker::Execute(const ExecutionProgress& progress) {
    // Wire C++ progress callback to send progress data to main thread
    if (!progress_callback_.IsEmpty()) {
        config_.progress_callback = [&progress](int phase, int prog) {
            ProgressData data{phase, prog};
            progress.Send(&data, 1);
        };
    }

    OfflinePipelineResult result;
    if (cache_) {
        result = offline_transcribe_with_cache(
            config_, cache_, audio_.data(), static_cast<int>(audio_.size()));
    } else {
        result = offline_transcribe(
            config_, audio_.data(), static_cast<int>(audio_.size()));
    }

    if (!result.valid) {
        SetError("Offline pipeline failed");
        return;
    }

    cb_data_.segments = std::move(result.segments);
}

void OfflineTranscribeWorker::OnProgress(const ProgressData* data, size_t count) {
    // Called on the main thread for each progress event
    if (progress_callback_.IsEmpty()) return;

    Napi::Env env = Env();
    for (size_t i = 0; i < count; i++) {
        progress_callback_.Value().Call({
            Napi::Number::New(env, data[i].phase),
            Napi::Number::New(env, data[i].progress)
        });
    }
}

void OfflineTranscribeWorker::OnOK() {
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

void OfflineTranscribeWorker::OnError(const Napi::Error& error) {
    model_->SetBusy(false);
    deferred_.Reject(error.Value());
}
