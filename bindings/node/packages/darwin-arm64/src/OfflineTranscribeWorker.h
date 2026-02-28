#pragma once
#include <napi.h>
#include <string>
#include <vector>
#include "offline_pipeline.h"
#include "aligner.h"

class PipelineModel;

struct ProgressData {
    int phase;
    int progress;
};

struct SegmentData {
    double start;
    double end;
    std::string text;
};

struct OfflineTranscribeCallbackData {
    std::vector<AlignedSegment> segments;
    std::vector<float> filtered_audio;
};

class OfflineTranscribeWorker : public Napi::AsyncProgressQueueWorker<ProgressData> {
public:
    OfflineTranscribeWorker(Napi::Env env,
                            PipelineModel* model,
                            std::vector<float>&& audio,
                            Napi::Promise::Deferred deferred,
                            Napi::Function progress_callback,
                            Napi::Function segment_callback);

    void Execute(const ExecutionProgress& progress) override;
    void OnOK() override;
    void OnError(const Napi::Error& error) override;
    void OnProgress(const ProgressData* data, size_t count) override;

private:
    PipelineModel* model_;
    OfflinePipelineConfig config_;
    std::vector<float> audio_;
    Napi::Promise::Deferred deferred_;
    OfflineTranscribeCallbackData cb_data_;
    ModelCache* cache_ = nullptr;
    Napi::FunctionReference progress_callback_;
    Napi::ThreadSafeFunction segment_tsfn_;
    bool has_segment_tsfn_ = false;
};
