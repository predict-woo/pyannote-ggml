#pragma once
#include <napi.h>
#include <vector>
#include "offline_pipeline.h"
#include "aligner.h"

class PipelineModel;

struct OfflineTranscribeCallbackData {
    std::vector<AlignedSegment> segments;
};

class OfflineTranscribeWorker : public Napi::AsyncWorker {
public:
    OfflineTranscribeWorker(Napi::Env env,
                            PipelineModel* model,
                            std::vector<float>&& audio,
                            Napi::Promise::Deferred deferred);

    void Execute() override;
    void OnOK() override;
    void OnError(const Napi::Error& error) override;

private:
    PipelineModel* model_;
    OfflinePipelineConfig config_;
    std::vector<float> audio_;
    Napi::Promise::Deferred deferred_;
    OfflineTranscribeCallbackData cb_data_;
    ModelCache* cache_ = nullptr;
};
