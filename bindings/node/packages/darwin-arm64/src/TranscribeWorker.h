#pragma once
#include <napi.h>
#include <vector>
#include "pipeline.h"
#include "aligner.h"

class PipelineModel;

struct TranscribeCallbackData {
    std::vector<AlignedSegment> segments;
};

class TranscribeWorker : public Napi::AsyncWorker {
public:
    TranscribeWorker(Napi::Env env,
                     PipelineModel* model,
                     std::vector<float>&& audio,
                     Napi::Promise::Deferred deferred);

    void Execute() override;
    void OnOK() override;
    void OnError(const Napi::Error& error) override;

private:
    static void OnPipelineCallback(const std::vector<AlignedSegment>& segments,
                                   const std::vector<float>& audio,
                                   void* user_data);

    PipelineModel* model_;
    PipelineConfig config_;
    std::vector<float> audio_;
    Napi::Promise::Deferred deferred_;
    TranscribeCallbackData cb_data_;
};
