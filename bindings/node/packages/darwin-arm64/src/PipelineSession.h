#pragma once
#include <napi.h>
#include <string>
#include <vector>
#include <mutex>
#include "pipeline.h"

class PipelineModel;

class PipelineSession : public Napi::ObjectWrap<PipelineSession> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    explicit PipelineSession(const Napi::CallbackInfo& info);
    ~PipelineSession();

    PipelineState* GetState() const { return state_; }
    void SetBusy(bool busy) { busy_ = busy; }

    std::vector<AlignedSegment> GetLastSegments() {
        std::lock_guard<std::mutex> lock(segments_mutex_);
        return last_segments_;
    }

    static Napi::FunctionReference constructor;

private:
    Napi::Value Push(const Napi::CallbackInfo& info);
    Napi::Value Finalize(const Napi::CallbackInfo& info);
    Napi::Value Close(const Napi::CallbackInfo& info);
    Napi::Value GetIsClosed(const Napi::CallbackInfo& info);

    void Cleanup();

    // Static callbacks for pipeline_init
    static void pipeline_cb(const std::vector<AlignedSegment>& segments,
                            void* user_data);
    static void audio_cb(const float* samples, int n_samples, void* user_data);

    std::string seg_model_path_;
    std::string emb_model_path_;
    std::string plda_path_;
    std::string coreml_path_;
    std::string seg_coreml_path_;
    std::string whisper_model_path_;
    std::string vad_model_path_;
    std::string language_;
    int n_threads_ = 4;

    bool use_gpu_ = true;
    bool flash_attn_ = true;
    int gpu_device_ = 0;
    bool use_coreml_ = false;
    bool no_prints_ = false;

    bool translate_ = false;
    bool detect_language_ = false;
    float temperature_ = 0.0f;
    float temperature_inc_ = 0.2f;
    bool no_fallback_ = false;
    int beam_size_ = -1;
    int best_of_ = 5;
    float entropy_thold_ = 2.4f;
    float logprob_thold_ = -1.0f;
    float no_speech_thold_ = 0.6f;
    std::string prompt_;
    bool no_context_ = true;
    bool suppress_blank_ = true;
    bool suppress_nst_ = false;

    PipelineState* state_ = nullptr;
    Napi::ThreadSafeFunction tsfn_;
    Napi::ThreadSafeFunction audio_tsfn_;
    bool closed_ = false;
    bool busy_ = false;
    bool tsfn_released_ = false;
    bool audio_tsfn_released_ = false;

    std::vector<AlignedSegment> last_segments_;
    std::mutex segments_mutex_;
};
