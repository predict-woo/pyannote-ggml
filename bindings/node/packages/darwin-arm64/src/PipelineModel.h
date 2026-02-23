#pragma once
#include <napi.h>
#include <string>
#include "pipeline.h"

class PipelineSession;

class PipelineModel : public Napi::ObjectWrap<PipelineModel> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    explicit PipelineModel(const Napi::CallbackInfo& info);
    ~PipelineModel();

    PipelineConfig BuildConfig() const;
    bool IsClosed() const { return closed_; }
    bool IsBusy() const { return busy_; }
    void SetBusy(bool busy) { busy_ = busy; }

    static Napi::FunctionReference constructor;

private:
    Napi::Value Transcribe(const Napi::CallbackInfo& info);
    Napi::Value CreateSession(const Napi::CallbackInfo& info);
    Napi::Value Close(const Napi::CallbackInfo& info);
    Napi::Value GetIsClosed(const Napi::CallbackInfo& info);
    Napi::Value SetLanguage(const Napi::CallbackInfo& info);
    Napi::Value SetDecodeOptions(const Napi::CallbackInfo& info);

    std::string seg_model_path_;
    std::string emb_model_path_;
    std::string plda_path_;
    std::string coreml_path_;
    std::string seg_coreml_path_;
    std::string whisper_model_path_;
    std::string vad_model_path_;
    std::string language_;
    int n_threads_ = 4;

    // Whisper context options
    bool use_gpu_ = true;
    bool flash_attn_ = true;
    int gpu_device_ = 0;
    bool use_coreml_ = false;
    bool no_prints_ = false;

    // Whisper decode options
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

    bool closed_ = false;
    bool busy_ = false;
};
