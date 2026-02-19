#pragma once
#include "transcriber_types.h"
#include <vector>

struct TranscriberConfig {
    // Model loading
    const char* whisper_model_path = nullptr;
    int n_threads = 4;  // whisper default: min(4, hw_concurrency) — we keep 4

    // Context params (model loading time) — defaults from whisper_context_default_params()
    bool use_gpu = true;           // whisper default: true
    bool flash_attn = true;        // whisper default: true — we keep true (was already hardcoded)
    int gpu_device = 0;            // whisper default: 0
    bool use_coreml = false;       // whisper default: false — REPLACES whisper_coreml_path
    bool no_prints = false;        // suppress whisper log output

    // Decode params — defaults from whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
    const char* language = "en";   // whisper default: "en"
    bool translate = false;        // whisper default: false
    bool detect_language = false;  // whisper default: false

    // Sampling
    float temperature = 0.0f;      // whisper default: 0.0
    float temperature_inc = 0.2f;  // whisper default: 0.2
    bool no_fallback = false;      // if true, sets temperature_inc = 0.0
    int beam_size = -1;            // whisper greedy default: -1 (uses greedy with best_of)
    int best_of = 5;               // whisper greedy default: 5

    // Thresholds
    float entropy_thold = 2.4f;    // whisper default: 2.4
    float logprob_thold = -1.0f;   // whisper default: -1.0
    float no_speech_thold = 0.6f;  // whisper default: 0.6

    // Context
    const char* prompt = nullptr;  // whisper default: nullptr
    bool no_context = true;        // whisper default: true
    bool suppress_blank = true;    // whisper default: true
    bool suppress_nst = false;     // whisper default: false
};

struct Transcriber;

Transcriber* transcriber_init(const TranscriberConfig& config);
void transcriber_submit(Transcriber* t, const float* audio, int n_samples, double buffer_start_time);
bool transcriber_try_get_result(Transcriber* t, TranscribeResult& result);
TranscribeResult transcriber_wait_result(Transcriber* t);
void transcriber_free(Transcriber* t);
