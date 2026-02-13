#pragma once
#include "transcriber_types.h"
#include <vector>

struct TranscriberConfig {
    const char* whisper_model_path;
    const char* whisper_coreml_path;  // nullptr if no CoreML
    int n_threads = 4;
    const char* language = "en";
};

struct Transcriber;

Transcriber* transcriber_init(const TranscriberConfig& config);
void transcriber_submit(Transcriber* t, const float* audio, int n_samples, double buffer_start_time);
bool transcriber_try_get_result(Transcriber* t, TranscribeResult& result);
TranscribeResult transcriber_wait_result(Transcriber* t);
void transcriber_free(Transcriber* t);
