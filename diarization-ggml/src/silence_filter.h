#pragma once

#include <vector>

struct whisper_vad_context;

struct SilenceFilterResult {
    std::vector<float> audio;
    bool flush_signal;
    std::vector<bool> vad_predictions;  // one per 512-sample window processed
};

struct SilenceFilter;

SilenceFilter* silence_filter_init(struct whisper_vad_context* vad_ctx, float threshold = 0.5f);
SilenceFilterResult silence_filter_push(SilenceFilter* sf, const float* samples, int n);
SilenceFilterResult silence_filter_flush(SilenceFilter* sf);
void silence_filter_free(SilenceFilter* sf);
