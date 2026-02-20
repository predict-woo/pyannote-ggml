#pragma once

#include "streaming_state.h"
#include "transcriber.h"
#include "aligner.h"

#include <vector>

struct PipelineConfig {
    StreamingConfig diarization;
    TranscriberConfig transcriber;
    const char* vad_model_path;  // Silero VAD model for silence filter (nullptr = fallback)
};

typedef void (*pipeline_callback)(const std::vector<AlignedSegment>& segments,
                                  void* user_data);

typedef void (*pipeline_audio_callback)(const float* samples, int n_samples, void* user_data);

struct PipelineState;

PipelineState* pipeline_init(const PipelineConfig& config,
                              pipeline_callback cb,
                              pipeline_audio_callback audio_cb,
                              void* user_data);
std::vector<bool> pipeline_push(PipelineState* state, const float* samples, int n_samples);
void pipeline_finalize(PipelineState* state);
void pipeline_free(PipelineState* state);
