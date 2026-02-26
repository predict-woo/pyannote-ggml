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
struct ModelCache;

PipelineState* pipeline_init(const PipelineConfig& config,
                              pipeline_callback cb,
                              pipeline_audio_callback audio_cb,
                              void* user_data);

// Initialize pipeline with pre-loaded models from a ModelCache.
// The cache must remain valid until pipeline_free() is called.
PipelineState* pipeline_init_with_cache(
    const PipelineConfig& config,
    ModelCache* cache,
    pipeline_callback cb,
    pipeline_audio_callback audio_cb,
    void* user_data);
std::vector<bool> pipeline_push(PipelineState* state, const float* samples, int n_samples);
void pipeline_finalize(PipelineState* state);
void pipeline_free(PipelineState* state);
void pipeline_set_decode_options(PipelineState* state, const DecodeOptions& opts);
