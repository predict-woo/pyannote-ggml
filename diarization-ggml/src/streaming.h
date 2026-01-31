#pragma once
#include "streaming_state.h"
#include "diarization.h"
#include <vector>

// Initialize streaming state. Returns nullptr on failure.
StreamingState* streaming_init(const StreamingConfig& config);

// Push audio samples. Returns provisional segments (full timeline from start).
// samples: float array of audio samples (16kHz mono)
// num_samples: number of samples (typically 16000 for 1s of audio)
std::vector<DiarizationResult::Segment> streaming_push(
    StreamingState* state,
    const float* samples,
    int num_samples);

// Force a recluster now (normally happens automatically every 60s)
void streaming_recluster(StreamingState* state);

// Finalize and get offline-identical result
DiarizationResult streaming_finalize(StreamingState* state);

// Free all resources
void streaming_free(StreamingState* state);
