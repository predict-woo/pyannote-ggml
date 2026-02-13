#pragma once
#include "streaming_state.h"
#include "diarization.h"
#include <vector>

// Combined VAD activity for newly covered frames from a processed chunk.
// In normal mode, chunk 0 returns all 589 frames; subsequent chunks return ~59-60 new frames.
// In zero_latency mode, chunk 0 (silence) is processed during init and not returned;
// the first real push returns ~59-60 frames with start_frame=0.
// Caller computes timestamps via: time = start_frame * 0.016875
struct VADChunk {
    int chunk_index;        // which internal chunk (0-based, including silence chunks)
    int start_frame;        // global frame index of the first new frame (adjusted for zero_latency)
    int num_frames;         // number of new frames (variable: ~59-60, or 589 for first chunk in normal mode)
    std::vector<float> vad; // [num_frames] combined speaker activity — 1.0 if ANY speaker active, 0.0 otherwise
};

// Initialize streaming state. Returns nullptr on failure.
StreamingState* streaming_init(const StreamingConfig& config);

// Push audio samples. Returns VAD chunks for each newly processed segmentation chunk.
// samples: float array of audio samples (16kHz mono)
// num_samples: number of samples (typically 16000 for 1s of audio)
std::vector<VADChunk> streaming_push(
    StreamingState* state,
    const float* samples,
    int num_samples);

// Force a recluster now. Returns full diarization result.
DiarizationResult streaming_recluster(StreamingState* state);

// Finalize: recluster + set finalized flag. Returns offline-identical result.
DiarizationResult streaming_finalize(StreamingState* state);

// Free all resources
void streaming_free(StreamingState* state);
