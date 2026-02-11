#pragma once
#include "streaming_state.h"
#include "diarization.h"
#include <vector>

// Combined VAD activity for a single processed chunk
struct VADChunk {
    int chunk_index;        // which chunk (0-based)
    double start_time;      // absolute start time of this chunk's audio window
    double duration;        // always 10.0s (CHUNK_SAMPLES / SAMPLE_RATE)
    int num_frames;         // always 589
    std::vector<float> vad; // [589] combined speaker activity — 1.0 if ANY speaker active, 0.0 otherwise
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
