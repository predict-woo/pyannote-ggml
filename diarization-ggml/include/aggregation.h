#pragma once
#include <vector>

namespace diarization {

struct SlidingWindowParams {
    double start;
    double duration;
    double step;
};

void aggregate_chunks(const float* chunks, int num_chunks, int num_frames, int num_classes,
                      const SlidingWindowParams& chunk_window,
                      const SlidingWindowParams& frame_window,
                      std::vector<float>& output, int& total_frames,
                      bool skip_average = false, float missing = 0.0f);

void compute_speaker_count(const float* binarized, int num_chunks, int num_frames,
                           int num_speakers,
                           const SlidingWindowParams& chunk_window,
                           const SlidingWindowParams& frame_window,
                           std::vector<int>& count, int& total_frames);

void to_diarization(const float* segmentations, int num_chunks, int num_frames, int num_speakers,
                    const int* count, int total_frames,
                    const SlidingWindowParams& chunk_window,
                    const SlidingWindowParams& frame_window,
                    std::vector<float>& output);

}  // namespace diarization
