#include "aggregation.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace diarization {

// Port of pyannote.core.SlidingWindow.closest_frame
// Python: int(np.rint((t - self.start - 0.5 * self.duration) / self.step))
static int closest_frame(double t, const SlidingWindowParams& sw) {
    return static_cast<int>(std::lround((t - sw.start - 0.5 * sw.duration) / sw.step));
}

void aggregate_chunks(const float* chunks, int num_chunks, int num_frames, int num_classes,
                      const SlidingWindowParams& chunk_window,
                      const SlidingWindowParams& frame_window,
                      std::vector<float>& output, int& total_frames,
                      bool skip_average, float missing) {
    // Python constructs: frames = SlidingWindow(start=chunks.start, duration=frames.duration, step=frames.step)
    // So frames.start = chunk_window.start, not frame_window.start
    const SlidingWindowParams frames = {chunk_window.start, frame_window.duration, frame_window.step};

    // 1. Compute total number of output frames
    // Python: frames.closest_frame(chunks.start + chunks.duration + (nc-1)*chunks.step + 0.5*frames.duration) + 1
    const double end_time = chunk_window.start
                          + chunk_window.duration
                          + (num_chunks - 1) * chunk_window.step
                          + 0.5 * frames.duration;
    total_frames = closest_frame(end_time, frames) + 1;

    // 2. Initialize arrays
    const int total_elements = total_frames * num_classes;
    output.assign(total_elements, 0.0f);
    std::vector<float> overlap_count(total_elements, 0.0f);
    std::vector<float> agg_mask(total_elements, 0.0f);

    // 3. Loop over chunks — overlap-add aggregation
    // No Hamming window (hamming=false → weight=1.0)
    // No warm-up (warm_up=(0.0,0.0) → weight=1.0)
    for (int c = 0; c < num_chunks; c++) {
        const double chunk_start = chunk_window.start + c * chunk_window.step;
        const int start_frame = closest_frame(chunk_start + 0.5 * frames.duration, frames);

        const float* score = chunks + (size_t)c * num_frames * num_classes;

        for (int f = 0; f < num_frames; f++) {
            const int out_f = start_frame + f;
            if (out_f < 0 || out_f >= total_frames) continue;

            for (int k = 0; k < num_classes; k++) {
                const float val = score[f * num_classes + k];
                const float mask = std::isnan(val) ? 0.0f : 1.0f;
                const float clean_val = std::isnan(val) ? 0.0f : val;

                output[out_f * num_classes + k] += clean_val * mask;
                overlap_count[out_f * num_classes + k] += mask;
                agg_mask[out_f * num_classes + k] = std::max(
                    agg_mask[out_f * num_classes + k], mask);
            }
        }
    }

    // 4. Average (unless skip_average)
    if (!skip_average) {
        constexpr float epsilon = 1e-12f;
        for (int i = 0; i < total_elements; i++) {
            output[i] /= std::max(overlap_count[i], epsilon);
        }
    }

    // 5. Replace frames with no contributing data with missing value
    for (int i = 0; i < total_elements; i++) {
        if (agg_mask[i] == 0.0f) {
            output[i] = missing;
        }
    }
}

void compute_speaker_count(const float* binarized, int num_chunks, int num_frames,
                           int num_speakers,
                           const SlidingWindowParams& chunk_window,
                           const SlidingWindowParams& frame_window,
                           std::vector<int>& count, int& total_frames) {
    // 1. Sum across speakers per frame: (num_chunks, num_frames, num_speakers) → (num_chunks, num_frames, 1)
    const int chunk_size = num_frames;  // frames per chunk
    std::vector<float> summed(num_chunks * chunk_size);
    for (int c = 0; c < num_chunks; c++) {
        for (int f = 0; f < chunk_size; f++) {
            float sum = 0.0f;
            for (int s = 0; s < num_speakers; s++) {
                sum += binarized[(size_t)c * chunk_size * num_speakers + f * num_speakers + s];
            }
            summed[c * chunk_size + f] = sum;
        }
    }

    // 2. No warm-up trimming (warm_up = (0.0, 0.0))

    // 3. Aggregate with hamming=false, missing=0.0, skip_average=false, num_classes=1
    std::vector<float> aggregated;
    aggregate_chunks(summed.data(), num_chunks, chunk_size, /*num_classes=*/1,
                     chunk_window, frame_window,
                     aggregated, total_frames,
                     /*skip_average=*/false, /*missing=*/0.0f);

    // 4. Round to nearest integer
    count.resize(total_frames);
    for (int i = 0; i < total_frames; i++) {
        count[i] = static_cast<int>(std::lround(aggregated[i]));
    }
}

void to_diarization(const float* segmentations, int num_chunks, int num_frames, int num_speakers,
                    const int* count, int total_frames,
                    const SlidingWindowParams& chunk_window,
                    const SlidingWindowParams& frame_window,
                    std::vector<float>& output) {
    // 1. Aggregate segmentations with skip_average=true (sum, not average)
    std::vector<float> activations;
    int agg_total_frames = 0;
    aggregate_chunks(segmentations, num_chunks, num_frames, num_speakers,
                     chunk_window, frame_window,
                     activations, agg_total_frames,
                     /*skip_average=*/true, /*missing=*/0.0f);

    // Use minimum of count's total_frames and aggregated total_frames
    const int out_frames = std::min(total_frames, agg_total_frames);

    // 2. For each frame, select top-N speakers based on count
    output.assign((size_t)out_frames * num_speakers, 0.0f);

    std::vector<int> sorted_speakers(num_speakers);

    for (int t = 0; t < out_frames; t++) {
        const int c = count[t];  // number of active speakers at this frame

        // argsort descending by activation value
        std::iota(sorted_speakers.begin(), sorted_speakers.end(), 0);
        std::sort(sorted_speakers.begin(), sorted_speakers.end(),
                  [&](int a, int b) {
                      return activations[t * num_speakers + a] > activations[t * num_speakers + b];
                  });

        // Set top-c speakers to 1.0
        const int select = std::min(c, num_speakers);
        for (int i = 0; i < select; i++) {
            output[t * num_speakers + sorted_speakers[i]] = 1.0f;
        }
    }
}

}  // namespace diarization
