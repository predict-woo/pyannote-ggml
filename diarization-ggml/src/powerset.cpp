#include "powerset.h"

namespace diarization {

// Powerset-to-multilabel mapping for num_classes=3, max_set_size=2
// Rows: {∅}, {0}, {1}, {2}, {0,1}, {0,2}, {1,2}  →  columns: speaker 0, 1, 2
static constexpr float MAPPING[7][3] = {
    {0.0f, 0.0f, 0.0f},
    {1.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f},
    {0.0f, 0.0f, 1.0f},
    {1.0f, 1.0f, 0.0f},
    {1.0f, 0.0f, 1.0f},
    {0.0f, 1.0f, 1.0f},
};

static constexpr int NUM_POWERSET_CLASSES = 7;
static constexpr int NUM_SPEAKERS = 3;

void powerset_to_multilabel(const float* logits, int num_chunks, int num_frames,
                            float* output) {
    for (int c = 0; c < num_chunks; ++c) {
        for (int f = 0; f < num_frames; ++f) {
            const float* frame_logits = logits + (c * num_frames + f) * NUM_POWERSET_CLASSES;

            int best_class = 0;
            float best_val = frame_logits[0];
            for (int k = 1; k < NUM_POWERSET_CLASSES; ++k) {
                if (frame_logits[k] > best_val) {
                    best_val = frame_logits[k];
                    best_class = k;
                }
            }

            float* frame_output = output + (c * num_frames + f) * NUM_SPEAKERS;
            frame_output[0] = MAPPING[best_class][0];
            frame_output[1] = MAPPING[best_class][1];
            frame_output[2] = MAPPING[best_class][2];
        }
    }
}

}  // namespace diarization
