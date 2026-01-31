#pragma once
#include <vector>

namespace diarization {

// Convert powerset logits (num_chunks * 589 * 7) to multilabel (num_chunks * 589 * 3)
void powerset_to_multilabel(const float* logits, int num_chunks, int num_frames,
                            float* output);

}  // namespace diarization
