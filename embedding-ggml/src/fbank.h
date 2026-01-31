#ifndef EMBEDDING_GGML_FBANK_H
#define EMBEDDING_GGML_FBANK_H

#include <vector>

namespace embedding {

struct fbank_result {
    std::vector<float> data;  // Flattened T×80 matrix (row-major)
    int num_frames;           // T
    int num_bins;             // 80
};

/**
 * Compute 80-dim mel fbank features from raw audio.
 *
 * Parameters match WeSpeaker/pyannote embedding model requirements:
 * - 80 mel bins
 * - 25ms frame length, 10ms frame shift
 * - Hamming window
 * - No dithering (dither=0.0)
 * - snip_edges=True (Kaldi default)
 * - sample_rate=16000
 * - Waveform must be pre-scaled by 32768 (int16 range) before calling
 * - After fbank, subtract global mean per utterance (CMN)
 */
fbank_result compute_fbank(const float* audio, int num_samples, int sample_rate = 16000);

} // namespace embedding

#endif // EMBEDDING_GGML_FBANK_H
