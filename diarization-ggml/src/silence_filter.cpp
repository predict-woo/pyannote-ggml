#include "silence_filter.h"

#include "whisper.h"

#include <cstdint>
#include <deque>
#include <vector>

namespace {

constexpr int kSampleRate = 16000;
constexpr int kVadWindowSize = 512;
constexpr int64_t kSilencePassthrough = 16000;
constexpr int64_t kEndSilenceCapacity = 16000;
constexpr int64_t kFlushThreshold = 80000;

struct SilenceFilterImpl {
    whisper_vad_context* vad_ctx = nullptr;
    float threshold = 0.5f;

    std::vector<float> pending;
    int64_t silence_started = -1;
    std::deque<float> end_silence;
    int64_t consecutive_discarded_frames = 0;
    int64_t total_frames_processed = 0;
};

float detect_probability(SilenceFilterImpl* sf, const float* samples, int n) {
    if (sf->vad_ctx == nullptr) {
        for (int i = 0; i < n; ++i) {
            if (samples[i] != 0.0f) {
                return 1.0f;
            }
        }
        return 0.0f;
    }

    return whisper_vad_detect_speech_single_frame(sf->vad_ctx, samples, n);
}

void append_end_silence_to_output(SilenceFilterImpl* sf, std::vector<float>& out) {
    if (sf->end_silence.empty()) {
        return;
    }

    out.insert(out.end(), sf->end_silence.begin(), sf->end_silence.end());
    sf->end_silence.clear();
}

void process_silent_sample(SilenceFilterImpl* sf, float sample, std::vector<float>& out, bool& flush_signal) {
    if (sf->silence_started == -1) {
        sf->silence_started = sf->total_frames_processed;
    }

    const int64_t elapsed = sf->total_frames_processed - sf->silence_started;
    if (elapsed <= kSilencePassthrough) {
        out.push_back(sample);
    } else {
        if (static_cast<int64_t>(sf->end_silence.size()) >= kEndSilenceCapacity) {
            sf->end_silence.pop_front();
            sf->consecutive_discarded_frames += 1;
            if (sf->consecutive_discarded_frames >= kFlushThreshold) {
                flush_signal = true;
                sf->consecutive_discarded_frames = 0;
            }
        }
        sf->end_silence.push_back(sample);
    }

    sf->total_frames_processed += 1;
}

void process_speech_block(SilenceFilterImpl* sf, const float* samples, int n, std::vector<float>& out) {
    if (sf->silence_started != -1) {
        sf->silence_started = -1;
        sf->consecutive_discarded_frames = 0;
        append_end_silence_to_output(sf, out);
    }

    out.insert(out.end(), samples, samples + n);
    sf->total_frames_processed += n;
}

}

struct SilenceFilter {
    SilenceFilterImpl impl;
};

SilenceFilter* silence_filter_init(struct whisper_vad_context* vad_ctx, float threshold) {
    auto* sf = new SilenceFilter();
    sf->impl.vad_ctx = vad_ctx;
    sf->impl.threshold = threshold;
    sf->impl.pending.reserve(kVadWindowSize * 4);

    if (sf->impl.vad_ctx != nullptr) {
        const int n_window = whisper_vad_n_window(sf->impl.vad_ctx);
        if (n_window > 0 && n_window != kVadWindowSize) {
            sf->impl.pending.reserve(n_window * 4);
        }
        whisper_vad_reset_state(sf->impl.vad_ctx);
    }

    return sf;
}

SilenceFilterResult silence_filter_push(SilenceFilter* sf, const float* samples, int n) {
    SilenceFilterResult result;
    result.flush_signal = false;

    if (sf == nullptr || samples == nullptr || n <= 0) {
        return result;
    }

    auto& impl = sf->impl;
    impl.pending.insert(impl.pending.end(), samples, samples + n);

    while (static_cast<int>(impl.pending.size()) >= kVadWindowSize) {
        const float* frame = impl.pending.data();
        const float prob = detect_probability(&impl, frame, kVadWindowSize);
        const bool is_speech = prob >= impl.threshold;

        if (is_speech) {
            process_speech_block(&impl, frame, kVadWindowSize, result.audio);
        } else {
            for (int i = 0; i < kVadWindowSize; ++i) {
                process_silent_sample(&impl, frame[i], result.audio, result.flush_signal);
            }
        }

        impl.pending.erase(impl.pending.begin(), impl.pending.begin() + kVadWindowSize);
    }

    return result;
}

SilenceFilterResult silence_filter_flush(SilenceFilter* sf) {
    SilenceFilterResult result;
    result.flush_signal = true;

    if (sf == nullptr) {
        return result;
    }

    auto& impl = sf->impl;
    if (!impl.pending.empty()) {
        const float prob = detect_probability(&impl, impl.pending.data(), static_cast<int>(impl.pending.size()));
        const bool is_speech = prob >= impl.threshold;

        if (is_speech) {
            process_speech_block(&impl, impl.pending.data(), static_cast<int>(impl.pending.size()), result.audio);
        } else {
            for (float sample : impl.pending) {
                process_silent_sample(&impl, sample, result.audio, result.flush_signal);
            }
        }

        impl.pending.clear();
    }

    append_end_silence_to_output(&impl, result.audio);
    impl.consecutive_discarded_frames = 0;

    return result;
}

void silence_filter_free(SilenceFilter* sf) {
    delete sf;
}
