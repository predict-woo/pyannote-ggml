#include "../src/silence_filter.h"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace {

constexpr int kSampleRate = 16000;

struct FilterRun {
    std::vector<float> output;
    int flush_count_from_push = 0;
    int64_t first_flush_input_frame = -1;
};

FilterRun run_filter(const std::vector<float>& input, int chunk_size) {
    SilenceFilter* sf = silence_filter_init(nullptr, 0.5f);

    FilterRun run;
    int64_t pushed_frames = 0;

    for (size_t offset = 0; offset < input.size(); offset += static_cast<size_t>(chunk_size)) {
        const int n = static_cast<int>(std::min(static_cast<size_t>(chunk_size), input.size() - offset));
        SilenceFilterResult result = silence_filter_push(sf, input.data() + offset, n);
        run.output.insert(run.output.end(), result.audio.begin(), result.audio.end());

        pushed_frames += n;
        if (result.flush_signal) {
            run.flush_count_from_push += 1;
            if (run.first_flush_input_frame < 0) {
                run.first_flush_input_frame = pushed_frames;
            }
        }
    }

    SilenceFilterResult tail = silence_filter_flush(sf);
    run.output.insert(run.output.end(), tail.audio.begin(), tail.audio.end());
    silence_filter_free(sf);

    return run;
}

bool expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        return false;
    }
    return true;
}

}

int main() {
    bool ok = true;

    {
        std::vector<float> speech(10 * kSampleRate, 0.5f);
        FilterRun run = run_filter(speech, 700);
        ok &= expect(run.output.size() == speech.size(), "all-speech should keep all samples");
    }

    {
        std::vector<float> mixed;
        mixed.insert(mixed.end(), 10 * kSampleRate, 0.5f);
        mixed.insert(mixed.end(), 8 * kSampleRate, 0.0f);
        mixed.insert(mixed.end(), 5 * kSampleRate, 0.5f);

        FilterRun run = run_filter(mixed, 731);
        const double out_sec = static_cast<double>(run.output.size()) / static_cast<double>(kSampleRate);
        ok &= expect(out_sec > 16.7 && out_sec < 17.3, "8s silence should compress to about 2s (total about 17s)");
    }

    {
        std::vector<float> silence(10 * kSampleRate, 0.0f);
        FilterRun run = run_filter(silence, 512);

        ok &= expect(run.flush_count_from_push >= 1, "flush signal should trigger during long discarded silence");
        const double first_flush_sec = static_cast<double>(run.first_flush_input_frame) / static_cast<double>(kSampleRate);
        ok &= expect(first_flush_sec > 6.5 && first_flush_sec < 7.5,
                     "flush should fire after about 5s of discarded frames");
    }

    {
        std::vector<float> alternating;
        for (int i = 0; i < 10; ++i) {
            alternating.insert(alternating.end(), kSampleRate / 2, 0.5f);
            alternating.insert(alternating.end(), kSampleRate / 2, 0.0f);
        }

        FilterRun run = run_filter(alternating, 777);
        ok &= expect(run.output.size() == alternating.size(), "alternating 0.5s speech/silence should pass through");
    }

    if (!ok) {
        return 1;
    }

    std::fprintf(stderr, "PASSED: silence filter tests\n");
    return 0;
}
