#include "../src/silence_filter.h"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace {

constexpr int kSampleRate = 16000;

struct FilterRun {
    std::vector<float> output;
    std::vector<bool> vad_predictions;
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
        run.vad_predictions.insert(run.vad_predictions.end(),
                                   result.vad_predictions.begin(), result.vad_predictions.end());

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
    run.vad_predictions.insert(run.vad_predictions.end(),
                               tail.vad_predictions.begin(), tail.vad_predictions.end());
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

    // --- vad_predictions tests ---

    {
        constexpr size_t N = 156 * 512;
        std::vector<float> speech(N, 0.5f);
        FilterRun run = run_filter(speech, 512);
        ok &= expect(run.vad_predictions.size() == N / 512,
                     "vad_predictions count should equal num_samples / 512");
        bool all_true = true;
        for (bool v : run.vad_predictions) { if (!v) all_true = false; }
        ok &= expect(all_true, "all-speech input should produce all-true vad_predictions");
    }

    {
        constexpr size_t N = 156 * 512;
        std::vector<float> silence(N, 0.0f);
        FilterRun run = run_filter(silence, 512);
        ok &= expect(run.vad_predictions.size() == N / 512,
                     "vad_predictions count for silence should equal num_samples / 512");
        bool all_false = true;
        for (bool v : run.vad_predictions) { if (v) all_false = false; }
        ok &= expect(all_false, "all-silence input should produce all-false vad_predictions");
    }

    {
        constexpr size_t N = 62 * 512;
        std::vector<float> speech(N, 0.5f);
        FilterRun run = run_filter(speech, 256);
        ok &= expect(run.vad_predictions.size() == N / 512,
                     "sub-window pushes should still produce correct prediction count");
    }

    {
        SilenceFilter* sf = silence_filter_init(nullptr, 0.5f);
        std::vector<float> partial(300, 0.5f);
        SilenceFilterResult r = silence_filter_push(sf, partial.data(), (int)partial.size());
        ok &= expect(r.vad_predictions.empty(),
                     "pushing < 512 samples should produce no predictions");

        std::vector<float> rest(212, 0.5f);
        SilenceFilterResult r2 = silence_filter_push(sf, rest.data(), (int)rest.size());
        ok &= expect(r2.vad_predictions.size() == 1,
                     "accumulating to 512 should produce exactly 1 prediction");
        ok &= expect(r2.vad_predictions[0] == true,
                     "speech samples should produce true prediction");
        silence_filter_free(sf);
    }

    {
        constexpr size_t WINDOWS = 62;
        std::vector<float> mixed;
        mixed.insert(mixed.end(), WINDOWS * 512, 0.5f);
        mixed.insert(mixed.end(), WINDOWS * 512, 0.0f);
        FilterRun run = run_filter(mixed, 512);

        ok &= expect(run.vad_predictions.size() == WINDOWS * 2,
                     "mixed input prediction count should match total windows");

        bool speech_ok = true;
        for (size_t i = 0; i < WINDOWS; ++i) {
            if (!run.vad_predictions[i]) speech_ok = false;
        }
        ok &= expect(speech_ok, "speech portion should have true predictions");

        bool silence_ok = true;
        for (size_t i = WINDOWS; i < run.vad_predictions.size(); ++i) {
            if (run.vad_predictions[i]) silence_ok = false;
        }
        ok &= expect(silence_ok, "silence portion should have false predictions");
    }

    if (!ok) {
        return 1;
    }

    std::fprintf(stderr, "PASSED: silence filter tests\n");
    return 0;
}
