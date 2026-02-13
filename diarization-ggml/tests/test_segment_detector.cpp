#include "segment_detector.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {
constexpr double FRAME_STEP = 0.016875;

bool nearly_equal(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) <= eps;
}

VADChunk make_chunk(int chunk_index, int start_frame, const std::vector<float>& vad) {
    VADChunk chunk;
    chunk.chunk_index = chunk_index;
    chunk.start_frame = start_frame;
    chunk.num_frames = static_cast<int>(vad.size());
    chunk.vad = vad;
    return chunk;
}

int require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        return 1;
    }
    return 0;
}
}

int main() {
    {
        SegmentDetector* sd = segment_detector_init();
        auto result = segment_detector_push(sd, make_chunk(0, 0, std::vector<float>(60, 1.0f)));
        if (require(result.segment_end_times.empty(), "Test 1 expected no segment end")) return 1;
        if (require(!result.flush_signal, "Test 1 expected flush_signal=false")) return 1;
        segment_detector_free(sd);
    }

    {
        SegmentDetector* sd = segment_detector_init();
        std::vector<float> vad(60, 0.0f);
        for (int i = 0; i < 30; ++i) vad[i] = 1.0f;
        auto result = segment_detector_push(sd, make_chunk(0, 0, vad));
        if (require(result.segment_end_times.size() == 1, "Test 2 expected one segment end")) return 1;
        if (require(nearly_equal(result.segment_end_times[0], 30 * FRAME_STEP), "Test 2 timestamp mismatch")) return 1;
        segment_detector_free(sd);
    }

    {
        SegmentDetector* sd = segment_detector_init();
        auto first = segment_detector_push(sd, make_chunk(0, 0, std::vector<float>(10, 1.0f)));
        if (require(first.segment_end_times.empty(), "Test 3 first chunk expected no segment end")) return 1;
        auto second = segment_detector_push(sd, make_chunk(1, 10, std::vector<float>(10, 0.0f)));
        if (require(second.segment_end_times.size() == 1, "Test 3 expected one cross-boundary segment end")) return 1;
        if (require(nearly_equal(second.segment_end_times[0], 10 * FRAME_STEP), "Test 3 boundary timestamp mismatch")) return 1;
        segment_detector_free(sd);
    }

    {
        SegmentDetector* sd = segment_detector_init();
        auto result = segment_detector_push(sd, make_chunk(0, 0, std::vector<float>(60, 0.0f)));
        if (require(result.segment_end_times.empty(), "Test 4 expected no segment end")) return 1;
        segment_detector_free(sd);
    }

    {
        SegmentDetector* sd = segment_detector_init();
        auto a = segment_detector_push(sd, make_chunk(0, 0, std::vector<float>(7, 1.0f)));
        if (require(a.segment_end_times.empty(), "Test 5 chunk A expected no segment end")) return 1;

        auto b = segment_detector_push(sd, make_chunk(1, 7, std::vector<float>(13, 1.0f)));
        if (require(b.segment_end_times.empty(), "Test 5 chunk B expected no segment end")) return 1;

        std::vector<float> vad_c(12, 0.0f);
        for (int i = 0; i < 6; ++i) vad_c[i] = 1.0f;
        auto c = segment_detector_push(sd, make_chunk(2, 20, vad_c));
        if (require(c.segment_end_times.size() == 1, "Test 5 chunk C expected one segment end")) return 1;
        if (require(nearly_equal(c.segment_end_times[0], (20 + 6) * FRAME_STEP), "Test 5 accumulated timestamp mismatch")) return 1;
        segment_detector_free(sd);
    }

    {
        SegmentDetector* sd = segment_detector_init();
        auto flush = segment_detector_flush(sd);
        if (require(flush.segment_end_times.empty(), "Test 6 flush expected no timestamp")) return 1;
        if (require(flush.flush_signal, "Test 6 flush expected flush_signal=true")) return 1;
        segment_detector_free(sd);
    }

    std::fprintf(stderr, "PASS: test_segment_detector\n");
    return 0;
}
