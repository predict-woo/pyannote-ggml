#include "segment_detector.h"

namespace {
constexpr double FRAME_STEP = 0.016875;
constexpr float VAD_THRESHOLD = 0.5f;
}

struct SegmentDetector {
    int current_prediction_frame;
    bool last_frame_was_speech;
};

SegmentDetector* segment_detector_init() {
    auto* sd = new SegmentDetector();
    sd->current_prediction_frame = 0;
    sd->last_frame_was_speech = false;
    return sd;
}

SegmentDetectorResult segment_detector_push(SegmentDetector* sd, const VADChunk& chunk) {
    SegmentDetectorResult result{{}, false};
    if (!sd) {
        return result;
    }

    if (chunk.num_frames <= 0 || chunk.vad.empty()) {
        return result;
    }

    if (sd->last_frame_was_speech && chunk.vad[0] < VAD_THRESHOLD) {
        result.segment_end_times.push_back(sd->current_prediction_frame * FRAME_STEP);
    }

    for (int i = 0; i + 1 < chunk.num_frames; ++i) {
        if (chunk.vad[i] >= VAD_THRESHOLD && chunk.vad[i + 1] < VAD_THRESHOLD) {
            result.segment_end_times.push_back((sd->current_prediction_frame + i + 1) * FRAME_STEP);
            break;
        }
    }

    sd->last_frame_was_speech = chunk.vad[chunk.num_frames - 1] >= VAD_THRESHOLD;
    sd->current_prediction_frame += chunk.num_frames;

    return result;
}

SegmentDetectorResult segment_detector_flush(SegmentDetector* sd) {
    (void) sd;
    return SegmentDetectorResult{{}, true};
}

void segment_detector_free(SegmentDetector* sd) {
    delete sd;
}
