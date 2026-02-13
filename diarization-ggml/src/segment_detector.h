#pragma once

#include "streaming.h"

#include <vector>

struct SegmentDetectorResult {
    std::vector<double> segment_end_times;
    bool flush_signal;
};

struct SegmentDetector;

SegmentDetector* segment_detector_init();
SegmentDetectorResult segment_detector_push(SegmentDetector* sd, const VADChunk& chunk);
SegmentDetectorResult segment_detector_flush(SegmentDetector* sd);
void segment_detector_free(SegmentDetector* sd);
