#pragma once

#include "diarization.h"
#include "transcriber_types.h"

#include <string>
#include <vector>

struct AlignedSegment {
    std::string speaker;
    double start;
    double duration;
    std::string text;
};

// WhisperX-style segment-level alignment:
// Each TranscribeSegment gets assigned a speaker based on maximum overlap
// with diarization segments. Consecutive same-speaker segments are merged.
std::vector<AlignedSegment> align_segments(
    const std::vector<TranscribeSegment>& segments,
    const DiarizationResult& diarization);
