#pragma once

#include "diarization.h"
#include "transcriber_types.h"

#include <string>
#include <vector>

struct AlignedWord {
    std::string text;
    double start;
    double end;
    std::string speaker;
};

struct AlignedSegment {
    std::string speaker;
    double start;
    double duration;
    std::vector<AlignedWord> words;
};

// WhisperX-style segment-level alignment:
// Each TranscribeSegment gets assigned a speaker based on maximum overlap
// with diarization segments. All words in a segment inherit that speaker.
// Consecutive same-speaker segments are merged.
std::vector<AlignedSegment> align_segments(
    const std::vector<TranscribeSegment>& segments,
    const DiarizationResult& diarization);
