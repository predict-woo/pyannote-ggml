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

std::vector<AlignedSegment> align_words(
    const std::vector<TranscribeToken>& tokens,
    const DiarizationResult& diarization);
