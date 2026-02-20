#pragma once
#include <string>
#include <vector>

struct TranscribeSegment {
    double start;  // segment start (absolute seconds)
    double end;    // segment end (absolute seconds)
    std::string text;  // full segment text from Whisper
};

struct TranscribeResult {
    std::vector<TranscribeSegment> segments;
    bool valid = false;
};
