#pragma once
#include <string>
#include <vector>

// Keep TranscribeToken for backward compat (used nowhere after refactor, but harmless)
struct TranscribeToken {
    std::string text;
    double start;
    double end;
};

struct TranscribeWord {
    std::string text;
    double start;  // absolute seconds in filtered timeline
    double end;    // absolute seconds in filtered timeline
};

struct TranscribeSegment {
    double start;  // segment start (absolute seconds)
    double end;    // segment end (absolute seconds)
    std::string text;  // concatenated text of all words
    std::vector<TranscribeWord> words;
};

struct TranscribeResult {
    std::vector<TranscribeSegment> segments;
    bool valid = false;
};
