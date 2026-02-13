#pragma once
#include <string>
#include <vector>

struct TranscribeToken {
    std::string text;
    double start;  // absolute filtered-timeline seconds
    double end;    // absolute filtered-timeline seconds
};

struct TranscribeResult {
    std::vector<TranscribeToken> tokens;
    bool valid = false;  // true when result is ready
};
