#pragma once
#include <string>
#include <vector>

namespace diarization {

struct RTTMSegment {
    double start;
    double duration;
    std::string speaker;
};

void write_rttm(const std::vector<RTTMSegment>& segments, const std::string& uri,
                const std::string& output_path);

void write_rttm_stdout(const std::vector<RTTMSegment>& segments, const std::string& uri);

}  // namespace diarization
