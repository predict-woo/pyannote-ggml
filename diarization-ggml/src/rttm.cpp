#include "rttm.h"

#include <cstdio>

namespace diarization {

// RTTM format: SPEAKER <uri> 1 <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>
void write_rttm(const std::vector<RTTMSegment>& segments, const std::string& uri,
                const std::string& output_path) {
    FILE* f = fopen(output_path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "Error: cannot open output file '%s'\n", output_path.c_str());
        return;
    }

    for (const auto& seg : segments) {
        fprintf(f, "SPEAKER %s 1 %.3f %.3f <NA> <NA> %s <NA> <NA>\n",
                uri.c_str(), seg.start, seg.duration, seg.speaker.c_str());
    }

    fclose(f);
}

void write_rttm_stdout(const std::vector<RTTMSegment>& segments, const std::string& uri) {
    for (const auto& seg : segments) {
        printf("SPEAKER %s 1 %.3f %.3f <NA> <NA> %s <NA> <NA>\n",
               uri.c_str(), seg.start, seg.duration, seg.speaker.c_str());
    }
}

}  // namespace diarization
