#include "aligner.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

int require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        return 1;
    }
    return 0;
}

DiarizationResult make_diarization(const std::vector<DiarizationResult::Segment>& segments) {
    DiarizationResult diarization;
    diarization.segments = segments;
    return diarization;
}

TranscribeSegment make_segment(const std::string& text, double start, double end) {
    TranscribeSegment seg;
    seg.start = start;
    seg.end = end;
    seg.text = text;
    seg.words.push_back({text, start, end});
    return seg;
}

}

int main() {
    {
        DiarizationResult diarization = make_diarization({
            {0.0, 5.0, "SPEAKER_00"},
            {5.0, 5.0, "SPEAKER_01"},
        });
        std::vector<TranscribeSegment> segments = {
            make_segment("hello", 1.0, 2.0),
            make_segment("world", 3.0, 4.0),
            make_segment("foo", 6.0, 7.0),
            make_segment("bar", 8.0, 9.0),
        };

        auto aligned = align_segments(segments, diarization);
        if (require(aligned.size() == 2, "Test 1 expected 2 aligned segments")) return 1;
        if (require(aligned[0].speaker == "SPEAKER_00", "Test 1 first segment speaker")) return 1;
        if (require(aligned[1].speaker == "SPEAKER_01", "Test 1 second segment speaker")) return 1;
        if (require(aligned[0].words.size() == 2, "Test 1 first segment word count")) return 1;
        if (require(aligned[1].words.size() == 2, "Test 1 second segment word count")) return 1;
        if (require(aligned[0].words[0].speaker == "SPEAKER_00", "Test 1 hello speaker")) return 1;
        if (require(aligned[0].words[1].speaker == "SPEAKER_00", "Test 1 world speaker")) return 1;
        if (require(aligned[1].words[0].speaker == "SPEAKER_01", "Test 1 foo speaker")) return 1;
        if (require(aligned[1].words[1].speaker == "SPEAKER_01", "Test 1 bar speaker")) return 1;
    }

    {
        DiarizationResult diarization = make_diarization({
            {0.0, 5.0, "SPEAKER_00"},
            {5.0, 5.0, "SPEAKER_01"},
        });
        std::vector<TranscribeSegment> segments = {
            make_segment("overlap", 4.5, 5.5),
            make_segment("mostly01", 4.8, 5.5),
        };

        auto aligned = align_segments(segments, diarization);
        if (require(aligned.size() >= 1, "Test 2 expected at least one aligned segment")) return 1;

        const std::string overlap_speaker = aligned[0].words[0].speaker;
        const bool overlap_ok = overlap_speaker == "SPEAKER_00" || overlap_speaker == "SPEAKER_01";
        if (require(overlap_ok, "Test 2 overlap speaker must be 00 or 01")) return 1;

        const std::string mostly01_speaker = (aligned.size() == 1) ? aligned[0].words[1].speaker : aligned[1].words[0].speaker;
        if (require(mostly01_speaker == "SPEAKER_01", "Test 2 mostly01 should be SPEAKER_01")) return 1;
    }

    {
        DiarizationResult diarization = make_diarization({
            {0.0, 3.0, "SPEAKER_00"},
            {7.0, 3.0, "SPEAKER_01"},
        });
        std::vector<TranscribeSegment> segments = {
            make_segment("gap", 4.5, 5.5),
        };

        auto aligned = align_segments(segments, diarization);
        if (require(aligned.size() == 1, "Test 3 expected one aligned segment")) return 1;
        if (require(aligned[0].words.size() == 1, "Test 3 expected one word")) return 1;
        if (require(aligned[0].words[0].speaker == "SPEAKER_00", "Test 3 gap word nearest speaker")) return 1;
    }

    {
        DiarizationResult diarization;
        std::vector<TranscribeSegment> empty_segments;
        auto aligned_empty = align_segments(empty_segments, diarization);
        if (require(aligned_empty.empty(), "Test 4 empty segments should produce empty output")) return 1;

        std::vector<TranscribeSegment> segments = { make_segment("lonely", 1.0, 2.0) };
        auto aligned_empty_diar = align_segments(segments, diarization);
        if (require(aligned_empty_diar.size() == 1, "Test 4 empty diarization should still produce one segment")) return 1;
        if (require(aligned_empty_diar[0].speaker == "UNKNOWN", "Test 4 empty diarization speaker should be UNKNOWN")) return 1;
    }

    {
        DiarizationResult diarization = make_diarization({
            {0.0, 6.0, "SPEAKER_00"},
            {4.0, 6.0, "SPEAKER_01"},
        });
        std::vector<TranscribeSegment> segments = {
            make_segment("in_overlap", 4.5, 5.5),
            make_segment("more_00", 2.0, 3.0),
        };

        auto aligned = align_segments(segments, diarization);
        if (require(aligned.size() == 2, "Test 5 expected two aligned segments")) return 1;
        const std::string first_speaker = aligned[0].words[0].speaker;
        const bool first_ok = first_speaker == "SPEAKER_00" || first_speaker == "SPEAKER_01";
        if (require(first_ok, "Test 5 overlap token speaker must be 00 or 01")) return 1;
        if (require(aligned[1].words[0].speaker == "SPEAKER_00", "Test 5 more_00 should be SPEAKER_00")) return 1;
    }

    std::fprintf(stderr, "PASS: test_aligner\n");
    return 0;
}
