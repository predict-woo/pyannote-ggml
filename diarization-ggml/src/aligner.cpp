#include "aligner.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

std::string assign_speaker_for_token(
    const TranscribeToken& token,
    const std::vector<DiarizationResult::Segment>& sorted_segments) {
    if (sorted_segments.empty()) {
        return "UNKNOWN";
    }

    const auto right_it = std::lower_bound(
        sorted_segments.begin(),
        sorted_segments.end(),
        token.end,
        [](const DiarizationResult::Segment& seg, double token_end) {
            return seg.start < token_end;
        });

    std::unordered_map<std::string, double> overlap_by_speaker;

    for (auto it = sorted_segments.begin(); it != right_it; ++it) {
        const double seg_start = it->start;
        const double seg_end = it->start + it->duration;
        const double overlap = std::min(seg_end, token.end) - std::max(seg_start, token.start);
        if (overlap >= 0.0) {
            overlap_by_speaker[it->speaker] += overlap;
        }
    }

    if (!overlap_by_speaker.empty()) {
        std::string best_speaker;
        double best_overlap = -std::numeric_limits<double>::infinity();
        for (const auto& [speaker, overlap] : overlap_by_speaker) {
            if (overlap > best_overlap) {
                best_overlap = overlap;
                best_speaker = speaker;
            }
        }
        return best_speaker;
    }

    const double token_mid = 0.5 * (token.start + token.end);
    std::string nearest_speaker = sorted_segments.front().speaker;
    double best_dist = std::numeric_limits<double>::infinity();

    for (const auto& seg : sorted_segments) {
        const double seg_mid = seg.start + 0.5 * seg.duration;
        const double dist = std::abs(token_mid - seg_mid);
        if (dist < best_dist) {
            best_dist = dist;
            nearest_speaker = seg.speaker;
        }
    }

    return nearest_speaker;
}

}

std::vector<AlignedSegment> align_words(
    const std::vector<TranscribeToken>& tokens,
    const DiarizationResult& diarization) {
    if (tokens.empty()) {
        return {};
    }

    std::vector<DiarizationResult::Segment> sorted_segments = diarization.segments;
    std::sort(sorted_segments.begin(), sorted_segments.end(), [](const auto& a, const auto& b) {
        if (a.start == b.start) {
            return a.duration < b.duration;
        }
        return a.start < b.start;
    });

    std::vector<AlignedSegment> aligned_segments;
    aligned_segments.reserve(tokens.size());

    for (const auto& token : tokens) {
        AlignedWord word{token.text, token.start, token.end, assign_speaker_for_token(token, sorted_segments)};

        if (aligned_segments.empty() || aligned_segments.back().speaker != word.speaker) {
            AlignedSegment seg;
            seg.speaker = word.speaker;
            seg.start = word.start;
            seg.duration = word.end - word.start;
            seg.words.push_back(std::move(word));
            aligned_segments.push_back(std::move(seg));
            continue;
        }

        AlignedSegment& current = aligned_segments.back();
        current.words.push_back(std::move(word));
        current.duration = current.words.back().end - current.start;
    }

    return aligned_segments;
}
