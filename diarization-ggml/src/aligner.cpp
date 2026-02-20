#include "aligner.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

namespace {

std::string assign_speaker_for_segment(
    double seg_start,
    double seg_end,
    const std::vector<DiarizationResult::Segment>& sorted_segments) {

    if (sorted_segments.empty()) {
        return "UNKNOWN";
    }

    std::unordered_map<std::string, double> overlap_by_speaker;

    for (const auto& diar_seg : sorted_segments) {
        double diar_start = diar_seg.start;
        double diar_end = diar_seg.start + diar_seg.duration;

        if (diar_end <= seg_start) continue;
        if (diar_start >= seg_end) break;

        double overlap = std::min(diar_end, seg_end) - std::max(diar_start, seg_start);
        if (overlap > 0.0) {
            overlap_by_speaker[diar_seg.speaker] += overlap;
        }
    }

    if (!overlap_by_speaker.empty()) {
        std::string best_speaker;
        double best_overlap = -1.0;
        for (const auto& [speaker, overlap] : overlap_by_speaker) {
            if (overlap > best_overlap) {
                best_overlap = overlap;
                best_speaker = speaker;
            }
        }
        return best_speaker;
    }

    double seg_mid = 0.5 * (seg_start + seg_end);
    std::string nearest_speaker = sorted_segments.front().speaker;
    double best_dist = std::numeric_limits<double>::infinity();

    for (const auto& diar_seg : sorted_segments) {
        double diar_mid = diar_seg.start + 0.5 * diar_seg.duration;
        double dist = std::abs(seg_mid - diar_mid);
        if (dist < best_dist) {
            best_dist = dist;
            nearest_speaker = diar_seg.speaker;
        }
    }

    return nearest_speaker;
}

} // namespace

std::vector<AlignedSegment> align_segments(
    const std::vector<TranscribeSegment>& segments,
    const DiarizationResult& diarization) {

    if (segments.empty()) {
        return {};
    }

    std::vector<DiarizationResult::Segment> sorted_diar = diarization.segments;
    std::sort(sorted_diar.begin(), sorted_diar.end(), [](const auto& a, const auto& b) {
        return a.start < b.start;
    });

    std::vector<AlignedSegment> result;
    result.reserve(segments.size());

    for (const auto& seg : segments) {
        std::string speaker = assign_speaker_for_segment(seg.start, seg.end, sorted_diar);

        AlignedSegment aligned_seg;
        aligned_seg.speaker = speaker;
        aligned_seg.start = seg.start;
        aligned_seg.duration = seg.end - seg.start;
        aligned_seg.text = seg.text;
        result.push_back(std::move(aligned_seg));
    }

    return result;
}
