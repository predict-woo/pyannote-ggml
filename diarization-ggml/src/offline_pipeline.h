#pragma once

#include "diarization.h"
#include "transcriber.h"
#include "aligner.h"

#include <functional>
#include <string>
#include <vector>

struct OfflinePipelineConfig {
    // Diarization model paths
    std::string seg_model_path;
    std::string emb_model_path;
    std::string plda_path;
    std::string coreml_path;       // embedding CoreML
    std::string seg_coreml_path;   // segmentation CoreML

    // Progress callback: phase (0=whisper, 1=diarization, 2=alignment), progress (0-100)
    // Optional — null check before calling.
    std::function<void(int phase, int progress)> progress_callback;

    // New segment callback: (start_seconds, end_seconds, text)
    // Called for each new Whisper segment as it's produced. Optional.
    std::function<void(double start, double end, const std::string& text)> new_segment_callback;

    // Whisper config
    TranscriberConfig transcriber;
};

struct OfflinePipelineResult {
    std::vector<AlignedSegment> segments;
    DiarizationResult diarization;  // raw diarization result
    std::vector<float> filtered_audio;  // silence-compressed audio (empty if no VAD)
    bool valid = false;
};

// Run offline transcription + diarization on entire audio buffer.
// 1. Runs whisper_full() on entire audio → TranscribeSegments
// 2. Runs offline diarization → DiarizationResult
// 3. Aligns Whisper segments with diarization → AlignedSegments
OfflinePipelineResult offline_transcribe(
    const OfflinePipelineConfig& config,
    const float* audio,
    int n_samples);

struct ModelCache;

// Run offline transcription + diarization using pre-loaded models from a ModelCache.
// The cache must remain valid for the duration of this call.
OfflinePipelineResult offline_transcribe_with_cache(
    const OfflinePipelineConfig& config,
    ModelCache* cache,
    const float* audio,
    int n_samples);
