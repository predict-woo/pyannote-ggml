#pragma once

#include "diarization.h"
#include "transcriber.h"
#include "aligner.h"

#include <string>
#include <vector>

struct OfflinePipelineConfig {
    // Diarization model paths
    std::string seg_model_path;
    std::string emb_model_path;
    std::string plda_path;
    std::string coreml_path;       // embedding CoreML
    std::string seg_coreml_path;   // segmentation CoreML

    // Whisper config
    TranscriberConfig transcriber;
};

struct OfflinePipelineResult {
    std::vector<AlignedSegment> segments;
    DiarizationResult diarization;  // raw diarization result
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
