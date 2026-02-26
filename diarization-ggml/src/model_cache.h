#pragma once

#include "plda.h"         // for diarization::PLDAModel
#include "transcriber.h"  // for TranscriberConfig

#include <string>

// Forward declarations
struct whisper_context;
struct whisper_vad_context;
struct segmentation_coreml_context;
struct embedding_coreml_context;

struct ModelCacheConfig {
    // Diarization models
    std::string seg_model_path;
    std::string emb_model_path;
    std::string plda_path;
    std::string seg_coreml_path;
    std::string coreml_path;  // embedding CoreML

    // Whisper
    TranscriberConfig transcriber;  // contains whisper_model_path + context params

    // VAD (optional)
    const char* vad_model_path = nullptr;
};

struct ModelCache {
    // Diarization models (CoreML)
    segmentation_coreml_context* seg_coreml_ctx = nullptr;
    embedding_coreml_context* emb_coreml_ctx = nullptr;
    diarization::PLDAModel plda;
    bool plda_loaded = false;

    // Whisper
    whisper_context* whisper_ctx = nullptr;

    // VAD (optional)
    whisper_vad_context* vad_ctx = nullptr;
};

// Load all models. Returns nullptr on failure.
ModelCache* model_cache_load(const ModelCacheConfig& config);

// Free all models.
void model_cache_free(ModelCache* cache);
