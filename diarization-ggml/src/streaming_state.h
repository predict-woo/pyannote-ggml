#pragma once
#include <vector>
#include "plda.h"  // for PLDAModel

// Forward declarations for CoreML contexts
struct embedding_coreml_context;
struct segmentation_coreml_context;

struct StreamingConfig {
    // Base configuration (model paths)
    std::string seg_model_path;
    std::string emb_model_path;
    std::string plda_path;
    std::string coreml_path;       // embedding CoreML
    std::string seg_coreml_path;   // segmentation CoreML
    
};

struct StreamingState {
    // Configuration
    StreamingConfig config;
    
    // Accumulated audio buffer (grows over time)
    std::vector<float> audio_buffer;
    
    // Accumulated embeddings [N × 256]
    std::vector<float> embeddings;
    
    // Tracking which chunk/speaker each embedding came from
    std::vector<int> chunk_idx;
    std::vector<int> local_speaker_idx;
    
    // Binarized segmentation [num_chunks × 589 × 3]
    std::vector<float> binarized;
    
    // Provisional clustering state
    std::vector<float> centroids;      // [K × 256] provisional centroids
    std::vector<int> centroid_counts;  // How many embeddings contributed to each centroid
    int num_provisional_speakers = 0;
    
    // Global labels from last recluster
    std::vector<int> global_labels;    // [N] global speaker for each embedding
    int num_speakers = 0;
    
    // Bookkeeping
    int chunks_processed = 0;
    int last_recluster_chunk = 0;
    double audio_time_processed = 0.0;
    bool finalized = false;
    int samples_trimmed = 0;  // absolute sample offset trimmed from audio_buffer front
    
    // Model contexts (owned, freed in streaming_free)
    struct segmentation_coreml_context* seg_coreml_ctx = nullptr;
    struct embedding_coreml_context* emb_coreml_ctx = nullptr;
    
    // PLDA model
    diarization::PLDAModel plda;
};
