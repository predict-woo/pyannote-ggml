#include "streaming.h"
#include "plda.h"

#ifdef SEGMENTATION_USE_COREML
#include "segmentation_coreml_bridge.h"
#endif
#ifdef EMBEDDING_USE_COREML
#include "coreml_bridge.h"
#endif

#include <cstdio>

StreamingState* streaming_init(const StreamingConfig& config) {
    StreamingState* state = new StreamingState();
    state->config = config;
    
    // Load segmentation CoreML model
#ifdef SEGMENTATION_USE_COREML
    if (!config.seg_coreml_path.empty()) {
        state->seg_coreml_ctx = segmentation_coreml_init(config.seg_coreml_path.c_str());
        if (!state->seg_coreml_ctx) {
            fprintf(stderr, "Error: failed to load CoreML segmentation model '%s'\n",
                    config.seg_coreml_path.c_str());
            delete state;
            return nullptr;
        }
    }
#endif
    
    // Load embedding CoreML model
#ifdef EMBEDDING_USE_COREML
    if (!config.coreml_path.empty()) {
        state->emb_coreml_ctx = embedding_coreml_init(config.coreml_path.c_str());
        if (!state->emb_coreml_ctx) {
            fprintf(stderr, "Error: failed to load CoreML embedding model '%s'\n",
                    config.coreml_path.c_str());
#ifdef SEGMENTATION_USE_COREML
            if (state->seg_coreml_ctx) segmentation_coreml_free(state->seg_coreml_ctx);
#endif
            delete state;
            return nullptr;
        }
    }
#endif
    
    // Load PLDA model
    if (!config.plda_path.empty()) {
        if (!diarization::plda_load(config.plda_path, state->plda)) {
            fprintf(stderr, "Error: failed to load PLDA model '%s'\n",
                    config.plda_path.c_str());
#ifdef EMBEDDING_USE_COREML
            if (state->emb_coreml_ctx) embedding_coreml_free(state->emb_coreml_ctx);
#endif
#ifdef SEGMENTATION_USE_COREML
            if (state->seg_coreml_ctx) segmentation_coreml_free(state->seg_coreml_ctx);
#endif
            delete state;
            return nullptr;
        }
    }
    
    // Initialize bookkeeping
    state->chunks_processed = 0;
    state->last_recluster_chunk = 0;
    state->audio_time_processed = 0.0;
    state->finalized = false;
    state->num_speakers = 0;
    state->num_provisional_speakers = 0;
    
    return state;
}

std::vector<DiarizationResult::Segment> streaming_push(
    StreamingState* state,
    const float* samples,
    int num_samples) {
    (void)state; (void)samples; (void)num_samples;
    // TODO: Implement in Task 3
    return {};
}

void streaming_recluster(StreamingState* state) {
    (void)state;
    // TODO: Implement in Task 6
}

DiarizationResult streaming_finalize(StreamingState* state) {
    (void)state;
    // TODO: Implement in Task 8
    return {};
}

void streaming_free(StreamingState* state) {
    if (!state) return;
    
#ifdef SEGMENTATION_USE_COREML
    if (state->seg_coreml_ctx) {
        segmentation_coreml_free(state->seg_coreml_ctx);
        state->seg_coreml_ctx = nullptr;
    }
#endif

#ifdef EMBEDDING_USE_COREML
    if (state->emb_coreml_ctx) {
        embedding_coreml_free(state->emb_coreml_ctx);
        state->emb_coreml_ctx = nullptr;
    }
#endif
    
    // Clear all vectors (automatic with delete, but explicit for clarity)
    state->audio_buffer.clear();
    state->embeddings.clear();
    state->chunk_idx.clear();
    state->local_speaker_idx.clear();
    state->binarized.clear();
    state->centroids.clear();
    state->centroid_counts.clear();
    state->global_labels.clear();
    
    delete state;
}
