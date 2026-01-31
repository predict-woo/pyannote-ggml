#pragma once
#include <string>
#include <vector>

struct DiarizationConfig {
    std::string seg_model_path;
    std::string emb_model_path;
    std::string audio_path;
    std::string plda_path;
    std::string coreml_path;
    std::string seg_coreml_path;
    std::string output_path;
    std::string dump_stage;  // empty = none
};

struct DiarizationResult {
    struct Segment {
        double start;
        double duration;
        std::string speaker;
    };
    std::vector<Segment> segments;
};

bool diarize(const DiarizationConfig& config, DiarizationResult& result);

struct embedding_coreml_context;
namespace embedding {
    struct embedding_model;
    struct embedding_state;
}

bool extract_embeddings(
    const float* audio,
    int          num_samples,
    const float* binarized_segmentations,
    int          num_chunks,
    int          num_frames_per_chunk,
    int          num_speakers,
#ifdef EMBEDDING_USE_COREML
    struct embedding_coreml_context* coreml_ctx,
#else
    embedding::embedding_model& emb_model,
    embedding::embedding_state& emb_state,
#endif
    float*       embeddings_out);
