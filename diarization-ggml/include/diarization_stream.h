#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct StreamingState streaming_state_t;

typedef struct {
    const char* seg_model_path;
    const char* emb_model_path;
    const char* plda_path;
    const char* coreml_path;
    const char* seg_coreml_path;
} streaming_config_t;

typedef struct {
    double start;
    double duration;
    char speaker[32];
} streaming_segment_t;

typedef struct {
    int chunk_index;
    double start_time;
    double duration;
    int num_frames;
    float* vad;     // [num_frames] combined speaker activity (caller must free via streaming_free_vad_chunks_c)
} streaming_vad_chunk_t;

// API
streaming_state_t* streaming_init_c(const streaming_config_t* config);
int streaming_push_c(streaming_state_t* state, const float* samples, int num_samples,
                     streaming_vad_chunk_t** chunks_out, int* num_chunks_out);
int streaming_recluster_c(streaming_state_t* state,
                          streaming_segment_t** segments_out, int* num_segments_out);
int streaming_finalize_c(streaming_state_t* state,
                         streaming_segment_t** segments_out, int* num_segments_out);
void streaming_free_c(streaming_state_t* state);
void streaming_free_segments_c(streaming_segment_t* segments);
void streaming_free_vad_chunks_c(streaming_vad_chunk_t* chunks, int num_chunks);

#ifdef __cplusplus
}
#endif
