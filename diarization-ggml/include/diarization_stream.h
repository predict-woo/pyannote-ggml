#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle
typedef struct StreamingState streaming_state_t;

// Configuration
typedef struct {
    const char* seg_model_path;
    const char* emb_model_path;
    const char* plda_path;
    const char* coreml_path;
    const char* seg_coreml_path;
    int recluster_interval_sec;
    float new_speaker_threshold;
    int provisional_output;  // 0 = false, 1 = true
} streaming_config_t;

// Segment result
typedef struct {
    double start;
    double duration;
    char speaker[32];  // "SPEAKER_00", etc.
} streaming_segment_t;

// API
streaming_state_t* streaming_init_c(const streaming_config_t* config);
int streaming_push_c(streaming_state_t* state, const float* samples, int num_samples,
                     streaming_segment_t** segments_out, int* num_segments_out);
void streaming_recluster_c(streaming_state_t* state);
int streaming_finalize_c(streaming_state_t* state,
                         streaming_segment_t** segments_out, int* num_segments_out);
void streaming_free_c(streaming_state_t* state);
void streaming_free_segments_c(streaming_segment_t* segments);

#ifdef __cplusplus
}
#endif
