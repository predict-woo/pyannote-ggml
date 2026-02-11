#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct embedding_coreml_context;

// Initialize CoreML model from .mlpackage path
struct embedding_coreml_context * embedding_coreml_init(const char * path_model);

// Free CoreML context
void embedding_coreml_free(struct embedding_coreml_context * ctx);

// Run inference: fbank features → 256-dim embedding
// fbank_data: row-major float array of shape (num_frames, 80)
// embedding_out: output buffer for 256 floats
void embedding_coreml_encode(
    const struct embedding_coreml_context * ctx,
    int64_t num_frames,
    float * fbank_data,
    float * embedding_out);

#ifdef __cplusplus
}
#endif
