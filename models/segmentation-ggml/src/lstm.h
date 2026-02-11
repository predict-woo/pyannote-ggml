#ifndef SEGMENTATION_GGML_LSTM_H
#define SEGMENTATION_GGML_LSTM_H

#include "model.h"

namespace segmentation {

// ============================================================================
// LSTM Configuration Constants
// ============================================================================

// LSTM architecture from pyannote model
constexpr int LSTM_INPUT_DIM = 60;       // Input features from SincNet
constexpr int LSTM_HIDDEN_DIM = 128;     // Hidden size per direction
constexpr int LSTM_NUM_LAYERS = 4;       // Number of stacked layers
constexpr int LSTM_OUTPUT_DIM = 256;     // Output = 2 * LSTM_HIDDEN_DIM (bidirectional)

// ============================================================================
// LSTM Layer Functions
// ============================================================================

struct ggml_tensor* lstm_layer_bidirectional(
    struct ggml_context* ctx,
    struct ggml_tensor* input,
    struct ggml_tensor* weight_ih_fwd,
    struct ggml_tensor* weight_hh_fwd,
    struct ggml_tensor* bias_ih_fwd,
    struct ggml_tensor* bias_hh_fwd,
    struct ggml_tensor* weight_ih_rev,
    struct ggml_tensor* weight_hh_rev,
    struct ggml_tensor* bias_ih_rev,
    struct ggml_tensor* bias_hh_rev,
    int hidden_size);

struct ggml_tensor* lstm_forward(
    struct ggml_context* ctx,
    const segmentation_model& model,
    struct ggml_tensor* input);

void lstm_init_weight_cache(lstm_weight_cache& cache, const segmentation_model& model);
void lstm_set_active_cache(const lstm_weight_cache* cache);
void lstm_print_profile();
void lstm_reset_profile();
class LSTM {
public:
    LSTM(int input_dim, int hidden_dim, int num_layers, bool bidirectional = true);
    ~LSTM();
    
    bool init(struct ggml_context* ctx);
    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* input);
    int get_output_dim() const;

private:
    int input_dim_;
    int hidden_dim_;
    int num_layers_;
    bool bidirectional_;
};

} // namespace segmentation

#endif // SEGMENTATION_GGML_LSTM_H
