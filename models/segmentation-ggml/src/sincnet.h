#ifndef SEGMENTATION_GGML_SINCNET_H
#define SEGMENTATION_GGML_SINCNET_H

#include "model.h"

namespace segmentation {

// ============================================================================
// SincNet Configuration Constants
// ============================================================================

// Stage configurations from pyannote architecture
constexpr int SINCNET_STAGE0_OUT_CHANNELS = 80;
constexpr int SINCNET_STAGE0_KERNEL_SIZE = 251;
constexpr int SINCNET_STAGE0_STRIDE = 10;
constexpr int SINCNET_POOL_SIZE = 3;

constexpr int SINCNET_STAGE1_OUT_CHANNELS = 60;
constexpr int SINCNET_STAGE1_KERNEL_SIZE = 5;
constexpr int SINCNET_STAGE1_STRIDE = 1;

constexpr int SINCNET_STAGE2_OUT_CHANNELS = 60;
constexpr int SINCNET_STAGE2_KERNEL_SIZE = 5;
constexpr int SINCNET_STAGE2_STRIDE = 1;

// LeakyReLU negative slope
constexpr float LEAKY_RELU_SLOPE = 0.01f;

// InstanceNorm1d epsilon
constexpr float INSTANCE_NORM_EPS = 1e-5f;

// ============================================================================
// InstanceNorm1d Implementation
// ============================================================================

/**
 * @brief Instance Normalization 1D using GGML primitives
 * 
 * Normalizes over the time dimension (ne[0]) for each (batch, channel) pair.
 * Unlike LayerNorm which normalizes over features, InstanceNorm normalizes
 * over the spatial/temporal dimension independently for each channel.
 * 
 * For input shape [time, channels, batch] (GGML ordering):
 *   mean = sum(x, dim=time) / time
 *   var = sum((x - mean)^2, dim=time) / time
 *   x_norm = (x - mean) / sqrt(var + eps)
 *   output = weight * x_norm + bias
 * 
 * @param ctx GGML context for tensor allocation
 * @param x Input tensor [time, channels, batch]
 * @param weight Affine weight (gamma) [channels], can be nullptr
 * @param bias Affine bias (beta) [channels], can be nullptr
 * @param eps Epsilon for numerical stability (default 1e-5)
 * @return Normalized output tensor [time, channels, batch]
 */
struct ggml_tensor* ggml_instance_norm_1d(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* weight,
    struct ggml_tensor* bias,
    float eps = INSTANCE_NORM_EPS);

// ============================================================================
// SincNet Layer Functions
// ============================================================================

/**
 * @brief Single SincNet stage: Conv1d → [Abs] → MaxPool1d → InstanceNorm1d → LeakyReLU
 * 
 * @param ctx GGML context for tensor allocation
 * @param input Input tensor [time, channels_in, batch]
 * @param conv_weight Convolution kernel [kernel_size, channels_in, channels_out]
 * @param conv_bias Convolution bias [channels_out], can be nullptr
 * @param norm_weight InstanceNorm weight [channels_out]
 * @param norm_bias InstanceNorm bias [channels_out]
 * @param stride Convolution stride
 * @param pool_size Max pooling kernel size (stride = pool_size)
 * @param apply_abs Whether to apply abs() after convolution (stage 0 only)
 * @return Output tensor [time_out, channels_out, batch]
 */
struct ggml_tensor* sincnet_stage(
    struct ggml_context* ctx,
    struct ggml_tensor* input,
    struct ggml_tensor* conv_weight,
    struct ggml_tensor* conv_bias,
    struct ggml_tensor* norm_weight,
    struct ggml_tensor* norm_bias,
    int stride,
    int pool_size,
    bool apply_abs);

/**
 * @brief Full SincNet forward pass (3 stages)
 * 
 * Processes raw waveform through:
 * 1. Input InstanceNorm1d
 * 2. Stage 0: Conv1d(1→80, k=251, s=10) → abs() → MaxPool(3) → InstanceNorm → LeakyReLU
 * 3. Stage 1: Conv1d(80→60, k=5, s=1) → MaxPool(3) → InstanceNorm → LeakyReLU
 * 4. Stage 2: Conv1d(60→60, k=5, s=1) → MaxPool(3) → InstanceNorm → LeakyReLU
 * 
 * @param ctx GGML context for tensor allocation
 * @param model Model containing all weight tensors
 * @param waveform Input waveform [samples, 1, batch]
 * @return Feature tensor [time_out, 60, batch]
 */
struct ggml_tensor* sincnet_forward(
    struct ggml_context* ctx,
    const segmentation_model& model,
    struct ggml_tensor* waveform);

/**
 * @brief Compute output time dimension for a given input length
 * 
 * Calculates the temporal dimension after passing through all SincNet stages.
 * Useful for pre-allocating output buffers.
 * 
 * @param num_samples Number of input audio samples
 * @param stride First stage stride (default 10)
 * @return Number of output time frames
 */
int sincnet_output_frames(int num_samples, int stride = SINCNET_STAGE0_STRIDE);

// ============================================================================
// Legacy SincNet Class (Deprecated)
// ============================================================================

/**
 * @class SincNet
 * @brief Legacy SincNet class (deprecated, use sincnet_forward instead)
 */
class SincNet {
public:
    SincNet(int sample_rate, int num_filters);
    ~SincNet();
    
    bool init(struct ggml_context* ctx);
    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* input);
    int get_output_dim() const;

private:
    int sample_rate_;
    int num_filters_;
};

} // namespace segmentation

#endif // SEGMENTATION_GGML_SINCNET_H
