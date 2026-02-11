#include "sincnet.h"
#include <iostream>

namespace segmentation {

// ============================================================================
// InstanceNorm1d Implementation
// ============================================================================

struct ggml_tensor* ggml_instance_norm_1d(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* weight,
    struct ggml_tensor* bias,
    float eps)
{
    // Input shape in GGML: [time, channels, batch] (ne[0]=time, ne[1]=channels, ne[2]=batch)
    // ggml_norm normalizes along rows (ne[0]) which is the time dimension - exactly what we need
    struct ggml_tensor* x_norm = ggml_norm(ctx, x, eps);
    
    // Apply affine transformation: output = weight * x_norm + bias
    // weight and bias are [channels], x_norm is [time, channels, batch]
    // GGML mul/add broadcast from rightmost dimensions
    if (weight != nullptr) {
        // Create view of weight that can broadcast: [1, channels, 1] matches [time, channels, batch]
        // Reshape weight to add dimensions for time and batch
        int64_t ne[4] = {1, weight->ne[0], 1, 1};
        struct ggml_tensor* weight_3d = ggml_reshape_3d(ctx, weight, ne[0], ne[1], ne[2]);
        x_norm = ggml_mul(ctx, x_norm, weight_3d);
    }
    
    if (bias != nullptr) {
        int64_t ne[4] = {1, bias->ne[0], 1, 1};
        struct ggml_tensor* bias_3d = ggml_reshape_3d(ctx, bias, ne[0], ne[1], ne[2]);
        x_norm = ggml_add(ctx, x_norm, bias_3d);
    }
    
    return x_norm;
}

// ============================================================================
// SincNet Stage Implementation
// ============================================================================

struct ggml_tensor* sincnet_stage(
    struct ggml_context* ctx,
    struct ggml_tensor* input,
    struct ggml_tensor* conv_weight,
    struct ggml_tensor* conv_bias,
    struct ggml_tensor* norm_weight,
    struct ggml_tensor* norm_bias,
    int stride,
    int pool_size,
    bool apply_abs)
{
    // ggml_conv_1d: kernel shape [kernel_size, channels_in, channels_out], data [time, channels_in, batch]
    struct ggml_tensor* x = ggml_conv_1d(ctx, conv_weight, input, stride, 0, 1);
    
    // Add bias: reshape [channels] to [1, channels, 1] for broadcasting with [time, channels, batch]
    if (conv_bias != nullptr) {
        struct ggml_tensor* bias_3d = ggml_reshape_3d(ctx, conv_bias, 1, conv_bias->ne[0], 1);
        x = ggml_add(ctx, x, bias_3d);
    }
    
    // Apply abs() for stage 0 only - this creates rectified representation for sinc filters
    if (apply_abs) {
        x = ggml_abs(ctx, x);
    }
    
    x = ggml_pool_1d(ctx, x, GGML_OP_POOL_MAX, pool_size, pool_size, 0);
    x = ggml_instance_norm_1d(ctx, x, norm_weight, norm_bias, INSTANCE_NORM_EPS);
    x = ggml_leaky_relu(ctx, x, LEAKY_RELU_SLOPE, false);
    
    return x;
}

// ============================================================================
// Full SincNet Forward Pass
// ============================================================================

struct ggml_tensor* sincnet_forward(
    struct ggml_context* ctx,
    const segmentation_model& model,
    struct ggml_tensor* waveform)
{
    // Input waveform: [samples, 1, batch]
    // Expected output: [time_out, 60, batch]
    
    // Step 1: Input waveform normalization (InstanceNorm1d)
    // Note: wav_norm may be optional, check if weights exist
    struct ggml_tensor* x = waveform;
    if (model.wav_norm_weight != nullptr && model.wav_norm_bias != nullptr) {
        x = ggml_instance_norm_1d(ctx, x, model.wav_norm_weight, model.wav_norm_bias, INSTANCE_NORM_EPS);
    }
    
    // Step 2: Stage 0 - SincNet conv with abs()
    // Conv1d(1 → 80, kernel=251, stride=10) → abs() → MaxPool(3) → InstanceNorm → LeakyReLU
    x = sincnet_stage(
        ctx, x,
        model.sincnet_conv_weight[0],
        model.sincnet_conv_bias[0],  // nullptr for stage 0 (sinc filters have no bias)
        model.sincnet_norm_weight[0],
        model.sincnet_norm_bias[0],
        SINCNET_STAGE0_STRIDE,  // stride = 10
        SINCNET_POOL_SIZE,      // pool = 3
        true                     // apply_abs = true for stage 0
    );
    
    // Step 3: Stage 1 - Standard conv (no abs)
    // Conv1d(80 → 60, kernel=5, stride=1) → MaxPool(3) → InstanceNorm → LeakyReLU
    x = sincnet_stage(
        ctx, x,
        model.sincnet_conv_weight[1],
        model.sincnet_conv_bias[1],
        model.sincnet_norm_weight[1],
        model.sincnet_norm_bias[1],
        SINCNET_STAGE1_STRIDE,  // stride = 1
        SINCNET_POOL_SIZE,      // pool = 3
        false                    // apply_abs = false
    );
    
    // Step 4: Stage 2 - Standard conv (no abs)
    // Conv1d(60 → 60, kernel=5, stride=1) → MaxPool(3) → InstanceNorm → LeakyReLU
    x = sincnet_stage(
        ctx, x,
        model.sincnet_conv_weight[2],
        model.sincnet_conv_bias[2],
        model.sincnet_norm_weight[2],
        model.sincnet_norm_bias[2],
        SINCNET_STAGE2_STRIDE,  // stride = 1
        SINCNET_POOL_SIZE,      // pool = 3
        false                    // apply_abs = false
    );

    // Output: [time, channels, batch] in GGML convention
    // This matches PyTorch's (batch, channels, time) when interpreted correctly
    return x;
}

// ============================================================================
// Helper Functions
// ============================================================================

int sincnet_output_frames(int num_samples, int stride)
{
    // Stage 0: Conv(kernel=251, stride=stride) → Pool(3)
    // output_after_conv = (num_samples - 251) / stride + 1
    // output_after_pool = output_after_conv / 3
    int after_stage0_conv = (num_samples - SINCNET_STAGE0_KERNEL_SIZE) / stride + 1;
    int after_stage0_pool = after_stage0_conv / SINCNET_POOL_SIZE;
    
    // Stage 1: Conv(kernel=5, stride=1) → Pool(3)
    int after_stage1_conv = after_stage0_pool - SINCNET_STAGE1_KERNEL_SIZE + 1;
    int after_stage1_pool = after_stage1_conv / SINCNET_POOL_SIZE;
    
    // Stage 2: Conv(kernel=5, stride=1) → Pool(3)
    int after_stage2_conv = after_stage1_pool - SINCNET_STAGE2_KERNEL_SIZE + 1;
    int after_stage2_pool = after_stage2_conv / SINCNET_POOL_SIZE;
    
    return after_stage2_pool;
}

// ============================================================================
// Legacy SincNet Class Implementation
// ============================================================================

SincNet::SincNet(int sample_rate, int num_filters)
    : sample_rate_(sample_rate), num_filters_(num_filters) {
}

SincNet::~SincNet() {
}

bool SincNet::init(struct ggml_context* ctx) {
    (void)ctx;
    std::cout << "Initializing SincNet with " << num_filters_ << " filters" << std::endl;
    std::cout << "NOTE: Use sincnet_forward() instead of this legacy class" << std::endl;
    return true;
}

struct ggml_tensor* SincNet::forward(struct ggml_context* ctx, struct ggml_tensor* input) {
    (void)ctx;
    std::cout << "WARNING: Legacy SincNet::forward() called, returning input unchanged" << std::endl;
    std::cout << "Please use sincnet_forward() with model weights instead" << std::endl;
    return input;
}

int SincNet::get_output_dim() const {
    return num_filters_;
}

} // namespace segmentation
