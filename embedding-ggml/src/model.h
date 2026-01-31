#ifndef EMBEDDING_GGML_MODEL_H
#define EMBEDDING_GGML_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif
#include <ggml.h>
#include <gguf.h>
#include <ggml-backend.h>
#ifdef __cplusplus
}
#endif

#include <string>
#include <vector>

namespace embedding {

// Maximum graph nodes (ResNet34 ~200-300 nodes, 4096 is plenty)
constexpr int MAX_GRAPH_NODES = 4096;

// ResNet34 layer configuration
constexpr int NUM_LAYERS = 4;
constexpr int MAX_BLOCKS = 6;  // layer3 has 6 blocks
constexpr int LAYER_BLOCKS[4] = {3, 4, 6, 3};
constexpr int LAYER_CHANNELS[4] = {32, 64, 128, 256};

/**
 * @struct embedding_hparams
 * @brief Model hyperparameters loaded from GGUF metadata
 *
 * WeSpeaker ResNet34 with TSTP pooling, embed_dim=256
 * Input: 16kHz audio -> 80-dim fbank features -> ResNet -> 256-dim embedding
 */
struct embedding_hparams {
    int sample_rate   = 16000;
    int num_mel_bins  = 80;
    int frame_length  = 25;    // ms
    int frame_shift   = 10;    // ms
    int embed_dim     = 256;
    int feat_dim      = 80;
};

/**
 * @struct embedding_model
 * @brief Model weights loaded from GGUF file
 *
 * WeSpeaker ResNet34 architecture:
 * - Initial Conv2d(1→32, 3×3) + BN + ReLU
 * - Layer1: 3× BasicBlock(32→32, stride=1)
 * - Layer2: 4× BasicBlock(32→64, stride=2) [block 0 has shortcut]
 * - Layer3: 6× BasicBlock(64→128, stride=2) [block 0 has shortcut]
 * - Layer4: 3× BasicBlock(128→256, stride=2) [block 0 has shortcut]
 * - TSTP pooling → Linear(5120→256)
 */
struct embedding_model {
    embedding_hparams hparams;

    // ========================
    // Initial Conv + BN
    // ========================
    struct ggml_tensor* conv1_weight = nullptr;     // "resnet.conv1.weight" [32, 1, 3, 3]
    struct ggml_tensor* bn1_weight = nullptr;       // "resnet.bn1.weight" [32]
    struct ggml_tensor* bn1_bias = nullptr;         // "resnet.bn1.bias" [32]
    struct ggml_tensor* bn1_running_mean = nullptr; // "resnet.bn1.running_mean" [32]
    struct ggml_tensor* bn1_running_var = nullptr;  // "resnet.bn1.running_var" [32]

    // ========================
    // ResNet Layers 1-4
    // ========================
    // For each layer L (0-3), block B (0-N):
    //   conv1 + bn1 + conv2 + bn2

    struct ggml_tensor* layer_conv1_weight[NUM_LAYERS][MAX_BLOCKS] = {};
    struct ggml_tensor* layer_bn1_weight[NUM_LAYERS][MAX_BLOCKS] = {};
    struct ggml_tensor* layer_bn1_bias[NUM_LAYERS][MAX_BLOCKS] = {};
    struct ggml_tensor* layer_bn1_mean[NUM_LAYERS][MAX_BLOCKS] = {};
    struct ggml_tensor* layer_bn1_var[NUM_LAYERS][MAX_BLOCKS] = {};

    struct ggml_tensor* layer_conv2_weight[NUM_LAYERS][MAX_BLOCKS] = {};
    struct ggml_tensor* layer_bn2_weight[NUM_LAYERS][MAX_BLOCKS] = {};
    struct ggml_tensor* layer_bn2_bias[NUM_LAYERS][MAX_BLOCKS] = {};
    struct ggml_tensor* layer_bn2_mean[NUM_LAYERS][MAX_BLOCKS] = {};
    struct ggml_tensor* layer_bn2_var[NUM_LAYERS][MAX_BLOCKS] = {};

    // Shortcut conv+BN (only for layer2.0, layer3.0, layer4.0)
    // Index 0 = layer1 (no shortcut), 1 = layer2, 2 = layer3, 3 = layer4
    struct ggml_tensor* shortcut_conv_weight[NUM_LAYERS] = {};
    struct ggml_tensor* shortcut_bn_weight[NUM_LAYERS] = {};
    struct ggml_tensor* shortcut_bn_bias[NUM_LAYERS] = {};
    struct ggml_tensor* shortcut_bn_mean[NUM_LAYERS] = {};
    struct ggml_tensor* shortcut_bn_var[NUM_LAYERS] = {};

    // ========================
    // Embedding Head
    // ========================
    struct ggml_tensor* seg1_weight = nullptr;  // "resnet.seg_1.weight" [256, 5120]
    struct ggml_tensor* seg1_bias = nullptr;    // "resnet.seg_1.bias" [256]

    // ========================
    // Weight Storage (whisper.cpp pattern)
    // ========================
    struct ggml_context* ctx = nullptr;
    struct gguf_context* gguf_ctx = nullptr;
    std::vector<ggml_backend_buffer_t> weight_buffers;
};

/**
 * @struct embedding_state
 * @brief Mutable inference state (can be recreated without reloading model)
 *
 * Follows whisper.cpp pattern: state holds backends, scheduler,
 * and graph metadata buffer.
 */
struct embedding_state {
    // Backends (CPU-only)
    std::vector<ggml_backend_t> backends;

    // Backend scheduler for automatic multi-backend orchestration
    ggml_backend_sched_t sched = nullptr;

    // Graph metadata buffer (temporary context for graph building)
    std::vector<uint8_t> graph_meta;

    // Number of frames for current graph (varies with audio duration)
    int num_frames = 0;
};

// ============================================================================
// Model Loading and Management
// ============================================================================

/**
 * @brief Load model weights from GGUF file
 */
bool model_load(const std::string& fname, embedding_model& model, bool verbose = true);

/**
 * @brief Free model resources
 */
void model_free(embedding_model& model);

/**
 * @brief Print model tensor information
 */
void model_print_info(const embedding_model& model);

/**
 * @brief Verify all tensors loaded correctly
 * @return true if all tensors valid
 */
bool model_verify(const embedding_model& model);

/**
 * @brief Print memory usage breakdown (weights, compute, RSS)
 */
void model_print_memory_usage(const embedding_model& model, const embedding_state& state);

// ============================================================================
// State Management
// ============================================================================

/**
 * @brief Initialize inference state with backends and scheduler
 */
bool state_init(embedding_state& state, embedding_model& model, bool verbose = true);

/**
 * @brief Free state resources
 */
void state_free(embedding_state& state);

// ============================================================================
// Graph Building
// ============================================================================

/**
 * @brief Build computation graph for ResNet34 forward pass
 *
 * @param model Model with weight tensors
 * @param state State with graph metadata buffer
 * @param num_frames Number of fbank frames (T)
 * @return Computation graph (owned by temporary context)
 */
struct ggml_cgraph* build_graph(
    embedding_model& model,
    embedding_state& state,
    int num_frames);

/**
 * @brief Forward pass through the complete embedding model (graph building only)
 *
 * @param ctx GGML context for computation graph
 * @param model Embedding model with all weight tensors
 * @param fbank_input Input fbank tensor [T, 80, 1, 1]
 * @return Output tensor [256] — speaker embedding
 */
struct ggml_tensor* model_forward(
    struct ggml_context* ctx,
    const embedding_model& model,
    struct ggml_tensor* fbank_input);

// ============================================================================
// Inference
// ============================================================================

/**
 * @brief Run inference using backend scheduler
 *
 * @param model Model with weights
 * @param state State with scheduler
 * @param fbank_data Fbank features (T×80 row-major float array)
 * @param num_frames Number of fbank frames (T)
 * @param output Output buffer for 256-dim embedding
 * @param output_size Size of output buffer in floats (should be 256)
 * @return true if successful
 */
bool model_infer(
    embedding_model& model,
    embedding_state& state,
    const float* fbank_data,
    int num_frames,
    float* output,
    size_t output_size);

} // namespace embedding

#endif // EMBEDDING_GGML_MODEL_H
