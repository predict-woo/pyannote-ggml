#ifndef SEGMENTATION_GGML_MODEL_H
#define SEGMENTATION_GGML_MODEL_H

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

namespace segmentation {

// Number of SincNet stages
constexpr int SINCNET_STAGES = 3;

// Number of LSTM layers
constexpr int LSTM_LAYERS = 4;

// Number of linear layers  
constexpr int LINEAR_LAYERS = 2;

// Maximum graph nodes (custom op LSTM collapses each direction into 1 node, ~50-100 total)
constexpr int MAX_GRAPH_NODES = 2048;

/**
 * @struct segmentation_hparams
 * @brief Model hyperparameters loaded from GGUF metadata
 */
struct segmentation_hparams {
    int sample_rate       = 16000;   // Audio sample rate
    int num_classes       = 7;       // Number of output classes
    int lstm_layers       = 4;       // Number of LSTM layers
    int lstm_hidden       = 128;     // LSTM hidden size (per direction)
    int sincnet_kernel    = 251;     // SincNet first conv kernel size
    int sincnet_stride    = 10;      // SincNet first conv stride
};

/**
 * @struct segmentation_model
 * @brief Immutable model weights and architecture
 * 
 * Contains all weight tensors for:
 * - SincNet: Feature extraction (3 stages)
 * - LSTM: Temporal modeling (4 layers, bidirectional)
 * - Linear: Feed-forward layers (2 layers)
 * - Classifier: Final classification head
 * 
 * Follows whisper.cpp pattern: model holds immutable weights,
 * state holds mutable inference data.
 */
struct segmentation_model {
    // Model hyperparameters
    segmentation_hparams hparams;
    
    // ========================
    // SincNet Tensors
    // ========================
    
    // Input waveform normalization (InstanceNorm1d)
    struct ggml_tensor* wav_norm_weight = nullptr;    // [1]
    struct ggml_tensor* wav_norm_bias   = nullptr;    // [1]
    
    // SincNet stage 0: Pre-computed sinc filters
    // Shape: [80, 1, 251] - 80 output channels, 1 input channel, 251 kernel
    struct ggml_tensor* sincnet_conv_weight[SINCNET_STAGES] = {nullptr};
    
    // SincNet stages 1-2: Standard Conv1d (stage 0 has no learnable bias from conversion)
    struct ggml_tensor* sincnet_conv_bias[SINCNET_STAGES] = {nullptr};
    
    // InstanceNorm1d after each stage
    struct ggml_tensor* sincnet_norm_weight[SINCNET_STAGES] = {nullptr};
    struct ggml_tensor* sincnet_norm_bias[SINCNET_STAGES]   = {nullptr};
    
    // ========================
    // LSTM Tensors (Bidirectional, 4 layers)
    // ========================
    
    // Forward direction
    struct ggml_tensor* lstm_weight_ih[LSTM_LAYERS] = {nullptr};  // Input-hidden weights
    struct ggml_tensor* lstm_weight_hh[LSTM_LAYERS] = {nullptr};  // Hidden-hidden weights
    struct ggml_tensor* lstm_bias_ih[LSTM_LAYERS]   = {nullptr};  // Input-hidden biases
    struct ggml_tensor* lstm_bias_hh[LSTM_LAYERS]   = {nullptr};  // Hidden-hidden biases
    
    // Reverse direction
    struct ggml_tensor* lstm_weight_ih_reverse[LSTM_LAYERS] = {nullptr};
    struct ggml_tensor* lstm_weight_hh_reverse[LSTM_LAYERS] = {nullptr};
    struct ggml_tensor* lstm_bias_ih_reverse[LSTM_LAYERS]   = {nullptr};
    struct ggml_tensor* lstm_bias_hh_reverse[LSTM_LAYERS]   = {nullptr};
    
    // ========================
    // Linear Layers (2 layers with LeakyReLU)
    // ========================
    
    struct ggml_tensor* linear_weight[LINEAR_LAYERS] = {nullptr};
    struct ggml_tensor* linear_bias[LINEAR_LAYERS]   = {nullptr};
    
    // ========================
    // Classifier (final output layer)
    // ========================
    
    struct ggml_tensor* classifier_weight = nullptr;  // [7, 128]
    struct ggml_tensor* classifier_bias   = nullptr;  // [7]
    
    // ========================
    // Weight Storage (whisper.cpp pattern)
    // ========================
    
    struct ggml_context* ctx = nullptr;                     // Context holding weight tensor metadata
    struct gguf_context* gguf_ctx = nullptr;                // GGUF file context
    std::vector<ggml_backend_buffer_t> weight_buffers;      // Backend buffers holding weight data
};

/**
 * @struct lstm_weight_cache
 * @brief Pre-converted F32 weights for LSTM (avoids F16→F32 on every inference)
 */
struct lstm_weight_cache {
    bool initialized = false;
    std::vector<float> w_ih[LSTM_LAYERS];
    std::vector<float> w_hh[LSTM_LAYERS];
    std::vector<float> w_ih_reverse[LSTM_LAYERS];
    std::vector<float> w_hh_reverse[LSTM_LAYERS];
    std::vector<float> bias_combined[LSTM_LAYERS];
    std::vector<float> bias_combined_reverse[LSTM_LAYERS];
    std::vector<float> ih_all_buf;
    std::vector<float> ih_all_buf_rev;
};

/**
 * @struct segmentation_state
 * @brief Mutable inference state (can be recreated without reloading model)
 * 
 * Follows whisper.cpp pattern: state holds backends, scheduler,
 * and graph metadata buffer. Can create multiple states for
 * parallel inference from the same model.
 */
struct segmentation_state {
    std::vector<ggml_backend_t> backends;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> graph_meta;
    lstm_weight_cache lstm_cache;
};

// ============================================================================
// Model Loading and Management
// ============================================================================

/**
 * @brief Load model weights from GGUF file using backend allocation
 * 
 * Uses no_alloc=true pattern from whisper.cpp:
 * 1. Open GGUF with no_alloc=true (metadata only)
 * 2. Map tensor pointers
 * 3. Allocate weight buffers on CPU backend
 * 4. Load weight data from file
 * 
 * @param fname Path to the GGUF model file
 * @param model Pointer to model struct to populate
 * @return true if successful, false otherwise
 */
bool model_load(const std::string& fname, segmentation_model& model, bool verbose = true);

/**
 * @brief Free model resources
 * @param model Model to free
 */
void model_free(segmentation_model& model);

/**
 * @brief Print model information
 * @param model Model to print info for
 */
void model_print_info(const segmentation_model& model);

/**
 * @brief Print memory usage breakdown (weights, compute, RSS)
 */
void model_print_memory_usage(const segmentation_model& model, const segmentation_state& state);

/**
 * @brief Verify all tensors are loaded and shapes match expected
 * @param model Model to verify
 * @return true if all tensors valid, false otherwise
 */
bool model_verify(const segmentation_model& model);

// ============================================================================
// State Management
// ============================================================================

/**
 * @brief Initialize inference state with backends and scheduler
 * 
 * Creates CPU backend, initializes the backend scheduler,
 * and pre-allocates by building graph once.
 * 
 * @param state State to initialize
 * @param model Model to use for pre-allocation
 * @return true if successful, false otherwise
 */
bool state_init(segmentation_state& state, segmentation_model& model, bool verbose = true);

/**
 * @brief Free state resources
 * @param state State to free
 */
void state_free(segmentation_state& state);

// ============================================================================
// Graph Building (Pure Functions)
// ============================================================================

/**
 * @brief Build computation graph as a pure function
 * 
 * Uses temporary context from state.graph_meta buffer.
 * All input/output tensors are named with ggml_set_name,
 * ggml_set_input, and ggml_set_output for backend scheduler.
 * 
 * @param model Model with weight tensors
 * @param state State with graph metadata buffer
 * @return Computation graph (owned by temporary context)
 */
struct ggml_cgraph* build_graph(
    segmentation_model& model,
    segmentation_state& state);

// ============================================================================
// Inference
// ============================================================================

/**
 * @brief Run inference using backend scheduler
 * 
 * Follows whisper.cpp pattern:
 * 1. Build graph (pure function)
 * 2. Allocate compute buffers via scheduler
 * 3. Set input data via ggml_backend_tensor_set
 * 4. Compute via ggml_backend_sched_graph_compute
 * 5. Get output data via ggml_backend_tensor_get
 * 6. Reset scheduler
 * 
 * @param model Model with weights
 * @param state State with scheduler
 * @param audio Input audio samples
 * @param n_samples Number of audio samples
 * @param output Output buffer for results
 * @param output_size Size of output buffer in floats
 * @return true if successful, false otherwise
 */
void model_enable_profile(bool enable);

bool model_infer(
    segmentation_model& model,
    segmentation_state& state,
    const float* audio,
    size_t n_samples,
    float* output,
    size_t output_size);

/**
 * @brief Forward pass through the complete segmentation model (graph building only)
 * 
 * This is the graph-building portion used by build_graph.
 * 
 * @param ctx GGML context for computation graph
 * @param model Segmentation model with all weight tensors
 * @param waveform Input waveform tensor [num_samples, 1, 1]
 * @return Output tensor [seq_len, num_classes, 1] with log-probabilities
 */
struct ggml_tensor* model_forward(
    struct ggml_context* ctx,
    const segmentation_model& model,
    struct ggml_tensor* waveform);

// ============================================================================
// Legacy Model Class (kept for backward compatibility)
// ============================================================================

/**
 * @class Model
 * @brief High-level C++ wrapper for the segmentation model
 */
class Model {
public:
    Model();
    ~Model();

    bool load(const std::string& model_path);
    std::vector<float> infer(const float* audio_data, int num_samples);
    std::string get_info() const;
    bool is_loaded() const { return loaded_; }

private:
    segmentation_model model_;
    segmentation_state state_;
    bool loaded_ = false;
};

} // namespace segmentation

#endif // SEGMENTATION_GGML_MODEL_H
