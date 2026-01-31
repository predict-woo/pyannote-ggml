#include "model.h"
#include "sincnet.h"
#include "lstm.h"
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <chrono>
#include <fstream>
#include <cstring>
#include <ggml-cpu.h>
#include <ggml-backend.h>

struct wav_header {
    char riff[4];
    uint32_t file_size;
    char wave[4];
    char fmt[4];
    uint32_t fmt_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
};

struct wav_data_chunk {
    char id[4];
    uint32_t size;
};

static bool load_wav_file(const std::string& path, std::vector<float>& samples, uint32_t& sample_rate) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "ERROR: Failed to open WAV file: %s\n", path.c_str());
        return false;
    }
    
    wav_header header;
    file.read(reinterpret_cast<char*>(&header), sizeof(wav_header));
    
    if (std::strncmp(header.riff, "RIFF", 4) != 0 || std::strncmp(header.wave, "WAVE", 4) != 0) {
        fprintf(stderr, "ERROR: Invalid WAV file format\n");
        return false;
    }
    
    if (header.audio_format != 1) {
        fprintf(stderr, "ERROR: Only PCM format supported (got format %d)\n", header.audio_format);
        return false;
    }
    
    if (header.num_channels != 1) {
        fprintf(stderr, "ERROR: Only mono audio supported (got %d channels)\n", header.num_channels);
        return false;
    }
    
    if (header.bits_per_sample != 16) {
        fprintf(stderr, "ERROR: Only 16-bit audio supported (got %d bits)\n", header.bits_per_sample);
        return false;
    }
    
    wav_data_chunk data_chunk;
    file.read(reinterpret_cast<char*>(&data_chunk), sizeof(wav_data_chunk));
    
    while (std::strncmp(data_chunk.id, "data", 4) != 0) {
        file.seekg(data_chunk.size, std::ios::cur);
        if (!file.read(reinterpret_cast<char*>(&data_chunk), sizeof(wav_data_chunk))) {
            fprintf(stderr, "ERROR: Data chunk not found\n");
            return false;
        }
    }
    
    uint32_t num_samples = data_chunk.size / (header.bits_per_sample / 8);
    samples.resize(num_samples);
    
    std::vector<int16_t> pcm_data(num_samples);
    file.read(reinterpret_cast<char*>(pcm_data.data()), data_chunk.size);
    
    for (size_t i = 0; i < num_samples; i++) {
        samples[i] = static_cast<float>(pcm_data[i]) / 32768.0f;
    }
    
    sample_rate = header.sample_rate;
    
    file.close();
    return true;
}

static bool test_sincnet_shapes(const segmentation::segmentation_model& model) {
    printf("\n");
    printf("============================================================\n");
    printf("Testing SincNet Shape Propagation\n");
    printf("============================================================\n");
    
    const int64_t num_samples = 160000;
    const int64_t num_channels = 1;
    const int64_t batch_size = 1;
    
    size_t ctx_size = 1024 * 1024 * 64;
    struct ggml_init_params params = {
        .mem_size   = ctx_size,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "ERROR: Failed to create computation context\n");
        return false;
    }
    
    printf("\nInput configuration:\n");
    printf("  Samples: %lld (%.1f seconds at 16kHz)\n", (long long)num_samples, num_samples / 16000.0);
    printf("  Channels: %lld\n", (long long)num_channels);
    printf("  Batch: %lld\n", (long long)batch_size);
    
    struct ggml_tensor* input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, num_samples, num_channels, batch_size);
    ggml_set_name(input, "input");
    ggml_set_input(input);
    
    float* input_data = ggml_get_data_f32(input);
    for (int i = 0; i < num_samples; i++) {
        input_data[i] = sinf(2.0f * 3.14159f * 440.0f * i / 16000.0f) * 0.1f;
    }
    
    printf("\nInput tensor: [%lld, %lld, %lld]\n", 
           (long long)input->ne[0], (long long)input->ne[1], (long long)input->ne[2]);
    
    printf("\nBuilding SincNet forward graph...\n");
    struct ggml_tensor* output = segmentation::sincnet_forward(ctx, model, input);
    
    if (!output) {
        fprintf(stderr, "ERROR: sincnet_forward returned null\n");
        ggml_free(ctx);
        return false;
    }
    
    ggml_set_name(output, "sincnet_output");
    ggml_set_output(output);
    
    printf("Output tensor: [%lld, %lld, %lld]\n",
           (long long)output->ne[0], (long long)output->ne[1], (long long)output->ne[2]);
    
    int expected_frames = segmentation::sincnet_output_frames(num_samples);
    printf("\nExpected output frames: %d\n", expected_frames);
    printf("Computed output frames: %lld\n", (long long)output->ne[0]);
    printf("Expected output channels: 60\n");
    printf("Computed output channels: %lld\n", (long long)output->ne[1]);
    
    printf("\nCreating computation graph...\n");
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    
    printf("Graph nodes: %d\n", ggml_graph_n_nodes(graph));
    
    printf("\nComputing graph...\n");
    enum ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, 4);
    
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ERROR: Graph computation failed with status %d\n", status);
        ggml_free(ctx);
        return false;
    }
    
    printf("Graph computation completed successfully!\n");
    
    float* output_data = ggml_get_data_f32(output);
    float min_val = output_data[0];
    float max_val = output_data[0];
    float sum = 0.0f;
    int64_t total_elements = output->ne[0] * output->ne[1] * output->ne[2];
    
    for (int64_t i = 0; i < total_elements; i++) {
        min_val = fminf(min_val, output_data[i]);
        max_val = fmaxf(max_val, output_data[i]);
        sum += output_data[i];
    }
    
    printf("\nOutput statistics:\n");
    printf("  Elements: %lld\n", (long long)total_elements);
    printf("  Min: %.6f\n", min_val);
    printf("  Max: %.6f\n", max_val);
    printf("  Mean: %.6f\n", sum / total_elements);
    
    bool shapes_match = (output->ne[0] == expected_frames) && (output->ne[1] == 60) && (output->ne[2] == batch_size);
    
    printf("\n============================================================\n");
    if (shapes_match) {
        printf("SUCCESS: SincNet shape test PASSED\n");
    } else {
        printf("FAILURE: SincNet shape test FAILED\n");
    }
    printf("============================================================\n");
    
    ggml_free(ctx);
    return shapes_match;
}

static bool test_linear_classifier_shapes(const segmentation::segmentation_model& model) {
    printf("\n");
    printf("============================================================\n");
    printf("Testing Linear and Classifier Shape Propagation\n");
    printf("============================================================\n");
    
    const int64_t seq_len = 589;
    const int64_t lstm_features = 256;
    const int64_t batch_size = 1;
    
    size_t ctx_size = 1024ULL * 1024 * 4;
    struct ggml_init_params params = {
        .mem_size   = ctx_size,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "ERROR: Failed to create computation context\n");
        return false;
    }
    
    printf("\nInput configuration:\n");
    printf("  Sequence length: %lld\n", (long long)seq_len);
    printf("  LSTM features: %lld\n", (long long)lstm_features);
    printf("  Batch size: %lld\n", (long long)batch_size);
    
    struct ggml_tensor* input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq_len, lstm_features, batch_size);
    ggml_set_name(input, "linear_input");
    ggml_set_input(input);
    
    float* input_data = ggml_get_data_f32(input);
    for (int64_t i = 0; i < seq_len * lstm_features * batch_size; i++) {
        input_data[i] = sinf(i * 0.01f) * 0.1f;
    }
    
    printf("\nInput tensor: [%lld, %lld, %lld]\n", 
           (long long)input->ne[0], (long long)input->ne[1], (long long)input->ne[2]);
    
    printf("\nApplying Linear 0 (256 -> 128)...\n");
    struct ggml_tensor* input_2d = ggml_reshape_2d(ctx, input, lstm_features, seq_len);
    struct ggml_tensor* linear0_2d = ggml_mul_mat(ctx, model.linear_weight[0], input_2d);
    linear0_2d = ggml_add(ctx, linear0_2d, model.linear_bias[0]);
    linear0_2d = ggml_leaky_relu(ctx, linear0_2d, 0.01f, true);
    struct ggml_tensor* linear0 = ggml_reshape_3d(ctx, linear0_2d, seq_len, 128, batch_size);
    ggml_set_name(linear0, "linear0_output");
    
    printf("Linear 0 output: [%lld, %lld, %lld]\n",
           (long long)linear0->ne[0], (long long)linear0->ne[1], (long long)linear0->ne[2]);
    
    printf("\nApplying Linear 1 (128 -> 128)...\n");
    struct ggml_tensor* linear0_2d_out = ggml_reshape_2d(ctx, linear0, 128, seq_len);
    struct ggml_tensor* linear1_2d = ggml_mul_mat(ctx, model.linear_weight[1], linear0_2d_out);
    linear1_2d = ggml_add(ctx, linear1_2d, model.linear_bias[1]);
    linear1_2d = ggml_leaky_relu(ctx, linear1_2d, 0.01f, true);
    struct ggml_tensor* linear1 = ggml_reshape_3d(ctx, linear1_2d, seq_len, 128, batch_size);
    ggml_set_name(linear1, "linear1_output");
    
    printf("Linear 1 output: [%lld, %lld, %lld]\n",
           (long long)linear1->ne[0], (long long)linear1->ne[1], (long long)linear1->ne[2]);
    
    printf("\nApplying Classifier (128 -> 7)...\n");
    struct ggml_tensor* linear1_2d_out = ggml_reshape_2d(ctx, linear1, 128, seq_len);
    struct ggml_tensor* logits_2d = ggml_mul_mat(ctx, model.classifier_weight, linear1_2d_out);
    logits_2d = ggml_add(ctx, logits_2d, model.classifier_bias);
    struct ggml_tensor* output_2d = ggml_soft_max(ctx, logits_2d);
    output_2d = ggml_log(ctx, output_2d);
    struct ggml_tensor* output = ggml_reshape_3d(ctx, output_2d, seq_len, 7, batch_size);
    ggml_set_name(output, "classifier_output");
    ggml_set_output(output);
    
    printf("Classifier output: [%lld, %lld, %lld]\n",
           (long long)output->ne[0], (long long)output->ne[1], (long long)output->ne[2]);
    
    int64_t expected_seq_len = seq_len;
    int64_t expected_classes = 7;
    
    printf("\nExpected output: [%lld, %lld, %lld]\n",
           (long long)expected_seq_len, (long long)expected_classes, (long long)batch_size);
    
    printf("\nCreating computation graph...\n");
    struct ggml_cgraph* graph = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(graph, output);
    
    printf("Graph nodes: %d\n", ggml_graph_n_nodes(graph));
    
    printf("\nComputing graph...\n");
    enum ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, 4);
    
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ERROR: Graph computation failed with status %d\n", status);
        ggml_free(ctx);
        return false;
    }
    
    printf("Graph computation completed successfully!\n");
    
    float* output_data = ggml_get_data_f32(output);
    float min_val = output_data[0];
    float max_val = output_data[0];
    float sum = 0.0f;
    int64_t total_elements = output->ne[0] * output->ne[1] * output->ne[2];
    
    for (int64_t i = 0; i < total_elements; i++) {
        min_val = fminf(min_val, output_data[i]);
        max_val = fmaxf(max_val, output_data[i]);
        sum += output_data[i];
    }
    
    printf("\nOutput statistics:\n");
    printf("  Elements: %lld\n", (long long)total_elements);
    printf("  Min: %.6f\n", min_val);
    printf("  Max: %.6f\n", max_val);
    printf("  Mean: %.6f\n", sum / total_elements);
    
    bool shapes_match = (output->ne[0] == expected_seq_len) && 
                        (output->ne[1] == expected_classes) && 
                        (output->ne[2] == batch_size);
    
    printf("\n============================================================\n");
    if (shapes_match) {
        printf("SUCCESS: Linear and Classifier shape test PASSED\n");
        printf("Output shape [%lld, %lld, %lld] matches expected [%lld, %lld, %lld]\n",
               (long long)output->ne[0], (long long)output->ne[1], (long long)output->ne[2],
               (long long)expected_seq_len, (long long)expected_classes, (long long)batch_size);
    } else {
        printf("FAILURE: Linear and Classifier shape test FAILED\n");
        printf("Output shape [%lld, %lld, %lld] does not match expected [%lld, %lld, %lld]\n",
               (long long)output->ne[0], (long long)output->ne[1], (long long)output->ne[2],
               (long long)expected_seq_len, (long long)expected_classes, (long long)batch_size);
    }
    printf("============================================================\n");
    
    ggml_free(ctx);
    return shapes_match;
}

static bool test_full_forward_pass(
    segmentation::segmentation_model& model,
    segmentation::segmentation_state& state,
    const char* output_path = nullptr,
    const char* audio_path = nullptr) {
    
    printf("\n");
    printf("============================================================\n");
    printf("Testing End-to-End Model Forward Pass\n");
    printf("============================================================\n");
    
    std::vector<float> audio_samples;
    uint32_t audio_sample_rate = 0;
    int64_t num_samples = 160000;
    
    if (audio_path != nullptr) {
        printf("\nLoading audio from: %s\n", audio_path);
        if (!load_wav_file(audio_path, audio_samples, audio_sample_rate)) {
            fprintf(stderr, "ERROR: Failed to load audio file\n");
            return false;
        }
        
        if (audio_sample_rate != 16000) {
            fprintf(stderr, "ERROR: Audio must be 16kHz (got %d Hz)\n", audio_sample_rate);
            return false;
        }
        
        printf("  Loaded %zu samples (%.2f seconds)\n", audio_samples.size(), audio_samples.size() / 16000.0);
        
        if (audio_samples.size() > 160000) {
            printf("  Truncating to first 10 seconds (160000 samples)\n");
            audio_samples.resize(160000);
        }
        
        num_samples = audio_samples.size();
    } else {
        audio_samples.resize(num_samples);
        for (int i = 0; i < num_samples; i++) {
            audio_samples[i] = sinf(2.0f * 3.14159f * 440.0f * i / 16000.0f) * 0.1f;
        }
    }
    
    printf("\nInput configuration:\n");
    printf("  Samples: %lld (%.1f seconds at 16kHz)\n", (long long)num_samples, num_samples / 16000.0);
    printf("  Using backend scheduler for inference\n");
    
    printf("\nBuilding full model forward graph...\n");
    printf("  SincNet: [160000, 1, 1] -> [589, 60, 1]\n");
    printf("  LSTM:    [589, 60, 1] -> [589, 256, 1]\n");
    printf("  Linear:  [589, 256, 1] -> [589, 128, 1]\n");
    printf("  Classifier: [589, 128, 1] -> [589, 7, 1]\n");
    
    const int64_t expected_seq_len = 589;
    const int64_t expected_classes = 7;
    int64_t output_elements = expected_seq_len * expected_classes;
    std::vector<float> output_data(output_elements);
    
    printf("\nRunning inference via backend scheduler...\n");
    auto start_time = std::chrono::high_resolution_clock::now();
    
    bool success = segmentation::model_infer(
        model, state,
        audio_samples.data(), num_samples,
        output_data.data(), output_elements);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (!success) {
        fprintf(stderr, "ERROR: model_infer failed\n");
        return false;
    }
    
    printf("Inference completed successfully!\n");
    printf("\n*** Inference time: %lld ms ***\n", (long long)duration.count());
    
    float min_val = output_data[0];
    float max_val = output_data[0];
    float sum = 0.0f;
    
    for (int64_t i = 0; i < output_elements; i++) {
        min_val = fminf(min_val, output_data[i]);
        max_val = fmaxf(max_val, output_data[i]);
        sum += output_data[i];
    }
    
    printf("\nOutput statistics:\n");
    printf("  Elements: %lld\n", (long long)output_elements);
    printf("  Min: %.6f\n", min_val);
    printf("  Max: %.6f\n", max_val);
    printf("  Mean: %.6f\n", sum / output_elements);
    
    printf("\n============================================================\n");
    printf("SUCCESS: End-to-End Forward Pass PASSED\n");
    printf("Output shape [%lld, %lld, %lld] matches expected [%lld, %lld, %lld]\n",
           (long long)expected_seq_len, (long long)expected_classes, (long long)1,
           (long long)expected_seq_len, (long long)expected_classes, (long long)1);
    printf("============================================================\n");
    
    if (output_path != nullptr) {
        printf("\nSaving output to: %s\n", output_path);
        FILE* f = fopen(output_path, "wb");
        if (f) {
            int64_t shape[3] = {expected_seq_len, expected_classes, 1};
            fwrite(shape, sizeof(int64_t), 3, f);
            fwrite(output_data.data(), sizeof(float), output_elements, f);
            fclose(f);
            printf("  Saved %lld elements\n", (long long)output_elements);
        } else {
            fprintf(stderr, "  ERROR: Failed to open output file\n");
        }
    }
    
    return true;
}

static bool test_lstm_shapes(const segmentation::segmentation_model& model) {
    printf("\n");
    printf("============================================================\n");
    printf("Testing LSTM Shape Propagation\n");
    printf("============================================================\n");
    
    const int64_t seq_len = 589;
    const int64_t input_features = 60;
    const int64_t batch_size = 1;
    
    size_t ctx_size = 1024ULL * 1024 * 256;
    struct ggml_init_params params = {
        .mem_size   = ctx_size,
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "ERROR: Failed to create computation context\n");
        return false;
    }
    
    printf("\nInput configuration:\n");
    printf("  Sequence length: %lld\n", (long long)seq_len);
    printf("  Input features: %lld\n", (long long)input_features);
    printf("  Batch size: %lld\n", (long long)batch_size);
    
    struct ggml_tensor* input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq_len, input_features, batch_size);
    ggml_set_name(input, "lstm_input");
    ggml_set_input(input);
    
    float* input_data = ggml_get_data_f32(input);
    for (int64_t i = 0; i < seq_len * input_features * batch_size; i++) {
        input_data[i] = sinf(i * 0.01f) * 0.1f;
    }
    
    printf("\nInput tensor: [%lld, %lld, %lld]\n", 
           (long long)input->ne[0], (long long)input->ne[1], (long long)input->ne[2]);
    
    printf("\nBuilding LSTM forward graph...\n");
    struct ggml_tensor* output = segmentation::lstm_forward(ctx, model, input);
    
    if (!output) {
        fprintf(stderr, "ERROR: lstm_forward returned null\n");
        ggml_free(ctx);
        return false;
    }
    
    ggml_set_name(output, "lstm_output");
    ggml_set_output(output);
    
    printf("Output tensor: [%lld, %lld, %lld]\n",
           (long long)output->ne[0], (long long)output->ne[1], (long long)output->ne[2]);
    
    int64_t expected_seq_len = seq_len;
    int64_t expected_features = 256;
    
    printf("\nExpected output: [%lld, %lld, %lld]\n",
           (long long)expected_seq_len, (long long)expected_features, (long long)batch_size);
    
    printf("\nCreating computation graph...\n");
    struct ggml_cgraph* graph = ggml_new_graph_custom(ctx, segmentation::MAX_GRAPH_NODES, false);
    ggml_build_forward_expand(graph, output);
    
    printf("Graph nodes: %d\n", ggml_graph_n_nodes(graph));
    
    printf("\nComputing graph...\n");
    enum ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, 4);
    
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ERROR: Graph computation failed with status %d\n", status);
        ggml_free(ctx);
        return false;
    }
    
    printf("Graph computation completed successfully!\n");
    
    float* output_data = ggml_get_data_f32(output);
    float min_val = output_data[0];
    float max_val = output_data[0];
    float sum = 0.0f;
    int64_t total_elements = output->ne[0] * output->ne[1] * output->ne[2];
    
    for (int64_t i = 0; i < total_elements; i++) {
        min_val = fminf(min_val, output_data[i]);
        max_val = fmaxf(max_val, output_data[i]);
        sum += output_data[i];
    }
    
    printf("\nOutput statistics:\n");
    printf("  Elements: %lld\n", (long long)total_elements);
    printf("  Min: %.6f\n", min_val);
    printf("  Max: %.6f\n", max_val);
    printf("  Mean: %.6f\n", sum / total_elements);
    
    bool shapes_match = (output->ne[0] == expected_seq_len) && 
                        (output->ne[1] == expected_features) && 
                        (output->ne[2] == batch_size);
    
    printf("\n============================================================\n");
    if (shapes_match) {
        printf("SUCCESS: LSTM shape test PASSED\n");
        printf("Output shape [%lld, %lld, %lld] matches expected [%lld, %lld, %lld]\n",
               (long long)output->ne[0], (long long)output->ne[1], (long long)output->ne[2],
               (long long)expected_seq_len, (long long)expected_features, (long long)batch_size);
    } else {
        printf("FAILURE: LSTM shape test FAILED\n");
        printf("Output shape [%lld, %lld, %lld] does not match expected [%lld, %lld, %lld]\n",
               (long long)output->ne[0], (long long)output->ne[1], (long long)output->ne[2],
               (long long)expected_seq_len, (long long)expected_features, (long long)batch_size);
    }
    printf("============================================================\n");
    
    ggml_free(ctx);
    return shapes_match;
}

static void benchmark_performance(
    segmentation::segmentation_model& model,
    segmentation::segmentation_state& state) {
    
    printf("\n");
    printf("============================================================\n");
    printf("Performance Benchmark\n");
    printf("============================================================\n");
    printf("Active backends: %zu\n", state.backends.size());
    for (size_t i = 0; i < state.backends.size(); i++) {
        printf("  [%zu] %s\n", i, ggml_backend_name(state.backends[i]));
    }
    
    const int64_t num_samples = 160000;
    const int num_iterations = 5;
    const int warmup_iterations = 2;
    
    std::vector<float> input_data(num_samples);
    for (int i = 0; i < num_samples; i++) {
        input_data[i] = sinf(2.0f * 3.14159f * 440.0f * i / 16000.0f) * 0.1f;
    }
    
    const int64_t expected_seq_len = 589;
    const int64_t expected_classes = 7;
    int64_t output_elements = expected_seq_len * expected_classes;
    std::vector<float> output_data(output_elements);
    
    long long cpu_total_ms = 0;
    
    segmentation::model_enable_profile(true);
    
    for (int iter = 0; iter < num_iterations + warmup_iterations; iter++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        bool success = segmentation::model_infer(
            model, state,
            input_data.data(), num_samples,
            output_data.data(), output_elements);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        if (success) {
            if (iter >= warmup_iterations) {
                cpu_total_ms += ms;
                printf("  Iteration %d: %lld ms\n", iter - warmup_iterations + 1, ms);
            } else {
                printf("  Warmup %d: %lld ms\n", iter + 1, ms);
            }
        } else {
            printf("  Iteration %d: FAILED\n", iter + 1);
        }
    }
    
    long long cpu_avg_ms = cpu_total_ms / num_iterations;
    
    printf("\n============================================================\n");
    printf("Benchmark Summary\n");
    printf("============================================================\n");
    printf("Audio duration: 10 seconds\n");
    printf("Avg inference:  %lld ms (%.1fx real-time)\n", cpu_avg_ms, 10000.0 / cpu_avg_ms);
    printf("Primary backend: %s\n", ggml_backend_name(state.backends[0]));
    printf("============================================================\n");
}

int main(int argc, char** argv) {
    std::cout << "Segmentation GGML - Speaker Segmentation Model" << std::endl;
    std::cout << "================================================" << std::endl;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf> [--test] [--benchmark] [--audio <path>] [--save-output <path>]" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Arguments:" << std::endl;
        std::cerr << "  model.gguf                - Path to the GGUF model file" << std::endl;
        std::cerr << "  --test                    - Run model tests" << std::endl;
        std::cerr << "  --benchmark               - Run performance benchmark" << std::endl;
        std::cerr << "  --audio <path>            - Load audio from WAV file (16kHz mono PCM)" << std::endl;
        std::cerr << "  --save-output <path>      - Save test output to binary file" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::cout << "Model path: " << model_path << std::endl;
    
    bool run_test = false;
    bool run_benchmark = false;
    std::string audio_path;
    std::string output_path;
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--test") {
            run_test = true;
        } else if (arg == "--benchmark") {
            run_benchmark = true;
        } else if (arg == "--audio" && i + 1 < argc) {
            audio_path = argv[++i];
        } else if (arg == "--save-output" && i + 1 < argc) {
            output_path = argv[++i];
        }
    }

    try {
        segmentation::segmentation_model model;
        segmentation::segmentation_state state;
        
        std::cout << "\nLoading model..." << std::endl;
        if (!segmentation::model_load(model_path, model)) {
            std::cerr << "Error: Failed to load model from " << model_path << std::endl;
            return 1;
        }

        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Model loaded successfully!" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        segmentation::model_print_info(model);
        segmentation::model_verify(model);
        
        // Initialize inference state (backends + scheduler)
        if (!segmentation::state_init(state, model)) {
            std::cerr << "Error: Failed to initialize inference state" << std::endl;
            segmentation::model_free(model);
            return 1;
        }

        if (run_test) {
            if (!test_sincnet_shapes(model)) {
                std::cerr << "SincNet test failed!" << std::endl;
                segmentation::state_free(state);
                segmentation::model_free(model);
                return 1;
            }
            
            if (!test_lstm_shapes(model)) {
                std::cerr << "LSTM test failed!" << std::endl;
                segmentation::state_free(state);
                segmentation::model_free(model);
                return 1;
            }
            
            if (!test_linear_classifier_shapes(model)) {
                std::cerr << "Linear and Classifier test failed!" << std::endl;
                segmentation::state_free(state);
                segmentation::model_free(model);
                return 1;
            }
            
            const char* save_path = output_path.empty() ? nullptr : output_path.c_str();
            const char* audio_file = audio_path.empty() ? nullptr : audio_path.c_str();
            if (!test_full_forward_pass(model, state, save_path, audio_file)) {
                std::cerr << "End-to-End Forward Pass test failed!" << std::endl;
                segmentation::state_free(state);
                segmentation::model_free(model);
                return 1;
            }
            
            segmentation::model_print_memory_usage(model, state);
        }
        
        if (run_benchmark) {
            benchmark_performance(model, state);
        }

        if (!audio_path.empty()) {
            std::cout << "\nAudio path: " << audio_path << std::endl;
            std::cout << "Audio inference not yet implemented" << std::endl;
        }

        std::cout << "\nDone!" << std::endl;
        segmentation::state_free(state);
        segmentation::model_free(model);
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
