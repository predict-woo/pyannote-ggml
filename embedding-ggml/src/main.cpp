#include "model.h"
#include "fbank.h"
#ifdef EMBEDDING_USE_COREML
#include "coreml_bridge.h"
#endif
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdint>
#include <chrono>

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

static bool test_fbank(const std::string& audio_path) {
    printf("\n");
    printf("============================================================\n");
    printf("Testing Fbank Feature Extraction\n");
    printf("============================================================\n");

    std::vector<float> audio_samples;
    uint32_t sr = 0;

    printf("Loading audio from: %s\n", audio_path.c_str());
    if (!load_wav_file(audio_path, audio_samples, sr)) {
        fprintf(stderr, "ERROR: Failed to load audio file\n");
        return false;
    }

    if (sr != 16000) {
        fprintf(stderr, "ERROR: Audio must be 16kHz (got %d Hz)\n", sr);
        return false;
    }

    printf("  Loaded %zu samples (%.2f seconds)\n", audio_samples.size(), audio_samples.size() / 16000.0);

    int max_samples = 5 * 16000;
    if (static_cast<int>(audio_samples.size()) > max_samples) {
        printf("  Truncating to first 5 seconds (%d samples)\n", max_samples);
        audio_samples.resize(max_samples);
    }

    printf("\nComputing fbank features...\n");
    embedding::fbank_result result = embedding::compute_fbank(audio_samples.data(),
                                                              static_cast<int>(audio_samples.size()),
                                                              16000);

    printf("  Shape: [%d, %d]\n", result.num_frames, result.num_bins);

    float min_val = result.data[0];
    float max_val = result.data[0];
    float sum = 0.0f;
    int total = result.num_frames * result.num_bins;
    for (int i = 0; i < total; i++) {
        if (result.data[i] < min_val) min_val = result.data[i];
        if (result.data[i] > max_val) max_val = result.data[i];
        sum += result.data[i];
    }
    printf("  Min: %.6f\n", min_val);
    printf("  Max: %.6f\n", max_val);
    printf("  Mean: %.6f\n", sum / total);

    std::string output_path = "fbank_output.bin";
    printf("\nSaving output to: %s\n", output_path.c_str());
    FILE* f = fopen(output_path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "ERROR: Failed to open output file\n");
        return false;
    }

    int32_t num_frames = result.num_frames;
    int32_t num_bins = result.num_bins;
    fwrite(&num_frames, sizeof(int32_t), 1, f);
    fwrite(&num_bins, sizeof(int32_t), 1, f);
    fwrite(result.data.data(), sizeof(float), total, f);
    fclose(f);

    printf("  Saved [%d, %d] = %d floats\n", num_frames, num_bins, total);
    printf("\n============================================================\n");
    printf("SUCCESS: Fbank test completed\n");
    printf("============================================================\n");

    return true;
}

static void benchmark_performance(
    embedding::embedding_model& model,
    embedding::embedding_state& state,
    const float* fbank_data,
    int num_frames) {

    printf("\n");
    printf("============================================================\n");
    printf("Performance Benchmark\n");
    printf("============================================================\n");

    const int num_iterations = 5;
    const int warmup_iterations = 2;
    const int embed_dim = model.hparams.embed_dim;
    std::vector<float> embedding(embed_dim);

    long long total_ms = 0;

    for (int iter = 0; iter < num_iterations + warmup_iterations; iter++) {
        auto start = std::chrono::high_resolution_clock::now();

        bool success = embedding::model_infer(
            model, state,
            fbank_data, num_frames,
            embedding.data(), embed_dim);

        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (success) {
            if (iter >= warmup_iterations) {
                total_ms += ms;
                printf("  Iteration %d: %lld ms\n", iter - warmup_iterations + 1, (long long)ms);
            } else {
                printf("  Warmup %d: %lld ms\n", iter + 1, (long long)ms);
            }
        } else {
            printf("  Iteration %d: FAILED\n", iter + 1);
        }
    }

    long long avg_ms = total_ms / num_iterations;
    float audio_duration_s = (float)num_frames * 10.0f / 1000.0f;

    printf("\n============================================================\n");
    printf("Benchmark Summary\n");
    printf("============================================================\n");
    printf("Audio duration: %.1f seconds (fbank frames: %d)\n", audio_duration_s, num_frames);
    printf("Avg inference:  %lld ms (%.1fx real-time)\n", avg_ms, (audio_duration_s * 1000.0) / avg_ms);
    printf("Primary backend: CPU\n");
    printf("============================================================\n");
}

int main(int argc, char** argv) {
    std::cout << "Embedding GGML - WeSpeaker ResNet34 Speaker Embedding Model" << std::endl;
    std::cout << "============================================================" << std::endl;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf> [options]" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  --test               Print tensor summary and verify model" << std::endl;
        std::cerr << "  --test-fbank         Run fbank feature extraction test (requires --audio)" << std::endl;
        std::cerr << "  --test-inference     Run full inference test (requires --audio)" << std::endl;
        std::cerr << "  --benchmark          Run performance benchmark (requires --audio)" << std::endl;
        std::cerr << "  --audio <path>       Path to WAV file (16kHz mono PCM)" << std::endl;
        std::cerr << "  --coreml <path>      Use CoreML model (.mlpackage) for inference" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::cout << "Model path: " << model_path << std::endl;

    bool run_test = false;
    bool run_test_fbank = false;
    bool run_test_inference = false;
    bool run_benchmark = false;
    std::string audio_path;
    std::string coreml_path;
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--test") {
            run_test = true;
        } else if (arg == "--test-fbank") {
            run_test_fbank = true;
        } else if (arg == "--test-inference") {
            run_test_inference = true;
        } else if (arg == "--benchmark") {
            run_benchmark = true;
        } else if (arg == "--audio" && i + 1 < argc) {
            audio_path = argv[++i];
        } else if (arg == "--coreml" && i + 1 < argc) {
            coreml_path = argv[++i];
        }
    }

    bool use_coreml = !coreml_path.empty();
    if (use_coreml) {
#ifndef EMBEDDING_USE_COREML
        fprintf(stderr, "ERROR: CoreML support not compiled. Rebuild with -DEMBEDDING_COREML=ON\n");
        return 1;
#endif
    }

    if (run_test_fbank) {
        if (audio_path.empty()) {
            std::cerr << "Error: --test-fbank requires --audio <path>" << std::endl;
            return 1;
        }
        bool ok = test_fbank(audio_path);
        return ok ? 0 : 1;
    }

    embedding::embedding_model model;

    std::cout << "\nLoading model..." << std::endl;
    if (!embedding::model_load(model_path, model)) {
        std::cerr << "Error: Failed to load model from " << model_path << std::endl;
        return 1;
    }

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Model loaded successfully!" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    embedding::model_print_info(model);

    if (run_test_inference) {
        if (audio_path.empty()) {
            std::cerr << "Error: --test-inference requires --audio <path>" << std::endl;
            embedding::model_free(model);
            return 1;
        }

        printf("\n");
        printf("============================================================\n");
        printf("Running Full Inference Test\n");
        printf("============================================================\n");

        std::vector<float> audio_samples;
        uint32_t sr = 0;

        printf("Loading audio from: %s\n", audio_path.c_str());
        if (!load_wav_file(audio_path, audio_samples, sr)) {
            fprintf(stderr, "ERROR: Failed to load audio file\n");
            embedding::model_free(model);
            return 1;
        }

        if (sr != 16000) {
            fprintf(stderr, "ERROR: Audio must be 16kHz (got %d Hz)\n", sr);
            embedding::model_free(model);
            return 1;
        }

        printf("  Loaded %zu samples (%.2f seconds)\n", audio_samples.size(), audio_samples.size() / 16000.0);

        int max_samples = 10 * 16000;
        if (static_cast<int>(audio_samples.size()) > max_samples) {
            printf("  Truncating to first 10 seconds (%d samples)\n", max_samples);
            audio_samples.resize(max_samples);
        }

        printf("\nComputing fbank features...\n");
        auto fbank_start = std::chrono::high_resolution_clock::now();
        embedding::fbank_result fbank = embedding::compute_fbank(
            audio_samples.data(), static_cast<int>(audio_samples.size()), 16000);
        auto fbank_end = std::chrono::high_resolution_clock::now();
        auto fbank_ms = std::chrono::duration_cast<std::chrono::milliseconds>(fbank_end - fbank_start).count();
        printf("  Fbank shape: [%d, %d]\n", fbank.num_frames, fbank.num_bins);
        printf("  Fbank time: %lld ms\n", (long long)fbank_ms);

        const int embed_dim = model.hparams.embed_dim;
        std::vector<float> embedding(embed_dim);
        long long infer_ms = 0;

#ifdef EMBEDDING_USE_COREML
        if (use_coreml) {
            printf("\n[CoreML] Loading model from: %s\n", coreml_path.c_str());
            auto coreml_load_start = std::chrono::high_resolution_clock::now();
            struct embedding_coreml_context * coreml_ctx = embedding_coreml_init(coreml_path.c_str());
            auto coreml_load_end = std::chrono::high_resolution_clock::now();
            auto coreml_load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(coreml_load_end - coreml_load_start).count();

            if (!coreml_ctx) {
                fprintf(stderr, "[CoreML] ERROR: Failed to load CoreML model\n");
                embedding::model_free(model);
                return 1;
            }
            printf("[CoreML] Model loaded in %lld ms\n", (long long)coreml_load_ms);

            printf("\n[CoreML] Running inference...\n");
            auto infer_start = std::chrono::high_resolution_clock::now();
            embedding_coreml_encode(coreml_ctx, fbank.num_frames,
                                    fbank.data.data(), embedding.data());
            auto infer_end = std::chrono::high_resolution_clock::now();
            infer_ms = std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - infer_start).count();

            embedding_coreml_free(coreml_ctx);
            printf("[CoreML] Inference completed\n");
        } else
#endif
        {
            printf("\nInitializing inference state...\n");
            embedding::embedding_state state;
            if (!embedding::state_init(state, model)) {
                fprintf(stderr, "ERROR: Failed to initialize inference state\n");
                embedding::model_free(model);
                return 1;
            }

            printf("\nRunning inference...\n");
            auto infer_start = std::chrono::high_resolution_clock::now();
            if (!embedding::model_infer(model, state, fbank.data.data(), fbank.num_frames,
                                         embedding.data(), embed_dim)) {
                fprintf(stderr, "ERROR: Inference failed\n");
                embedding::state_free(state);
                embedding::model_free(model);
                return 1;
            }
            auto infer_end = std::chrono::high_resolution_clock::now();
            infer_ms = std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - infer_start).count();

            embedding::model_print_memory_usage(model, state);
            embedding::state_free(state);
        }

        const char * backend_label = use_coreml ? "[CoreML] " : "";

        printf("\n%s*** Inference time: %lld ms ***\n", backend_label, (long long)infer_ms);
        printf("%s*** Total time (fbank + inference): %lld ms ***\n", backend_label, (long long)(fbank_ms + infer_ms));

        printf("\n============================================================\n");
        printf("%sEmbedding Output (%d dimensions)\n", backend_label, embed_dim);
        printf("============================================================\n");

        printf("  First 10 values: [");
        for (int i = 0; i < 10 && i < embed_dim; i++) {
            printf("%.6f%s", embedding[i], i < 9 ? ", " : "");
        }
        printf("]\n");

        float min_val = embedding[0], max_val = embedding[0], sum = 0.0f, sum_sq = 0.0f;
        for (int i = 0; i < embed_dim; i++) {
            if (embedding[i] < min_val) min_val = embedding[i];
            if (embedding[i] > max_val) max_val = embedding[i];
            sum += embedding[i];
            sum_sq += embedding[i] * embedding[i];
        }
        float mean = sum / embed_dim;
        float l2_norm = std::sqrt(sum_sq);

        printf("  Min: %.6f\n", min_val);
        printf("  Max: %.6f\n", max_val);
        printf("  Mean: %.6f\n", mean);
        printf("  L2 norm: %.6f\n", l2_norm);

        std::string output_path = "embedding_output.bin";
        printf("\nSaving to: %s\n", output_path.c_str());
        FILE* f = fopen(output_path.c_str(), "wb");
        if (f) {
            int32_t dim = embed_dim;
            fwrite(&dim, sizeof(int32_t), 1, f);
            fwrite(embedding.data(), sizeof(float), embed_dim, f);
            fclose(f);
            printf("  Saved [int32 dim=%d, float32 data[%d]]\n", dim, embed_dim);
        } else {
            fprintf(stderr, "WARNING: Failed to save output file\n");
        }

        printf("\n============================================================\n");
        printf("SUCCESS: Inference test completed\n");
        printf("============================================================\n");

        embedding::model_free(model);
        return 0;
    }

    if (run_benchmark) {
        if (audio_path.empty()) {
            std::cerr << "Error: --benchmark requires --audio <path>" << std::endl;
            embedding::model_free(model);
            return 1;
        }

        std::vector<float> audio_samples;
        uint32_t sr = 0;

        printf("\nLoading audio from: %s\n", audio_path.c_str());
        if (!load_wav_file(audio_path, audio_samples, sr)) {
            fprintf(stderr, "ERROR: Failed to load audio file\n");
            embedding::model_free(model);
            return 1;
        }

        if (sr != 16000) {
            fprintf(stderr, "ERROR: Audio must be 16kHz (got %d Hz)\n", sr);
            embedding::model_free(model);
            return 1;
        }

        int max_samples = 10 * 16000;
        if (static_cast<int>(audio_samples.size()) > max_samples) {
            printf("  Truncating to first 10 seconds (%d samples)\n", max_samples);
            audio_samples.resize(max_samples);
        }

        embedding::fbank_result fbank = embedding::compute_fbank(
            audio_samples.data(), static_cast<int>(audio_samples.size()), 16000);
        printf("  Fbank shape: [%d, %d]\n", fbank.num_frames, fbank.num_bins);

        embedding::embedding_state state;
        if (!embedding::state_init(state, model)) {
            fprintf(stderr, "ERROR: Failed to initialize inference state\n");
            embedding::model_free(model);
            return 1;
        }

        benchmark_performance(model, state, fbank.data.data(), fbank.num_frames);

        embedding::model_print_memory_usage(model, state);

        embedding::state_free(state);
        embedding::model_free(model);
        return 0;
    }

    if (run_test) {
        bool ok = embedding::model_verify(model);
        if (!ok) {
            std::cerr << "Model verification failed!" << std::endl;
            embedding::model_free(model);
            return 1;
        }

        std::cout << "\nHyperparameters:" << std::endl;
        std::cout << "  Architecture: WeSpeaker ResNet34" << std::endl;
        std::cout << "  Sample rate: " << model.hparams.sample_rate << " Hz" << std::endl;
        std::cout << "  Mel bins: " << model.hparams.num_mel_bins << std::endl;
        std::cout << "  Frame length: " << model.hparams.frame_length << " ms" << std::endl;
        std::cout << "  Frame shift: " << model.hparams.frame_shift << " ms" << std::endl;
        std::cout << "  Embedding dim: " << model.hparams.embed_dim << std::endl;
        std::cout << "  Feature dim: " << model.hparams.feat_dim << std::endl;

        std::cout << "\nAll tests passed!" << std::endl;
    }

    std::cout << "\nDone!" << std::endl;
    embedding::model_free(model);
    return 0;
}
