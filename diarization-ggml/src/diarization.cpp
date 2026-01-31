#include "diarization.h"
#include "aggregation.h"
#include "clustering.h"
#include "plda.h"
#include "powerset.h"
#include "rttm.h"
#include "vbx.h"

// Both model.h files have unique include guards (SEGMENTATION_GGML_MODEL_H, EMBEDDING_GGML_MODEL_H).
// Use explicit relative paths since both libraries export "model.h" in their PUBLIC includes.
#include "../../segmentation-ggml/src/model.h"
#include "../../embedding-ggml/src/model.h"
#include "fbank.h"

#ifdef EMBEDDING_USE_COREML
#include "coreml_bridge.h"
#endif

#ifdef SEGMENTATION_USE_COREML
#include "segmentation_coreml_bridge.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

// ============================================================================
// Constants — hardcoded pipeline parameters for community-1
// ============================================================================

static constexpr int SAMPLE_RATE             = 16000;
static constexpr int CHUNK_SAMPLES           = 160000;   // 10s at 16kHz
static constexpr int STEP_SAMPLES            = 16000;    // 1s step at 16kHz
static constexpr int FRAMES_PER_CHUNK        = 589;      // segmentation frames per chunk
static constexpr int NUM_POWERSET_CLASSES    = 7;        // powerset output classes
static constexpr int NUM_LOCAL_SPEAKERS      = 3;        // speakers after powerset->multilabel
static constexpr int EMBEDDING_DIM           = 256;
static constexpr int PLDA_DIM                = 128;
static constexpr int FBANK_NUM_BINS          = 80;

static constexpr double AHC_THRESHOLD        = 0.6;
static constexpr double VBX_FA               = 0.07;
static constexpr double VBX_FB               = 0.8;
static constexpr int    VBX_MAX_ITERS        = 20;

static constexpr double FRAME_DURATION       = 0.0619375; // model receptive field duration (NOT the step)
static constexpr double FRAME_STEP           = 0.016875;  // seconds
static constexpr double CHUNK_DURATION       = 10.0;      // seconds
static constexpr double CHUNK_STEP           = 1.0;       // seconds

// ============================================================================
// WAV loading — adapted from embedding-ggml/src/main.cpp
// ============================================================================

struct wav_header {
    char     riff[4];
    uint32_t file_size;
    char     wave[4];
    char     fmt[4];
    uint32_t fmt_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
};

struct wav_data_chunk {
    char     id[4];
    uint32_t size;
};

static bool load_wav_file(const std::string& path, std::vector<float>& samples,
                          uint32_t& sample_rate) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "ERROR: Failed to open WAV file: %s\n", path.c_str());
        return false;
    }

    wav_header header;
    file.read(reinterpret_cast<char*>(&header), sizeof(wav_header));

    if (std::strncmp(header.riff, "RIFF", 4) != 0 ||
        std::strncmp(header.wave, "WAVE", 4) != 0) {
        fprintf(stderr, "ERROR: Invalid WAV file format\n");
        return false;
    }

    if (header.audio_format != 1) {
        fprintf(stderr, "ERROR: Only PCM format supported (got format %d)\n",
                header.audio_format);
        return false;
    }

    if (header.num_channels != 1) {
        fprintf(stderr, "ERROR: Only mono audio supported (got %d channels)\n",
                header.num_channels);
        return false;
    }

    if (header.bits_per_sample != 16) {
        fprintf(stderr, "ERROR: Only 16-bit audio supported (got %d bits)\n",
                header.bits_per_sample);
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

// ============================================================================
// Embedding extraction (uses either CoreML or GGML backend)
// ============================================================================

bool extract_embeddings(
    const float* audio,
    int          num_samples,
    const float* binarized_segmentations,
    int          num_chunks,
    int          num_frames_per_chunk,
    int          num_speakers,
#ifdef EMBEDDING_USE_COREML
    struct embedding_coreml_context* coreml_ctx,
#else
    embedding::embedding_model& emb_model,
    embedding::embedding_state& emb_state,
#endif
    float*       embeddings_out)
{
    std::vector<float> cropped(CHUNK_SAMPLES, 0.0f);

    for (int c = 0; c < num_chunks; c++) {
        const int chunk_start = c * STEP_SAMPLES;
        int copy_len = num_samples - chunk_start;
        if (copy_len > CHUNK_SAMPLES) copy_len = CHUNK_SAMPLES;
        if (copy_len < 0) copy_len = 0;

        std::fill(cropped.begin(), cropped.end(), 0.0f);
        if (copy_len > 0) {
            std::memcpy(cropped.data(), audio + chunk_start,
                        static_cast<size_t>(copy_len) * sizeof(float));
        }

        embedding::fbank_result fbank =
            embedding::compute_fbank(cropped.data(), CHUNK_SAMPLES, SAMPLE_RATE);
        const int num_fbank_frames = fbank.num_frames;

        // seg layout: [num_chunks, num_frames_per_chunk, num_speakers] row-major
        const float* seg_chunk =
            binarized_segmentations + c * num_frames_per_chunk * num_speakers;

        for (int s = 0; s < num_speakers; s++) {
            float* emb_out =
                embeddings_out + (c * num_speakers + s) * EMBEDDING_DIM;

            bool all_zero = true;
            for (int f = 0; f < num_frames_per_chunk; f++) {
                if (seg_chunk[f * num_speakers + s] != 0.0f) {
                    all_zero = false;
                    break;
                }
            }

            if (all_zero) {
                const float nan_val = std::nanf("");
                for (int d = 0; d < EMBEDDING_DIM; d++) {
                    emb_out[d] = nan_val;
                }
                continue;
            }

            std::vector<float> masked_fbank(fbank.data);

            for (int ft = 0; ft < num_fbank_frames; ft++) {
                // fbank has ~998 frames, seg has 589; map via integer arithmetic
                int seg_frame =
                    static_cast<int>(
                        static_cast<long long>(ft) * num_frames_per_chunk
                        / num_fbank_frames);
                if (seg_frame >= num_frames_per_chunk) {
                    seg_frame = num_frames_per_chunk - 1;
                }

                const float mask_val = seg_chunk[seg_frame * num_speakers + s];
                if (mask_val == 0.0f) {
                    std::memset(&masked_fbank[ft * FBANK_NUM_BINS], 0,
                                FBANK_NUM_BINS * sizeof(float));
                }
            }

#ifdef EMBEDDING_USE_COREML
            embedding_coreml_encode(coreml_ctx,
                                    static_cast<int64_t>(num_fbank_frames),
                                    masked_fbank.data(),
                                    emb_out);
#else
            embedding::model_infer(emb_model, emb_state,
                                   masked_fbank.data(), num_fbank_frames,
                                   emb_out, EMBEDDING_DIM);
#endif
        }
    }

    return true;
}

// ============================================================================
// URI extraction — filename without extension
// ============================================================================

static std::string extract_uri(const std::string& audio_path) {
    std::string uri = audio_path;
    size_t last_slash = uri.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        uri = uri.substr(last_slash + 1);
    }
    size_t last_dot = uri.find_last_of('.');
    if (last_dot != std::string::npos) {
        uri = uri.substr(0, last_dot);
    }
    return uri;
}

// ============================================================================
// Main pipeline — ports SpeakerDiarization.apply() from speaker_diarization.py
// ============================================================================

bool diarize(const DiarizationConfig& config, DiarizationResult& result) {
    using Clock = std::chrono::high_resolution_clock;
    
    // Timing variables for each stage
    double t_load_audio_ms = 0.0;
    double t_load_seg_model_ms = 0.0;
    double t_load_emb_model_ms = 0.0;
    double t_load_plda_ms = 0.0;
    double t_segmentation_ms = 0.0;
    double t_powerset_ms = 0.0;
    double t_speaker_count_ms = 0.0;
    double t_embeddings_ms = 0.0;
    double t_filter_plda_ms = 0.0;
    double t_clustering_ms = 0.0;
    double t_postprocess_ms = 0.0;
    
    auto t_total_start = Clock::now();

    // ====================================================================
    // Step 1: Load audio WAV file
    // ====================================================================

    auto t_stage_start = Clock::now();
    std::vector<float> audio_samples;
    uint32_t sample_rate = 0;
    if (!load_wav_file(config.audio_path, audio_samples, sample_rate)) {
        return false;
    }
    if (sample_rate != SAMPLE_RATE) {
        fprintf(stderr, "Error: expected %d Hz audio, got %u Hz\n",
                SAMPLE_RATE, sample_rate);
        return false;
    }

    const int num_samples = static_cast<int>(audio_samples.size());
    const double audio_duration = static_cast<double>(num_samples) / SAMPLE_RATE;
    
    auto t_stage_end = Clock::now();
    t_load_audio_ms = std::chrono::duration<double, std::milli>(t_stage_end - t_stage_start).count();
    
    int audio_mins = static_cast<int>(audio_duration) / 60;
    int audio_secs = static_cast<int>(audio_duration) % 60;
    fprintf(stderr, "Audio: %.2fs (%d:%02d), %d samples\n", 
            audio_duration, audio_mins, audio_secs, num_samples);

    // ====================================================================
    // Step 2: Load segmentation model (CoreML or GGML)
    // ====================================================================

    t_stage_start = Clock::now();
#ifdef SEGMENTATION_USE_COREML
    struct segmentation_coreml_context* seg_coreml_ctx = nullptr;
    bool use_seg_coreml = !config.seg_coreml_path.empty();
#else
    bool use_seg_coreml = false;
#endif

    segmentation::segmentation_model seg_model = {};
    segmentation::segmentation_state seg_state = {};

#ifdef SEGMENTATION_USE_COREML
    if (use_seg_coreml) {
        seg_coreml_ctx = segmentation_coreml_init(config.seg_coreml_path.c_str());
        if (!seg_coreml_ctx) {
            fprintf(stderr, "Error: failed to load CoreML segmentation model '%s'\n",
                    config.seg_coreml_path.c_str());
            return false;
        }
    } else
#endif
    {
        if (!segmentation::model_load(config.seg_model_path, seg_model, false)) {
            fprintf(stderr, "Error: failed to load segmentation model '%s'\n",
                    config.seg_model_path.c_str());
            return false;
        }
        if (!segmentation::state_init(seg_state, seg_model, false)) {
            fprintf(stderr, "Error: failed to initialize segmentation state\n");
            segmentation::model_free(seg_model);
            return false;
        }
    }
    
    t_stage_end = Clock::now();
    t_load_seg_model_ms = std::chrono::duration<double, std::milli>(t_stage_end - t_stage_start).count();

    // ====================================================================
    // Step 3: Load embedding model (CoreML or GGML)
    // ====================================================================

    t_stage_start = Clock::now();
#ifdef EMBEDDING_USE_COREML
    struct embedding_coreml_context* coreml_ctx = nullptr;
    if (!config.coreml_path.empty()) {
        coreml_ctx = embedding_coreml_init(config.coreml_path.c_str());
    }
    if (!coreml_ctx) {
        fprintf(stderr, "Error: failed to load CoreML embedding model '%s'\n",
                config.coreml_path.c_str());
        if (seg_state.sched) segmentation::state_free(seg_state);
        if (seg_model.ctx) segmentation::model_free(seg_model);
        return false;
    }
#else
    embedding::embedding_model emb_model;
    embedding::embedding_state emb_state;

    if (!embedding::model_load(config.emb_model_path, emb_model, false)) {
        fprintf(stderr, "Error: failed to load embedding model '%s'\n",
                config.emb_model_path.c_str());
        if (seg_state.sched) segmentation::state_free(seg_state);
        if (seg_model.ctx) segmentation::model_free(seg_model);
        return false;
    }
    if (!embedding::state_init(emb_state, emb_model, false)) {
        fprintf(stderr, "Error: failed to initialize embedding state\n");
        embedding::model_free(emb_model);
        if (seg_state.sched) segmentation::state_free(seg_state);
        if (seg_model.ctx) segmentation::model_free(seg_model);
        return false;
    }
#endif
    
    t_stage_end = Clock::now();
    t_load_emb_model_ms = std::chrono::duration<double, std::milli>(t_stage_end - t_stage_start).count();

    // ====================================================================
    // Step 4: Load PLDA model
    // ====================================================================

    t_stage_start = Clock::now();
    diarization::PLDAModel plda;
    if (config.plda_path.empty()) {
        fprintf(stderr, "Error: --plda path is required for VBx clustering\n");
#ifdef EMBEDDING_USE_COREML
        embedding_coreml_free(coreml_ctx);
#else
        embedding::state_free(emb_state);
        embedding::model_free(emb_model);
#endif
        if (seg_state.sched) segmentation::state_free(seg_state);
        if (seg_model.ctx) segmentation::model_free(seg_model);
        return false;
    }
    if (!diarization::plda_load(config.plda_path, plda)) {
        fprintf(stderr, "Error: failed to load PLDA model '%s'\n",
                config.plda_path.c_str());
#ifdef EMBEDDING_USE_COREML
        embedding_coreml_free(coreml_ctx);
#else
        embedding::state_free(emb_state);
        embedding::model_free(emb_model);
#endif
        if (seg_state.sched) segmentation::state_free(seg_state);
        if (seg_model.ctx) segmentation::model_free(seg_model);
        return false;
    }
    
    t_stage_end = Clock::now();
    t_load_plda_ms = std::chrono::duration<double, std::milli>(t_stage_end - t_stage_start).count();

    // ====================================================================
    // Step 5: Sliding window segmentation
    // ====================================================================

    t_stage_start = Clock::now();
    int num_chunks;
    std::vector<float> seg_logits;

    num_chunks = std::max(1,
        1 + static_cast<int>(std::ceil((audio_duration - CHUNK_DURATION) / CHUNK_STEP)));
    fprintf(stderr, "Segmentation: %d chunks... ", num_chunks);
    fflush(stderr);

    seg_logits.resize(
        static_cast<size_t>(num_chunks) * FRAMES_PER_CHUNK * NUM_POWERSET_CLASSES);

    {
        std::vector<float> cropped(CHUNK_SAMPLES, 0.0f);
        for (int c = 0; c < num_chunks; c++) {
            const int chunk_start = c * STEP_SAMPLES;
            int copy_len = num_samples - chunk_start;
            if (copy_len > CHUNK_SAMPLES) copy_len = CHUNK_SAMPLES;
            if (copy_len < 0) copy_len = 0;

            std::fill(cropped.begin(), cropped.end(), 0.0f);
            if (copy_len > 0) {
                std::memcpy(cropped.data(), audio_samples.data() + chunk_start,
                            static_cast<size_t>(copy_len) * sizeof(float));
            }

            float* output = seg_logits.data() +
                static_cast<size_t>(c) * FRAMES_PER_CHUNK * NUM_POWERSET_CLASSES;

#ifdef SEGMENTATION_USE_COREML
            if (use_seg_coreml) {
                segmentation_coreml_infer(seg_coreml_ctx,
                                          cropped.data(), CHUNK_SAMPLES,
                                          output,
                                          FRAMES_PER_CHUNK * NUM_POWERSET_CLASSES);
                // CoreML output is already frame-major [589][7] — no transpose needed
            } else
#endif
            {
                if (!segmentation::model_infer(seg_model, seg_state,
                                               cropped.data(), CHUNK_SAMPLES,
                                               output,
                                               FRAMES_PER_CHUNK * NUM_POWERSET_CLASSES)) {
                    fprintf(stderr, "Error: segmentation failed at chunk %d/%d\n",
                            c + 1, num_chunks);
                    goto cleanup;
                }

                // Transpose GGML output from [class, frame] to [frame, class] layout
                // GGML stores [589, 7, 1] with ne[0]=589 contiguous, so memory is [7][589]
                // powerset_to_multilabel expects [589][7] (frame-major)
                {
                    std::vector<float> tmp(FRAMES_PER_CHUNK * NUM_POWERSET_CLASSES);
                    std::memcpy(tmp.data(), output, tmp.size() * sizeof(float));
                    for (int f = 0; f < FRAMES_PER_CHUNK; f++) {
                        for (int k = 0; k < NUM_POWERSET_CLASSES; k++) {
                            output[f * NUM_POWERSET_CLASSES + k] = tmp[k * FRAMES_PER_CHUNK + f];
                        }
                    }
                }
            }

            if ((c + 1) % 50 == 0 || c + 1 == num_chunks) {
                fprintf(stderr, "%d/%d chunks\r", c + 1, num_chunks);
                fflush(stderr);
            }
        }
    }

#ifdef SEGMENTATION_USE_COREML
    if (use_seg_coreml) {
        segmentation_coreml_free(seg_coreml_ctx);
        seg_coreml_ctx = nullptr;
    } else
#endif
    {
        segmentation::state_free(seg_state);
        segmentation::model_free(seg_model);
        seg_state.sched = nullptr;
        seg_model.ctx = nullptr;
    }
    
    t_stage_end = Clock::now();
    t_segmentation_ms = std::chrono::duration<double, std::milli>(t_stage_end - t_stage_start).count();
    fprintf(stderr, "done [%.0f ms]\n", t_segmentation_ms);

    {
        // ================================================================
        // Step 6: Powerset -> multilabel
        // (num_chunks, 589, 7) -> (num_chunks, 589, 3)
        // ================================================================

        t_stage_start = Clock::now();
        std::vector<float> binarized(
            static_cast<size_t>(num_chunks) * FRAMES_PER_CHUNK * NUM_LOCAL_SPEAKERS);
        diarization::powerset_to_multilabel(
            seg_logits.data(), num_chunks, FRAMES_PER_CHUNK, binarized.data());

        // Free logits — no longer needed
        seg_logits.clear();
        seg_logits.shrink_to_fit();
        
        t_stage_end = Clock::now();
        t_powerset_ms = std::chrono::duration<double, std::milli>(t_stage_end - t_stage_start).count();
        fprintf(stderr, "Powerset: done [%.0f ms]\n", t_powerset_ms);

        // ================================================================
        // Step 7: Compute frame-level speaker count
        // ================================================================

        t_stage_start = Clock::now();
        diarization::SlidingWindowParams chunk_window = {0.0, CHUNK_DURATION, CHUNK_STEP};
        diarization::SlidingWindowParams frame_window = {chunk_window.start, FRAME_DURATION, FRAME_STEP};

        std::vector<int> count;
        int total_frames = 0;
        diarization::compute_speaker_count(
            binarized.data(), num_chunks, FRAMES_PER_CHUNK, NUM_LOCAL_SPEAKERS,
            chunk_window, frame_window, count, total_frames);

        // Early exit if no speaker is ever active
        int max_count = 0;
        for (int i = 0; i < total_frames; i++) {
            if (count[i] > max_count) max_count = count[i];
        }
        
        t_stage_end = Clock::now();
        t_speaker_count_ms = std::chrono::duration<double, std::milli>(t_stage_end - t_stage_start).count();
        
        if (max_count == 0) {
            fprintf(stderr, "Speaker count: no speakers detected\n");
            result.segments.clear();
            std::string uri = extract_uri(config.audio_path);
            std::vector<diarization::RTTMSegment> empty;
            if (config.output_path.empty()) {
                diarization::write_rttm_stdout(empty, uri);
            } else {
                diarization::write_rttm(empty, uri, config.output_path);
            }
            goto cleanup_emb;
        }

        fprintf(stderr, "Speaker count: max %d speakers, %d frames [%.0f ms]\n",
                max_count, total_frames, t_speaker_count_ms);

        // ================================================================
        // Step 8: Extract embeddings
        // (num_chunks, 3, 256)
        // ================================================================

        t_stage_start = Clock::now();
        std::vector<float> embeddings(
            static_cast<size_t>(num_chunks) * NUM_LOCAL_SPEAKERS * EMBEDDING_DIM);

        fprintf(stderr, "Embeddings: %d chunks... ", num_chunks);
        fflush(stderr);
        if (!extract_embeddings(
                audio_samples.data(), num_samples,
                binarized.data(), num_chunks, FRAMES_PER_CHUNK, NUM_LOCAL_SPEAKERS,
#ifdef EMBEDDING_USE_COREML
                coreml_ctx,
#else
                emb_model, emb_state,
#endif
                embeddings.data())) {
            fprintf(stderr, "Error: embedding extraction failed\n");
            goto cleanup;
        }
        
        t_stage_end = Clock::now();
        t_embeddings_ms = std::chrono::duration<double, std::milli>(t_stage_end - t_stage_start).count();
        fprintf(stderr, "done [%.0f ms]\n", t_embeddings_ms);

        // Free embedding model — no longer needed
#ifdef EMBEDDING_USE_COREML
        embedding_coreml_free(coreml_ctx);
        coreml_ctx = nullptr;
#else
        embedding::state_free(emb_state);
        embedding::model_free(emb_model);
        emb_state.sched = nullptr;
        emb_model.ctx = nullptr;
#endif

        // Can also free audio samples now
        audio_samples.clear();
        audio_samples.shrink_to_fit();

        // ================================================================
        // Step 9: Filter embeddings
        // ================================================================

        t_stage_start = Clock::now();
        std::vector<float> filtered_emb;
        std::vector<int> filt_chunk_idx, filt_speaker_idx;
        diarization::filter_embeddings(
            embeddings.data(), num_chunks, NUM_LOCAL_SPEAKERS, EMBEDDING_DIM,
            binarized.data(), FRAMES_PER_CHUNK,
            filtered_emb, filt_chunk_idx, filt_speaker_idx);

        const int num_filtered = static_cast<int>(filt_chunk_idx.size());
        fprintf(stderr, "Filter: %d embeddings ", num_filtered);
        fflush(stderr);

        // ================================================================
        // Steps 10-17: Clustering + assignment
        // ================================================================

        std::vector<int> hard_clusters(
            static_cast<size_t>(num_chunks) * NUM_LOCAL_SPEAKERS, 0);
        int num_clusters = 1;

        if (num_filtered < 2) {
            t_stage_end = Clock::now();
            t_filter_plda_ms = std::chrono::duration<double, std::milli>(t_stage_end - t_stage_start).count();
            fprintf(stderr, "[%.0f ms]\n", t_filter_plda_ms);
            
            // Too few embeddings — assign all to single cluster
            fprintf(stderr, "Too few embeddings for clustering, using single cluster\n");

        } else {
            // Step 11: L2-normalize filtered embeddings (for AHC)
            std::vector<double> filtered_normed(
                static_cast<size_t>(num_filtered) * EMBEDDING_DIM);
            for (int i = 0; i < num_filtered; i++) {
                const float* src = filtered_emb.data() + i * EMBEDDING_DIM;
                double* dst = filtered_normed.data() + i * EMBEDDING_DIM;
                double norm = 0.0;
                for (int d = 0; d < EMBEDDING_DIM; d++) {
                    dst[d] = static_cast<double>(src[d]);
                    norm += dst[d] * dst[d];
                }
                norm = std::sqrt(norm);
                if (norm > 0.0) {
                    const double inv = 1.0 / norm;
                    for (int d = 0; d < EMBEDDING_DIM; d++) {
                        dst[d] *= inv;
                    }
                }
            }

            // Step 12: PLDA transform (on original filtered embeddings, NOT normalized)
            std::vector<double> filtered_double(
                static_cast<size_t>(num_filtered) * EMBEDDING_DIM);
            for (int i = 0; i < num_filtered * EMBEDDING_DIM; i++) {
                filtered_double[i] = static_cast<double>(filtered_emb[i]);
            }

            std::vector<double> plda_features(
                static_cast<size_t>(num_filtered) * PLDA_DIM);
            diarization::plda_transform(
                plda, filtered_double.data(), num_filtered, plda_features.data());
            
            t_stage_end = Clock::now();
            t_filter_plda_ms = std::chrono::duration<double, std::milli>(t_stage_end - t_stage_start).count();
            fprintf(stderr, "[%.0f ms]\n", t_filter_plda_ms);
            
            // Step 13: AHC cluster (on normalized embeddings)
            t_stage_start = Clock::now();
            std::vector<int> ahc_clusters;
            diarization::ahc_cluster(
                filtered_normed.data(), num_filtered, EMBEDDING_DIM,
                AHC_THRESHOLD, ahc_clusters);

            int num_ahc_clusters = 0;
            for (int i = 0; i < num_filtered; i++) {
                if (ahc_clusters[i] + 1 > num_ahc_clusters)
                    num_ahc_clusters = ahc_clusters[i] + 1;
            }
            fprintf(stderr, "PLDA: done ");
            fflush(stderr);

            // Free intermediates
            filtered_normed.clear();
            filtered_normed.shrink_to_fit();
            filtered_double.clear();
            filtered_double.shrink_to_fit();

            fprintf(stderr, "AHC: %d clusters ", num_ahc_clusters);
            fflush(stderr);
            
            auto t_ahc_end = Clock::now();
            double t_ahc_ms = std::chrono::duration<double, std::milli>(t_ahc_end - t_stage_start).count();
            fprintf(stderr, "[%.0f ms]\n", t_ahc_ms);

            // Step 14: VBx clustering
            t_stage_start = Clock::now();
            diarization::VBxResult vbx_result;
            if (!diarization::vbx_cluster(
                    ahc_clusters.data(), num_filtered, num_ahc_clusters,
                    plda_features.data(), PLDA_DIM,
                    plda.plda_psi.data(), VBX_FA, VBX_FB, VBX_MAX_ITERS,
                    vbx_result)) {
                fprintf(stderr, "Error: VBx clustering failed\n");
                result.segments.clear();
                return false;
            }

            // Step 15: Compute centroids from VBx soft assignments
            // W = gamma[:, pi > 1e-7], centroids = W.T @ train_emb / W.sum(0).T
            const int vbx_S = vbx_result.num_speakers;
            const int vbx_T = vbx_result.num_frames;

            std::vector<int> sig_speakers;
            for (int s = 0; s < vbx_S; s++) {
                if (vbx_result.pi[s] > 1e-7) {
                    sig_speakers.push_back(s);
                }
            }
            num_clusters = static_cast<int>(sig_speakers.size());
            if (num_clusters == 0) num_clusters = 1;  // fallback

            t_stage_end = Clock::now();
            double t_vbx_ms = std::chrono::duration<double, std::milli>(t_stage_end - t_stage_start).count();
            t_clustering_ms = t_ahc_ms + t_vbx_ms;
            fprintf(stderr, "VBx: %d speakers [%.0f ms]\n", num_clusters, t_vbx_ms);

            // centroids: (num_clusters, EMBEDDING_DIM)
            std::vector<float> centroids(
                static_cast<size_t>(num_clusters) * EMBEDDING_DIM, 0.0f);
            std::vector<double> w_col_sum(num_clusters, 0.0);

            for (int k = 0; k < num_clusters; k++) {
                const int s = sig_speakers[k];
                for (int t = 0; t < vbx_T; t++) {
                    const double w = vbx_result.gamma[t * vbx_S + s];
                    w_col_sum[k] += w;
                    const float* emb = filtered_emb.data() + t * EMBEDDING_DIM;
                    float* cent = centroids.data() + k * EMBEDDING_DIM;
                    for (int d = 0; d < EMBEDDING_DIM; d++) {
                        cent[d] += static_cast<float>(w * static_cast<double>(emb[d]));
                    }
                }
            }
            for (int k = 0; k < num_clusters; k++) {
                if (w_col_sum[k] > 0.0) {
                    float* cent = centroids.data() + k * EMBEDDING_DIM;
                    const float inv = static_cast<float>(1.0 / w_col_sum[k]);
                    for (int d = 0; d < EMBEDDING_DIM; d++) {
                        cent[d] *= inv;
                    }
                }
            }

            // Step 16: Compute soft clusters and assign all embeddings
            // cosine distance: all embeddings vs centroids
            // soft_clusters = 2 - cosine_distance (similarity, range [0, 2])
            const int total_emb = num_chunks * NUM_LOCAL_SPEAKERS;
            std::vector<float> soft_clusters(
                static_cast<size_t>(total_emb) * num_clusters);

            for (int e = 0; e < total_emb; e++) {
                const float* emb = embeddings.data() + e * EMBEDDING_DIM;
                for (int k = 0; k < num_clusters; k++) {
                    const float* cent = centroids.data() + k * EMBEDDING_DIM;
                    double dist = diarization::cosine_distance(emb, cent, EMBEDDING_DIM);
                    soft_clusters[e * num_clusters + k] =
                        static_cast<float>(2.0 - dist);
                }
            }

            // Constrained argmax (Hungarian per chunk)
            diarization::constrained_argmax(
                soft_clusters.data(), num_chunks, NUM_LOCAL_SPEAKERS, num_clusters,
                hard_clusters);
        }

        // Step 17: Mark inactive speakers as -2
        // inactive = sum(binarized[c, :, s]) == 0
        t_stage_start = Clock::now();
        for (int c = 0; c < num_chunks; c++) {
            const float* seg_chunk =
                binarized.data() + c * FRAMES_PER_CHUNK * NUM_LOCAL_SPEAKERS;
            for (int s = 0; s < NUM_LOCAL_SPEAKERS; s++) {
                float sum = 0.0f;
                for (int f = 0; f < FRAMES_PER_CHUNK; f++) {
                    sum += seg_chunk[f * NUM_LOCAL_SPEAKERS + s];
                }
                if (sum == 0.0f) {
                    hard_clusters[c * NUM_LOCAL_SPEAKERS + s] = -2;
                }
            }
        }

        // ================================================================
        // Step 18: Reconstruct — build clustered segmentations
        // (num_chunks, 589, num_clusters) from binarized + hard_clusters
        // ================================================================

        std::vector<float> clustered_seg(
            static_cast<size_t>(num_chunks) * FRAMES_PER_CHUNK * num_clusters);
        {
            const float nan_val = std::nanf("");
            std::fill(clustered_seg.begin(), clustered_seg.end(), nan_val);
        }

        for (int c = 0; c < num_chunks; c++) {
            const int* chunk_clusters = hard_clusters.data() + c * NUM_LOCAL_SPEAKERS;
            const float* seg_chunk =
                binarized.data() + c * FRAMES_PER_CHUNK * NUM_LOCAL_SPEAKERS;

            for (int k = 0; k < num_clusters; k++) {
                // Find all local speakers assigned to cluster k in this chunk
                for (int s = 0; s < NUM_LOCAL_SPEAKERS; s++) {
                    if (chunk_clusters[s] != k) continue;

                    for (int f = 0; f < FRAMES_PER_CHUNK; f++) {
                        const float val =
                            seg_chunk[f * NUM_LOCAL_SPEAKERS + s];
                        float& out =
                            clustered_seg[
                                (c * FRAMES_PER_CHUNK + f) * num_clusters + k];
                        if (std::isnan(out)) {
                            out = val;
                        } else {
                            out = std::max(out, val);
                        }
                    }
                }
            }
        }

        // ================================================================
        // Step 19: to_diarization — aggregate + select top-count speakers
        // ================================================================

        std::vector<float> discrete_diarization;
        diarization::to_diarization(
            clustered_seg.data(), num_chunks, FRAMES_PER_CHUNK, num_clusters,
            count.data(), total_frames,
            chunk_window, frame_window,
            discrete_diarization);

        // Free intermediates
        clustered_seg.clear();
        clustered_seg.shrink_to_fit();

        // ================================================================
        // Step 20: Convert to RTTM segments
        // Each contiguous run of 1.0 in a speaker column -> one segment
        // ================================================================

        const int out_frames =
            static_cast<int>(discrete_diarization.size()) / num_clusters;

        std::vector<diarization::RTTMSegment> rttm_segments;

        for (int k = 0; k < num_clusters; k++) {
            char speaker_label[16];
            snprintf(speaker_label, sizeof(speaker_label), "SPEAKER_%02d", k);

            bool in_segment = false;
            int seg_start_frame = 0;

            for (int f = 0; f <= out_frames; f++) {
                bool active = false;
                if (f < out_frames) {
                    active = (discrete_diarization[f * num_clusters + k] == 1.0f);
                }

                if (active && !in_segment) {
                    seg_start_frame = f;
                    in_segment = true;
                } else if (!active && in_segment) {
                    // Segment: [seg_start_frame, f) frames
                    // Use frame midpoint (matches Python Binarize: timestamps = [frames[i].middle for i in ...])
                    const double start_time =
                        chunk_window.start + seg_start_frame * FRAME_STEP + 0.5 * FRAME_DURATION;
                    const double duration =
                        (f - seg_start_frame) * FRAME_STEP;
                    if (duration > 0.0) {
                        rttm_segments.push_back(
                            {start_time, duration, speaker_label});
                    }
                    in_segment = false;
                }
            }
        }

        // Sort segments by start time
        std::sort(rttm_segments.begin(), rttm_segments.end(),
                  [](const diarization::RTTMSegment& a,
                     const diarization::RTTMSegment& b) {
                      return a.start < b.start;
                  });

        // ================================================================
        // Step 21: Write RTTM output
        // ================================================================

        std::string uri = extract_uri(config.audio_path);

        if (config.output_path.empty()) {
            diarization::write_rttm_stdout(rttm_segments, uri);
        } else {
            diarization::write_rttm(rttm_segments, uri, config.output_path);
        }

        // Populate result struct
        result.segments.clear();
        result.segments.reserve(rttm_segments.size());
        for (const auto& seg : rttm_segments) {
            result.segments.push_back({seg.start, seg.duration, seg.speaker});
        }

        t_stage_end = Clock::now();
        t_postprocess_ms = std::chrono::duration<double, std::milli>(t_stage_end - t_stage_start).count();
        fprintf(stderr, "Assignment + reconstruction: done [%.0f ms]\n", t_postprocess_ms);
        fprintf(stderr, "RTTM: %zu segments [0 ms]\n", rttm_segments.size());
        
        auto t_total_end = Clock::now();
        double t_total_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();
        
        fprintf(stderr, "\n=== Timing Summary ===\n");
        fprintf(stderr, "  Load audio:      %6.0f ms\n", t_load_audio_ms);
        fprintf(stderr, "  Load seg model:  %6.0f ms\n", t_load_seg_model_ms);
        fprintf(stderr, "  Load emb model:  %6.0f ms\n", t_load_emb_model_ms);
        fprintf(stderr, "  Load PLDA:       %6.0f ms\n", t_load_plda_ms);
        fprintf(stderr, "  Segmentation:    %6.0f ms  (%.1f%%)\n", 
                t_segmentation_ms, 100.0 * t_segmentation_ms / t_total_ms);
        fprintf(stderr, "  Powerset:        %6.0f ms\n", t_powerset_ms);
        fprintf(stderr, "  Speaker count:   %6.0f ms\n", t_speaker_count_ms);
        fprintf(stderr, "  Embeddings:      %6.0f ms  (%.1f%%)\n", 
                t_embeddings_ms, 100.0 * t_embeddings_ms / t_total_ms);
        fprintf(stderr, "  Filter+PLDA:     %6.0f ms\n", t_filter_plda_ms);
        fprintf(stderr, "  AHC+VBx:         %6.0f ms\n", t_clustering_ms);
        fprintf(stderr, "  Post-process:    %6.0f ms\n", t_postprocess_ms);
        fprintf(stderr, "  ─────────────────────────\n");
        
        int total_mins = static_cast<int>(t_total_ms / 1000.0) / 60;
        double total_secs = (t_total_ms / 1000.0) - (total_mins * 60);
        fprintf(stderr, "  Total:           %6.0f ms  (%d:%.1f)\n", 
                t_total_ms, total_mins, total_secs);
        
        fprintf(stderr, "\nDiarization complete: %zu segments, %d speakers\n",
                rttm_segments.size(), num_clusters);
        return true;
    }

    // --- cleanup labels for error/early-exit paths ---
cleanup_emb:
#ifdef EMBEDDING_USE_COREML
    if (coreml_ctx) embedding_coreml_free(coreml_ctx);
#else
    if (emb_state.sched) embedding::state_free(emb_state);
    if (emb_model.ctx) embedding::model_free(emb_model);
#endif
    return true;

cleanup:
#ifdef EMBEDDING_USE_COREML
    if (coreml_ctx) embedding_coreml_free(coreml_ctx);
#else
    if (emb_state.sched) embedding::state_free(emb_state);
    if (emb_model.ctx) embedding::model_free(emb_model);
#endif
#ifdef SEGMENTATION_USE_COREML
    if (seg_coreml_ctx) segmentation_coreml_free(seg_coreml_ctx);
#endif
    if (seg_state.sched) segmentation::state_free(seg_state);
    if (seg_model.ctx) segmentation::model_free(seg_model);
    return false;
}
