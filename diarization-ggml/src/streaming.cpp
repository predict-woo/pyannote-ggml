#include "streaming.h"
#include "aggregation.h"
#include "clustering.h"
#include "diarization.h"
#include "plda.h"
#include "powerset.h"
#include "vbx.h"
#include "../../models/embedding-ggml/src/fbank.h"

#ifdef SEGMENTATION_USE_COREML
#include "segmentation_coreml_bridge.h"
#endif
#ifdef EMBEDDING_USE_COREML
#include "coreml_bridge.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

static constexpr int SAMPLE_RATE             = 16000;
static constexpr int CHUNK_SAMPLES           = 160000;
static constexpr int STEP_SAMPLES            = 16000;
static constexpr int FRAMES_PER_CHUNK        = 589;
static constexpr int NUM_POWERSET_CLASSES    = 7;
static constexpr int NUM_LOCAL_SPEAKERS      = 3;
static constexpr int EMBEDDING_DIM           = 256;
static constexpr int FBANK_NUM_BINS          = 80;
static constexpr double FRAME_STEP_SEC       = 0.016875;

static int chunk_global_start_frame(int chunk_idx) {
    return static_cast<int>(std::lround(
        static_cast<double>(chunk_idx) * 1.0 / FRAME_STEP_SEC));
}

// Process a single chunk at state->chunks_processed. The chunk audio is cropped
// from audio_buffer with zero-padding if it extends past total_samples.
// Returns true on success, false on segmentation/embedding failure.
static bool process_one_chunk(StreamingState* state, int total_samples) {
    const int chunk_start = state->chunks_processed * STEP_SAMPLES - state->samples_trimmed;
    int copy_start = chunk_start;
    int copy_len = CHUNK_SAMPLES;
    int pad_front = 0;

    if (copy_start < 0) {
        pad_front = -copy_start;
        copy_start = 0;
        copy_len = CHUNK_SAMPLES - pad_front;
    }

    if (copy_start + copy_len > total_samples) {
        copy_len = total_samples - copy_start;
    }
    if (copy_len < 0) copy_len = 0;

    std::vector<float> cropped(CHUNK_SAMPLES, 0.0f);
    if (copy_len > 0) {
        std::memcpy(cropped.data() + pad_front,
                   state->audio_buffer.data() + copy_start,
                   static_cast<size_t>(copy_len) * sizeof(float));
    }

    std::vector<float> chunk_logits(FRAMES_PER_CHUNK * NUM_POWERSET_CLASSES);

#ifdef SEGMENTATION_USE_COREML
    if (state->seg_coreml_ctx) {
        segmentation_coreml_infer(state->seg_coreml_ctx,
                                  cropped.data(), CHUNK_SAMPLES,
                                  chunk_logits.data(),
                                  FRAMES_PER_CHUNK * NUM_POWERSET_CLASSES);
    } else {
        fprintf(stderr, "Error: no segmentation model available\n");
        return false;
    }
#else
    fprintf(stderr, "Error: non-CoreML segmentation not supported in streaming\n");
    return false;
#endif

    std::vector<float> chunk_binarized(FRAMES_PER_CHUNK * NUM_LOCAL_SPEAKERS);
    diarization::powerset_to_multilabel(
        chunk_logits.data(), 1, FRAMES_PER_CHUNK, chunk_binarized.data());

    state->binarized.insert(
        state->binarized.end(),
        chunk_binarized.begin(),
        chunk_binarized.end());

    embedding::fbank_result fbank =
        embedding::compute_fbank(cropped.data(), CHUNK_SAMPLES, SAMPLE_RATE);
    const int num_fbank_frames = fbank.num_frames;

    for (int s = 0; s < NUM_LOCAL_SPEAKERS; s++) {
        bool all_zero = true;
        for (int f = 0; f < FRAMES_PER_CHUNK; f++) {
            if (chunk_binarized[f * NUM_LOCAL_SPEAKERS + s] != 0.0f) {
                all_zero = false;
                break;
            }
        }

        std::vector<float> embedding(EMBEDDING_DIM);

        if (all_zero) {
            const float nan_val = std::nanf("");
            for (int d = 0; d < EMBEDDING_DIM; d++) {
                embedding[d] = nan_val;
            }
        } else {
            std::vector<float> masked_fbank(fbank.data);

            for (int ft = 0; ft < num_fbank_frames; ft++) {
                int seg_frame =
                    static_cast<int>(
                        static_cast<long long>(ft) * FRAMES_PER_CHUNK / num_fbank_frames);
                if (seg_frame >= FRAMES_PER_CHUNK) {
                    seg_frame = FRAMES_PER_CHUNK - 1;
                }

                const float mask_val = chunk_binarized[seg_frame * NUM_LOCAL_SPEAKERS + s];
                if (mask_val == 0.0f) {
                    std::memset(&masked_fbank[ft * FBANK_NUM_BINS], 0,
                               FBANK_NUM_BINS * sizeof(float));
                }
            }

#ifdef EMBEDDING_USE_COREML
            if (state->emb_coreml_ctx) {
                embedding_coreml_encode(state->emb_coreml_ctx,
                                       static_cast<int64_t>(num_fbank_frames),
                                       masked_fbank.data(),
                                       embedding.data());
            } else {
                fprintf(stderr, "Error: no embedding model available\n");
                continue;
            }
#else
            fprintf(stderr, "Error: non-CoreML embedding not supported in streaming\n");
            continue;
#endif
        }

        state->embeddings.insert(
            state->embeddings.end(),
            embedding.begin(),
            embedding.end());
        state->chunk_idx.push_back(state->chunks_processed);
        state->local_speaker_idx.push_back(s);
    }

    state->chunks_processed++;
    state->audio_time_processed =
        static_cast<double>(state->chunks_processed) *
        static_cast<double>(STEP_SAMPLES) / static_cast<double>(SAMPLE_RATE);

    const int abs_next_start = state->chunks_processed * STEP_SAMPLES;
    const int buf_next_start = abs_next_start - state->samples_trimmed;
    if (buf_next_start > 0 && buf_next_start < static_cast<int>(state->audio_buffer.size())) {
        state->audio_buffer.erase(state->audio_buffer.begin(), state->audio_buffer.begin() + buf_next_start);
        state->samples_trimmed = abs_next_start;
    }

    return true;
}

StreamingState* streaming_init(const StreamingConfig& config) {
    StreamingState* state = new StreamingState();
    state->config = config;
    
    // Load segmentation CoreML model
#ifdef SEGMENTATION_USE_COREML
    if (!config.seg_coreml_path.empty()) {
        state->seg_coreml_ctx = segmentation_coreml_init(config.seg_coreml_path.c_str());
        if (!state->seg_coreml_ctx) {
            fprintf(stderr, "Error: failed to load CoreML segmentation model '%s'\n",
                    config.seg_coreml_path.c_str());
            delete state;
            return nullptr;
        }
    }
#endif
    
    // Load embedding CoreML model
#ifdef EMBEDDING_USE_COREML
    if (!config.coreml_path.empty()) {
        state->emb_coreml_ctx = embedding_coreml_init(config.coreml_path.c_str());
        if (!state->emb_coreml_ctx) {
            fprintf(stderr, "Error: failed to load CoreML embedding model '%s'\n",
                    config.coreml_path.c_str());
#ifdef SEGMENTATION_USE_COREML
            if (state->seg_coreml_ctx) segmentation_coreml_free(state->seg_coreml_ctx);
#endif
            delete state;
            return nullptr;
        }
    }
#endif
    
    // Load PLDA model
    if (!config.plda_path.empty()) {
        if (!diarization::plda_load(config.plda_path, state->plda)) {
            fprintf(stderr, "Error: failed to load PLDA model '%s'\n",
                    config.plda_path.c_str());
#ifdef EMBEDDING_USE_COREML
            if (state->emb_coreml_ctx) embedding_coreml_free(state->emb_coreml_ctx);
#endif
#ifdef SEGMENTATION_USE_COREML
            if (state->seg_coreml_ctx) segmentation_coreml_free(state->seg_coreml_ctx);
#endif
            delete state;
            return nullptr;
        }
    }
    
    state->chunks_processed = 0;
    state->last_recluster_chunk = 0;
    state->audio_time_processed = 0.0;
    state->finalized = false;
    state->num_speakers = 0;
    state->num_provisional_speakers = 0;
    state->samples_trimmed = 0;
    state->silence_frames_offset = 0;

    if (config.zero_latency) {
        state->audio_buffer.resize(CHUNK_SAMPLES, 0.0f);
        if (!process_one_chunk(state, CHUNK_SAMPLES)) {
            fprintf(stderr, "Error: failed to process silence chunk in zero_latency mode\n");
#ifdef EMBEDDING_USE_COREML
            if (state->emb_coreml_ctx) embedding_coreml_free(state->emb_coreml_ctx);
#endif
#ifdef SEGMENTATION_USE_COREML
            if (state->seg_coreml_ctx) segmentation_coreml_free(state->seg_coreml_ctx);
#endif
            delete state;
            return nullptr;
        }
        state->silence_frames_offset = FRAMES_PER_CHUNK;
    }

    return state;
}

StreamingState* streaming_init_with_models(
    const StreamingConfig& config,
    struct segmentation_coreml_context* seg_ctx,
    struct embedding_coreml_context* emb_ctx,
    const diarization::PLDAModel& plda)
{
    StreamingState* state = new StreamingState();
    state->config = config;
    state->owns_models = false;

    // Use borrowed model handles (not freed in streaming_free)
    state->seg_coreml_ctx = seg_ctx;
    state->emb_coreml_ctx = emb_ctx;
    state->plda = plda;  // copy the PLDA model

    state->chunks_processed = 0;
    state->last_recluster_chunk = 0;
    state->audio_time_processed = 0.0;
    state->finalized = false;
    state->num_speakers = 0;
    state->num_provisional_speakers = 0;
    state->samples_trimmed = 0;
    state->silence_frames_offset = 0;

    if (config.zero_latency) {
        state->audio_buffer.resize(CHUNK_SAMPLES, 0.0f);
        if (!process_one_chunk(state, CHUNK_SAMPLES)) {
            fprintf(stderr, "Error: failed to process silence chunk in zero_latency mode\n");
            delete state;
            return nullptr;
        }
        state->silence_frames_offset = FRAMES_PER_CHUNK;
    }

    return state;
}

std::vector<VADChunk> streaming_push(
    StreamingState* state,
    const float* samples,
    int num_samples) {
    
    if (!state || !samples || num_samples <= 0) {
        return {};
    }
    
    state->audio_buffer.insert(
        state->audio_buffer.end(),
        samples,
        samples + num_samples);
    
    int samples_needed_for_next_chunk;
    if (state->chunks_processed == 0) {
        samples_needed_for_next_chunk = CHUNK_SAMPLES;
    } else {
        samples_needed_for_next_chunk = 
            state->chunks_processed * STEP_SAMPLES + CHUNK_SAMPLES;
    }
    
    std::vector<VADChunk> result;
    
    int total_abs_samples = static_cast<int>(state->audio_buffer.size()) + state->samples_trimmed;
    while (total_abs_samples >= samples_needed_for_next_chunk) {
        const int chunk_idx_before = state->chunks_processed;
        const int buf_samples = static_cast<int>(state->audio_buffer.size());
        
        if (!process_one_chunk(state, buf_samples)) {
            break;
        }
        
        int new_start_frame;
        int new_frame_count;

        if (chunk_idx_before == 0) {
            new_start_frame = 0;
            new_frame_count = FRAMES_PER_CHUNK;
        } else {
            int prev_global_start = chunk_global_start_frame(chunk_idx_before - 1);
            int cur_global_start = chunk_global_start_frame(chunk_idx_before);
            new_frame_count = cur_global_start - prev_global_start;
            new_start_frame = prev_global_start + FRAMES_PER_CHUNK;
        }

        if (state->config.zero_latency && chunk_idx_before == 0) {
            total_abs_samples = static_cast<int>(state->audio_buffer.size()) + state->samples_trimmed;
            samples_needed_for_next_chunk =
                state->chunks_processed * STEP_SAMPLES + CHUNK_SAMPLES;
            continue;
        }

        if (state->config.zero_latency) {
            new_start_frame -= state->silence_frames_offset;
        }

        const float* chunk_binarized = state->binarized.data() +
            static_cast<size_t>(chunk_idx_before) * FRAMES_PER_CHUNK * NUM_LOCAL_SPEAKERS;

        VADChunk vad_chunk;
        vad_chunk.chunk_index = chunk_idx_before;
        vad_chunk.start_frame = new_start_frame;
        vad_chunk.num_frames = new_frame_count;
        vad_chunk.vad.resize(new_frame_count);

        int frame_offset = FRAMES_PER_CHUNK - new_frame_count;
        for (int f = 0; f < new_frame_count; f++) {
            float any_active = 0.0f;
            for (int s = 0; s < NUM_LOCAL_SPEAKERS; s++) {
                if (chunk_binarized[(frame_offset + f) * NUM_LOCAL_SPEAKERS + s] != 0.0f) {
                    any_active = 1.0f;
                    break;
                }
            }
            vad_chunk.vad[f] = any_active;
        }
        result.push_back(std::move(vad_chunk));
        
        total_abs_samples = static_cast<int>(state->audio_buffer.size()) + state->samples_trimmed;
        samples_needed_for_next_chunk = 
            state->chunks_processed * STEP_SAMPLES + CHUNK_SAMPLES;
    }
    
    return result;
}

DiarizationResult streaming_recluster(StreamingState* state) {
    if (!state || state->embeddings.empty()) {
        return {};
    }
    
    const int num_chunks = state->chunks_processed;
    
    // Filter embeddings (same as offline pipeline)
    std::vector<float> filtered_emb;
    std::vector<int> filt_chunk_idx, filt_speaker_idx;
    diarization::filter_embeddings(
        state->embeddings.data(),
        num_chunks,
        NUM_LOCAL_SPEAKERS,
        EMBEDDING_DIM,
        state->binarized.data(),
        FRAMES_PER_CHUNK,
        filtered_emb,
        filt_chunk_idx,
        filt_speaker_idx);

    
    const int num_filtered = static_cast<int>(filt_chunk_idx.size());
    
    if (num_filtered < 2) {
        state->num_speakers = (num_filtered > 0) ? 1 : 0;
        state->last_recluster_chunk = state->chunks_processed;
        return {};
    }
    
    // Constants (match diarization.cpp)
    constexpr int PLDA_DIM = 128;
    constexpr double AHC_THRESHOLD = 0.6;
    constexpr double VBX_FA = 0.07;
    constexpr double VBX_FB = 0.8;
    constexpr int VBX_MAX_ITERS = 20;
    
    // Step 1: L2-normalize filtered embeddings for AHC (double precision)
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
    
    // Step 2: Convert to double for PLDA (original filtered embeddings, NOT normalized)
    std::vector<double> filtered_double(
        static_cast<size_t>(num_filtered) * EMBEDDING_DIM);
    for (int i = 0; i < num_filtered * EMBEDDING_DIM; i++) {
        filtered_double[i] = static_cast<double>(filtered_emb[i]);
    }
    
    // Step 3: PLDA transform
    std::vector<double> plda_features(
        static_cast<size_t>(num_filtered) * PLDA_DIM);
    diarization::plda_transform(
        state->plda, filtered_double.data(), num_filtered, plda_features.data());
    
    // Step 4: AHC cluster (on normalized embeddings)
    std::vector<int> ahc_clusters;
    diarization::ahc_cluster(
        filtered_normed.data(), num_filtered, EMBEDDING_DIM,
        AHC_THRESHOLD, ahc_clusters);
    
    int num_ahc_clusters = 0;
    for (int i = 0; i < num_filtered; i++) {
        if (ahc_clusters[i] + 1 > num_ahc_clusters) {
            num_ahc_clusters = ahc_clusters[i] + 1;
        }
    }
    
    // Free L2-normalized embeddings
    filtered_normed.clear();
    filtered_normed.shrink_to_fit();
    filtered_double.clear();
    filtered_double.shrink_to_fit();
    
    // Step 5: VBx clustering
    diarization::VBxResult vbx_result;
    if (!diarization::vbx_cluster(
            ahc_clusters.data(), num_filtered, num_ahc_clusters,
            plda_features.data(), PLDA_DIM,
            state->plda.plda_psi.data(), VBX_FA, VBX_FB, VBX_MAX_ITERS,
     vbx_result)) {
         state->num_speakers = 1;
         state->last_recluster_chunk = state->chunks_processed;
         return {};
     }
    
    // Step 6: Compute centroids from VBx soft assignments
    const int vbx_S = vbx_result.num_speakers;
    const int vbx_T = vbx_result.num_frames;
    
    // Find significant speakers (pi > 1e-7)
    std::vector<int> sig_speakers;
    for (int s = 0; s < vbx_S; s++) {
        if (vbx_result.pi[s] > 1e-7) {
            sig_speakers.push_back(s);
        }
    }
    int num_clusters = static_cast<int>(sig_speakers.size());
    if (num_clusters == 0) num_clusters = 1;  // fallback
    
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
    
    const int total_emb = num_chunks * NUM_LOCAL_SPEAKERS;
    std::vector<float> soft_clusters(
        static_cast<size_t>(total_emb) * num_clusters);
    
    for (int e = 0; e < total_emb; e++) {
        const float* emb = state->embeddings.data() + e * EMBEDDING_DIM;
        for (int k = 0; k < num_clusters; k++) {
            const float* cent = centroids.data() + k * EMBEDDING_DIM;
            double dist = diarization::cosine_distance(emb, cent, EMBEDDING_DIM);
            soft_clusters[e * num_clusters + k] = static_cast<float>(2.0 - dist);
        }
    }
    
    std::vector<int> hard_clusters(static_cast<size_t>(total_emb), 0);
    diarization::constrained_argmax(
        soft_clusters.data(), num_chunks, NUM_LOCAL_SPEAKERS, num_clusters,
        hard_clusters);
    
    for (int c = 0; c < num_chunks; c++) {
        for (int s = 0; s < NUM_LOCAL_SPEAKERS; s++) {
            bool active = false;
            const float* seg_chunk = state->binarized.data() + 
                c * FRAMES_PER_CHUNK * NUM_LOCAL_SPEAKERS;
            for (int f = 0; f < FRAMES_PER_CHUNK; f++) {
                if (seg_chunk[f * NUM_LOCAL_SPEAKERS + s] != 0.0f) {
                    active = true;
                    break;
                }
            }
            if (!active) {
                hard_clusters[c * NUM_LOCAL_SPEAKERS + s] = -2;
            }
        }
     }
     
     // Keep state->embeddings/chunk_idx/local_speaker_idx unfiltered (3 per chunk)
    // so subsequent push() calls can safely append to them.
    
    state->num_speakers = num_clusters;
    state->centroids = std::move(centroids);
    state->last_recluster_chunk = state->chunks_processed;
    
    // --- Reconstruction: convert clustering result to DiarizationResult ---
    // Use hard_clusters directly (includes active-but-not-filtered speakers,
    // matching offline pipeline behavior)
    
    DiarizationResult result;
    
    if (num_chunks == 0 || num_clusters == 0) {
        return result;
    }
    
    constexpr double CHUNK_DURATION = 10.0;
    constexpr double CHUNK_STEP = 1.0;
    constexpr double FRAME_DURATION = 0.0619375;
    constexpr double FRAME_STEP = 0.016875;
    
    diarization::SlidingWindowParams chunk_window = {0.0, CHUNK_DURATION, CHUNK_STEP};
    diarization::SlidingWindowParams frame_window = {chunk_window.start, FRAME_DURATION, FRAME_STEP};
    
    std::vector<int> count;
    int total_frames = 0;
    diarization::compute_speaker_count(
        state->binarized.data(), num_chunks, FRAMES_PER_CHUNK, NUM_LOCAL_SPEAKERS,
        chunk_window, frame_window,
        count, total_frames);
    
    std::vector<float> clustered_seg(
        static_cast<size_t>(num_chunks) * FRAMES_PER_CHUNK * num_clusters);
    {
        const float nan_val = std::nanf("");
        std::fill(clustered_seg.begin(), clustered_seg.end(), nan_val);
    }
    
    for (int c = 0; c < num_chunks; c++) {
        const int* chunk_clusters = hard_clusters.data() + c * NUM_LOCAL_SPEAKERS;
        const float* seg_chunk =
            state->binarized.data() + c * FRAMES_PER_CHUNK * NUM_LOCAL_SPEAKERS;
        
        for (int k = 0; k < num_clusters; k++) {
            for (int s = 0; s < NUM_LOCAL_SPEAKERS; s++) {
                if (chunk_clusters[s] != k) continue;
                
                for (int f = 0; f < FRAMES_PER_CHUNK; f++) {
                    const float val = seg_chunk[f * NUM_LOCAL_SPEAKERS + s];
                    float& out = clustered_seg[
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
    
    std::vector<float> discrete_diarization;
    diarization::to_diarization(
        clustered_seg.data(), num_chunks, FRAMES_PER_CHUNK, num_clusters,
        count.data(), total_frames,
        chunk_window, frame_window,
        discrete_diarization);
    
    clustered_seg.clear();
    clustered_seg.shrink_to_fit();
    
    const int out_frames =
        static_cast<int>(discrete_diarization.size()) / num_clusters;
    
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
                const double start_time =
                    chunk_window.start + seg_start_frame * FRAME_STEP + 0.5 * FRAME_DURATION;
                const double duration =
                    (f - seg_start_frame) * FRAME_STEP;
                if (duration > 0.0) {
                    result.segments.push_back({start_time, duration, speaker_label});
                }
                in_segment = false;
            }
        }
    }
    
    std::sort(result.segments.begin(), result.segments.end(),
              [](const DiarizationResult::Segment& a,
                 const DiarizationResult::Segment& b) {
                  return a.start < b.start;
              });
    
    // Zero-latency: subtract silence pre-fill (10s) so real audio starts at t=0
    if (state->config.zero_latency) {
        const double silence_duration = static_cast<double>(CHUNK_SAMPLES) / static_cast<double>(SAMPLE_RATE);
        std::vector<DiarizationResult::Segment> adjusted;
        adjusted.reserve(result.segments.size());
        for (auto& seg : result.segments) {
            seg.start -= silence_duration;
            if (seg.start + seg.duration <= 0.0) continue;
            if (seg.start < 0.0) {
                seg.duration += seg.start;
                seg.start = 0.0;
            }
            adjusted.push_back(std::move(seg));
        }
        result.segments = std::move(adjusted);
    }
    
    return result;
}

DiarizationResult streaming_finalize(StreamingState* state) {
    if (!state || state->finalized) {
        return {};
    }
    
    const int total_abs_samples = static_cast<int>(state->audio_buffer.size()) + state->samples_trimmed;
    const double audio_duration = static_cast<double>(total_abs_samples) /
                                   static_cast<double>(SAMPLE_RATE);
    const int offline_num_chunks = std::max(1,
        1 + static_cast<int>(std::ceil((audio_duration -
            static_cast<double>(CHUNK_SAMPLES) / static_cast<double>(SAMPLE_RATE)) /
            (static_cast<double>(STEP_SAMPLES) / static_cast<double>(SAMPLE_RATE)))));
    
    while (state->chunks_processed < offline_num_chunks) {
        if (!process_one_chunk(state, static_cast<int>(state->audio_buffer.size()))) {
            break;
        }
    }
    
    DiarizationResult result = streaming_recluster(state);
    state->finalized = true;
    return result;
}

void streaming_free(StreamingState* state) {
    if (!state) return;
    
#ifdef SEGMENTATION_USE_COREML
    if (state->owns_models && state->seg_coreml_ctx) {
        segmentation_coreml_free(state->seg_coreml_ctx);
        state->seg_coreml_ctx = nullptr;
    }
#endif

#ifdef EMBEDDING_USE_COREML
    if (state->owns_models && state->emb_coreml_ctx) {
        embedding_coreml_free(state->emb_coreml_ctx);
        state->emb_coreml_ctx = nullptr;
    }
#endif
    
    // Clear all vectors (automatic with delete, but explicit for clarity)
     state->audio_buffer.clear();
     state->embeddings.clear();
     state->chunk_idx.clear();
     state->local_speaker_idx.clear();
     state->binarized.clear();
     state->centroids.clear();
     state->centroid_counts.clear();
     
     delete state;
}
