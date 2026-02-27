#include "offline_pipeline.h"
#include "whisper.h"
#include "model_cache.h"

#include <cstdint>
#include <cstdio>
#include <vector>

OfflinePipelineResult offline_transcribe(
    const OfflinePipelineConfig& config,
    const float* audio,
    int n_samples)
{
    OfflinePipelineResult out;

    // ================================================================
    // Step 1: Initialize Whisper
    // ================================================================

    auto cparams = whisper_context_default_params();
    cparams.use_gpu     = config.transcriber.use_gpu;
    cparams.flash_attn  = config.transcriber.flash_attn;
    cparams.gpu_device  = config.transcriber.gpu_device;
    cparams.use_coreml  = config.transcriber.use_coreml;

    if (config.transcriber.no_prints) {
        whisper_log_set([](enum ggml_log_level, const char*, void*){}, nullptr);
    }

    whisper_context* ctx = whisper_init_from_file_with_params(
        config.transcriber.whisper_model_path, cparams);
    if (!ctx) {
        fprintf(stderr, "ERROR: offline_transcribe: failed to load whisper model '%s'\n",
                config.transcriber.whisper_model_path
                    ? config.transcriber.whisper_model_path : "(null)");
        return out;
    }

    fprintf(stderr, "Offline pipeline: Whisper model loaded\n");

    // ================================================================
    // Step 2: Run whisper_full() on entire audio
    // ================================================================

    auto strategy = (config.transcriber.beam_size > 1)
        ? WHISPER_SAMPLING_BEAM_SEARCH
        : WHISPER_SAMPLING_GREEDY;
    auto params = whisper_full_default_params(strategy);

    // Pipeline-required overrides (not user-controllable)
    params.print_progress   = false;
    params.print_realtime   = false;
    params.print_timestamps = false;
    params.token_timestamps = false;

    // User-configurable params from TranscriberConfig
    params.language         = config.transcriber.language;
    params.n_threads        = config.transcriber.n_threads;
    params.translate        = config.transcriber.translate;
    params.detect_language  = config.transcriber.detect_language;
    if (config.transcriber.detect_language) {
        params.language = "auto";
    }

    // Sampling
    params.temperature      = config.transcriber.temperature;
    params.temperature_inc  = config.transcriber.no_fallback
        ? 0.0f : config.transcriber.temperature_inc;
    if (strategy == WHISPER_SAMPLING_BEAM_SEARCH) {
        params.beam_search.beam_size = config.transcriber.beam_size;
    } else {
        params.greedy.best_of = config.transcriber.best_of;
    }

    // Thresholds
    params.entropy_thold    = config.transcriber.entropy_thold;
    params.logprob_thold    = config.transcriber.logprob_thold;
    params.no_speech_thold  = config.transcriber.no_speech_thold;

    // Context
    params.initial_prompt   = config.transcriber.prompt;
    params.no_context       = config.transcriber.no_context;
    params.suppress_blank   = config.transcriber.suppress_blank;
    params.suppress_nst     = config.transcriber.suppress_nst;

    // Progress callback — wire phase 0 (whisper) to whisper_full's progress_callback
    struct ProgressBridge {
        std::function<void(int,int)> cb;
    };
    ProgressBridge progress_bridge{config.progress_callback};

    if (progress_bridge.cb) {
        params.progress_callback = [](struct whisper_context*, struct whisper_state*, int progress, void* user_data) {
            auto* bridge = static_cast<ProgressBridge*>(user_data);
            if (bridge->cb) bridge->cb(0, progress);
        };
        params.progress_callback_user_data = &progress_bridge;
    }

    const double audio_duration = static_cast<double>(n_samples) / 16000.0;
    fprintf(stderr, "Offline pipeline: running Whisper on %.1fs of audio...\n",
            audio_duration);

    int ret = whisper_full(ctx, params, audio, n_samples);
    if (ret != 0) {
        fprintf(stderr, "ERROR: offline_transcribe: whisper_full failed (code %d)\n", ret);
        whisper_free(ctx);
        return out;
    }

    // ================================================================
    // Step 3: Extract TranscribeSegments
    // ================================================================

    std::vector<TranscribeSegment> transcribe_segments;
    int n_seg = whisper_full_n_segments(ctx);

    for (int i = 0; i < n_seg; i++) {
        int64_t seg_t0 = whisper_full_get_segment_t0(ctx, i);
        int64_t seg_t1 = whisper_full_get_segment_t1(ctx, i);
        const char* text = whisper_full_get_segment_text(ctx, i);
        if (!text || text[0] == '\0') continue;

        TranscribeSegment ts;
        ts.start = seg_t0 * 0.01;  // centiseconds → seconds (no offset — full audio)
        ts.end   = seg_t1 * 0.01;
        ts.text  = text;
        transcribe_segments.push_back(std::move(ts));
    }

    fprintf(stderr, "Offline pipeline: %zu Whisper segments extracted\n",
            transcribe_segments.size());

    // Done with Whisper — free before heavy diarization work
    whisper_free(ctx);
    ctx = nullptr;

    // ================================================================
    // Step 4: Run offline diarization
    // ================================================================

    // Notify phase 1 — diarization start
    if (config.progress_callback) config.progress_callback(1, 0);

    DiarizationConfig diar_config;
    diar_config.seg_model_path  = config.seg_model_path;
    diar_config.emb_model_path  = config.emb_model_path;
    diar_config.plda_path       = config.plda_path;
    diar_config.coreml_path     = config.coreml_path;
    diar_config.seg_coreml_path = config.seg_coreml_path;
    // output_path left empty — no RTTM file output

    DiarizationResult diar_result;
    if (!diarize_from_samples(diar_config, audio, n_samples, diar_result)) {
        fprintf(stderr, "ERROR: offline_transcribe: diarization failed\n");
        return out;
    }

    fprintf(stderr, "Offline pipeline: diarization complete — %zu segments\n",
            diar_result.segments.size());

    // ================================================================
    // Step 5: Align Whisper segments with diarization
    // ================================================================

    // Notify phase 2 — alignment start
    if (config.progress_callback) config.progress_callback(2, 0);

    out.segments = align_segments(transcribe_segments, diar_result);
    out.diarization = std::move(diar_result);
    out.valid = true;

    fprintf(stderr, "Offline pipeline: %zu aligned segments\n", out.segments.size());

    return out;
}

OfflinePipelineResult offline_transcribe_with_cache(
    const OfflinePipelineConfig& config,
    ModelCache* cache,
    const float* audio,
    int n_samples)
{
    OfflinePipelineResult out;

    if (!cache || !cache->whisper_ctx) {
        fprintf(stderr, "ERROR: offline_transcribe_with_cache: invalid cache or missing whisper_ctx\n");
        return out;
    }

    if (config.transcriber.no_prints) {
        whisper_log_set([](enum ggml_log_level, const char*, void*){}, nullptr);
    }

    fprintf(stderr, "Offline pipeline (cached): using pre-loaded Whisper context\n");

    // ================================================================
    // Step 1: Run whisper_full() on entire audio (using cached context)
    // ================================================================

    auto strategy = (config.transcriber.beam_size > 1)
        ? WHISPER_SAMPLING_BEAM_SEARCH
        : WHISPER_SAMPLING_GREEDY;
    auto params = whisper_full_default_params(strategy);

    // Pipeline-required overrides (not user-controllable)
    params.print_progress   = false;
    params.print_realtime   = false;
    params.print_timestamps = false;
    params.token_timestamps = false;

    // User-configurable params from TranscriberConfig
    params.language         = config.transcriber.language;
    params.n_threads        = config.transcriber.n_threads;
    params.translate        = config.transcriber.translate;
    params.detect_language  = config.transcriber.detect_language;
    if (config.transcriber.detect_language) {
        params.language = "auto";
    }

    // Sampling
    params.temperature      = config.transcriber.temperature;
    params.temperature_inc  = config.transcriber.no_fallback
        ? 0.0f : config.transcriber.temperature_inc;
    if (strategy == WHISPER_SAMPLING_BEAM_SEARCH) {
        params.beam_search.beam_size = config.transcriber.beam_size;
    } else {
        params.greedy.best_of = config.transcriber.best_of;
    }

    // Thresholds
    params.entropy_thold    = config.transcriber.entropy_thold;
    params.logprob_thold    = config.transcriber.logprob_thold;
    params.no_speech_thold  = config.transcriber.no_speech_thold;

    // Context
    params.initial_prompt   = config.transcriber.prompt;
    params.no_context       = config.transcriber.no_context;
    params.suppress_blank   = config.transcriber.suppress_blank;
    params.suppress_nst     = config.transcriber.suppress_nst;

    // Progress callback — wire phase 0 (whisper) to whisper_full's progress_callback
    struct ProgressBridge {
        std::function<void(int,int)> cb;
    };
    ProgressBridge progress_bridge{config.progress_callback};

    if (progress_bridge.cb) {
        params.progress_callback = [](struct whisper_context*, struct whisper_state*, int progress, void* user_data) {
            auto* bridge = static_cast<ProgressBridge*>(user_data);
            if (bridge->cb) bridge->cb(0, progress);
        };
        params.progress_callback_user_data = &progress_bridge;
    }

    const double audio_duration = static_cast<double>(n_samples) / 16000.0;
    fprintf(stderr, "Offline pipeline (cached): running Whisper on %.1fs of audio...\n",
            audio_duration);

    int ret = whisper_full(cache->whisper_ctx, params, audio, n_samples);
    if (ret != 0) {
        fprintf(stderr, "ERROR: offline_transcribe_with_cache: whisper_full failed (code %d)\n", ret);
        return out;
    }

    // ================================================================
    // Step 2: Extract TranscribeSegments
    // ================================================================

    std::vector<TranscribeSegment> transcribe_segments;
    int n_seg = whisper_full_n_segments(cache->whisper_ctx);

    for (int i = 0; i < n_seg; i++) {
        int64_t seg_t0 = whisper_full_get_segment_t0(cache->whisper_ctx, i);
        int64_t seg_t1 = whisper_full_get_segment_t1(cache->whisper_ctx, i);
        const char* text = whisper_full_get_segment_text(cache->whisper_ctx, i);
        if (!text || text[0] == '\0') continue;

        TranscribeSegment ts;
        ts.start = seg_t0 * 0.01;
        ts.end   = seg_t1 * 0.01;
        ts.text  = text;
        transcribe_segments.push_back(std::move(ts));
    }

    fprintf(stderr, "Offline pipeline (cached): %zu Whisper segments extracted\n",
            transcribe_segments.size());

    // ================================================================
    // Step 3: Run offline diarization (using cached models)
    // ================================================================

    // Notify phase 1 — diarization start
    if (config.progress_callback) config.progress_callback(1, 0);

    DiarizationConfig diar_config;
    diar_config.seg_model_path  = config.seg_model_path;
    diar_config.emb_model_path  = config.emb_model_path;
    diar_config.plda_path       = config.plda_path;
    diar_config.coreml_path     = config.coreml_path;
    diar_config.seg_coreml_path = config.seg_coreml_path;
    // output_path left empty — no RTTM file output

    DiarizationResult diar_result;

#if defined(SEGMENTATION_USE_COREML) && defined(EMBEDDING_USE_COREML)
    if (!diarize_from_samples_with_models(diar_config, audio, n_samples,
                                          cache->seg_coreml_ctx, cache->emb_coreml_ctx,
                                          cache->plda, diar_result)) {
        fprintf(stderr, "ERROR: offline_transcribe_with_cache: diarization failed\n");
        return out;
    }
#else
    // Fallback: use path-based loading (models in cache may not be available)
    if (!diarize_from_samples(diar_config, audio, n_samples, diar_result)) {
        fprintf(stderr, "ERROR: offline_transcribe_with_cache: diarization failed\n");
        return out;
    }
#endif

    fprintf(stderr, "Offline pipeline (cached): diarization complete — %zu segments\n",
            diar_result.segments.size());

    // ================================================================
    // Step 4: Align Whisper segments with diarization
    // ================================================================

    // Notify phase 2 — alignment start
    if (config.progress_callback) config.progress_callback(2, 0);

    out.segments = align_segments(transcribe_segments, diar_result);
    out.diarization = std::move(diar_result);
    out.valid = true;

    fprintf(stderr, "Offline pipeline (cached): %zu aligned segments\n", out.segments.size());

    return out;
}
