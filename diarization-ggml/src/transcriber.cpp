#include "transcriber.h"
#include "whisper.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

static constexpr int SAMPLE_RATE = 16000;
static constexpr int MIN_SAMPLES = SAMPLE_RATE;        // 1s minimum

struct Transcriber {
    whisper_context* ctx = nullptr;
    std::thread worker;
    std::mutex mtx;
    std::condition_variable cv_submit;
    std::condition_variable cv_result;

    std::vector<float> pending_audio;
    double pending_start_time = 0.0;
    bool has_pending = false;

    TranscribeResult result;
    bool has_result = false;

    bool shutdown = false;
    bool owns_ctx = true;
    TranscriberConfig config;
    DecodeOptions decode_opts;
};

static void worker_loop(Transcriber* t) {
    while (true) {
        std::vector<float> audio;
        double start_time = 0.0;
        DecodeOptions opts;

        {
            std::unique_lock<std::mutex> lock(t->mtx);
            t->cv_submit.wait(lock, [t]{ return t->has_pending || t->shutdown; });

            if (t->shutdown && !t->has_pending) return;

            audio = std::move(t->pending_audio);
            start_time = t->pending_start_time;
            t->has_pending = false;
            opts = t->decode_opts;  // snapshot ALL decode options under lock
        }

        TranscribeResult res;

        if ((int)audio.size() < MIN_SAMPLES) {
            res.valid = true;
            std::lock_guard<std::mutex> lock(t->mtx);
            t->result = std::move(res);
            t->has_result = true;
            t->cv_result.notify_one();
            continue;
        }

        // Choose strategy based on beam_size
        auto strategy = (opts.beam_size > 1) ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;
        auto params = whisper_full_default_params(strategy);

        // Pipeline-required overrides (never user-controllable)
        params.print_progress   = false;
        params.print_realtime   = false;
        params.print_timestamps = false;
        params.token_timestamps = false;

        // User-configurable params (from snapshotted DecodeOptions)
        params.language         = opts.language.empty() ? nullptr : opts.language.c_str();
        params.n_threads        = opts.n_threads;
        params.translate        = opts.translate;
        params.detect_language  = opts.detect_language;
        if (opts.detect_language) {
            params.language = "auto";
        }

        // Sampling
        params.temperature      = opts.temperature;
        params.temperature_inc  = opts.no_fallback ? 0.0f : opts.temperature_inc;
        if (strategy == WHISPER_SAMPLING_BEAM_SEARCH) {
            params.beam_search.beam_size = opts.beam_size;
        } else {
            params.greedy.best_of = opts.best_of;
        }

        // Thresholds
        params.entropy_thold    = opts.entropy_thold;
        params.logprob_thold    = opts.logprob_thold;
        params.no_speech_thold  = opts.no_speech_thold;

        // Context
        params.initial_prompt   = opts.prompt.empty() ? nullptr : opts.prompt.c_str();
        params.no_context       = opts.no_context;
        params.suppress_blank   = opts.suppress_blank;
        params.suppress_nst     = opts.suppress_nst;

        int ret = whisper_full(t->ctx, params, audio.data(), (int)audio.size());
        if (ret != 0) {
            fprintf(stderr, "ERROR: whisper_full failed with code %d\n", ret);
            res.valid = false;
            std::lock_guard<std::mutex> lock(t->mtx);
            t->result = std::move(res);
            t->has_result = true;
            t->cv_result.notify_one();
            continue;
        }

        int n_segments = whisper_full_n_segments(t->ctx);
        for (int seg = 0; seg < n_segments; seg++) {
            int64_t seg_t0 = whisper_full_get_segment_t0(t->ctx, seg);
            int64_t seg_t1 = whisper_full_get_segment_t1(t->ctx, seg);
            const char* text = whisper_full_get_segment_text(t->ctx, seg);
            if (!text || text[0] == '\0') continue;

            TranscribeSegment ts;
            ts.start = seg_t0 * 0.01 + start_time;
            ts.end   = seg_t1 * 0.01 + start_time;
            ts.text  = text;
            res.segments.push_back(std::move(ts));
        }

        res.valid = true;

        {
            std::lock_guard<std::mutex> lock(t->mtx);
            t->result = std::move(res);
            t->has_result = true;
            t->cv_result.notify_one();
        }
    }
}

Transcriber* transcriber_init(const TranscriberConfig& config) {
    auto cparams = whisper_context_default_params();
    cparams.use_gpu = config.use_gpu;
    cparams.flash_attn = config.flash_attn;
    cparams.gpu_device = config.gpu_device;
    cparams.use_coreml = config.use_coreml;

    if (config.no_prints) {
        whisper_log_set([](enum ggml_log_level, const char*, void*){}, nullptr);
    }

    whisper_context* ctx = whisper_init_from_file_with_params(config.whisper_model_path, cparams);
    if (!ctx) {
        fprintf(stderr, "ERROR: failed to load whisper model from '%s'\n", config.whisper_model_path);
        return nullptr;
    }

    auto* t = new Transcriber();
    t->ctx = ctx;
    t->config = config;
    t->decode_opts.language = config.language ? config.language : "";
    t->decode_opts.translate = config.translate;
    t->decode_opts.detect_language = config.detect_language;
    t->decode_opts.n_threads = config.n_threads;
    t->decode_opts.temperature = config.temperature;
    t->decode_opts.temperature_inc = config.temperature_inc;
    t->decode_opts.no_fallback = config.no_fallback;
    t->decode_opts.beam_size = config.beam_size;
    t->decode_opts.best_of = config.best_of;
    t->decode_opts.entropy_thold = config.entropy_thold;
    t->decode_opts.logprob_thold = config.logprob_thold;
    t->decode_opts.no_speech_thold = config.no_speech_thold;
    t->decode_opts.prompt = config.prompt ? config.prompt : "";
    t->decode_opts.no_context = config.no_context;
    t->decode_opts.suppress_blank = config.suppress_blank;
    t->decode_opts.suppress_nst = config.suppress_nst;
    t->worker = std::thread(worker_loop, t);
    return t;
}

Transcriber* transcriber_init_with_context(const TranscriberConfig& config, whisper_context* ctx) {
    if (!ctx) {
        fprintf(stderr, "ERROR: transcriber_init_with_context called with null context\n");
        return nullptr;
    }

    if (config.no_prints) {
        whisper_log_set([](enum ggml_log_level, const char*, void*){}, nullptr);
    }

    auto* t = new Transcriber();
    t->ctx = ctx;
    t->owns_ctx = false;
    t->config = config;
    t->decode_opts.language = config.language ? config.language : "";
    t->decode_opts.translate = config.translate;
    t->decode_opts.detect_language = config.detect_language;
    t->decode_opts.n_threads = config.n_threads;
    t->decode_opts.temperature = config.temperature;
    t->decode_opts.temperature_inc = config.temperature_inc;
    t->decode_opts.no_fallback = config.no_fallback;
    t->decode_opts.beam_size = config.beam_size;
    t->decode_opts.best_of = config.best_of;
    t->decode_opts.entropy_thold = config.entropy_thold;
    t->decode_opts.logprob_thold = config.logprob_thold;
    t->decode_opts.no_speech_thold = config.no_speech_thold;
    t->decode_opts.prompt = config.prompt ? config.prompt : "";
    t->decode_opts.no_context = config.no_context;
    t->decode_opts.suppress_blank = config.suppress_blank;
    t->decode_opts.suppress_nst = config.suppress_nst;
    t->worker = std::thread(worker_loop, t);
    return t;
}

void transcriber_submit(Transcriber* t, const float* audio, int n_samples, double buffer_start_time) {
    std::lock_guard<std::mutex> lock(t->mtx);
    t->pending_audio.assign(audio, audio + n_samples);
    t->pending_start_time = buffer_start_time;
    t->has_pending = true;
    t->has_result = false;
    t->cv_submit.notify_one();
}

bool transcriber_try_get_result(Transcriber* t, TranscribeResult& result) {
    std::lock_guard<std::mutex> lock(t->mtx);
    if (!t->has_result) return false;
    result = std::move(t->result);
    t->has_result = false;
    return true;
}

TranscribeResult transcriber_wait_result(Transcriber* t) {
    std::unique_lock<std::mutex> lock(t->mtx);
    t->cv_result.wait(lock, [t]{ return t->has_result; });
    TranscribeResult res = std::move(t->result);
    t->has_result = false;
    return res;
}

void transcriber_set_decode_options(Transcriber* t, const DecodeOptions& opts) {
    if (!t) return;
    std::lock_guard<std::mutex> lock(t->mtx);
    t->decode_opts = opts;
}

void transcriber_free(Transcriber* t) {
    if (!t) return;

    {
        std::lock_guard<std::mutex> lock(t->mtx);
        t->shutdown = true;
        t->cv_submit.notify_one();
    }

    if (t->worker.joinable()) {
        t->worker.join();
    }

    if (t->ctx && t->owns_ctx) {
        whisper_free(t->ctx);
    }

    delete t;
}
