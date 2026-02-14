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
static constexpr int MAX_SAMPLES = 30 * SAMPLE_RATE;  // 30s hard cap
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
    TranscriberConfig config;
};

static void worker_loop(Transcriber* t) {
    while (true) {
        std::vector<float> audio;
        double start_time = 0.0;

        {
            std::unique_lock<std::mutex> lock(t->mtx);
            t->cv_submit.wait(lock, [t]{ return t->has_pending || t->shutdown; });

            if (t->shutdown && !t->has_pending) return;

            audio = std::move(t->pending_audio);
            start_time = t->pending_start_time;
            t->has_pending = false;
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

        if ((int)audio.size() > MAX_SAMPLES) {
            fprintf(stderr, "WARNING: transcriber audio truncated from %d to %d samples (30s cap)\n",
                    (int)audio.size(), MAX_SAMPLES);
            audio.resize(MAX_SAMPLES);
        }

        auto params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        params.print_progress   = false;
        params.print_realtime   = false;
        params.print_timestamps = false;
        params.token_timestamps = true;
        params.max_len          = 1;
        params.split_on_word    = true;
        params.language         = t->config.language;
        params.n_threads        = t->config.n_threads;

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
            int n_tokens = whisper_full_n_tokens(t->ctx, seg);
            for (int tok = 0; tok < n_tokens; tok++) {
                const char* text = whisper_full_get_token_text(t->ctx, seg, tok);
                if (!text || text[0] == '\0' || text[0] == '[') continue;

                bool all_ws = true;
                for (const char* p = text; *p; p++) {
                    if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') {
                        all_ws = false;
                        break;
                    }
                }
                if (all_ws) continue;

                auto tdata = whisper_full_get_token_data(t->ctx, seg, tok);
                double t0 = (tdata.t_dtw >= 0 ? tdata.t_dtw : tdata.t0) * 0.01 + start_time;
                double t1 = tdata.t1 * 0.01 + start_time;
                res.tokens.push_back({text, t0, t1});
            }
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
    cparams.use_gpu = true;
    cparams.flash_attn = true;
    cparams.dtw_token_timestamps = true;
    cparams.dtw_aheads_preset = WHISPER_AHEADS_N_TOP_MOST_NORM;
    cparams.dtw_n_top = 2;
    cparams.dtw_mem_size = 1024 * 1024 * 512;
    if (config.whisper_coreml_path) {
        cparams.use_coreml = true;
    }

    whisper_context* ctx = whisper_init_from_file_with_params(config.whisper_model_path, cparams);
    if (!ctx) {
        fprintf(stderr, "ERROR: failed to load whisper model from '%s'\n", config.whisper_model_path);
        return nullptr;
    }

    auto* t = new Transcriber();
    t->ctx = ctx;
    t->config = config;
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

    if (t->ctx) {
        whisper_free(t->ctx);
    }

    delete t;
}
