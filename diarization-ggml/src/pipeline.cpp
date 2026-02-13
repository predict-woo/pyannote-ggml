#include "pipeline.h"
#include "silence_filter.h"
#include "audio_buffer.h"
#include "segment_detector.h"
#include "streaming.h"
#include "aligner.h"
#include "whisper.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static constexpr int SAMPLE_RATE = 16000;
static constexpr double MIN_SEGMENT_DURATION = 20.0;
static constexpr int MAX_WHISPER_SAMPLES = 30 * SAMPLE_RATE;

struct PipelineState {
    SilenceFilter* silence_filter;
    AudioBuffer audio_buffer;
    StreamingState* streaming_state;
    SegmentDetector* segment_detector;
    Transcriber* transcriber;

    double buffer_start_time;
    bool whisper_in_flight;

    pipeline_callback callback;
    void* user_data;

    std::vector<AlignedSegment> all_segments;

    whisper_vad_context* vad_ctx;
};

static void submit_audio_chunk(PipelineState* state, int64_t abs_start, int64_t abs_end) {
    std::vector<float> audio;
    state->audio_buffer.read_range(abs_start, abs_end, audio);

    if (audio.empty()) return;

    if ((int)audio.size() > MAX_WHISPER_SAMPLES) {
        audio.resize(MAX_WHISPER_SAMPLES);
    }

    transcriber_submit(state->transcriber, audio.data(), (int)audio.size(), state->buffer_start_time);

    state->audio_buffer.dequeue_up_to(abs_end);
    state->buffer_start_time = static_cast<double>(state->audio_buffer.dequeued_frames()) / SAMPLE_RATE;
    state->whisper_in_flight = true;
}

static void handle_whisper_result(PipelineState* state, const TranscribeResult& result, const DiarizationResult& diarization) {
    if (!result.valid || result.tokens.empty()) return;

    std::vector<AlignedSegment> segments = align_words(result.tokens, diarization);
    state->all_segments.insert(state->all_segments.end(), segments.begin(), segments.end());

    if (state->callback) {
        state->callback(state->all_segments, state->user_data);
    }
}

PipelineState* pipeline_init(const PipelineConfig& config, pipeline_callback cb, void* user_data) {
    auto* state = new PipelineState();
    state->callback = cb;
    state->user_data = user_data;
    state->buffer_start_time = 0.0;
    state->whisper_in_flight = false;
    state->vad_ctx = nullptr;

    if (config.vad_model_path) {
        state->vad_ctx = whisper_vad_init_from_file_with_params(
            config.vad_model_path, whisper_vad_default_context_params());
    }

    state->silence_filter = silence_filter_init(state->vad_ctx, 0.5f);
    if (!state->silence_filter) {
        fprintf(stderr, "ERROR: failed to init silence filter\n");
        if (state->vad_ctx) whisper_vad_free(state->vad_ctx);
        delete state;
        return nullptr;
    }

    StreamingConfig diar_config = config.diarization;
    diar_config.zero_latency = true;
    state->streaming_state = streaming_init(diar_config);
    if (!state->streaming_state) {
        fprintf(stderr, "ERROR: failed to init streaming diarization\n");
        silence_filter_free(state->silence_filter);
        if (state->vad_ctx) whisper_vad_free(state->vad_ctx);
        delete state;
        return nullptr;
    }

    state->segment_detector = segment_detector_init();
    if (!state->segment_detector) {
        fprintf(stderr, "ERROR: failed to init segment detector\n");
        streaming_free(state->streaming_state);
        silence_filter_free(state->silence_filter);
        if (state->vad_ctx) whisper_vad_free(state->vad_ctx);
        delete state;
        return nullptr;
    }

    state->transcriber = transcriber_init(config.transcriber);
    if (!state->transcriber) {
        fprintf(stderr, "ERROR: failed to init transcriber\n");
        segment_detector_free(state->segment_detector);
        streaming_free(state->streaming_state);
        silence_filter_free(state->silence_filter);
        if (state->vad_ctx) whisper_vad_free(state->vad_ctx);
        delete state;
        return nullptr;
    }

    return state;
}

void pipeline_push(PipelineState* state, const float* samples, int n_samples) {
    if (!state || !samples || n_samples <= 0) return;

    SilenceFilterResult sf_result = silence_filter_push(state->silence_filter, samples, n_samples);

    bool need_flush = sf_result.flush_signal;

    if (!sf_result.audio.empty()) {
        state->audio_buffer.enqueue(sf_result.audio.data(), (int)sf_result.audio.size());

        std::vector<VADChunk> vad_chunks = streaming_push(
            state->streaming_state, sf_result.audio.data(), (int)sf_result.audio.size());

        for (const auto& chunk : vad_chunks) {
            SegmentDetectorResult sd_result = segment_detector_push(state->segment_detector, chunk);

            if (sd_result.flush_signal) need_flush = true;

            for (double seg_end_time : sd_result.segment_end_times) {
                double segment_duration = seg_end_time - state->buffer_start_time;

                if (segment_duration >= MIN_SEGMENT_DURATION) {
                    int64_t buffer_start_abs = state->audio_buffer.dequeued_frames();
                    int64_t cut_abs = static_cast<int64_t>(std::round(seg_end_time * SAMPLE_RATE));
                    submit_audio_chunk(state, buffer_start_abs, cut_abs);
                }
            }
        }
    }

    if (need_flush && state->audio_buffer.size() > 0 && !state->whisper_in_flight) {
        int64_t abs_start = state->audio_buffer.dequeued_frames();
        int64_t abs_end = state->audio_buffer.total_frames();
        submit_audio_chunk(state, abs_start, abs_end);
    }

    TranscribeResult result;
    if (state->whisper_in_flight && transcriber_try_get_result(state->transcriber, result)) {
        DiarizationResult diarization = streaming_recluster(state->streaming_state);
        handle_whisper_result(state, result, diarization);
        state->whisper_in_flight = false;
    }
}

void pipeline_finalize(PipelineState* state) {
    if (!state) return;

    SilenceFilterResult sf_flush = silence_filter_flush(state->silence_filter);
    if (!sf_flush.audio.empty()) {
        state->audio_buffer.enqueue(sf_flush.audio.data(), (int)sf_flush.audio.size());
        streaming_push(state->streaming_state, sf_flush.audio.data(), (int)sf_flush.audio.size());
    }

    DiarizationResult final_diarization = streaming_finalize(state->streaming_state);

    if (state->whisper_in_flight) {
        TranscribeResult inflight_result = transcriber_wait_result(state->transcriber);
        handle_whisper_result(state, inflight_result, final_diarization);
        state->whisper_in_flight = false;
    }

    if (state->audio_buffer.size() > 0) {
        int64_t abs_start = state->audio_buffer.dequeued_frames();
        int64_t abs_end = state->audio_buffer.total_frames();

        std::vector<float> audio;
        state->audio_buffer.read_range(abs_start, abs_end, audio);

        if (!audio.empty()) {
            if ((int)audio.size() > MAX_WHISPER_SAMPLES) {
                audio.resize(MAX_WHISPER_SAMPLES);
            }

            transcriber_submit(state->transcriber, audio.data(), (int)audio.size(), state->buffer_start_time);
            state->audio_buffer.dequeue_up_to(abs_end);

            TranscribeResult final_result = transcriber_wait_result(state->transcriber);
            handle_whisper_result(state, final_result, final_diarization);
        }
    }
}

void pipeline_free(PipelineState* state) {
    if (!state) return;

    transcriber_free(state->transcriber);
    segment_detector_free(state->segment_detector);
    streaming_free(state->streaming_state);
    silence_filter_free(state->silence_filter);

    if (state->vad_ctx) {
        whisper_vad_free(state->vad_ctx);
    }

    delete state;
}
