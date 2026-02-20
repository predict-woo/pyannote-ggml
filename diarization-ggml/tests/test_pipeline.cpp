#include "pipeline.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

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
        fprintf(stderr, "ERROR: Only PCM format supported\n");
        return false;
    }

    if (header.num_channels != 1) {
        fprintf(stderr, "ERROR: Only mono audio supported\n");
        return false;
    }

    if (header.bits_per_sample != 16) {
        fprintf(stderr, "ERROR: Only 16-bit audio supported\n");
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

static void print_usage(const char* prog) {
    fprintf(stderr,
            "Usage: %s <audio.wav> <whisper-model> <seg-model> <emb-model> <plda>\n"
            "       --seg-coreml <path> --emb-coreml <path>\n"
            "       [--vad-model <path>]\n", prog);
}

int main(int argc, char* argv[]) {
    if (argc < 6) {
        print_usage(argv[0]);
        fprintf(stderr, "SKIP: missing model paths (expected for CI without models)\n");
        return 0;
    }

    const char* audio_path   = argv[1];
    const char* whisper_path = argv[2];
    const char* seg_path     = argv[3];
    const char* emb_path     = argv[4];
    const char* plda_path    = argv[5];

    const char* seg_coreml   = nullptr;
    const char* emb_coreml   = nullptr;
    const char* vad_model    = nullptr;

    for (int i = 6; i < argc; i++) {
        if (strcmp(argv[i], "--seg-coreml") == 0 && i + 1 < argc) {
            seg_coreml = argv[++i];
        } else if (strcmp(argv[i], "--emb-coreml") == 0 && i + 1 < argc) {
            emb_coreml = argv[++i];
        } else if (strcmp(argv[i], "--vad-model") == 0 && i + 1 < argc) {
            vad_model = argv[++i];
        }
    }

    std::vector<float> audio;
    uint32_t sample_rate;
    if (!load_wav_file(audio_path, audio, sample_rate)) {
        fprintf(stderr, "FAIL: could not load WAV file\n");
        return 1;
    }

    fprintf(stderr, "Loaded %s: %zu samples (%.1fs) at %u Hz\n",
            audio_path, audio.size(), (double)audio.size() / sample_rate, sample_rate);

    PipelineConfig config{};

    config.diarization.seg_model_path = seg_path;
    config.diarization.emb_model_path = emb_path;
    config.diarization.plda_path = plda_path;
    if (seg_coreml) config.diarization.seg_coreml_path = seg_coreml;
    if (emb_coreml) config.diarization.coreml_path = emb_coreml;

    config.transcriber.whisper_model_path = whisper_path;
    config.transcriber.n_threads = 4;
    config.transcriber.language = "en";

    config.vad_model_path = vad_model;

    struct CallbackCtx {
        std::vector<AlignedSegment> results;
        int count = 0;
        int audio_received_count = 0;
        size_t last_audio_size = 0;
        size_t max_audio_size = 0;
    };

    CallbackCtx cb_ctx;

    auto cb_fn = [](const std::vector<AlignedSegment>& segs, const std::vector<float>& audio, void* ud) {
        auto* ctx = static_cast<CallbackCtx*>(ud);
        ctx->results = segs;
        ctx->count++;
        if (!audio.empty()) {
            ctx->audio_received_count++;
            ctx->last_audio_size = audio.size();
            if (audio.size() > ctx->max_audio_size) {
                ctx->max_audio_size = audio.size();
            }
        }
    };

    PipelineState* state = pipeline_init(config, cb_fn, &cb_ctx);

    if (!state) {
        fprintf(stderr, "FAIL: pipeline_init returned nullptr\n");
        return 1;
    }

    constexpr int CHUNK_SIZE = 16000;
    size_t total_vad_predictions = 0;
    for (size_t offset = 0; offset < audio.size(); offset += CHUNK_SIZE) {
        int n = static_cast<int>(std::min<size_t>(CHUNK_SIZE, audio.size() - offset));
        std::vector<bool> vad = pipeline_push(state, audio.data() + offset, n);
        total_vad_predictions += vad.size();
    }

    pipeline_finalize(state);

    fprintf(stderr, "\n=== Pipeline Test Results ===\n");
    fprintf(stderr, "Callback invocations: %d\n", cb_ctx.count);
    fprintf(stderr, "Total aligned segments: %zu\n", cb_ctx.results.size());
    fprintf(stderr, "Total VAD predictions from push: %zu\n", total_vad_predictions);
    fprintf(stderr, "Callbacks with audio: %d\n", cb_ctx.audio_received_count);
    fprintf(stderr, "Max callback audio size: %zu samples (%.2fs)\n",
            cb_ctx.max_audio_size, static_cast<double>(cb_ctx.max_audio_size) / 16000.0);

    bool ok = true;

    if (cb_ctx.count == 0) {
        fprintf(stderr, "FAIL: callback was never invoked\n");
        ok = false;
    }

    std::set<std::string> speakers;
    bool has_text = false;

    for (const auto& seg : cb_ctx.results) {
        speakers.insert(seg.speaker);
        if (!seg.text.empty()) {
            has_text = true;
        }
    }

    fprintf(stderr, "Speakers found: %zu (", speakers.size());
    for (const auto& s : speakers) fprintf(stderr, " %s", s.c_str());
    fprintf(stderr, " )\n");

    if (speakers.size() < 2) {
        fprintf(stderr, "FAIL: expected >=2 speakers, got %zu\n", speakers.size());
        ok = false;
    }

    if (!has_text) {
        fprintf(stderr, "FAIL: no text produced\n");
        ok = false;
    }

    if (total_vad_predictions == 0) {
        fprintf(stderr, "FAIL: pipeline_push returned no VAD predictions\n");
        ok = false;
    } else {
        size_t expected_min = audio.size() / 512 / 2;
        if (total_vad_predictions < expected_min) {
            fprintf(stderr, "FAIL: too few VAD predictions: %zu (expected at least %zu)\n",
                    total_vad_predictions, expected_min);
            ok = false;
        }
    }

    if (cb_ctx.audio_received_count == 0) {
        fprintf(stderr, "FAIL: callback never received audio\n");
        ok = false;
    }

    constexpr size_t MAX_WHISPER_SAMPLES = 30 * 16000;
    if (cb_ctx.max_audio_size > MAX_WHISPER_SAMPLES) {
        fprintf(stderr, "FAIL: callback audio exceeds 30s cap: %zu samples\n", cb_ctx.max_audio_size);
        ok = false;
    }

    for (const auto& seg : cb_ctx.results) {
        fprintf(stderr, "  [%.2f - %.2f] %s: %s\n",
                seg.start, seg.start + seg.duration,
                seg.speaker.c_str(), seg.text.c_str());
    }

    pipeline_free(state);

    if (ok) {
        fprintf(stderr, "\nPASS: pipeline integration test\n");
        return 0;
    } else {
        fprintf(stderr, "\nFAIL: pipeline integration test\n");
        return 1;
    }
}
