#include "pipeline.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static constexpr int SAMPLE_RATE = 16000;
static constexpr int CHUNK_SIZE = 16000;

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
    if (!file.good()) {
        fprintf(stderr, "ERROR: Failed to read WAV header\n");
        return false;
    }

    if (std::strncmp(header.riff, "RIFF", 4) != 0 || std::strncmp(header.wave, "WAVE", 4) != 0) {
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
    if (!file.good()) {
        fprintf(stderr, "ERROR: Failed to read WAV chunks\n");
        return false;
    }

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
    if (!file.good()) {
        fprintf(stderr, "ERROR: Failed to read WAV sample data\n");
        return false;
    }

    for (size_t i = 0; i < num_samples; ++i) {
        samples[i] = static_cast<float>(pcm_data[i]) / 32768.0f;
    }

    sample_rate = header.sample_rate;
    return true;
}

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '\\') {
            out += "\\\\";
        } else if (c == '"') {
            out += "\\\"";
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == '\t') {
            out += "\\t";
        } else {
            out += c;
        }
    }
    return out;
}

static void print_usage(const char* program) {
    fprintf(stderr, "Usage: %s <audio.wav> [options]\n", program);
    fprintf(stderr, "\n");
    fprintf(stderr, "Positional arguments:\n");
    fprintf(stderr, "  audio.wav               Input audio file (16kHz mono PCM WAV)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --seg-model <path>      Segmentation GGUF model path\n");
    fprintf(stderr, "  --emb-model <path>      Embedding GGUF model path\n");
    fprintf(stderr, "  --whisper-model <path>  Whisper GGUF model path\n");
    fprintf(stderr, "  --plda <path>           PLDA model path\n");
    fprintf(stderr, "  --seg-coreml <path>     Segmentation CoreML model path\n");
    fprintf(stderr, "  --emb-coreml <path>     Embedding CoreML model path\n");
    fprintf(stderr, "  --vad-model <path>      Optional Silero VAD model path\n");
    fprintf(stderr, "  --language <lang>       Whisper language code (default: en)\n");
    fprintf(stderr, "  -o, --output <path>     Output JSON file (default: stdout)\n");
    fprintf(stderr, "  --help                  Print this help message\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    if (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
        print_usage(argv[0]);
        return 0;
    }

    std::string audio_path;
    const char* seg_model = nullptr;
    const char* emb_model = nullptr;
    const char* whisper_model = nullptr;
    const char* plda_model = nullptr;
    const char* seg_coreml = nullptr;
    const char* emb_coreml = nullptr;
    const char* vad_model = nullptr;
    const char* language = "en";
    std::string output_path;

    int i = 1;
    if (i < argc) {
        std::string first = argv[i];
        if (first.rfind("-", 0) != 0) {
            audio_path = first;
            ++i;
        }
    }

    for (; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--seg-model" && i + 1 < argc) {
            seg_model = argv[++i];
        } else if (arg == "--emb-model" && i + 1 < argc) {
            emb_model = argv[++i];
        } else if (arg == "--whisper-model" && i + 1 < argc) {
            whisper_model = argv[++i];
        } else if (arg == "--plda" && i + 1 < argc) {
            plda_model = argv[++i];
        } else if (arg == "--seg-coreml" && i + 1 < argc) {
            seg_coreml = argv[++i];
        } else if (arg == "--emb-coreml" && i + 1 < argc) {
            emb_coreml = argv[++i];
        } else if (arg == "--vad-model" && i + 1 < argc) {
            vad_model = argv[++i];
        } else if (arg == "--language" && i + 1 < argc) {
            language = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Error: unknown or incomplete option '%s'\n\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (audio_path.empty()) {
        fprintf(stderr, "Error: missing positional argument <audio.wav>\n\n");
        print_usage(argv[0]);
        return 1;
    }
    if (audio_path.size() < 4 || audio_path.substr(audio_path.size() - 4) != ".wav") {
        fprintf(stderr, "Error: input must be a .wav file\n");
        return 1;
    }
    if (!seg_model || !emb_model || !whisper_model || !plda_model) {
        fprintf(stderr, "Error: --seg-model, --emb-model, --whisper-model, and --plda are required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    std::vector<float> audio;
    uint32_t sample_rate = 0;
    if (!load_wav_file(audio_path, audio, sample_rate)) {
        return 1;
    }
    if (sample_rate != SAMPLE_RATE) {
        fprintf(stderr, "Error: expected %d Hz audio, got %u Hz\n", SAMPLE_RATE, sample_rate);
        return 1;
    }

    struct CallbackCtx {
        std::vector<AlignedSegment> segments;
    } callback_ctx;

    auto callback = [](const std::vector<AlignedSegment>& segments, void* user_data) {
        auto* ctx = static_cast<CallbackCtx*>(user_data);
        ctx->segments = segments;
    };

    PipelineConfig config{};
    config.diarization.seg_model_path = seg_model;
    config.diarization.emb_model_path = emb_model;
    config.diarization.plda_path = plda_model;
    if (seg_coreml) {
        config.diarization.seg_coreml_path = seg_coreml;
    }
    if (emb_coreml) {
        config.diarization.coreml_path = emb_coreml;
    }
    config.transcriber.whisper_model_path = whisper_model;
    config.transcriber.whisper_coreml_path = nullptr;
    config.transcriber.n_threads = 4;
    config.transcriber.language = language;
    config.vad_model_path = vad_model;

    const auto t0 = std::chrono::steady_clock::now();

    PipelineState* state = pipeline_init(config, callback, &callback_ctx);
    if (!state) {
        fprintf(stderr, "Error: pipeline initialization failed\n");
        return 1;
    }

    for (size_t offset = 0; offset < audio.size(); offset += CHUNK_SIZE) {
        const int n = static_cast<int>(std::min<size_t>(CHUNK_SIZE, audio.size() - offset));
        pipeline_push(state, audio.data() + offset, n);
    }
    pipeline_finalize(state);

    const auto t1 = std::chrono::steady_clock::now();
    pipeline_free(state);

    FILE* out = stdout;
    if (!output_path.empty()) {
        out = std::fopen(output_path.c_str(), "wb");
        if (!out) {
            fprintf(stderr, "Error: could not open output file '%s'\n", output_path.c_str());
            return 1;
        }
    }

    std::fprintf(out, "{\n  \"segments\": [\n");
    for (size_t s = 0; s < callback_ctx.segments.size(); ++s) {
        const AlignedSegment& seg = callback_ctx.segments[s];
        std::fprintf(out, "    {\"speaker\": \"%s\", \"start\": %.6f, \"duration\": %.6f, \"words\": [",
                     json_escape(seg.speaker).c_str(), seg.start, seg.duration);

        for (size_t w = 0; w < seg.words.size(); ++w) {
            const AlignedWord& word = seg.words[w];
            std::fprintf(out, "%s{\"text\": \"%s\", \"start\": %.6f, \"end\": %.6f}",
                         (w == 0 ? "" : ", "),
                         json_escape(word.text).c_str(),
                         word.start,
                         word.end);
        }

        std::fprintf(out, "]}%s\n", (s + 1 == callback_ctx.segments.size() ? "" : ","));
    }
    std::fprintf(out, "  ]\n}\n");

    if (out != stdout) {
        std::fclose(out);
    }

    const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    const double audio_duration_s = static_cast<double>(audio.size()) / SAMPLE_RATE;
    const double rtf = audio_duration_s > 0.0 ? elapsed_s / audio_duration_s : 0.0;
    fprintf(stderr, "Timing: total=%.3fs audio=%.3fs rtf=%.4f\n", elapsed_s, audio_duration_s, rtf);

    return 0;
}
