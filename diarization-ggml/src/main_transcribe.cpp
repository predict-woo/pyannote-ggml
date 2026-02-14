#include "pipeline.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#ifdef __APPLE__
#include <mach/mach.h>
#endif
#include <sys/resource.h>
#include <vector>

static constexpr int SAMPLE_RATE = 16000;
static constexpr int CHUNK_SIZE = 16000;

struct ResourceStats {
    std::atomic<bool> running{false};
    std::thread sampler_thread;

    std::mutex mtx;
    size_t min_rss = std::numeric_limits<size_t>::max();
    size_t max_rss = 0;
    size_t sum_rss = 0;
    size_t sample_count = 0;
};

static size_t get_current_rss() {
#ifdef __APPLE__
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        return info.resident_size;
    }
#endif
    return 0;
}

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

struct WavReader {
    std::ifstream file;
    uint32_t sample_rate = 0;
    uint32_t total_samples = 0;
    uint32_t samples_read = 0;
    bool valid = false;
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

static bool wav_reader_open(WavReader& reader, const std::string& path) {
    reader.file.open(path, std::ios::binary);
    if (!reader.file.is_open()) {
        fprintf(stderr, "ERROR: Failed to open WAV file: %s\n", path.c_str());
        return false;
    }

    wav_header header;
    reader.file.read(reinterpret_cast<char*>(&header), sizeof(wav_header));
    if (!reader.file.good()) {
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
    reader.file.read(reinterpret_cast<char*>(&data_chunk), sizeof(wav_data_chunk));
    if (!reader.file.good()) {
        fprintf(stderr, "ERROR: Failed to read WAV chunks\n");
        return false;
    }

    while (std::strncmp(data_chunk.id, "data", 4) != 0) {
        reader.file.seekg(data_chunk.size, std::ios::cur);
        if (!reader.file.read(reinterpret_cast<char*>(&data_chunk), sizeof(wav_data_chunk))) {
            fprintf(stderr, "ERROR: Data chunk not found\n");
            return false;
        }
    }

    reader.total_samples = data_chunk.size / 2;
    reader.sample_rate = header.sample_rate;
    reader.samples_read = 0;
    reader.valid = true;
    return true;
}

static int wav_reader_read(WavReader& reader, std::vector<float>& out, int n) {
    if (!reader.valid) {
        return 0;
    }

    const int remaining = static_cast<int>(reader.total_samples - reader.samples_read);
    const int to_read = std::min(n, remaining);
    if (to_read <= 0) {
        return 0;
    }

    std::vector<int16_t> pcm(to_read);
    reader.file.read(reinterpret_cast<char*>(pcm.data()), to_read * static_cast<int>(sizeof(int16_t)));
    const int actually_read = static_cast<int>(reader.file.gcount() / static_cast<std::streamsize>(sizeof(int16_t)));

    out.resize(actually_read);
    for (int i = 0; i < actually_read; ++i) {
        out[i] = static_cast<float>(pcm[i]) / 32768.0f;
    }

    reader.samples_read += static_cast<uint32_t>(actually_read);
    return actually_read;
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

static std::string trim_ascii_whitespace(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start])) != 0) {
        ++start;
    }

    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1])) != 0) {
        --end;
    }

    return s.substr(start, end - start);
}

static std::string segment_text(const AlignedSegment& seg) {
    std::string raw;
    for (const AlignedWord& word : seg.words) {
        raw += word.text;
    }
    return trim_ascii_whitespace(raw);
}

static void write_segments_json(FILE* out, const std::vector<AlignedSegment>& segments) {
    std::fprintf(out, "{\n  \"segments\": [\n");
    for (size_t s = 0; s < segments.size(); ++s) {
        const AlignedSegment& seg = segments[s];
        const std::string combined_text = segment_text(seg);
        std::fprintf(out, "    {\"speaker\": \"%s\", \"start\": %.6f, \"duration\": %.6f, \"text\": \"%s\", \"words\": [",
                     json_escape(seg.speaker).c_str(),
                     seg.start,
                     seg.duration,
                     json_escape(combined_text).c_str());

        for (size_t w = 0; w < seg.words.size(); ++w) {
            const AlignedWord& word = seg.words[w];
            std::fprintf(out, "%s{\"text\": \"%s\", \"start\": %.6f, \"end\": %.6f}",
                         (w == 0 ? "" : ", "),
                         json_escape(word.text).c_str(),
                         word.start,
                         word.end);
        }

        std::fprintf(out, "]}%s\n", (s + 1 == segments.size() ? "" : ","));
    }
    std::fprintf(out, "  ]\n}\n");
}

static bool write_segments_json_file(const std::string& path, const std::vector<AlignedSegment>& segments) {
    FILE* out = std::fopen(path.c_str(), "wb");
    if (!out) return false;

    write_segments_json(out, segments);
    std::fclose(out);
    return true;
}

static std::string audio_file_id_from_path(const std::string& path) {
    size_t sep = path.find_last_of("/\\");
    std::string base = (sep == std::string::npos) ? path : path.substr(sep + 1);
    size_t dot = base.find_last_of('.');
    if (dot != std::string::npos) {
        base = base.substr(0, dot);
    }
    return base;
}

static bool write_segments_rttm_file(
    const std::string& path,
    const std::string& file_id,
    const std::vector<AlignedSegment>& segments) {
    FILE* out = std::fopen(path.c_str(), "wb");
    if (!out) return false;

    for (const AlignedSegment& seg : segments) {
        std::fprintf(out,
                     "SPEAKER %s 1 %.3f %.3f <NA> <NA> %s <NA> <NA>\n",
                     file_id.c_str(),
                     seg.start,
                     seg.duration,
                     seg.speaker.c_str());
    }

    std::fclose(out);
    return true;
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
    fprintf(stderr, "  --rttm <path>           Output RTTM diarization file\n");
    fprintf(stderr, "  --realtime              Pace audio at 1x real-time speed (for profiling)\n");
    fprintf(stderr, "  --stats                 Print memory and CPU usage statistics\n");
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
    const char* rttm_path = nullptr;
    const char* language = "en";
    bool realtime = false;
    bool stats = false;
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
        } else if (arg == "--rttm" && i + 1 < argc) {
            rttm_path = argv[++i];
        } else if (arg == "--realtime") {
            realtime = true;
        } else if (arg == "--stats") {
            stats = true;
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

    struct CallbackCtx {
        std::vector<AlignedSegment> segments;
        std::string output_path;
        std::string rttm_path;
        std::string file_id;
    } callback_ctx;
    callback_ctx.output_path = output_path;
    if (rttm_path) {
        callback_ctx.rttm_path = rttm_path;
        callback_ctx.file_id = audio_file_id_from_path(audio_path);
    }

    auto callback = [](const std::vector<AlignedSegment>& segments, void* user_data) {
        auto* ctx = static_cast<CallbackCtx*>(user_data);
        ctx->segments = segments;

        if (!ctx->output_path.empty()) {
            if (!write_segments_json_file(ctx->output_path, ctx->segments)) {
                fprintf(stderr, "Error: could not write incremental output file '%s'\n", ctx->output_path.c_str());
            }
        }

        if (!ctx->rttm_path.empty()) {
            if (!write_segments_rttm_file(ctx->rttm_path, ctx->file_id, ctx->segments)) {
                fprintf(stderr, "Error: could not write incremental RTTM file '%s'\n", ctx->rttm_path.c_str());
            }
        }
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

    ResourceStats resource_stats;
    if (stats) {
        resource_stats.running = true;
        resource_stats.sampler_thread = std::thread([&resource_stats]() {
            while (resource_stats.running.load()) {
                size_t rss = get_current_rss();
                if (rss > 0) {
                    std::lock_guard<std::mutex> lock(resource_stats.mtx);
                    resource_stats.min_rss = std::min(resource_stats.min_rss, rss);
                    resource_stats.max_rss = std::max(resource_stats.max_rss, rss);
                    resource_stats.sum_rss += rss;
                    resource_stats.sample_count++;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        });
    }

    const auto fail_after_init = [&](int code) {
        pipeline_free(state);
        if (stats) {
            resource_stats.running = false;
            if (resource_stats.sampler_thread.joinable()) {
                resource_stats.sampler_thread.join();
            }
        }
        return code;
    };

    double total_audio_s = 0.0;

    if (realtime) {
        WavReader wav;
        if (!wav_reader_open(wav, audio_path)) {
            return fail_after_init(1);
        }
        if (wav.sample_rate != SAMPLE_RATE) {
            fprintf(stderr, "Error: expected %d Hz audio, got %u Hz\n", SAMPLE_RATE, wav.sample_rate);
            return fail_after_init(1);
        }

        total_audio_s = static_cast<double>(wav.total_samples) / SAMPLE_RATE;
        std::vector<float> chunk_buf;
        double next_progress_s = 1.0;

        while (true) {
            const int n = wav_reader_read(wav, chunk_buf, CHUNK_SIZE);
            if (n <= 0) {
                break;
            }

            pipeline_push(state, chunk_buf.data(), n);
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));

            const double pushed_s = static_cast<double>(wav.samples_read) / SAMPLE_RATE;
            if (pushed_s + 1e-9 >= next_progress_s) {
                fprintf(stderr, "[push] %.1fs / %.1fs\n", pushed_s, total_audio_s);
                next_progress_s += 1.0;
            }
        }
    } else {
        std::vector<float> audio;
        uint32_t sample_rate = 0;
        if (!load_wav_file(audio_path, audio, sample_rate)) {
            return fail_after_init(1);
        }
        if (sample_rate != SAMPLE_RATE) {
            fprintf(stderr, "Error: expected %d Hz audio, got %u Hz\n", SAMPLE_RATE, sample_rate);
            return fail_after_init(1);
        }

        total_audio_s = static_cast<double>(audio.size()) / SAMPLE_RATE;
        const double progress_interval = 5.0;
        double next_progress_s = progress_interval;

        for (size_t offset = 0; offset < audio.size(); offset += CHUNK_SIZE) {
            const int n = static_cast<int>(std::min<size_t>(CHUNK_SIZE, audio.size() - offset));
            pipeline_push(state, audio.data() + offset, n);

            const double pushed_s = static_cast<double>(offset + n) / SAMPLE_RATE;
            if (pushed_s + 1e-9 >= next_progress_s) {
                fprintf(stderr, "[push] %.1fs / %.1fs\n", pushed_s, total_audio_s);
                next_progress_s += progress_interval;
            }
        }
    }

    pipeline_finalize(state);

    const auto t1 = std::chrono::steady_clock::now();
    pipeline_free(state);

    if (stats) {
        resource_stats.running = false;
        if (resource_stats.sampler_thread.joinable()) {
            resource_stats.sampler_thread.join();
        }
    }

    if (output_path.empty()) {
        write_segments_json(stdout, callback_ctx.segments);
    } else {
        if (!write_segments_json_file(output_path, callback_ctx.segments)) {
            fprintf(stderr, "Error: could not open output file '%s'\n", output_path.c_str());
            return 1;
        }
    }

    if (rttm_path) {
        const std::string file_id = audio_file_id_from_path(audio_path);
        if (!write_segments_rttm_file(rttm_path, file_id, callback_ctx.segments)) {
            fprintf(stderr, "Error: could not open RTTM output file '%s'\n", rttm_path);
            return 1;
        }
    }

    const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    const double audio_duration_s = total_audio_s;
    const double rtf = audio_duration_s > 0.0 ? elapsed_s / audio_duration_s : 0.0;
    fprintf(stderr, "Timing: total=%.3fs audio=%.3fs rtf=%.4f\n", elapsed_s, audio_duration_s, rtf);

    if (stats) {
        struct rusage usage;
        std::memset(&usage, 0, sizeof(usage));
        getrusage(RUSAGE_SELF, &usage);

        const double user_s = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6;
        const double sys_s = usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6;
        const size_t peak_rss = usage.ru_maxrss;
        const size_t current_rss = get_current_rss();

        fprintf(stderr, "\nResource usage:\n");
        fprintf(stderr, "  CPU time:      %.2fs (user: %.2fs, sys: %.2fs)\n", user_s + sys_s, user_s, sys_s);
        fprintf(stderr, "  Peak RSS:      %zu MB\n", peak_rss / (1024 * 1024));
        fprintf(stderr, "  Current RSS:   %zu MB\n", current_rss / (1024 * 1024));

        std::lock_guard<std::mutex> lock(resource_stats.mtx);
        if (resource_stats.sample_count > 0) {
            const size_t avg_rss = resource_stats.sum_rss / resource_stats.sample_count;
            fprintf(stderr,
                    "  RSS samples:   min=%zu MB, max=%zu MB, avg=%zu MB (%zu samples)\n",
                    resource_stats.min_rss / (1024 * 1024),
                    resource_stats.max_rss / (1024 * 1024),
                    avg_rss / (1024 * 1024),
                    resource_stats.sample_count);
        }
    }

    return 0;
}
