// test_offline_progress.cpp — Tests progress_callback in offline_transcribe_with_cache()
//
// Verifies:
// 1. Phase 0 (whisper) is called with progress 0-100
// 2. Phase 1 (diarization) is called once with progress 0
// 3. Phase 2 (alignment) is called once with progress 0
// 4. Phases arrive in order: all phase-0 before phase-1 before phase-2
// 5. No callback is fine (null progress_callback still works)

#include "offline_pipeline.h"
#include "model_cache.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
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

    if (header.audio_format != 1 || header.num_channels != 1 || header.bits_per_sample != 16) {
        fprintf(stderr, "ERROR: Only mono 16-bit PCM supported\n");
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
    return true;
}

struct ProgressEvent {
    int phase;
    int progress;
};

static void print_usage(const char* prog) {
    fprintf(stderr,
            "Usage: %s <audio.wav> <whisper-model> <seg-model> <emb-model> <plda>\n"
            "       --seg-coreml <path> --emb-coreml <path>\n", prog);
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

    for (int i = 6; i < argc; i++) {
        if (strcmp(argv[i], "--seg-coreml") == 0 && i + 1 < argc) {
            seg_coreml = argv[++i];
        } else if (strcmp(argv[i], "--emb-coreml") == 0 && i + 1 < argc) {
            emb_coreml = argv[++i];
        }
    }

    // ================================================================
    // Load audio
    // ================================================================

    std::vector<float> audio;
    uint32_t sample_rate;
    if (!load_wav_file(audio_path, audio, sample_rate)) {
        fprintf(stderr, "FAIL: could not load WAV file\n");
        return 1;
    }

    fprintf(stderr, "Loaded %s: %zu samples (%.1fs) at %u Hz\n",
            audio_path, audio.size(), (double)audio.size() / sample_rate, sample_rate);

    // ================================================================
    // Build ModelCache (same pattern as node binding)
    // ================================================================

    OfflinePipelineConfig config;
    config.seg_model_path  = seg_path;
    config.emb_model_path  = emb_path;
    config.plda_path       = plda_path;
    if (seg_coreml) config.seg_coreml_path = seg_coreml;
    if (emb_coreml) config.coreml_path     = emb_coreml;

    config.transcriber.whisper_model_path = whisper_path;
    config.transcriber.n_threads = 4;
    config.transcriber.language  = "en";
    config.transcriber.no_prints = true;

    // ================================================================
    // Test 1: offline_transcribe_with_cache + progress_callback
    // ================================================================

    fprintf(stderr, "\n=== Test 1: progress_callback receives correct phases ===\n");

    // Build cache
    ModelCacheConfig cache_config{};
    cache_config.seg_model_path  = config.seg_model_path;
    cache_config.emb_model_path  = config.emb_model_path;
    cache_config.plda_path       = config.plda_path;
    cache_config.seg_coreml_path = config.seg_coreml_path;
    cache_config.coreml_path     = config.coreml_path;
    cache_config.transcriber     = config.transcriber;

    ModelCache* cache = model_cache_load(cache_config);
    if (!cache) {
        fprintf(stderr, "FAIL: model_cache_init returned nullptr\n");
        return 1;
    }

    // Collect progress events
    std::vector<ProgressEvent> events;
    std::mutex events_mutex;

    config.progress_callback = [&](int phase, int progress) {
        std::lock_guard<std::mutex> lock(events_mutex);
        events.push_back({phase, progress});
    };

    OfflinePipelineResult result = offline_transcribe_with_cache(
        config, cache, audio.data(), static_cast<int>(audio.size()));

    if (!result.valid) {
        fprintf(stderr, "FAIL: offline_transcribe_with_cache returned invalid result\n");
        model_cache_free(cache);
        return 1;
    }

    fprintf(stderr, "Pipeline produced %zu aligned segments\n", result.segments.size());
    fprintf(stderr, "Collected %zu progress events\n", events.size());

    // ----------------------------------------------------------------
    // Validate progress events
    // ----------------------------------------------------------------

    bool ok = true;

    // Must have at least some events
    if (events.empty()) {
        fprintf(stderr, "FAIL: no progress events received\n");
        ok = false;
    }

    // Count events per phase
    int phase0_count = 0, phase1_count = 0, phase2_count = 0;
    int phase0_min = 999, phase0_max = -1;

    for (const auto& e : events) {
        if (e.phase == 0) {
            phase0_count++;
            if (e.progress < phase0_min) phase0_min = e.progress;
            if (e.progress > phase0_max) phase0_max = e.progress;
        } else if (e.phase == 1) {
            phase1_count++;
        } else if (e.phase == 2) {
            phase2_count++;
        }
    }

    fprintf(stderr, "Phase 0 (whisper): %d events, progress range [%d, %d]\n",
            phase0_count, phase0_min, phase0_max);
    fprintf(stderr, "Phase 1 (diarization): %d events\n", phase1_count);
    fprintf(stderr, "Phase 2 (alignment): %d events\n", phase2_count);

    // Phase 0 should have multiple calls from whisper's progress_callback
    if (phase0_count == 0) {
        fprintf(stderr, "FAIL: phase 0 (whisper) never called\n");
        ok = false;
    }

    // Phase 0 should reach 100
    if (phase0_max < 100) {
        fprintf(stderr, "FAIL: phase 0 max progress = %d (expected 100)\n", phase0_max);
        ok = false;
    }

    // Phase 1 should be called exactly once
    if (phase1_count != 1) {
        fprintf(stderr, "FAIL: phase 1 (diarization) called %d times (expected 1)\n", phase1_count);
        ok = false;
    }

    // Phase 2 should be called exactly once
    if (phase2_count != 1) {
        fprintf(stderr, "FAIL: phase 2 (alignment) called %d times (expected 1)\n", phase2_count);
        ok = false;
    }

    // Phases must appear in order: all 0s, then 1, then 2
    int last_phase = -1;
    for (const auto& e : events) {
        if (e.phase < last_phase) {
            fprintf(stderr, "FAIL: out-of-order phase: saw phase %d after phase %d\n",
                    e.phase, last_phase);
            ok = false;
            break;
        }
        last_phase = e.phase;
    }

    // Phase 0 progress values should be non-decreasing
    int last_progress = -1;
    for (const auto& e : events) {
        if (e.phase != 0) break;
        if (e.progress < last_progress) {
            fprintf(stderr, "FAIL: phase 0 progress went backwards: %d -> %d\n",
                    last_progress, e.progress);
            ok = false;
            break;
        }
        last_progress = e.progress;
    }

    // Result should still be valid (callback didn't break anything)
    if (result.segments.empty()) {
        fprintf(stderr, "FAIL: no segments produced\n");
        ok = false;
    }

    // ================================================================
    // Test 2: No progress_callback (null) still works
    // ================================================================

    fprintf(stderr, "\n=== Test 2: null progress_callback (backward compat) ===\n");

    OfflinePipelineConfig config_no_cb = config;
    config_no_cb.progress_callback = nullptr;

    OfflinePipelineResult result2 = offline_transcribe_with_cache(
        config_no_cb, cache, audio.data(), static_cast<int>(audio.size()));

    if (!result2.valid) {
        fprintf(stderr, "FAIL: null progress_callback broke the pipeline\n");
        ok = false;
    } else {
        fprintf(stderr, "PASS: null callback produced %zu segments\n", result2.segments.size());
    }

    // ================================================================
    // Test 3: offline_transcribe (non-cached) with progress_callback
    // ================================================================

    fprintf(stderr, "\n=== Test 3: offline_transcribe (non-cached) with progress_callback ===\n");

    std::vector<ProgressEvent> events3;
    OfflinePipelineConfig config3 = config;
    config3.progress_callback = [&](int phase, int progress) {
        events3.push_back({phase, progress});
    };

    OfflinePipelineResult result3 = offline_transcribe(
        config3, audio.data(), static_cast<int>(audio.size()));

    if (!result3.valid) {
        fprintf(stderr, "FAIL: offline_transcribe with progress_callback failed\n");
        ok = false;
    } else {
        int p0 = 0, p1 = 0, p2 = 0;
        for (const auto& e : events3) {
            if (e.phase == 0) p0++;
            else if (e.phase == 1) p1++;
            else if (e.phase == 2) p2++;
        }
        fprintf(stderr, "Non-cached: %zu events (phase0=%d, phase1=%d, phase2=%d)\n",
                events3.size(), p0, p1, p2);

        if (p0 == 0 || p1 != 1 || p2 != 1) {
            fprintf(stderr, "FAIL: non-cached path has wrong event counts\n");
            ok = false;
        } else {
            fprintf(stderr, "PASS: non-cached progress_callback works correctly\n");
        }
    }

    // ================================================================
    // Cleanup & result
    // ================================================================

    model_cache_free(cache);

    if (ok) {
        fprintf(stderr, "\nPASS: offline progress callback test\n");
        return 0;
    } else {
        fprintf(stderr, "\nFAIL: offline progress callback test\n");
        return 1;
    }
}
