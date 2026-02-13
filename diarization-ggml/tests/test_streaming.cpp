#include "streaming.h"
#include "rttm.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// ============================================================================
// WAV loading (copied from diarization.cpp)
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

// ============================================================================
// Usage
// ============================================================================

// ============================================================================
// JSON output helpers for --json-stream mode
// ============================================================================

static void print_segments_json(const std::vector<DiarizationResult::Segment>& segments) {
    printf("[");
    for (size_t i = 0; i < segments.size(); i++) {
        const auto& seg = segments[i];
        printf("{\"start\":%.3f,\"duration\":%.3f,\"speaker\":\"%s\"}",
               seg.start, seg.duration, seg.speaker.c_str());
        if (i < segments.size() - 1) printf(",");
    }
    printf("]");
}

static void print_json_push(int chunk, double time, const std::vector<VADChunk>& vad_chunks) {
    int total_active = 0;
    int total_frames = 0;
    for (const auto& vc : vad_chunks) {
        total_frames += vc.num_frames;
        for (int f = 0; f < vc.num_frames; f++) {
            if (vc.vad[f] == 1.0f) total_active++;
        }
    }
    printf("{\"type\":\"push\",\"chunk\":%d,\"time\":%.1f,\"num_vad_frames\":%d,\"vad_active_frames\":%d}\n",
           chunk, time, total_frames, total_active);
    fflush(stdout);
}

static void print_json_recluster(int chunk, double time, const DiarizationResult& result) {
    printf("{\"type\":\"recluster\",\"chunk\":%d,\"time\":%.1f,\"segments\":", chunk, time);
    print_segments_json(result.segments);
    printf("}\n");
    fflush(stdout);
}

static void print_json_finalize(const std::vector<DiarizationResult::Segment>& segments) {
    printf("{\"type\":\"finalize\",\"segments\":");
    print_segments_json(segments);
    printf("}\n");
    fflush(stdout);
}

// ============================================================================
// Usage
// ============================================================================

static void print_usage(const char* program) {
    fprintf(stderr, "Usage: %s [options] [audio.wav]\n", program);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --help                Print this help message\n");
    fprintf(stderr, "  --init-only           Test init/free only (no audio processing)\n");
    fprintf(stderr, "  --push-only           Test push without finalize\n");
    fprintf(stderr, "  --provisional-only    Test first 30s only (no full recluster)\n");
    fprintf(stderr, "  --recluster-test      Test recluster behavior\n");
    fprintf(stderr, "  --benchmark           Measure push latency\n");
    fprintf(stderr, "  --json-stream         Output JSON events to stdout (for streaming viewer)\n");
    fprintf(stderr, "  --verify-frames       Verify incremental frame accumulation correctness\n");
    fprintf(stderr, "  --multi-recluster     Test push→recluster→push→recluster cycles\n");
    fprintf(stderr, "  --zero-latency        Enable zero-latency mode (pre-fill 10s silence)\n");
    fprintf(stderr, "  -o, --output <path>   Output RTTM file\n");
    fprintf(stderr, "  --plda <path>         Path to PLDA binary file\n");
    fprintf(stderr, "  --coreml <path>       Path to CoreML embedding model\n");
    fprintf(stderr, "  --seg-coreml <path>   Path to CoreML segmentation model\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Default paths (if not specified):\n");
    fprintf(stderr, "  PLDA: plda.gguf\n");
    fprintf(stderr, "  Embedding CoreML: ../models/embedding-ggml/embedding.mlpackage\n");
    fprintf(stderr, "  Segmentation CoreML: ../models/segmentation-ggml/segmentation.mlpackage\n");
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    // Parse arguments
    std::string audio_path;
    std::string output_path;
    std::string plda_path = "plda.gguf";
    std::string coreml_path = "../embedding-ggml/embedding.mlpackage";
    std::string seg_coreml_path = "../segmentation-ggml/segmentation.mlpackage";
    
    bool init_only = false;
    bool push_only = false;
    bool provisional_only = false;
    bool recluster_test = false;
    bool benchmark = false;
    bool json_stream = false;
    bool verify_frames = false;
    bool multi_recluster = false;
    bool zero_latency = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--init-only") {
            init_only = true;
        } else if (arg == "--push-only") {
            push_only = true;
        } else if (arg == "--provisional-only") {
            provisional_only = true;
        } else if (arg == "--recluster-test") {
            recluster_test = true;
        } else if (arg == "--benchmark") {
            benchmark = true;
        } else if (arg == "--json-stream") {
            json_stream = true;
        } else if (arg == "--verify-frames") {
            verify_frames = true;
        } else if (arg == "--multi-recluster") {
            multi_recluster = true;
        } else if (arg == "--zero-latency") {
            zero_latency = true;
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--plda" && i + 1 < argc) {
            plda_path = argv[++i];
        } else if (arg == "--coreml" && i + 1 < argc) {
            coreml_path = argv[++i];
        } else if (arg == "--seg-coreml" && i + 1 < argc) {
            seg_coreml_path = argv[++i];
        } else if (arg[0] != '-') {
            audio_path = arg;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    StreamingConfig config;
    config.plda_path = plda_path;
    config.coreml_path = coreml_path;
    config.seg_coreml_path = seg_coreml_path;
    config.zero_latency = zero_latency;
    
    // ========================================================================
    // Init-only test
    // ========================================================================
    if (init_only) {
        fprintf(stderr, "Testing init/free...\n");
        StreamingState* state = streaming_init(config);
        if (!state) {
            fprintf(stderr, "FAILED: streaming_init returned nullptr\n");
            return 1;
        }
        fprintf(stderr, "streaming_init: OK\n");
        streaming_free(state);
        fprintf(stderr, "streaming_free: OK\n");
        fprintf(stderr, "PASSED: init/free test\n");
        return 0;
    }
    
    if (audio_path.empty()) {
        fprintf(stderr, "Error: audio file required for this test mode\n");
        print_usage(argv[0]);
        return 1;
    }
    
    std::vector<float> audio_samples;
    uint32_t sample_rate;
    if (!load_wav_file(audio_path, audio_samples, sample_rate)) {
        return 1;
    }
    
    if (sample_rate != 16000) {
        fprintf(stderr, "Error: expected 16kHz audio, got %u Hz\n", sample_rate);
        return 1;
    }
    
    const int num_samples = static_cast<int>(audio_samples.size());
    const double audio_duration = static_cast<double>(num_samples) / 16000.0;
    fprintf(stderr, "Audio: %.2fs, %d samples\n", audio_duration, num_samples);
    
    // ========================================================================
    // Verify-frames mode
    // ========================================================================
    if (verify_frames) {
        fprintf(stderr, "\n=== Frame Verification (%s mode) ===\n",
                zero_latency ? "zero_latency" : "normal");
        
        StreamingState* vstate = streaming_init(config);
        if (!vstate) {
            fprintf(stderr, "FAILED: streaming_init returned nullptr\n");
            return 1;
        }

        constexpr int STEP = 16000;
        int pushes = 0;
        int total_accumulated_frames = 0;
        int expected_next_start = 0;
        bool first_vad_received = false;
        int first_vad_push = -1;
        bool contiguity_ok = true;
        bool size_ok = true;

        for (int offset = 0; offset < num_samples; offset += STEP) {
            int chunk_size = std::min(STEP, num_samples - offset);
            auto vad_chunks = streaming_push(vstate, audio_samples.data() + offset, chunk_size);
            pushes++;

            for (const auto& vc : vad_chunks) {
                if (!first_vad_received) {
                    first_vad_received = true;
                    first_vad_push = pushes;

                    if (vc.start_frame != 0) {
                        fprintf(stderr, "FAIL: first VADChunk start_frame=%d, expected 0\n",
                                vc.start_frame);
                        contiguity_ok = false;
                    }
                }

                if (vc.start_frame != expected_next_start) {
                    fprintf(stderr, "FAIL: chunk %d start_frame=%d, expected %d (gap=%d)\n",
                            vc.chunk_index, vc.start_frame, expected_next_start,
                            vc.start_frame - expected_next_start);
                    contiguity_ok = false;
                }

                if (vc.num_frames != static_cast<int>(vc.vad.size())) {
                    fprintf(stderr, "FAIL: chunk %d num_frames=%d but vad.size()=%zu\n",
                            vc.chunk_index, vc.num_frames, vc.vad.size());
                    size_ok = false;
                }

                if (!zero_latency && vc.chunk_index == 0) {
                    if (vc.num_frames != 589) {
                        fprintf(stderr, "FAIL: chunk 0 should have 589 frames, got %d\n",
                                vc.num_frames);
                        size_ok = false;
                    }
                } else {
                    if (vc.num_frames < 58 || vc.num_frames > 61) {
                        fprintf(stderr, "FAIL: chunk %d has %d frames (expected 59-60)\n",
                                vc.chunk_index, vc.num_frames);
                        size_ok = false;
                    }
                }

                expected_next_start = vc.start_frame + vc.num_frames;
                total_accumulated_frames += vc.num_frames;

                fprintf(stderr, "  chunk=%d start_frame=%d num_frames=%d total=%d\n",
                        vc.chunk_index, vc.start_frame, vc.num_frames,
                        total_accumulated_frames);
            }
        }

        fprintf(stderr, "\nResults:\n");
        fprintf(stderr, "  Pushes: %d\n", pushes);
        fprintf(stderr, "  Total accumulated frames: %d\n", total_accumulated_frames);
        fprintf(stderr, "  First VAD at push: %d\n", first_vad_push);
        fprintf(stderr, "  Contiguity: %s\n", contiguity_ok ? "OK" : "FAIL");
        fprintf(stderr, "  Frame sizes: %s\n", size_ok ? "OK" : "FAIL");

        if (zero_latency) {
            if (first_vad_push != 1) {
                fprintf(stderr, "FAIL: zero_latency should produce VAD on push 1, got %d\n",
                        first_vad_push);
                streaming_free(vstate);
                return 1;
            }
            fprintf(stderr, "  Zero-latency immediate response: OK\n");
        } else {
            if (first_vad_push > 11) {
                fprintf(stderr, "FAIL: normal mode should produce first VAD by push 10-11, got %d\n",
                        first_vad_push);
                streaming_free(vstate);
                return 1;
            }
        }

        streaming_free(vstate);

        if (contiguity_ok && size_ok) {
            fprintf(stderr, "\nPASSED: frame verification\n");
            return 0;
        } else {
            fprintf(stderr, "\nFAILED: frame verification\n");
            return 1;
        }
    }

    // ========================================================================
    // Multi-recluster test: push→recluster→push→recluster→finalize
    // ========================================================================
    if (multi_recluster) {
        fprintf(stderr, "\n=== Multi-Recluster Test (%s mode) ===\n",
                zero_latency ? "zero_latency" : "normal");

        StreamingState* mrstate = streaming_init(config);
        if (!mrstate) {
            fprintf(stderr, "FAILED: streaming_init returned nullptr\n");
            return 1;
        }

        constexpr int STEP = 16000;
        int offset = 0;

        // Phase 1: Push 15 seconds
        fprintf(stderr, "Phase 1: Pushing 15s of audio...\n");
        for (int i = 0; i < 15 && offset < num_samples; i++) {
            int chunk_size = std::min(STEP, num_samples - offset);
            streaming_push(mrstate, audio_samples.data() + offset, chunk_size);
            offset += chunk_size;
        }

        // Recluster 1
        fprintf(stderr, "Recluster 1...\n");
        DiarizationResult r1 = streaming_recluster(mrstate);
        fprintf(stderr, "  Recluster 1: %zu segments\n", r1.segments.size());
        // Validate segments
        for (const auto& seg : r1.segments) {
            if (seg.start < 0 || seg.duration <= 0 || seg.speaker.empty()) {
                fprintf(stderr, "FAIL: invalid segment after recluster 1\n");
                streaming_free(mrstate);
                return 1;
            }
        }

        // Phase 2: Push 15 more seconds (THIS is where the old bug would crash/corrupt)
        fprintf(stderr, "Phase 2: Pushing 15 more seconds...\n");
        for (int i = 0; i < 15 && offset < num_samples; i++) {
            int chunk_size = std::min(STEP, num_samples - offset);
            auto vad = streaming_push(mrstate, audio_samples.data() + offset, chunk_size);
            offset += chunk_size;
            // Verify VAD chunks are still valid
            for (const auto& vc : vad) {
                if (vc.num_frames <= 0 || vc.num_frames > 589) {
                    fprintf(stderr, "FAIL: invalid VAD chunk after recluster\n");
                    streaming_free(mrstate);
                    return 1;
                }
                if (static_cast<int>(vc.vad.size()) != vc.num_frames) {
                    fprintf(stderr, "FAIL: vad size mismatch after recluster\n");
                    streaming_free(mrstate);
                    return 1;
                }
            }
        }

        // Recluster 2
        fprintf(stderr, "Recluster 2...\n");
        DiarizationResult r2 = streaming_recluster(mrstate);
        fprintf(stderr, "  Recluster 2: %zu segments\n", r2.segments.size());
        // Should have MORE or equal segments than r1 (more audio processed)
        for (const auto& seg : r2.segments) {
            if (seg.start < 0 || seg.duration <= 0 || seg.speaker.empty()) {
                fprintf(stderr, "FAIL: invalid segment after recluster 2\n");
                streaming_free(mrstate);
                return 1;
            }
        }

        // Finalize
        fprintf(stderr, "Finalizing...\n");
        DiarizationResult rfinal = streaming_finalize(mrstate);
        fprintf(stderr, "  Finalize: %zu segments\n", rfinal.segments.size());
        if (rfinal.segments.empty()) {
            fprintf(stderr, "FAIL: finalize returned no segments\n");
            streaming_free(mrstate);
            return 1;
        }
        for (const auto& seg : rfinal.segments) {
            if (seg.start < 0 || seg.duration <= 0 || seg.speaker.empty()) {
                fprintf(stderr, "FAIL: invalid segment in finalize\n");
                streaming_free(mrstate);
                return 1;
            }
        }

        streaming_free(mrstate);
        fprintf(stderr, "\nPASSED: multi-recluster test\n");
        return 0;
    }

    // ========================================================================
    // Standard streaming test modes
    // ========================================================================
    StreamingState* state = streaming_init(config);
    if (!state) {
        fprintf(stderr, "FAILED: streaming_init returned nullptr\n");
        return 1;
    }
    
    constexpr int STEP_SAMPLES = 16000;
    int chunks_pushed = 0;
    double total_push_time_ms = 0.0;
    
    std::vector<VADChunk> last_vad_chunks;
    
    for (int offset = 0; offset < num_samples; offset += STEP_SAMPLES) {
        int chunk_size = std::min(STEP_SAMPLES, num_samples - offset);
        
        auto t_start = std::chrono::high_resolution_clock::now();
        auto vad_chunks = streaming_push(state, audio_samples.data() + offset, chunk_size);
        auto t_end = std::chrono::high_resolution_clock::now();
        
        double push_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        total_push_time_ms += push_ms;
        chunks_pushed++;
        
        if (!vad_chunks.empty()) {
            last_vad_chunks = std::move(vad_chunks);
        }
        
        double current_time = static_cast<double>(chunks_pushed);
        bool is_recluster_point = (chunks_pushed % 60 == 0) && chunks_pushed > 0;
        
        if (json_stream) {
            if (is_recluster_point) {
                DiarizationResult recluster_result = streaming_recluster(state);
                print_json_recluster(chunks_pushed, current_time, recluster_result);
            } else {
                print_json_push(chunks_pushed, current_time, last_vad_chunks);
            }
        } else if (benchmark) {
            fprintf(stderr, "Push %d: %d samples, %.1f ms, %zu vad_chunks\n",
                    chunks_pushed, chunk_size, push_ms, last_vad_chunks.size());
        }
        
        if (provisional_only && chunks_pushed >= 30) {
            break;
        }
        
        if (recluster_test && chunks_pushed == 60) {
            fprintf(stderr, "Forcing recluster at 60s...\n");
            DiarizationResult recluster_result = streaming_recluster(state);
            fprintf(stderr, "Recluster returned %zu segments\n", recluster_result.segments.size());
        }
    }
    
    double avg_push_ms = total_push_time_ms / chunks_pushed;
    if (!json_stream) {
        fprintf(stderr, "\nPushed %d chunks, avg push time: %.1f ms\n", chunks_pushed, avg_push_ms);
        fprintf(stderr, "VAD chunks from last push: %zu\n", last_vad_chunks.size());
    }
    
    if (!push_only && !provisional_only) {
        if (!json_stream) {
            fprintf(stderr, "\nFinalizing...\n");
        }
        auto t_start = std::chrono::high_resolution_clock::now();
        DiarizationResult result = streaming_finalize(state);
        auto t_end = std::chrono::high_resolution_clock::now();
        
        if (json_stream) {
            print_json_finalize(result.segments);
        } else {
            double finalize_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            fprintf(stderr, "Finalize time: %.1f ms\n", finalize_ms);
            fprintf(stderr, "Final segments: %zu\n", result.segments.size());
        }
        
        // Write RTTM if output path specified
        if (!output_path.empty()) {
            // Extract URI from audio path
            std::string uri = audio_path;
            size_t last_slash = uri.find_last_of("/\\");
            if (last_slash != std::string::npos) {
                uri = uri.substr(last_slash + 1);
            }
            size_t last_dot = uri.find_last_of('.');
            if (last_dot != std::string::npos) {
                uri = uri.substr(0, last_dot);
            }
            
            // Convert to RTTMSegment format
            std::vector<diarization::RTTMSegment> rttm_segments;
            for (const auto& seg : result.segments) {
                rttm_segments.push_back({seg.start, seg.duration, seg.speaker});
            }
            
            diarization::write_rttm(rttm_segments, uri, output_path);
            fprintf(stderr, "Wrote RTTM to: %s\n", output_path.c_str());
        }
    }
    
    streaming_free(state);
    
    if (!json_stream) {
        if (benchmark) {
            fprintf(stderr, "\n=== Benchmark Summary ===\n");
            fprintf(stderr, "Average push latency: %.1f ms\n", avg_push_ms);
            fprintf(stderr, "Real-time factor: %.1fx\n", 1000.0 / avg_push_ms);
        }
        fprintf(stderr, "\nPASSED\n");
    }
    return 0;
}
