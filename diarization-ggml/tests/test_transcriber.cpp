#include "transcriber.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool load_wav_pcm16(const char* path, std::vector<float>& out, int expected_rate) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open WAV file: %s\n", path);
        return false;
    }

    char header[44];
    if (fread(header, 1, 44, f) != 44) {
        fprintf(stderr, "WAV header too short: %s\n", path);
        fclose(f);
        return false;
    }

    if (memcmp(header, "RIFF", 4) != 0 || memcmp(header + 8, "WAVE", 4) != 0) {
        fprintf(stderr, "Not a WAV file: %s\n", path);
        fclose(f);
        return false;
    }

    int sample_rate = *(int*)(header + 24);
    short bits_per_sample = *(short*)(header + 34);

    if (sample_rate != expected_rate) {
        fprintf(stderr, "Expected %d Hz, got %d Hz\n", expected_rate, sample_rate);
        fclose(f);
        return false;
    }
    if (bits_per_sample != 16) {
        fprintf(stderr, "Expected 16-bit PCM, got %d-bit\n", bits_per_sample);
        fclose(f);
        return false;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 44, SEEK_SET);

    long data_bytes = file_size - 44;
    int n_samples = (int)(data_bytes / 2);

    std::vector<short> pcm16(n_samples);
    size_t read_count = fread(pcm16.data(), 2, n_samples, f);
    fclose(f);

    out.resize(read_count);
    for (size_t i = 0; i < read_count; i++) {
        out[i] = pcm16[i] / 32768.0f;
    }

    return true;
}

int main(int argc, char* argv[]) {
    const char* model_path = nullptr;
    const char* audio_path = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else {
            audio_path = argv[i];
        }
    }

    if (!model_path || !audio_path) {
        printf("Usage: test_transcriber --model <whisper-model.bin> <audio.wav>\n");
        printf("SKIP: No model provided\n");
        return 0;
    }

    std::vector<float> audio;
    if (!load_wav_pcm16(audio_path, audio, 16000)) {
        printf("FAIL: Could not load audio file\n");
        return 1;
    }
    printf("Loaded %zu samples (%.1fs)\n", audio.size(), audio.size() / 16000.0);

    TranscriberConfig config;
    config.whisper_model_path = model_path;
    config.n_threads = 4;
    config.language = "en";

    printf("Initializing transcriber...\n");
    Transcriber* t = transcriber_init(config);
    if (!t) {
        printf("FAIL: transcriber_init returned null\n");
        return 1;
    }
    printf("Transcriber initialized\n");

    printf("Submitting audio...\n");
    transcriber_submit(t, audio.data(), (int)audio.size(), 0.0);

    printf("Waiting for result...\n");
    TranscribeResult result = transcriber_wait_result(t);

    if (!result.valid) {
        printf("FAIL: result not valid\n");
        transcriber_free(t);
        return 1;
    }

    size_t total_words = 0;
    for (const auto& seg : result.segments) {
        total_words += seg.words.size();
    }
    printf("Got %zu segments (%zu total words)\n", result.segments.size(), total_words);

    int failures = 0;

    if (total_words < 10) {
        printf("FAIL: expected >=10 words, got %zu\n", total_words);
        failures++;
    }

    for (size_t s = 0; s < result.segments.size(); s++) {
        const auto& seg = result.segments[s];
        if (seg.words.empty()) {
            printf("FAIL: segment %zu has no words\n", s);
            failures++;
        }
        for (size_t w = 0; w < seg.words.size(); w++) {
            const auto& word = seg.words[w];
            if (word.text.empty()) {
                printf("FAIL: segment %zu word %zu has empty text\n", s, w);
                failures++;
            }
            if (word.start > word.end) {
                printf("FAIL: segment %zu word %zu has start (%.3f) > end (%.3f)\n", s, w, word.start, word.end);
                failures++;
            }
        }
    }

    printf("\nFirst 3 segments:\n");
    for (size_t s = 0; s < std::min(result.segments.size(), (size_t)3); s++) {
        const auto& seg = result.segments[s];
        printf("  Segment %zu [%.3f - %.3f]: '%s'\n", s, seg.start, seg.end, seg.text.c_str());
        for (size_t w = 0; w < std::min(seg.words.size(), (size_t)5); w++) {
            printf("    [%.3f - %.3f] '%s'\n", seg.words[w].start, seg.words[w].end, seg.words[w].text.c_str());
        }
    }

    transcriber_free(t);
    printf("Transcriber freed cleanly\n");

    if (failures > 0) {
        printf("\nFAIL: %d failures\n", failures);
        return 1;
    }

    printf("\nPASS\n");
    return 0;
}
