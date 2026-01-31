#include "diarization.h"
#include <cstdio>
#include <cstring>
#include <string>

static void print_usage(const char* program) {
    fprintf(stderr, "Usage: %s <seg.gguf> <emb.gguf> <audio.wav> [options]\n", program);
    fprintf(stderr, "\n");
    fprintf(stderr, "Positional arguments:\n");
    fprintf(stderr, "  seg.gguf              Path to segmentation model (GGUF)\n");
    fprintf(stderr, "  emb.gguf              Path to embedding model (GGUF)\n");
    fprintf(stderr, "  audio.wav             Path to audio file (16kHz mono PCM WAV)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --plda <path>         Path to PLDA binary file\n");
    fprintf(stderr, "  --coreml <path>       Path to CoreML embedding model (.mlpackage)\n");
    fprintf(stderr, "  --seg-coreml <path>   Path to CoreML segmentation model (.mlpackage)\n");
    fprintf(stderr, "  -o, --output <path>   Output RTTM file (default: stdout)\n");
    fprintf(stderr, "  --dump-stage <name>   Dump intermediate stage to binary file\n");
    fprintf(stderr, "  --help                Print this help message\n");
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

    if (argc < 4) {
        fprintf(stderr, "Error: expected 3 positional arguments (seg.gguf, emb.gguf, audio.wav)\n\n");
        print_usage(argv[0]);
        return 1;
    }

    DiarizationConfig config;
    config.seg_model_path = argv[1];
    config.emb_model_path = argv[2];
    config.audio_path     = argv[3];

    for (int i = 4; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--plda" && i + 1 < argc) {
            config.plda_path = argv[++i];
        } else if (arg == "--coreml" && i + 1 < argc) {
            config.coreml_path = argv[++i];
        } else if (arg == "--seg-coreml" && i + 1 < argc) {
            config.seg_coreml_path = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            config.output_path = argv[++i];
        } else if (arg == "--dump-stage" && i + 1 < argc) {
            config.dump_stage = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Error: unknown option '%s'\n\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    DiarizationResult result;
    if (!diarize(config, result)) {
        fprintf(stderr, "Error: diarization failed\n");
        return 1;
    }

    return 0;
}
