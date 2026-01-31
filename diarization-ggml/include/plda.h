#pragma once
#include <string>
#include <vector>

namespace diarization {

struct PLDAModel {
    std::vector<double> mean1;      // [256]
    std::vector<double> mean2;      // [128]
    std::vector<double> lda;        // [256*128] row-major
    std::vector<double> plda_mu;    // [128]
    std::vector<double> plda_tr;    // [128*128] row-major
    std::vector<double> plda_psi;   // [128]
    bool loaded = false;
};

bool plda_load(const std::string& path, PLDAModel& model);

// Full transform: embeddings (N, 256) -> features (N, 128)
void plda_transform(const PLDAModel& model, const double* embeddings, int n,
                    double* output);

}  // namespace diarization
