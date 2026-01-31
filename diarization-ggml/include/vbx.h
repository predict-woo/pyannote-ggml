#pragma once
#include <vector>

namespace diarization {

struct VBxResult {
    std::vector<double> gamma;  // (T, S) responsibilities
    std::vector<double> pi;     // (S,) speaker priors
    int num_frames;
    int num_speakers;
};

bool vbx_cluster(const int* ahc_clusters, int num_embeddings, int num_ahc_clusters,
                 const double* plda_features, int feature_dim,
                 const double* phi, double Fa, double Fb, int max_iters,
                 VBxResult& result);

}  // namespace diarization
