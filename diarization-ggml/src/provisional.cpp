#include "provisional.h"

namespace diarization {

void provisional_init(ProvisionalState& state, float threshold) {
    state.num_speakers = 0;
    state.threshold = threshold;
    state.centroids.clear();
    state.counts.clear();
}

int provisional_assign(ProvisionalState& state, const float* embedding) {
    return 0;
}

std::vector<int> provisional_assign_batch(ProvisionalState& state, 
                                          const float* embeddings, 
                                          int num_embeddings) {
    return std::vector<int>();
}

const float* provisional_get_centroids(const ProvisionalState& state) {
    return state.centroids.empty() ? nullptr : state.centroids.data();
}

std::vector<int> provisional_remap_labels(const ProvisionalState& state,
                                          const float* vbx_centroids,
                                          int num_vbx_clusters) {
    return std::vector<int>();
}

void provisional_update_from_vbx(ProvisionalState& state,
                                 const float* vbx_centroids,
                                 int num_vbx_clusters) {
}

void provisional_clear(ProvisionalState& state) {
    state.num_speakers = 0;
    state.centroids.clear();
    state.counts.clear();
}

}
