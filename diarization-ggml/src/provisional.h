#pragma once
#include <vector>

// Provisional clustering for streaming diarization
// Uses incremental cosine-based speaker assignment before full VBx recluster

namespace diarization {

// Provisional clustering state
struct ProvisionalState {
    std::vector<float> centroids;      // [K x 256] speaker centroids
    std::vector<int> counts;           // How many embeddings per centroid
    int num_speakers = 0;
    float threshold = 0.6f;            // Cosine distance threshold for new speaker
};

// Initialize provisional clustering state
void provisional_init(ProvisionalState& state, float threshold = 0.6f);

// Assign a single embedding to a speaker (creates new speaker if needed)
// Returns the assigned speaker index (0-based)
// embedding: 256-dim speaker embedding
int provisional_assign(ProvisionalState& state, const float* embedding);

// Assign multiple embeddings at once
// embeddings: [N x 256] speaker embeddings
// Returns vector of assigned speaker indices
std::vector<int> provisional_assign_batch(ProvisionalState& state, 
                                          const float* embeddings, 
                                          int num_embeddings);

// Get current centroids for label remapping after VBx recluster
// Returns pointer to centroid data [K x 256]
const float* provisional_get_centroids(const ProvisionalState& state);

// Remap VBx cluster labels to minimize churn with provisional labels
// Uses Hungarian assignment to find optimal mapping
// vbx_centroids: [M x 256] centroids from VBx clustering
// num_vbx_clusters: number of VBx clusters
// Returns mapping: vbx_label -> provisional_label (or new label if no match)
std::vector<int> provisional_remap_labels(const ProvisionalState& state,
                                          const float* vbx_centroids,
                                          int num_vbx_clusters);

// Update provisional centroids from VBx results (after recluster)
void provisional_update_from_vbx(ProvisionalState& state,
                                 const float* vbx_centroids,
                                 int num_vbx_clusters);

// Clear provisional state
void provisional_clear(ProvisionalState& state);

}  // namespace diarization
