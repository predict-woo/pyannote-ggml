#include "provisional.h"
#include "../include/clustering.h"
#include <cstring>
#include <algorithm>
#include <limits>

namespace diarization {

constexpr int EMBEDDING_DIM = 256;

void provisional_init(ProvisionalState& state, float threshold) {
    state.num_speakers = 0;
    state.threshold = threshold;
    state.centroids.clear();
    state.counts.clear();
}

int provisional_assign(ProvisionalState& state, const float* embedding) {
    // If no centroids exist, create first speaker
    if (state.num_speakers == 0) {
        state.centroids.resize(EMBEDDING_DIM);
        std::memcpy(state.centroids.data(), embedding, EMBEDDING_DIM * sizeof(float));
        state.counts.push_back(1);
        state.num_speakers = 1;
        return 0;
    }
    
    // Compute cosine distance to all existing centroids
    double min_distance = std::numeric_limits<double>::max();
    int best_speaker = -1;
    
    for (int i = 0; i < state.num_speakers; i++) {
        const float* centroid = state.centroids.data() + i * EMBEDDING_DIM;
        double dist = cosine_distance(embedding, centroid, EMBEDDING_DIM);
        if (dist < min_distance) {
            min_distance = dist;
            best_speaker = i;
        }
    }
    
    // If min_distance < threshold, assign to that speaker and update centroid
    if (min_distance < state.threshold) {
        float* centroid = state.centroids.data() + best_speaker * EMBEDDING_DIM;
        int count = state.counts[best_speaker];
        
        // Update centroid using running mean: centroid = (centroid * count + embedding) / (count + 1)
        for (int i = 0; i < EMBEDDING_DIM; i++) {
            centroid[i] = (centroid[i] * count + embedding[i]) / (count + 1);
        }
        state.counts[best_speaker]++;
        return best_speaker;
    }
    
    // Create new speaker
    int new_speaker_idx = state.num_speakers;
    state.centroids.resize((state.num_speakers + 1) * EMBEDDING_DIM);
    std::memcpy(state.centroids.data() + new_speaker_idx * EMBEDDING_DIM, 
                embedding, EMBEDDING_DIM * sizeof(float));
    state.counts.push_back(1);
    state.num_speakers++;
    return new_speaker_idx;
}

std::vector<int> provisional_assign_batch(ProvisionalState& state, 
                                          const float* embeddings, 
                                          int num_embeddings) {
    std::vector<int> assignments;
    assignments.reserve(num_embeddings);
    
    for (int i = 0; i < num_embeddings; i++) {
        const float* embedding = embeddings + i * EMBEDDING_DIM;
        int speaker_idx = provisional_assign(state, embedding);
        assignments.push_back(speaker_idx);
    }
    
    return assignments;
}

const float* provisional_get_centroids(const ProvisionalState& state) {
    return state.centroids.empty() ? nullptr : state.centroids.data();
}

std::vector<int> provisional_remap_labels(const ProvisionalState& state,
                                          const float* vbx_centroids,
                                          int num_vbx_clusters) {
    std::vector<int> mapping(num_vbx_clusters);
    
    if (state.num_speakers == 0) {
        // No provisional speakers, assign sequential labels
        for (int i = 0; i < num_vbx_clusters; i++) {
            mapping[i] = i;
        }
        return mapping;
    }
    
    // Compute cost matrix: cosine distance between VBx centroids and provisional centroids
    int rows = num_vbx_clusters;
    int cols = state.num_speakers;
    std::vector<double> cost_matrix(rows * cols);
    
    for (int i = 0; i < rows; i++) {
        const float* vbx_centroid = vbx_centroids + i * EMBEDDING_DIM;
        for (int j = 0; j < cols; j++) {
            const float* prov_centroid = state.centroids.data() + j * EMBEDDING_DIM;
            cost_matrix[i * cols + j] = cosine_distance(vbx_centroid, prov_centroid, EMBEDDING_DIM);
        }
    }
    
    // Use Hungarian assignment to find optimal mapping
    std::vector<int> row_assign, col_assign;
    hungarian_assign(cost_matrix.data(), rows, cols, false, row_assign, col_assign);
    
    // Build mapping: vbx_label -> provisional_label
    int num_matched = std::min(rows, cols);
    std::vector<bool> used_prov_labels(state.num_speakers, false);
    
    for (int i = 0; i < num_matched; i++) {
        int vbx_idx = row_assign[i];
        int prov_idx = col_assign[i];
        mapping[vbx_idx] = prov_idx;
        used_prov_labels[prov_idx] = true;
    }
    
    // For unmatched VBx clusters, assign new labels
    if (num_vbx_clusters > state.num_speakers) {
        int next_label = state.num_speakers;
        for (int i = 0; i < num_vbx_clusters; i++) {
            bool is_matched = false;
            for (int j = 0; j < num_matched; j++) {
                if (row_assign[j] == i) {
                    is_matched = true;
                    break;
                }
            }
            if (!is_matched) {
                mapping[i] = next_label++;
            }
        }
    }
    
    return mapping;
}

void provisional_update_from_vbx(ProvisionalState& state,
                                 const float* vbx_centroids,
                                 int num_vbx_clusters) {
    state.num_speakers = num_vbx_clusters;
    state.centroids.resize(num_vbx_clusters * EMBEDDING_DIM);
    std::memcpy(state.centroids.data(), vbx_centroids, 
                num_vbx_clusters * EMBEDDING_DIM * sizeof(float));
    
    state.counts.assign(num_vbx_clusters, 1);
}

void provisional_clear(ProvisionalState& state) {
    state.num_speakers = 0;
    state.centroids.clear();
    state.counts.clear();
}

}
