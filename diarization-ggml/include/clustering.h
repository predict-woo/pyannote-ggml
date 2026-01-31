#pragma once
#include <vector>

namespace diarization {

// --- Cosine distance ---
// Returns 1 - cosine_similarity(a, b). Range [0, 2].
double cosine_distance(const float* a, const float* b, int dim);

// --- Hungarian algorithm (linear_sum_assignment) ---
// Solves the rectangular linear assignment problem.
// Equivalent to scipy.optimize.linear_sum_assignment.
// cost_matrix: row-major, rows × cols.
// maximize: if true, find maximum-weight matching.
// row_assign: output, size = min(rows, cols). row_assign[i] = row index.
// col_assign: output, size = min(rows, cols). col_assign[i] = col index.
void hungarian_assign(const double* cost_matrix, int rows, int cols,
                      bool maximize, std::vector<int>& row_assign, std::vector<int>& col_assign);

// --- Constrained argmax (Hungarian per chunk) ---
// soft_clusters: (num_chunks, num_speakers, num_clusters) row-major float.
// hard_clusters: output (num_chunks * num_speakers), initialized to -2.
// Each chunk uses Hungarian to bijectively assign speakers to clusters.
void constrained_argmax(const float* soft_clusters, int num_chunks, int num_speakers, int num_clusters,
                        std::vector<int>& hard_clusters);

// --- Filter embeddings ---
void filter_embeddings(const float* embeddings, int num_chunks, int num_speakers, int embed_dim,
                       const float* segmentations, int num_frames,
                       std::vector<float>& filtered, std::vector<int>& chunk_idx,
                       std::vector<int>& speaker_idx, float min_active_ratio = 0.2f);

// --- AHC clustering ---
void ahc_cluster(const double* embeddings, int n, int dim,
                 double threshold, std::vector<int>& clusters);

// --- Assign all embeddings to clusters ---
// Ports BaseClustering.assign_embeddings() from clustering.py.
// embeddings: (num_chunks, num_speakers, dim) float32 row-major.
// train_chunk_idx, train_speaker_idx, train_clusters: arrays of size num_train.
// num_clusters: max(train_clusters) + 1.
// hard_clusters: output (num_chunks * num_speakers) int, -2 = unassigned.
// soft_clusters: output (num_chunks * num_speakers * num_clusters) float.
// centroids: output (num_clusters * dim) float.
void assign_embeddings(const float* embeddings, int num_chunks, int num_speakers, int dim,
                       const int* train_chunk_idx, const int* train_speaker_idx,
                       const int* train_clusters, int num_train, int num_clusters,
                       std::vector<int>& hard_clusters, std::vector<float>& soft_clusters,
                       std::vector<float>& centroids,
                       bool constrained = true);

}  // namespace diarization
