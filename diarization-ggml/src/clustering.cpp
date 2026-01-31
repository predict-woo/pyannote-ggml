#include "clustering.h"
#include "fastcluster/fastcluster.h"

#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <cstring>

namespace diarization {

// ============================================================================
// Cosine distance: 1 - (a·b) / (||a|| * ||b||)
// ============================================================================

double cosine_distance(const float* a, const float* b, int dim) {
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (int i = 0; i < dim; i++) {
        dot    += (double)a[i] * (double)b[i];
        norm_a += (double)a[i] * (double)a[i];
        norm_b += (double)b[i] * (double)b[i];
    }
    double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom < 1e-30) return 1.0;
    return 1.0 - dot / denom;
}

// ============================================================================
// Hungarian algorithm — adapted from scipy's rectangular_lsap.cpp
// (BSD-3-Clause, Copyright (c) 2001-2002 Enthought, Inc. 2003-2023 SciPy Developers)
// Based on: DF Crouse. "On implementing 2D rectangular assignment algorithms."
// IEEE Trans. Aerospace and Electronic Systems, 52(4):1679-1696, Aug 2016.
// ============================================================================

static intptr_t augmenting_path(
    intptr_t nc, double* cost,
    std::vector<double>& u, std::vector<double>& v,
    std::vector<intptr_t>& path, std::vector<intptr_t>& row4col,
    std::vector<double>& shortest_path_costs, intptr_t i,
    std::vector<bool>& SR, std::vector<bool>& SC,
    std::vector<intptr_t>& remaining, double* p_min_val)
{
    double min_val = 0;
    intptr_t num_remaining = nc;
    for (intptr_t it = 0; it < nc; it++) {
        remaining[it] = nc - it - 1;
    }

    std::fill(SR.begin(), SR.end(), false);
    std::fill(SC.begin(), SC.end(), false);
    std::fill(shortest_path_costs.begin(), shortest_path_costs.end(), INFINITY);

    intptr_t sink = -1;
    while (sink == -1) {
        intptr_t index = -1;
        double lowest = INFINITY;
        SR[i] = true;

        for (intptr_t it = 0; it < num_remaining; it++) {
            intptr_t j = remaining[it];
            double r = min_val + cost[i * nc + j] - u[i] - v[j];
            if (r < shortest_path_costs[j]) {
                path[j] = i;
                shortest_path_costs[j] = r;
            }
            if (shortest_path_costs[j] < lowest ||
                (shortest_path_costs[j] == lowest && row4col[j] == -1)) {
                lowest = shortest_path_costs[j];
                index = it;
            }
        }

        min_val = lowest;
        if (min_val == INFINITY) {
            return -1;
        }

        intptr_t j = remaining[index];
        if (row4col[j] == -1) {
            sink = j;
        } else {
            i = row4col[j];
        }
        SC[j] = true;
        remaining[index] = remaining[--num_remaining];
    }

    *p_min_val = min_val;
    return sink;
}

template <typename T>
static std::vector<intptr_t> argsort_iter(const std::vector<T>& v) {
    std::vector<intptr_t> index(v.size());
    std::iota(index.begin(), index.end(), 0);
    std::sort(index.begin(), index.end(),
              [&v](intptr_t i, intptr_t j) { return v[i] < v[j]; });
    return index;
}

// Rectangular linear sum assignment (Crouse/scipy algorithm).
// Returns 0 on success, -1 if infeasible, -2 if invalid entries.
static int solve_lsap(intptr_t nr, intptr_t nc, double* cost, bool maximize,
                       int64_t* a, int64_t* b)
{
    if (nr == 0 || nc == 0) return 0;

    bool transpose = nc < nr;
    std::vector<double> temp;

    if (transpose || maximize) {
        temp.resize(nr * nc);
        if (transpose) {
            for (intptr_t i = 0; i < nr; i++)
                for (intptr_t j = 0; j < nc; j++)
                    temp[j * nr + i] = cost[i * nc + j];
            std::swap(nr, nc);
        } else {
            std::copy(cost, cost + nr * nc, temp.begin());
        }
        if (maximize) {
            for (intptr_t i = 0; i < nr * nc; i++)
                temp[i] = -temp[i];
        }
        cost = temp.data();
    }

    for (intptr_t i = 0; i < nr * nc; i++) {
        if (cost[i] != cost[i] || cost[i] == -INFINITY) return -2;
    }

    std::vector<double> u(nr, 0), v(nc, 0);
    std::vector<double> shortest_path_costs(nc);
    std::vector<intptr_t> path(nc, -1);
    std::vector<intptr_t> col4row(nr, -1), row4col(nc, -1);
    std::vector<bool> SR(nr), SC(nc);
    std::vector<intptr_t> remaining(nc);

    for (intptr_t cur_row = 0; cur_row < nr; cur_row++) {
        double min_val;
        intptr_t sink = augmenting_path(nc, cost, u, v, path, row4col,
                                         shortest_path_costs, cur_row, SR, SC,
                                         remaining, &min_val);
        if (sink < 0) return -1;

        u[cur_row] += min_val;
        for (intptr_t i = 0; i < nr; i++) {
            if (SR[i] && i != cur_row)
                u[i] += min_val - shortest_path_costs[col4row[i]];
        }
        for (intptr_t j = 0; j < nc; j++) {
            if (SC[j])
                v[j] -= min_val - shortest_path_costs[j];
        }

        intptr_t j = sink;
        while (true) {
            intptr_t i = path[j];
            row4col[j] = i;
            std::swap(col4row[i], j);
            if (i == cur_row) break;
        }
    }

    if (transpose) {
        intptr_t idx = 0;
        for (auto vi : argsort_iter(col4row)) {
            a[idx] = col4row[vi];
            b[idx] = vi;
            idx++;
        }
    } else {
        for (intptr_t i = 0; i < nr; i++) {
            a[i] = i;
            b[i] = col4row[i];
        }
    }
    return 0;
}

// ============================================================================
// Public hungarian_assign wrapper
// ============================================================================

void hungarian_assign(const double* cost_matrix, int rows, int cols,
                      bool maximize, std::vector<int>& row_assign, std::vector<int>& col_assign) {
    int n = std::min(rows, cols);
    row_assign.resize(n);
    col_assign.resize(n);

    if (n == 0) return;

    std::vector<double> cost_copy(cost_matrix, cost_matrix + rows * cols);
    std::vector<int64_t> a(n), b(n);

    solve_lsap((intptr_t)rows, (intptr_t)cols, cost_copy.data(), maximize, a.data(), b.data());

    for (int i = 0; i < n; i++) {
        row_assign[i] = (int)a[i];
        col_assign[i] = (int)b[i];
    }
}

// ============================================================================
// Constrained argmax — Hungarian per chunk
// Ports clustering.py:127-140
// ============================================================================

void constrained_argmax(const float* soft_clusters, int num_chunks, int num_speakers, int num_clusters,
                        std::vector<int>& hard_clusters) {
    hard_clusters.assign(num_chunks * num_speakers, -2);

    // Find global minimum to replace NaN values
    float min_val = std::numeric_limits<float>::max();
    int total = num_chunks * num_speakers * num_clusters;
    for (int i = 0; i < total; i++) {
        if (!std::isnan(soft_clusters[i]) && soft_clusters[i] < min_val)
            min_val = soft_clusters[i];
    }
    if (min_val == std::numeric_limits<float>::max()) min_val = 0.0f;

    // Per-chunk Hungarian assignment
    std::vector<double> cost(num_speakers * num_clusters);
    std::vector<int> row_a, col_a;

    for (int c = 0; c < num_chunks; c++) {
        const float* chunk_sc = soft_clusters + c * num_speakers * num_clusters;

        // Build double cost matrix, replacing NaN with min_val
        for (int i = 0; i < num_speakers * num_clusters; i++) {
            cost[i] = std::isnan(chunk_sc[i]) ? (double)min_val : (double)chunk_sc[i];
        }

        hungarian_assign(cost.data(), num_speakers, num_clusters, true, row_a, col_a);

        int n_assigned = (int)row_a.size();
        for (int i = 0; i < n_assigned; i++) {
            int s = row_a[i];
            int k = col_a[i];
            hard_clusters[c * num_speakers + s] = k;
        }
    }
}

// ============================================================================
// Assign embeddings — ports clustering.py:142-212
// ============================================================================

void assign_embeddings(const float* embeddings, int num_chunks, int num_speakers, int dim,
                       const int* train_chunk_idx, const int* train_speaker_idx,
                       const int* train_clusters, int num_train, int num_clusters,
                       std::vector<int>& hard_clusters, std::vector<float>& soft_clusters_out,
                       std::vector<float>& centroids_out,
                       bool constrained) {
    // Compute centroids: mean of train embeddings per cluster
    centroids_out.assign(num_clusters * dim, 0.0f);
    std::vector<int> cluster_counts(num_clusters, 0);

    for (int t = 0; t < num_train; t++) {
        int ci = train_chunk_idx[t];
        int si = train_speaker_idx[t];
        int k  = train_clusters[t];
        const float* emb = embeddings + (ci * num_speakers + si) * dim;
        float* cent = centroids_out.data() + k * dim;
        for (int d = 0; d < dim; d++) {
            cent[d] += emb[d];
        }
        cluster_counts[k]++;
    }
    for (int k = 0; k < num_clusters; k++) {
        if (cluster_counts[k] > 0) {
            float* cent = centroids_out.data() + k * dim;
            float inv = 1.0f / (float)cluster_counts[k];
            for (int d = 0; d < dim; d++) {
                cent[d] *= inv;
            }
        }
    }

    // Compute cosine distance: all embeddings vs centroids
    // e2k_distance shape: (num_chunks, num_speakers, num_clusters)
    int total_emb = num_chunks * num_speakers;
    soft_clusters_out.resize(total_emb * num_clusters);

    for (int e = 0; e < total_emb; e++) {
        const float* emb = embeddings + e * dim;
        for (int k = 0; k < num_clusters; k++) {
            const float* cent = centroids_out.data() + k * dim;
            double dist = cosine_distance(emb, cent, dim);
            // soft_clusters = 2 - cosine_distance (similarity measure)
            soft_clusters_out[e * num_clusters + k] = (float)(2.0 - dist);
        }
    }

    // Assign: constrained (Hungarian per chunk) or unconstrained (argmax)
    if (constrained) {
        constrained_argmax(soft_clusters_out.data(), num_chunks, num_speakers, num_clusters,
                           hard_clusters);
    } else {
        hard_clusters.resize(total_emb);
        for (int e = 0; e < total_emb; e++) {
            int best_k = 0;
            float best_val = soft_clusters_out[e * num_clusters];
            for (int k = 1; k < num_clusters; k++) {
                float val = soft_clusters_out[e * num_clusters + k];
                if (val > best_val) {
                    best_val = val;
                    best_k = k;
                }
            }
            hard_clusters[e] = best_k;
        }
    }
}

// ============================================================================
// Filter embeddings — ports clustering.py:77-125
// Keeps embeddings with no NaN and sufficient single-speaker activity.
// segmentations: (num_chunks, num_frames, num_speakers) row-major
// embeddings:    (num_chunks, num_speakers, embed_dim)  row-major
// ============================================================================

void filter_embeddings(const float* embeddings, int num_chunks, int num_speakers, int embed_dim,
                       const float* segmentations, int num_frames,
                       std::vector<float>& filtered, std::vector<int>& chunk_idx,
                       std::vector<int>& speaker_idx, float min_active_ratio) {
    filtered.clear();
    chunk_idx.clear();
    speaker_idx.clear();

    const float active_threshold = min_active_ratio * (float)num_frames;

    for (int c = 0; c < num_chunks; c++) {
        const float* seg_chunk = segmentations + c * num_frames * num_speakers;

        // num_clean[s] = frames where speaker s is active AND exactly 1 speaker is active
        float num_clean[16] = {};
        for (int f = 0; f < num_frames; f++) {
            const float* frame = seg_chunk + f * num_speakers;
            float frame_sum = 0.0f;
            for (int s = 0; s < num_speakers; s++) {
                frame_sum += frame[s];
            }
            if (frame_sum == 1.0f) {
                for (int s = 0; s < num_speakers; s++) {
                    num_clean[s] += frame[s];
                }
            }
        }

        for (int s = 0; s < num_speakers; s++) {
            bool active = (num_clean[s] >= active_threshold);

            const float* emb = embeddings + (c * num_speakers + s) * embed_dim;
            bool valid = true;
            for (int d = 0; d < embed_dim; d++) {
                if (std::isnan(emb[d])) {
                    valid = false;
                    break;
                }
            }

            if (active && valid) {
                chunk_idx.push_back(c);
                speaker_idx.push_back(s);
                filtered.insert(filtered.end(), emb, emb + embed_dim);
            }
        }
    }
}

// ============================================================================
// AHC clustering — centroid linkage via fastcluster (Müllner, 2011)
// O(n² log n) via binary min-heap nearest-neighbor tracking.
// ============================================================================

void ahc_cluster(const double* embeddings, int n, int dim,
                 double threshold, std::vector<int>& clusters) {
    if (n <= 0) {
        clusters.clear();
        return;
    }
    if (n == 1) {
        clusters.assign(1, 0);
        return;
    }

    size_t condensed_size = (size_t)n * (n - 1) / 2;
    std::vector<double> distmat(condensed_size);

    size_t idx = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double dist_sq = 0.0;
            for (int d = 0; d < dim; d++) {
                double diff = embeddings[i * dim + d] - embeddings[j * dim + d];
                dist_sq += diff * diff;
            }
            distmat[idx++] = std::sqrt(dist_sq);
        }
    }

    std::vector<int> merge(2 * (n - 1));
    std::vector<double> height(n - 1);

    int rc = hclust_fast(n, distmat.data(), HCLUST_METHOD_CENTROID,
                         merge.data(), height.data());
    if (rc != 0) {
        clusters.assign(n, 0);
        return;
    }

    clusters.resize(n);
    cutree_cdist(n, merge.data(), height.data(), threshold, clusters.data());
}

}  // namespace diarization
