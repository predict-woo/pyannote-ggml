#include "vbx.h"
#include <Accelerate/Accelerate.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

namespace diarization {

static double logsumexp(const double* x, int n) {
    double max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += std::exp(x[i] - max_val);
    }
    return max_val + std::log(sum);
}

static void softmax_rows(double* mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        double* row = mat + i * cols;
        double max_val = row[0];
        for (int j = 1; j < cols; j++) {
            if (row[j] > max_val) max_val = row[j];
        }
        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            row[j] = std::exp(row[j] - max_val);
            sum += row[j];
        }
        for (int j = 0; j < cols; j++) {
            row[j] /= sum;
        }
    }
}

bool vbx_cluster(const int* ahc_clusters, int num_embeddings, int num_ahc_clusters,
                 const double* plda_features, int feature_dim,
                 const double* phi, double Fa, double Fb, int max_iters,
                 VBxResult& result) {

    const int T = num_embeddings;
    const int S = num_ahc_clusters;
    const int D = feature_dim;
    const double epsilon = 1e-4;
    const double init_smoothing = 7.0;

    if (T <= 0 || S <= 0 || D <= 0) return false;

    // gamma (T, S): one-hot from AHC clusters → softmax(qinit * 7.0)
    std::vector<double> gamma(T * S, 0.0);
    for (int i = 0; i < T; i++) {
        int c = ahc_clusters[i];
        if (c < 0 || c >= S) return false;
        gamma[i * S + c] = 1.0;
    }
    for (int i = 0; i < T * S; i++) {
        gamma[i] *= init_smoothing;
    }
    softmax_rows(gamma.data(), T, S);

    std::vector<double> pi(S, 1.0 / S);

    // G (T,): eq. constant = -0.5 * (||x_t||^2 + D*log(2π))
    std::vector<double> G(T);
    for (int i = 0; i < T; i++) {
        double sum_sq = 0.0;
        const double* xi = plda_features + i * D;
        for (int d = 0; d < D; d++) {
            sum_sq += xi[d] * xi[d];
        }
        G[i] = -0.5 * (sum_sq + D * std::log(2.0 * M_PI));
    }

    // V (D,) = sqrt(Phi); rho (T, D) = X ⊙ V — eq. (18)
    std::vector<double> V(D);
    for (int d = 0; d < D; d++) {
        V[d] = std::sqrt(phi[d]);
    }
    std::vector<double> rho(T * D);
    for (int i = 0; i < T; i++) {
        for (int d = 0; d < D; d++) {
            rho[i * D + d] = plda_features[i * D + d] * V[d];
        }
    }

    std::vector<double> invL(S * D);
    std::vector<double> alpha(S * D);
    std::vector<double> log_p(T * S);
    std::vector<double> gamma_col_sum(S);
    std::vector<double> gamma_T_rho(S * D);
    std::vector<double> rho_alphaT(T * S);
    std::vector<double> invL_alpha2(S * D);
    std::vector<double> invL_alpha2_phi(S);
    std::vector<double> log_p_x(T);
    std::vector<double> lpi(S);
    std::vector<double> row_buf(S);

    double prev_ELBO = -1e30;

    for (int ii = 0; ii < max_iters; ii++) {

        // Eq. (17): invL (S, D) = 1 / (1 + Fa/Fb * Σ_t γ[t,s] * Φ[d])
        std::fill(gamma_col_sum.begin(), gamma_col_sum.end(), 0.0);
        for (int t = 0; t < T; t++) {
            for (int s = 0; s < S; s++) {
                gamma_col_sum[s] += gamma[t * S + s];
            }
        }
        const double FaFb = Fa / Fb;
        for (int s = 0; s < S; s++) {
            for (int d = 0; d < D; d++) {
                invL[s * D + d] = 1.0 / (1.0 + FaFb * gamma_col_sum[s] * phi[d]);
            }
        }

        // Eq. (16): α (S, D) = Fa/Fb * invL ⊙ (γᵀ @ ρ)
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    S, D, T,
                    1.0, gamma.data(), S, rho.data(), D,
                    0.0, gamma_T_rho.data(), D);
        for (int s = 0; s < S; s++) {
            for (int d = 0; d < D; d++) {
                alpha[s * D + d] = FaFb * invL[s * D + d] * gamma_T_rho[s * D + d];
            }
        }

        // Eq. (23): log_p (T, S) = Fa * (ρ @ αᵀ - 0.5*(invL + α²)·Φ + G)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    T, S, D,
                    1.0, rho.data(), D, alpha.data(), D,
                    0.0, rho_alphaT.data(), S);

        for (int s = 0; s < S; s++) {
            for (int d = 0; d < D; d++) {
                invL_alpha2[s * D + d] = invL[s * D + d] + alpha[s * D + d] * alpha[s * D + d];
            }
        }
        // (invL + α²) @ Φ: (S, D) @ (D,) → (S,)
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    S, D,
                    1.0, invL_alpha2.data(), D, phi, 1,
                    0.0, invL_alpha2_phi.data(), 1);

        for (int t = 0; t < T; t++) {
            for (int s = 0; s < S; s++) {
                log_p[t * S + s] = Fa * (rho_alphaT[t * S + s]
                                         - 0.5 * invL_alpha2_phi[s]
                                         + G[t]);
            }
        }

        // GMM update: responsibilities via logsumexp
        const double eps = 1e-8;
        for (int s = 0; s < S; s++) {
            lpi[s] = std::log(pi[s] + eps);
        }

        double log_pX = 0.0;
        for (int t = 0; t < T; t++) {
            for (int s = 0; s < S; s++) {
                row_buf[s] = log_p[t * S + s] + lpi[s];
            }
            log_p_x[t] = logsumexp(row_buf.data(), S);
            log_pX += log_p_x[t];
        }

        for (int t = 0; t < T; t++) {
            for (int s = 0; s < S; s++) {
                gamma[t * S + s] = std::exp(log_p[t * S + s] + lpi[s] - log_p_x[t]);
            }
        }

        double pi_sum = 0.0;
        for (int s = 0; s < S; s++) {
            pi[s] = 0.0;
            for (int t = 0; t < T; t++) {
                pi[s] += gamma[t * S + s];
            }
            pi_sum += pi[s];
        }
        for (int s = 0; s < S; s++) {
            pi[s] /= pi_sum;
        }

        // Eq. (25): ELBO = Σ log p(x_t) + Fb/2 * Σ(log(invL) - invL - α² + 1)
        double reg_sum = 0.0;
        for (int s = 0; s < S; s++) {
            for (int d = 0; d < D; d++) {
                double il = invL[s * D + d];
                double a2 = alpha[s * D + d] * alpha[s * D + d];
                reg_sum += std::log(il) - il - a2 + 1.0;
            }
        }
        double ELBO = log_pX + Fb * 0.5 * reg_sum;

        if (ii > 0 && (ELBO - prev_ELBO) < epsilon) {
            break;
        }
        prev_ELBO = ELBO;
    }

    result.gamma = std::move(gamma);
    result.pi = std::move(pi);
    result.num_frames = T;
    result.num_speakers = S;

    return true;
}

}  // namespace diarization
