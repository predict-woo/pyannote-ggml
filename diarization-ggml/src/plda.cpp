#include "plda.h"

#include <Accelerate/Accelerate.h>
#include <ggml.h>
#include <gguf.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace diarization {

static constexpr int XVEC_DIM = 256;
static constexpr int LDA_DIM  = 128;

static bool load_f64_tensor(struct ggml_context* ctx, struct gguf_context* gguf_ctx,
                            FILE* f, const char* name, std::vector<double>& out, size_t expected) {
    struct ggml_tensor* tensor = ggml_get_tensor(ctx, name);
    if (!tensor) {
        fprintf(stderr, "plda_load: tensor '%s' not found\n", name);
        return false;
    }
    if (ggml_nelements(tensor) != static_cast<int64_t>(expected)) {
        fprintf(stderr, "plda_load: tensor '%s' has %lld elements, expected %zu\n",
                name, (long long)ggml_nelements(tensor), expected);
        return false;
    }

    int idx = gguf_find_tensor(gguf_ctx, name);
    size_t offset = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, idx);

    out.resize(expected);
    fseek(f, (long)offset, SEEK_SET);
    return fread(out.data(), sizeof(double), expected, f) == expected;
}

bool plda_load(const std::string& path, PLDAModel& model) {
    struct ggml_context* ctx = nullptr;
    struct gguf_init_params params = { .no_alloc = true, .ctx = &ctx };

    struct gguf_context* gguf_ctx = gguf_init_from_file(path.c_str(), params);
    if (!gguf_ctx) {
        fprintf(stderr, "plda_load: cannot open GGUF file '%s'\n", path.c_str());
        return false;
    }

    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        gguf_free(gguf_ctx);
        ggml_free(ctx);
        return false;
    }

    bool ok = true;
    ok = ok && load_f64_tensor(ctx, gguf_ctx, f, "plda.mean1", model.mean1,    XVEC_DIM);
    ok = ok && load_f64_tensor(ctx, gguf_ctx, f, "plda.mean2", model.mean2,    LDA_DIM);
    ok = ok && load_f64_tensor(ctx, gguf_ctx, f, "plda.lda",   model.lda,      XVEC_DIM * LDA_DIM);
    ok = ok && load_f64_tensor(ctx, gguf_ctx, f, "plda.mu",    model.plda_mu,  LDA_DIM);
    ok = ok && load_f64_tensor(ctx, gguf_ctx, f, "plda.tr",    model.plda_tr,  LDA_DIM * LDA_DIM);
    ok = ok && load_f64_tensor(ctx, gguf_ctx, f, "plda.psi",   model.plda_psi, LDA_DIM);

    fclose(f);
    gguf_free(gguf_ctx);
    ggml_free(ctx);

    model.loaded = ok;
    if (!ok) {
        fprintf(stderr, "plda_load: failed to load tensors from '%s'\n", path.c_str());
    }
    return ok;
}

static void l2_normalize_rows(double* data, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        double* row = data + i * cols;
        double norm = cblas_dnrm2(cols, row, 1);
        if (norm > 0.0) {
            cblas_dscal(cols, 1.0 / norm, row, 1);
        }
    }
}

// Full xvec + PLDA transform chain (vbx.py:211-217)
//
// xvec_transform:  sqrt(128) * l2_norm(lda.T @ (sqrt(256) * l2_norm(x - mean1)).T).T - mean2)
// plda_transform:  (x - plda_mu) @ plda_tr.T
//
// Steps:
//   1. x -= mean1              (N,256)          center
//   2. l2_norm(x)              (N,256)          per-row normalize
//   3. x *= sqrt(256)          (N,256)          scale
//   4. x = x @ lda             (N,256)@(256,128) → (N,128)  LDA projection
//   5. x -= mean2              (N,128)          center
//   6. l2_norm(x)              (N,128)          per-row normalize
//   7. x *= sqrt(128)          (N,128)          scale
//   8. x -= plda_mu            (N,128)          center
//   9. x = x @ plda_tr.T       (N,128)@(128,128)^T → (N,128)
void plda_transform(const PLDAModel& model, const double* embeddings, int n,
                    double* output) {
    const double sqrt_xvec = std::sqrt(static_cast<double>(XVEC_DIM));
    const double sqrt_lda  = std::sqrt(static_cast<double>(LDA_DIM));

    // Step 1: center with mean1
    std::vector<double> tmp(static_cast<size_t>(n) * XVEC_DIM);
    for (int i = 0; i < n; i++) {
        const double* src = embeddings + i * XVEC_DIM;
        double* dst = tmp.data() + i * XVEC_DIM;
        for (int j = 0; j < XVEC_DIM; j++) {
            dst[j] = src[j] - model.mean1[j];
        }
    }

    // Step 2+3: L2 normalize, scale by sqrt(256)
    l2_normalize_rows(tmp.data(), n, XVEC_DIM);
    cblas_dscal(n * XVEC_DIM, sqrt_xvec, tmp.data(), 1);

    // Step 4: LDA projection — (N,256) @ (256,128) → (N,128)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, LDA_DIM, XVEC_DIM,
                1.0, tmp.data(), XVEC_DIM,
                model.lda.data(), LDA_DIM,
                0.0, output, LDA_DIM);

    // Step 5: center with mean2
    for (int i = 0; i < n; i++) {
        double* row = output + i * LDA_DIM;
        for (int j = 0; j < LDA_DIM; j++) {
            row[j] -= model.mean2[j];
        }
    }

    // Step 6+7: L2 normalize, scale by sqrt(128)
    l2_normalize_rows(output, n, LDA_DIM);
    cblas_dscal(n * LDA_DIM, sqrt_lda, output, 1);

    // Step 8: center with plda_mu
    for (int i = 0; i < n; i++) {
        double* row = output + i * LDA_DIM;
        for (int j = 0; j < LDA_DIM; j++) {
            row[j] -= model.plda_mu[j];
        }
    }

    // Step 9: PLDA transform — (N,128) @ (128,128)^T → (N,128)
    std::vector<double> plda_out(static_cast<size_t>(n) * LDA_DIM);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n, LDA_DIM, LDA_DIM,
                1.0, output, LDA_DIM,
                model.plda_tr.data(), LDA_DIM,
                0.0, plda_out.data(), LDA_DIM);

    std::memcpy(output, plda_out.data(),
                static_cast<size_t>(n) * LDA_DIM * sizeof(double));
}

}  // namespace diarization
