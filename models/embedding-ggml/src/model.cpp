#include "model.h"
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#ifdef __APPLE__
#include <mach/mach.h>
#endif

namespace embedding {

static struct ggml_tensor* get_tensor(struct ggml_context* ctx, const char* name, bool required = true) {
    struct ggml_tensor* tensor = ggml_get_tensor(ctx, name);
    if (!tensor && required) {
        fprintf(stderr, "ERROR: Required tensor '%s' not found\n", name);
    }
    return tensor;
}

static uint32_t get_u32_meta(struct gguf_context* ctx, const char* key, uint32_t default_val) {
    int64_t key_id = gguf_find_key(ctx, key);
    if (key_id < 0) {
        fprintf(stderr, "WARNING: Metadata key '%s' not found, using default %u\n", key, default_val);
        return default_val;
    }
    return gguf_get_val_u32(ctx, key_id);
}

bool model_load(const std::string& fname, embedding_model& model, bool verbose) {
    if (verbose) printf("Loading model from: %s\n", fname.c_str());

    struct gguf_init_params gguf_params = {
        .no_alloc = true,
        .ctx = &model.ctx,
    };

    model.gguf_ctx = gguf_init_from_file(fname.c_str(), gguf_params);
    if (!model.gguf_ctx) {
        fprintf(stderr, "ERROR: Failed to open GGUF file: %s\n", fname.c_str());
        return false;
    }

    if (verbose) {
        printf("\nGGUF File Info:\n");
        printf("  Version: %u\n", gguf_get_version(model.gguf_ctx));
        printf("  Alignment: %zu\n", gguf_get_alignment(model.gguf_ctx));
        printf("  Tensors: %lld\n", (long long)gguf_get_n_tensors(model.gguf_ctx));
        printf("  Metadata keys: %lld\n", (long long)gguf_get_n_kv(model.gguf_ctx));
    }

    if (verbose) printf("\nLoading hyperparameters...\n");
    model.hparams.sample_rate   = get_u32_meta(model.gguf_ctx, "wespeaker.sample_rate", 16000);
    model.hparams.num_mel_bins  = get_u32_meta(model.gguf_ctx, "wespeaker.num_mel_bins", 80);
    model.hparams.frame_length  = get_u32_meta(model.gguf_ctx, "wespeaker.frame_length", 25);
    model.hparams.frame_shift   = get_u32_meta(model.gguf_ctx, "wespeaker.frame_shift", 10);
    model.hparams.embed_dim     = get_u32_meta(model.gguf_ctx, "wespeaker.embed_dim", 256);
    model.hparams.feat_dim      = get_u32_meta(model.gguf_ctx, "wespeaker.feat_dim", 80);

    if (verbose) {
        printf("  sample_rate: %d\n", model.hparams.sample_rate);
        printf("  num_mel_bins: %d\n", model.hparams.num_mel_bins);
        printf("  frame_length: %d ms\n", model.hparams.frame_length);
        printf("  frame_shift: %d ms\n", model.hparams.frame_shift);
        printf("  embed_dim: %d\n", model.hparams.embed_dim);
        printf("  feat_dim: %d\n", model.hparams.feat_dim);
    }

    if (!model.ctx) {
        fprintf(stderr, "ERROR: Failed to create ggml context\n");
        gguf_free(model.gguf_ctx);
        model.gguf_ctx = nullptr;
        return false;
    }

    if (verbose) printf("\nMapping tensor pointers...\n");

    // Initial conv + BN
    model.conv1_weight     = get_tensor(model.ctx, "resnet.conv1.weight");
    model.bn1_weight       = get_tensor(model.ctx, "resnet.bn1.weight");
    model.bn1_bias         = get_tensor(model.ctx, "resnet.bn1.bias");
    model.bn1_running_mean = get_tensor(model.ctx, "resnet.bn1.running_mean");
    model.bn1_running_var  = get_tensor(model.ctx, "resnet.bn1.running_var");

    // Layer blocks
    for (int L = 0; L < NUM_LAYERS; L++) {
        for (int B = 0; B < LAYER_BLOCKS[L]; B++) {
            char name[128];

            snprintf(name, sizeof(name), "resnet.layer%d.%d.conv1.weight", L + 1, B);
            model.layer_conv1_weight[L][B] = get_tensor(model.ctx, name);

            snprintf(name, sizeof(name), "resnet.layer%d.%d.bn1.weight", L + 1, B);
            model.layer_bn1_weight[L][B] = get_tensor(model.ctx, name);

            snprintf(name, sizeof(name), "resnet.layer%d.%d.bn1.bias", L + 1, B);
            model.layer_bn1_bias[L][B] = get_tensor(model.ctx, name);

            snprintf(name, sizeof(name), "resnet.layer%d.%d.bn1.running_mean", L + 1, B);
            model.layer_bn1_mean[L][B] = get_tensor(model.ctx, name);

            snprintf(name, sizeof(name), "resnet.layer%d.%d.bn1.running_var", L + 1, B);
            model.layer_bn1_var[L][B] = get_tensor(model.ctx, name);

            snprintf(name, sizeof(name), "resnet.layer%d.%d.conv2.weight", L + 1, B);
            model.layer_conv2_weight[L][B] = get_tensor(model.ctx, name);

            snprintf(name, sizeof(name), "resnet.layer%d.%d.bn2.weight", L + 1, B);
            model.layer_bn2_weight[L][B] = get_tensor(model.ctx, name);

            snprintf(name, sizeof(name), "resnet.layer%d.%d.bn2.bias", L + 1, B);
            model.layer_bn2_bias[L][B] = get_tensor(model.ctx, name);

            snprintf(name, sizeof(name), "resnet.layer%d.%d.bn2.running_mean", L + 1, B);
            model.layer_bn2_mean[L][B] = get_tensor(model.ctx, name);

            snprintf(name, sizeof(name), "resnet.layer%d.%d.bn2.running_var", L + 1, B);
            model.layer_bn2_var[L][B] = get_tensor(model.ctx, name);
        }

        // Shortcut (only layer2.0, layer3.0, layer4.0 → indices 1, 2, 3)
        if (L >= 1) {
            char name[128];

            snprintf(name, sizeof(name), "resnet.layer%d.0.shortcut.0.weight", L + 1);
            model.shortcut_conv_weight[L] = get_tensor(model.ctx, name);

            snprintf(name, sizeof(name), "resnet.layer%d.0.shortcut.1.weight", L + 1);
            model.shortcut_bn_weight[L] = get_tensor(model.ctx, name);

            snprintf(name, sizeof(name), "resnet.layer%d.0.shortcut.1.bias", L + 1);
            model.shortcut_bn_bias[L] = get_tensor(model.ctx, name);

            snprintf(name, sizeof(name), "resnet.layer%d.0.shortcut.1.running_mean", L + 1);
            model.shortcut_bn_mean[L] = get_tensor(model.ctx, name);

            snprintf(name, sizeof(name), "resnet.layer%d.0.shortcut.1.running_var", L + 1);
            model.shortcut_bn_var[L] = get_tensor(model.ctx, name);
        }
    }

    // Embedding head
    model.seg1_weight = get_tensor(model.ctx, "resnet.seg_1.weight");
    model.seg1_bias   = get_tensor(model.ctx, "resnet.seg_1.bias");

    if (verbose) printf("\nAllocating weight buffers...\n");

    ggml_backend_t weight_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!weight_backend) {
        fprintf(stderr, "ERROR: Failed to initialize CPU backend for weights\n");
        return false;
    }
    if (verbose) printf("  Using CPU backend for weights: %s\n", ggml_backend_name(weight_backend));

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(model.ctx, weight_backend);
    if (!buf) {
        fprintf(stderr, "ERROR: Failed to allocate weight buffer\n");
        ggml_backend_free(weight_backend);
        return false;
    }

    model.weight_buffers.push_back(buf);
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    if (verbose) printf("  Weight buffer: %.2f MB (%s)\n",
           ggml_backend_buffer_get_size(buf) / 1024.0 / 1024.0,
           ggml_backend_buffer_name(buf));

    ggml_backend_free(weight_backend);

    if (verbose) printf("\nLoading weight data from file...\n");

    FILE* f = fopen(fname.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Failed to open file for weight loading: %s\n", fname.c_str());
        return false;
    }

    int n_tensors = gguf_get_n_tensors(model.gguf_ctx);
    std::vector<uint8_t> read_buf;

    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(model.gguf_ctx, i);
        struct ggml_tensor* tensor = ggml_get_tensor(model.ctx, name);

        if (!tensor) {
            fprintf(stderr, "WARNING: Tensor '%s' not found in context\n", name);
            continue;
        }

        size_t offset = gguf_get_data_offset(model.gguf_ctx) + gguf_get_tensor_offset(model.gguf_ctx, i);
        size_t nbytes = ggml_nbytes(tensor);

        read_buf.resize(nbytes);

        fseek(f, (long)offset, SEEK_SET);
        size_t nread = fread(read_buf.data(), 1, nbytes, f);
        if (nread != nbytes) {
            fprintf(stderr, "ERROR: Failed to read tensor '%s' data (read %zu of %zu bytes)\n",
                    name, nread, nbytes);
            fclose(f);
            return false;
        }

        ggml_backend_tensor_set(tensor, read_buf.data(), 0, nbytes);
    }

    fclose(f);

    if (verbose) printf("  Loaded %d tensors\n", n_tensors);

    return true;
}

void model_free(embedding_model& model) {
    for (size_t i = 0; i < model.weight_buffers.size(); i++) {
        ggml_backend_buffer_free(model.weight_buffers[i]);
    }
    model.weight_buffers.clear();

    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    if (model.gguf_ctx) {
        gguf_free(model.gguf_ctx);
        model.gguf_ctx = nullptr;
    }
}

void model_print_info(const embedding_model& model) {
    printf("\n");
    printf("============================================================\n");
    printf("Embedding Model Tensor Summary\n");
    printf("============================================================\n");

    int n_tensors = gguf_get_n_tensors(model.gguf_ctx);

    size_t total_f16 = 0;
    size_t total_f32 = 0;
    int f16_count = 0;
    int f32_count = 0;

    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(model.gguf_ctx, i);
        struct ggml_tensor* tensor = ggml_get_tensor(model.ctx, name);

        if (!tensor) {
            printf("  %-55s: (null)\n", name);
            continue;
        }

        int n_dims = ggml_n_dims(tensor);
        char shape_str[128];
        int offset = 0;
        offset += snprintf(shape_str + offset, sizeof(shape_str) - offset, "[");
        for (int d = n_dims - 1; d >= 0; d--) {
            offset += snprintf(shape_str + offset, sizeof(shape_str) - offset,
                              "%lld%s", (long long)tensor->ne[d], d > 0 ? ", " : "");
        }
        snprintf(shape_str + offset, sizeof(shape_str) - offset, "]");

        const char* type_str = ggml_type_name(tensor->type);
        printf("  %-55s %-20s %s\n", name, shape_str, type_str);

        if (tensor->type == GGML_TYPE_F16) {
            f16_count++;
            total_f16 += ggml_nbytes(tensor);
        } else {
            f32_count++;
            total_f32 += ggml_nbytes(tensor);
        }
    }

    printf("\n  Total tensors: %d\n", n_tensors);
    printf("  F16 tensors: %d (%.2f MB)\n", f16_count, total_f16 / 1024.0 / 1024.0);
    printf("  F32 tensors: %d (%.2f MB)\n", f32_count, total_f32 / 1024.0 / 1024.0);
    printf("  Total size: %.2f MB\n", (total_f16 + total_f32) / 1024.0 / 1024.0);
    printf("============================================================\n");
}

bool model_verify(const embedding_model& model) {
    printf("\nVerifying model tensors...\n");

    int n_tensors = gguf_get_n_tensors(model.gguf_ctx);
    int null_count = 0;
    int loaded_count = 0;

    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(model.gguf_ctx, i);
        struct ggml_tensor* tensor = ggml_get_tensor(model.ctx, name);

        if (!tensor) {
            fprintf(stderr, "  MISSING: %s\n", name);
            null_count++;
        } else {
            loaded_count++;
        }
    }

    printf("  Loaded: %d / %d tensors\n", loaded_count, n_tensors);

    if (null_count > 0) {
        fprintf(stderr, "WARNING: %d tensors missing\n", null_count);
        return false;
    }

    printf("  All tensors loaded successfully!\n");
    return true;
}

void model_print_memory_usage(const embedding_model& model, const embedding_state& state) {
    printf("\n");
    printf("============================================================\n");
    printf("Memory Usage\n");
    printf("============================================================\n");

    size_t weight_size = 0;
    for (size_t i = 0; i < model.weight_buffers.size(); i++) {
        if (model.weight_buffers[i]) {
            weight_size += ggml_backend_buffer_get_size(model.weight_buffers[i]);
        }
    }
    printf("  Weight buffer:  %8.2f MB\n", weight_size / (1024.0 * 1024.0));

    size_t compute_size = 0;
    if (state.sched) {
        int n_backends = ggml_backend_sched_get_n_backends(state.sched);
        for (int i = 0; i < n_backends; i++) {
            ggml_backend_t backend = ggml_backend_sched_get_backend(state.sched, i);
            if (backend) {
                compute_size += ggml_backend_sched_get_buffer_size(state.sched, backend);
            }
        }
    }
    printf("  Compute buffer: %8.2f MB\n", compute_size / (1024.0 * 1024.0));

    size_t meta_size = state.graph_meta.size();
    printf("  Graph metadata: %8.2f MB\n", meta_size / (1024.0 * 1024.0));

    size_t total_ggml = weight_size + compute_size + meta_size;
    printf("  Total GGML:     %8.2f MB\n", total_ggml / (1024.0 * 1024.0));

#ifdef __APPLE__
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) == KERN_SUCCESS) {
        printf("  Peak RSS:       %8.2f MB\n", info.resident_size_max / (1024.0 * 1024.0));
    }
#endif
    printf("============================================================\n");
}

// ============================================================================
// Forward Pass Helper Functions
// ============================================================================

// BN inference: y = (x - mean) / sqrt(var + eps) * weight + bias
// Fused: y = x * scale + shift
//   scale = weight / sqrt(var + eps)
//   shift = bias - mean * scale
static struct ggml_tensor* batch_norm_2d(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* weight,
    struct ggml_tensor* bias,
    struct ggml_tensor* running_mean,
    struct ggml_tensor* running_var,
    struct ggml_tensor* eps_scalar) {

    int64_t C = weight->ne[0];

    // var + eps (eps_scalar is [1], broadcast via ggml_add1 not needed — use ggml_add with broadcast)
    struct ggml_tensor* var_plus_eps = ggml_add1(ctx, running_var, eps_scalar);
    struct ggml_tensor* inv_std = ggml_sqrt(ctx, var_plus_eps);
    struct ggml_tensor* scale = ggml_div(ctx, weight, inv_std);

    struct ggml_tensor* mean_scaled = ggml_mul(ctx, running_mean, scale);
    struct ggml_tensor* shift = ggml_sub(ctx, bias, mean_scaled);

    struct ggml_tensor* scale_4d = ggml_reshape_4d(ctx, scale, 1, 1, C, 1);
    struct ggml_tensor* shift_4d = ggml_reshape_4d(ctx, shift, 1, 1, C, 1);

    struct ggml_tensor* y = ggml_mul(ctx, x, scale_4d);
    y = ggml_add(ctx, y, shift_4d);

    return y;
}

static struct ggml_tensor* basic_block(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* conv1_w, struct ggml_tensor* bn1_w, struct ggml_tensor* bn1_b,
    struct ggml_tensor* bn1_mean, struct ggml_tensor* bn1_var,
    struct ggml_tensor* conv2_w, struct ggml_tensor* bn2_w, struct ggml_tensor* bn2_b,
    struct ggml_tensor* bn2_mean, struct ggml_tensor* bn2_var,
    struct ggml_tensor* sc_conv_w, struct ggml_tensor* sc_bn_w, struct ggml_tensor* sc_bn_b,
    struct ggml_tensor* sc_bn_mean, struct ggml_tensor* sc_bn_var,
    struct ggml_tensor* eps_scalar,
    int stride) {

    struct ggml_tensor* identity = x;

    struct ggml_tensor* out = ggml_conv_2d(ctx, conv1_w, x, stride, stride, 1, 1, 1, 1);
    out = batch_norm_2d(ctx, out, bn1_w, bn1_b, bn1_mean, bn1_var, eps_scalar);
    out = ggml_relu(ctx, out);

    out = ggml_conv_2d(ctx, conv2_w, out, 1, 1, 1, 1, 1, 1);
    out = batch_norm_2d(ctx, out, bn2_w, bn2_b, bn2_mean, bn2_var, eps_scalar);

    if (sc_conv_w != nullptr) {
        identity = ggml_conv_2d(ctx, sc_conv_w, x, stride, stride, 0, 0, 1, 1);
        identity = batch_norm_2d(ctx, identity, sc_bn_w, sc_bn_b, sc_bn_mean, sc_bn_var, eps_scalar);
    }

    out = ggml_add(ctx, out, identity);
    out = ggml_relu(ctx, out);

    return out;
}

struct ggml_tensor* model_forward(
    struct ggml_context* ctx,
    const embedding_model& model,
    struct ggml_tensor* fbank_input) {

    // Shared BN epsilon scalar [1] — used by all batch_norm_2d calls
    struct ggml_tensor* bn_eps = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(bn_eps, "bn_eps");
    ggml_set_input(bn_eps);

    // fbank_input: [T, 80, 1, 1] (W=T, H=80, C=1, B=1)
    struct ggml_tensor* x = ggml_conv_2d(ctx, model.conv1_weight, fbank_input, 1, 1, 1, 1, 1, 1);
    x = batch_norm_2d(ctx, x, model.bn1_weight, model.bn1_bias,
                      model.bn1_running_mean, model.bn1_running_var, bn_eps);
    x = ggml_relu(ctx, x);

    for (int L = 0; L < NUM_LAYERS; L++) {
        for (int B = 0; B < LAYER_BLOCKS[L]; B++) {
            int stride = (B == 0 && L > 0) ? 2 : 1;

            struct ggml_tensor* sc_conv = nullptr;
            struct ggml_tensor* sc_bn_w = nullptr;
            struct ggml_tensor* sc_bn_b = nullptr;
            struct ggml_tensor* sc_bn_m = nullptr;
            struct ggml_tensor* sc_bn_v = nullptr;

            if (B == 0 && L >= 1) {
                sc_conv = model.shortcut_conv_weight[L];
                sc_bn_w = model.shortcut_bn_weight[L];
                sc_bn_b = model.shortcut_bn_bias[L];
                sc_bn_m = model.shortcut_bn_mean[L];
                sc_bn_v = model.shortcut_bn_var[L];
            }

            x = basic_block(ctx, x,
                model.layer_conv1_weight[L][B],
                model.layer_bn1_weight[L][B], model.layer_bn1_bias[L][B],
                model.layer_bn1_mean[L][B], model.layer_bn1_var[L][B],
                model.layer_conv2_weight[L][B],
                model.layer_bn2_weight[L][B], model.layer_bn2_bias[L][B],
                model.layer_bn2_mean[L][B], model.layer_bn2_var[L][B],
                sc_conv, sc_bn_w, sc_bn_b, sc_bn_m, sc_bn_v,
                bn_eps,
                stride);
        }
    }

    // After layer4: x is [T/8, 10, 256, 1] (W=T/8, H=10, C=256, B=1)
    int64_t T8 = x->ne[0];   // T/8
    int64_t H  = x->ne[1];   // 10
    int64_t C  = x->ne[2];   // 256

    // TSTP: Temporal Statistics Pooling
    // Flatten H×C into one dimension: [T/8, 10, 256, 1] → [T/8, 2560, 1, 1]
    int64_t feat_dim = H * C;  // 10 * 256 = 2560
    struct ggml_tensor* flat = ggml_reshape_4d(ctx, x, T8, feat_dim, 1, 1);

    // Permute to [feat_dim, T/8, 1, 1] so time is ne[1] for reduction
    // Actually we need [feat_dim, T/8] as 2D for mean/variance computation
    struct ggml_tensor* flat_2d = ggml_reshape_2d(ctx, flat, T8, feat_dim);

    // Permute: [T8, feat_dim] → [feat_dim, T8]
    struct ggml_tensor* perm = ggml_permute(ctx, flat_2d, 1, 0, 2, 3);
    perm = ggml_cont(ctx, perm);
    // perm: [feat_dim, T8] — ne[0]=feat_dim, ne[1]=T8

    // Mean over time (ne[1]): reduce T8 dimension
    // ggml doesn't have a direct mean over axis, so we compute manually
    // mean = sum / T8
    // For each feature: sum across T8 time steps

    // Reshape to [feat_dim, T8, 1, 1] for ggml_pool_2d
    // Use 1D pooling over the time dimension
    // Actually, let's use ggml_mean which computes mean of all elements
    // We need per-feature mean, so we'll use matrix multiplication with ones vector

    // Alternative: use ggml_mul_mat with a ones vector
    // ones: [T8, 1] → mean = perm @ ones / T8 → [feat_dim, 1]
    struct ggml_tensor* ones = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T8, 1);
    ggml_set_name(ones, "tstp_ones");
    ggml_set_input(ones);

    // ggml_mul_mat(a, b) = a^T @ b
    // We want perm @ ones = [feat_dim, T8] @ [T8, 1] = [feat_dim, 1]
    // ggml_mul_mat(a, b) where a=[T8, feat_dim], b=[T8, 1] → a^T @ b = [feat_dim, 1]
    // So we need a = perm transposed = [T8, feat_dim]
    struct ggml_tensor* perm_t = ggml_permute(ctx, perm, 1, 0, 2, 3);
    perm_t = ggml_cont(ctx, perm_t);
    // perm_t: [T8, feat_dim]

    struct ggml_tensor* sum_vec = ggml_mul_mat(ctx, perm_t, ones);
    // sum_vec: [feat_dim, 1]

    struct ggml_tensor* t8_scalar = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(t8_scalar, "tstp_t8");
    ggml_set_input(t8_scalar);

    struct ggml_tensor* mean_vec = ggml_div(ctx, sum_vec, t8_scalar);
    // mean_vec: [feat_dim, 1]

    // Variance: var = E[x^2] - E[x]^2
    // x^2
    struct ggml_tensor* perm_sq = ggml_mul(ctx, perm, perm);
    // perm_sq: [feat_dim, T8]
    struct ggml_tensor* perm_sq_t = ggml_permute(ctx, perm_sq, 1, 0, 2, 3);
    perm_sq_t = ggml_cont(ctx, perm_sq_t);

    struct ggml_tensor* sum_sq_vec = ggml_mul_mat(ctx, perm_sq_t, ones);
    struct ggml_tensor* mean_sq_vec = ggml_div(ctx, sum_sq_vec, t8_scalar);

    // var = mean(x^2) - mean(x)^2
    struct ggml_tensor* mean_vec_sq = ggml_mul(ctx, mean_vec, mean_vec);
    struct ggml_tensor* var_vec = ggml_sub(ctx, mean_sq_vec, mean_vec_sq);

    // std = sqrt(var + eps) — use small eps for numerical stability
    struct ggml_tensor* var_eps = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(var_eps, "tstp_eps");
    ggml_set_input(var_eps);

    struct ggml_tensor* var_stable = ggml_add(ctx, var_vec, var_eps);
    struct ggml_tensor* std_vec = ggml_sqrt(ctx, var_stable);
    // std_vec: [feat_dim, 1]

    // For unbiased std (ddof=1): multiply var by T8/(T8-1) before sqrt
    // PyTorch std uses ddof=1 by default in TSTP
    // Actually, let's apply Bessel's correction:
    // unbiased_var = var * T8 / (T8 - 1)
    struct ggml_tensor* t8m1_scalar = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(t8m1_scalar, "tstp_t8m1");
    ggml_set_input(t8m1_scalar);

    // bessel = T8 / (T8 - 1)
    struct ggml_tensor* bessel = ggml_div(ctx, t8_scalar, t8m1_scalar);
    struct ggml_tensor* var_unbiased = ggml_mul(ctx, var_vec, bessel);
    struct ggml_tensor* var_unbiased_stable = ggml_add(ctx, var_unbiased, var_eps);
    std_vec = ggml_sqrt(ctx, var_unbiased_stable);

    // Concatenate mean + std: [feat_dim, 1] + [feat_dim, 1] → [2*feat_dim, 1]
    struct ggml_tensor* mean_1d = ggml_reshape_1d(ctx, mean_vec, feat_dim);
    struct ggml_tensor* std_1d = ggml_reshape_1d(ctx, std_vec, feat_dim);
    struct ggml_tensor* pooled = ggml_concat(ctx, mean_1d, std_1d, 0);
    // pooled: [2*feat_dim] = [5120]

    // Linear: y = weight @ x + bias
    // seg1_weight: GGML ne[0]=5120, ne[1]=256
    // ggml_mul_mat(weight, x) = weight^T @ x
    // weight is [5120, 256], x is [5120]
    // weight^T is [256, 5120], weight^T @ x = [256]
    struct ggml_tensor* pooled_2d = ggml_reshape_2d(ctx, pooled, 2 * feat_dim, 1);
    struct ggml_tensor* embed = ggml_mul_mat(ctx, model.seg1_weight, pooled_2d);
    embed = ggml_add(ctx, embed, model.seg1_bias);
    // embed: [256, 1]

    struct ggml_tensor* output = ggml_reshape_1d(ctx, embed, model.hparams.embed_dim);
    ggml_set_name(output, "embedding");

    return output;
}

// ============================================================================
// State Management
// ============================================================================

bool state_init(embedding_state& state, embedding_model& model, bool verbose) {
    if (verbose) printf("\nInitializing inference state...\n");

    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu_backend) {
        fprintf(stderr, "ERROR: Failed to initialize CPU backend\n");
        return false;
    }
    state.backends.push_back(cpu_backend);
    if (verbose) printf("  Using CPU backend: %s\n", ggml_backend_name(cpu_backend));

    size_t meta_size = ggml_tensor_overhead() * MAX_GRAPH_NODES + ggml_graph_overhead_custom(MAX_GRAPH_NODES, false);
    state.graph_meta.resize(meta_size);
    if (verbose) printf("  Graph metadata buffer: %.2f MB\n", meta_size / 1024.0 / 1024.0);

    state.sched = ggml_backend_sched_new(
        state.backends.data(),
        NULL,
        (int)state.backends.size(),
        MAX_GRAPH_NODES,
        false,
        true
    );

    if (!state.sched) {
        fprintf(stderr, "ERROR: Failed to create backend scheduler\n");
        return false;
    }

    // Pre-allocate with a representative graph (use 500 frames as typical)
    if (verbose) printf("  Pre-allocating compute buffers...\n");
    state.num_frames = 500;
    struct ggml_cgraph* graph = build_graph(model, state, state.num_frames);
    if (!graph) {
        fprintf(stderr, "ERROR: Failed to build graph for pre-allocation\n");
        return false;
    }

    if (!ggml_backend_sched_alloc_graph(state.sched, graph)) {
        fprintf(stderr, "ERROR: Failed to pre-allocate compute buffers\n");
        return false;
    }

    ggml_backend_sched_reset(state.sched);
    if (verbose) printf("  State initialized successfully\n");

    return true;
}

void state_free(embedding_state& state) {
    if (state.sched) {
        ggml_backend_sched_free(state.sched);
        state.sched = nullptr;
    }

    for (size_t i = 0; i < state.backends.size(); i++) {
        ggml_backend_free(state.backends[i]);
    }
    state.backends.clear();
    state.graph_meta.clear();
}

// ============================================================================
// Graph Building
// ============================================================================

struct ggml_cgraph* build_graph(
    embedding_model& model,
    embedding_state& state,
    int num_frames) {

    struct ggml_init_params params = {
        /*.mem_size   =*/ state.graph_meta.size(),
        /*.mem_buffer =*/ state.graph_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "ERROR: Failed to create graph context\n");
        return nullptr;
    }

    struct ggml_cgraph* graph = ggml_new_graph_custom(ctx, MAX_GRAPH_NODES, false);

    // Input: fbank features as 4D tensor [T, 80, 1, 1]
    struct ggml_tensor* input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                                    num_frames, 80, 1, 1);
    ggml_set_name(input, "fbank");
    ggml_set_input(input);

    struct ggml_tensor* output = model_forward(ctx, model, input);
    if (!output) {
        ggml_free(ctx);
        return nullptr;
    }

    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);

    ggml_free(ctx);

    return graph;
}

// ============================================================================
// Inference
// ============================================================================

bool model_infer(
    embedding_model& model,
    embedding_state& state,
    const float* fbank_data,
    int num_frames,
    float* output,
    size_t output_size) {

    state.num_frames = num_frames;

    struct ggml_cgraph* graph = build_graph(model, state, num_frames);
    if (!graph) {
        fprintf(stderr, "ERROR: Failed to build computation graph\n");
        return false;
    }

    if (!ggml_backend_sched_alloc_graph(state.sched, graph)) {
        fprintf(stderr, "ERROR: Failed to allocate compute buffers\n");
        return false;
    }

    // Set fbank input data
    // fbank_data is (T, 80) row-major: ne[0]=80 varies fastest
    // But our graph input is [T, 80, 1, 1] where ne[0]=T
    // We need to transpose: create a transposed copy
    struct ggml_tensor* input_tensor = ggml_graph_get_tensor(graph, "fbank");
    if (!input_tensor) {
        fprintf(stderr, "ERROR: Input tensor 'fbank' not found in graph\n");
        ggml_backend_sched_reset(state.sched);
        return false;
    }

    // fbank_data layout: frame0[80], frame1[80], ... (row-major, ne[0]=80 fastest)
    // We need [T, 80] in GGML where ne[0]=T (column-major)
    // So we need to transpose the data
    std::vector<float> transposed(num_frames * 80);
    for (int t = 0; t < num_frames; t++) {
        for (int b = 0; b < 80; b++) {
            transposed[b * num_frames + t] = fbank_data[t * 80 + b];
        }
    }
    ggml_backend_tensor_set(input_tensor, transposed.data(), 0, num_frames * 80 * sizeof(float));

    // Set helper input tensors by name
    auto set_input = [&](const char* name_to_find) -> struct ggml_tensor* {
        return ggml_graph_get_tensor(graph, name_to_find);
    };

    // BN epsilon = 1e-5
    struct ggml_tensor* bn_eps_t = set_input("bn_eps");
    if (bn_eps_t) {
        float eps_val = 1e-5f;
        ggml_backend_tensor_set(bn_eps_t, &eps_val, 0, sizeof(float));
    }

    // TSTP ones vector
    struct ggml_tensor* ones_t = set_input("tstp_ones");
    if (ones_t) {
        int64_t T8 = ones_t->ne[0];
        std::vector<float> ones_data(T8, 1.0f);
        ggml_backend_tensor_set(ones_t, ones_data.data(), 0, T8 * sizeof(float));
    }

    // TSTP T8 scalar (number of time steps after layer4)
    struct ggml_tensor* t8_t = set_input("tstp_t8");
    if (t8_t) {
        // T8 is determined by the actual graph — get from ones tensor
        float t8_val = ones_t ? (float)ones_t->ne[0] : (float)(num_frames / 8);
        ggml_backend_tensor_set(t8_t, &t8_val, 0, sizeof(float));
    }

    // TSTP T8-1 scalar (for Bessel's correction)
    struct ggml_tensor* t8m1_t = set_input("tstp_t8m1");
    if (t8m1_t) {
        float t8_val = ones_t ? (float)ones_t->ne[0] : (float)(num_frames / 8);
        float t8m1_val = t8_val - 1.0f;
        if (t8m1_val < 1.0f) t8m1_val = 1.0f;
        ggml_backend_tensor_set(t8m1_t, &t8m1_val, 0, sizeof(float));
    }

    // TSTP variance epsilon
    struct ggml_tensor* tstp_eps_t = set_input("tstp_eps");
    if (tstp_eps_t) {
        float eps_val = 1e-10f;
        ggml_backend_tensor_set(tstp_eps_t, &eps_val, 0, sizeof(float));
    }

    if (ggml_backend_sched_graph_compute(state.sched, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ERROR: Graph computation failed\n");
        ggml_backend_sched_reset(state.sched);
        return false;
    }

    struct ggml_tensor* output_tensor = ggml_graph_get_tensor(graph, "embedding");
    if (!output_tensor) {
        fprintf(stderr, "ERROR: Output tensor 'embedding' not found in graph\n");
        ggml_backend_sched_reset(state.sched);
        return false;
    }

    size_t output_bytes = ggml_nbytes(output_tensor);
    size_t requested_bytes = output_size * sizeof(float);
    size_t copy_bytes = (output_bytes < requested_bytes) ? output_bytes : requested_bytes;
    ggml_backend_tensor_get(output_tensor, output, 0, copy_bytes);

    ggml_backend_sched_reset(state.sched);

    return true;
}

} // namespace embedding
