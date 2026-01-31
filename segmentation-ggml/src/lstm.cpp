#include "lstm.h"
#include <iostream>
#include <cstdio>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

namespace segmentation {

struct lstm_bidir_params {
    int hidden_size;
    const float* cached_w_ih_fwd;
    const float* cached_w_hh_fwd;
    const float* cached_w_ih_rev;
    const float* cached_w_hh_rev;
    float* ih_all_buf_fwd;
    float* ih_all_buf_rev;
};

// dst_h_offset: 0 for fwd, hidden_size for rev (writes to correct half of bidir output)
static void lstm_compute_one_direction(
    float* dst_data,
    int dst_h_offset,
    float* input_data,
    int64_t seq_len,
    int64_t input_size,
    int hidden_size,
    bool reverse,
    const float* w_ih_ptr,
    const float* w_hh_ptr,
    float* bias_ih_data,
    float* bias_hh_data,
    struct ggml_tensor* weight_ih,
    struct ggml_tensor* weight_hh,
    float* ih_all_buf) {

    int gate_size = 4 * hidden_size;

    std::vector<float> w_ih_tmp, w_hh_tmp;
    if (!w_ih_ptr) {
        w_ih_tmp.resize(input_size * gate_size);
        w_hh_tmp.resize(hidden_size * gate_size);
        ggml_fp16_t* f16 = (ggml_fp16_t*)weight_ih->data;
        for (int64_t i = 0; i < input_size * gate_size; i++)
            w_ih_tmp[i] = ggml_fp16_to_fp32(f16[i]);
        f16 = (ggml_fp16_t*)weight_hh->data;
        for (int64_t i = 0; i < hidden_size * gate_size; i++)
            w_hh_tmp[i] = ggml_fp16_to_fp32(f16[i]);
        w_ih_ptr = w_ih_tmp.data();
        w_hh_ptr = w_hh_tmp.data();
    }

    std::vector<float> ih_all_local;
    float* ih_all;
    if (ih_all_buf) {
        ih_all = ih_all_buf;
    } else {
        ih_all_local.resize(gate_size * seq_len, 0.0f);
        ih_all = ih_all_local.data();
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        gate_size, (int)seq_len, (int)input_size,
        1.0f,
        w_ih_ptr, (int)input_size,
        input_data, (int)seq_len,
        0.0f,
        ih_all, (int)seq_len);

    float h_t[512], c_t[512], gates[512];
    memset(h_t, 0, hidden_size * sizeof(float));
    memset(c_t, 0, hidden_size * sizeof(float));

    for (int64_t step = 0; step < seq_len; step++) {
        int64_t t = reverse ? (seq_len - 1 - step) : step;

        cblas_sgemv(CblasRowMajor, CblasNoTrans,
            gate_size, hidden_size,
            1.0f, w_hh_ptr, hidden_size,
            h_t, 1,
            0.0f, gates, 1);

        for (int g = 0; g < gate_size; g++) {
            gates[g] += ih_all[g * seq_len + t] + bias_ih_data[g] + bias_hh_data[g];
        }

        for (int h = 0; h < hidden_size; h++) {
            float i_val = 1.0f / (1.0f + expf(-gates[h]));
            float f_val = 1.0f / (1.0f + expf(-gates[hidden_size + h]));
            float g_val = tanhf(gates[2 * hidden_size + h]);
            float o_val = 1.0f / (1.0f + expf(-gates[3 * hidden_size + h]));

            c_t[h] = f_val * c_t[h] + i_val * g_val;
            h_t[h] = o_val * tanhf(c_t[h]);
        }

        for (int h = 0; h < hidden_size; h++) {
            dst_data[(dst_h_offset + h) * seq_len + t] = h_t[h];
        }
    }
}

// n_tasks=2: ith=0 runs forward, ith=1 runs reverse — in parallel
static void lstm_bidirectional_custom_op(
    struct ggml_tensor* dst,
    int ith, int nth,
    void* userdata) {

    (void)nth;
    auto* params = (lstm_bidir_params*)userdata;
    struct ggml_tensor* input = dst->src[0];
    int hidden_size = params->hidden_size;
    bool reverse = (ith == 1);

    const float* w_ih = reverse ? params->cached_w_ih_rev : params->cached_w_ih_fwd;
    const float* w_hh = reverse ? params->cached_w_hh_rev : params->cached_w_hh_fwd;
    float* ih_buf = reverse ? params->ih_all_buf_rev : params->ih_all_buf_fwd;

    int fwd_bias_idx = 3;
    int rev_bias_idx = 7;
    float* bias_ih = (float*)dst->src[reverse ? rev_bias_idx : fwd_bias_idx]->data;
    float* bias_hh = (float*)dst->src[reverse ? (rev_bias_idx + 1) : (fwd_bias_idx + 1)]->data;

    lstm_compute_one_direction(
        (float*)dst->data, ith * hidden_size,
        (float*)input->data,
        input->ne[0], input->ne[1],
        hidden_size, reverse,
        w_ih, w_hh,
        bias_ih, bias_hh,
        dst->src[reverse ? 5 : 1],
        dst->src[reverse ? 6 : 2],
        ih_buf);
}

static lstm_bidir_params s_bidir_params[4];
static int s_lstm_param_idx = 0;
static const lstm_weight_cache* s_active_cache = nullptr;

void lstm_init_weight_cache(lstm_weight_cache& cache, const segmentation_model& model) {
    int hidden_size = model.hparams.lstm_hidden;
    int gate_size = 4 * hidden_size;
    
    for (int layer = 0; layer < LSTM_LAYERS; layer++) {
        int input_size = (layer == 0) ? 60 : (2 * hidden_size);
        
        cache.w_ih[layer].resize(input_size * gate_size);
        cache.w_hh[layer].resize(hidden_size * gate_size);
        cache.w_ih_reverse[layer].resize(input_size * gate_size);
        cache.w_hh_reverse[layer].resize(hidden_size * gate_size);
        
        {
            ggml_fp16_t* src = (ggml_fp16_t*)model.lstm_weight_ih[layer]->data;
            for (int64_t i = 0; i < input_size * gate_size; i++)
                cache.w_ih[layer][i] = ggml_fp16_to_fp32(src[i]);
        }
        {
            ggml_fp16_t* src = (ggml_fp16_t*)model.lstm_weight_hh[layer]->data;
            for (int64_t i = 0; i < hidden_size * gate_size; i++)
                cache.w_hh[layer][i] = ggml_fp16_to_fp32(src[i]);
        }
        {
            ggml_fp16_t* src = (ggml_fp16_t*)model.lstm_weight_ih_reverse[layer]->data;
            for (int64_t i = 0; i < input_size * gate_size; i++)
                cache.w_ih_reverse[layer][i] = ggml_fp16_to_fp32(src[i]);
        }
        {
            ggml_fp16_t* src = (ggml_fp16_t*)model.lstm_weight_hh_reverse[layer]->data;
            for (int64_t i = 0; i < hidden_size * gate_size; i++)
                cache.w_hh_reverse[layer][i] = ggml_fp16_to_fp32(src[i]);
        }
    }
    
    // Pre-allocate ih_all buffers (one per direction to avoid conflicts)
    int max_seq_len = 589;
    int max_gate_size = 4 * hidden_size;
    cache.ih_all_buf.resize(max_seq_len * max_gate_size);
    cache.ih_all_buf_rev.resize(max_seq_len * max_gate_size);
    cache.initialized = true;
}

void lstm_set_active_cache(const lstm_weight_cache* cache) {
    s_active_cache = cache;
}

void lstm_print_profile() {
}

void lstm_reset_profile() {
}

struct ggml_tensor* lstm_layer_bidirectional(
    struct ggml_context* ctx,
    struct ggml_tensor* input,
    struct ggml_tensor* weight_ih_fwd,
    struct ggml_tensor* weight_hh_fwd,
    struct ggml_tensor* bias_ih_fwd,
    struct ggml_tensor* bias_hh_fwd,
    struct ggml_tensor* weight_ih_rev,
    struct ggml_tensor* weight_hh_rev,
    struct ggml_tensor* bias_ih_rev,
    struct ggml_tensor* bias_hh_rev,
    int hidden_size) {
    
    int64_t seq_len = input->ne[0];
    int64_t batch = input->ne[2];
    
    int layer = s_lstm_param_idx / 2;
    lstm_bidir_params* params = &s_bidir_params[layer % 4];
    s_lstm_param_idx += 2;
    
    params->hidden_size = hidden_size;
    params->cached_w_ih_fwd = nullptr;
    params->cached_w_hh_fwd = nullptr;
    params->cached_w_ih_rev = nullptr;
    params->cached_w_hh_rev = nullptr;
    params->ih_all_buf_fwd = nullptr;
    params->ih_all_buf_rev = nullptr;
    
    if (s_active_cache && s_active_cache->initialized && layer < LSTM_LAYERS) {
        params->cached_w_ih_fwd = s_active_cache->w_ih[layer].data();
        params->cached_w_hh_fwd = s_active_cache->w_hh[layer].data();
        params->cached_w_ih_rev = s_active_cache->w_ih_reverse[layer].data();
        params->cached_w_hh_rev = s_active_cache->w_hh_reverse[layer].data();
        params->ih_all_buf_fwd = const_cast<float*>(s_active_cache->ih_all_buf.data());
        params->ih_all_buf_rev = const_cast<float*>(s_active_cache->ih_all_buf_rev.data());
    }
    
    struct ggml_tensor* args[] = {
        input,
        weight_ih_fwd, weight_hh_fwd, bias_ih_fwd, bias_hh_fwd,
        weight_ih_rev, weight_hh_rev, bias_ih_rev, bias_hh_rev
    };
    
    struct ggml_tensor* output = ggml_custom_4d(
        ctx, GGML_TYPE_F32,
        seq_len, 2 * hidden_size, batch, 1,
        args, 9,
        lstm_bidirectional_custom_op,
        2, params);
    
    return output;
}

struct ggml_tensor* lstm_forward(
    struct ggml_context* ctx,
    const segmentation_model& model,
    struct ggml_tensor* input) {
    
    s_lstm_param_idx = 0;
    
    if (!input) {
        fprintf(stderr, "ERROR: LSTM input is null\n");
        return nullptr;
    }
    
    int hidden_size = model.hparams.lstm_hidden;
    struct ggml_tensor* layer_input = input;
    
    for (int layer = 0; layer < LSTM_LAYERS; layer++) {
        if (!model.lstm_weight_ih[layer] || !model.lstm_weight_hh[layer] ||
            !model.lstm_bias_ih[layer] || !model.lstm_bias_hh[layer] ||
            !model.lstm_weight_ih_reverse[layer] || !model.lstm_weight_hh_reverse[layer] ||
            !model.lstm_bias_ih_reverse[layer] || !model.lstm_bias_hh_reverse[layer]) {
            fprintf(stderr, "ERROR: Missing LSTM weights for layer %d\n", layer);
            return nullptr;
        }
        
        layer_input = lstm_layer_bidirectional(
            ctx, layer_input,
            model.lstm_weight_ih[layer],
            model.lstm_weight_hh[layer],
            model.lstm_bias_ih[layer],
            model.lstm_bias_hh[layer],
            model.lstm_weight_ih_reverse[layer],
            model.lstm_weight_hh_reverse[layer],
            model.lstm_bias_ih_reverse[layer],
            model.lstm_bias_hh_reverse[layer],
            hidden_size);
    }
    
    return layer_input;
}

LSTM::LSTM(int input_dim, int hidden_dim, int num_layers, bool bidirectional)
    : input_dim_(input_dim), hidden_dim_(hidden_dim), 
      num_layers_(num_layers), bidirectional_(bidirectional) {
}

LSTM::~LSTM() {
}

bool LSTM::init(struct ggml_context* ctx) {
    return true;
}

struct ggml_tensor* LSTM::forward(struct ggml_context* ctx, struct ggml_tensor* input) {
    return input;
}

int LSTM::get_output_dim() const {
    return bidirectional_ ? hidden_dim_ * 2 : hidden_dim_;
}

} // namespace segmentation
