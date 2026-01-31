# Whisper.cpp GGML Architecture Analysis

**Repository**: https://github.com/ggml-org/whisper.cpp  
**Commit SHA**: 7aa8818647303b567c3a21fe4220b2681988e220  
**Analysis Date**: 2026-01-29

This document analyzes how whisper.cpp uses GGML to create a Metal-ready, production-quality inference engine. The patterns here are directly applicable to our segmentation-ggml project.

---

## 1. Backend Architecture: Multi-Backend Support

### Pattern: Separate Backend Initialization from Model Loading

**Location**: `src/whisper.cpp:1290-1359`

```cpp
// Initialize GPU backend (Metal on macOS)
static ggml_backend_t whisper_backend_init_gpu(const whisper_context_params & params) {
    ggml_backend_dev_t dev = nullptr;
    
    if (params.use_gpu) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev_cur = ggml_backend_dev_get(i);
            enum ggml_backend_dev_type dev_type = ggml_backend_dev_type(dev_cur);
            
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || 
                dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                if (cnt == params.gpu_device) {
                    dev = dev_cur;
                }
            }
        }
    }
    
    return dev ? ggml_backend_dev_init(dev, nullptr) : nullptr;
}

// Initialize all backends in priority order: GPU -> ACCEL -> CPU
static std::vector<ggml_backend_t> whisper_backend_init(const whisper_context_params & params) {
    std::vector<ggml_backend_t> result;
    
    // 1. GPU backend (Metal on macOS)
    ggml_backend_t backend_gpu = whisper_backend_init_gpu(params);
    if (backend_gpu) {
        result.push_back(backend_gpu);
    }
    
    // 2. ACCEL backends (special accelerators)
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (backend) {
                result.push_back(backend);
            }
        }
    }
    
    // 3. CPU backend (always present as fallback)
    ggml_backend_t backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    result.push_back(backend_cpu);
    
    return result;
}
```

**Key Insights**:
- **Priority order**: GPU → ACCEL → CPU ensures optimal performance
- **Fallback strategy**: CPU backend is always initialized as fallback
- **Device enumeration**: Uses `ggml_backend_dev_count()` and `ggml_backend_dev_get()` to discover available devices
- **Type checking**: Distinguishes between GPU, IGPU (integrated GPU), ACCEL, and CPU

**Recommendation for segmentation-ggml**:
```cpp
// In segmentation_model.cpp
std::vector<ggml_backend_t> init_backends(bool use_gpu) {
    std::vector<ggml_backend_t> backends;
    
    if (use_gpu) {
        // Try Metal on macOS
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                auto backend = ggml_backend_dev_init(dev, nullptr);
                if (backend) {
                    backends.push_back(backend);
                    break; // Use first GPU
                }
            }
        }
    }
    
    // Always add CPU fallback
    backends.push_back(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    return backends;
}
```

---

## 2. Buffer Type Selection: Smart Weight Placement

### Pattern: Per-Tensor Buffer Type Selection Based on Operation Support

**Location**: `src/whisper.cpp:1361-1461`

```cpp
using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;

// Create prioritized list of buffer types
static buft_list_t make_buft_list(whisper_context_params & params) {
    buft_list_t buft_list;
    
    // 1. GPU buffer types
    if (params.use_gpu) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                auto * buft = ggml_backend_dev_buffer_type(dev);
                if (buft) {
                    buft_list.emplace_back(dev, buft);
                }
            }
        }
    }
    
    // 2. CPU extra buffer types (e.g., BLAS)
    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
        ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
    
    if (get_extra_bufts_fn) {
        ggml_backend_buffer_type_t * extra_bufts = get_extra_bufts_fn(cpu_dev);
        for (int i = 0; extra_bufts[i] != nullptr; ++i) {
            buft_list.emplace_back(cpu_dev, extra_bufts[i]);
        }
    }
    
    // 3. Standard CPU buffer type
    buft_list.emplace_back(cpu_dev, ggml_backend_cpu_buffer_type());
    
    return buft_list;
}

// Check if a specific operation is supported on a buffer type
static bool weight_buft_supported(
    const whisper_hparams & hparams,
    ggml_tensor * w,
    ggml_op op,
    ggml_backend_buffer_type_t buft,
    ggml_backend_dev_t dev) {
    
    // Create a temporary tensor to test operation support
    w->buffer = ggml_backend_buft_alloc_buffer(buft, 0);
    bool op_supported = ggml_backend_dev_supports_op(dev, op_tensor);
    ggml_backend_buffer_free(w->buffer);
    
    return op_supported;
}

// Select best buffer type for a weight tensor
static ggml_backend_buffer_type_t select_weight_buft(
    const whisper_hparams & hparams,
    ggml_tensor * w,
    ggml_op op,
    buft_list_t buft_list) {
    
    for (auto & p : buft_list) {
        ggml_backend_dev_t dev = p.first;
        ggml_backend_buffer_type_t buft = p.second;
        
        if (weight_buft_supported(hparams, w, op, buft, dev)) {
            return buft;
        }
    }
    
    return nullptr; // No compatible buffer type found
}
```

**Key Insights**:
- **Per-tensor selection**: Each weight tensor gets the best buffer type for its operation
- **Operation testing**: Actually tests if the backend supports the operation before committing
- **Graceful degradation**: Falls back through priority list (GPU → CPU BLAS → CPU)
- **Extra buffer types**: Leverages CPU BLAS or other accelerated CPU buffers when available

**Recommendation for segmentation-ggml**:
```cpp
// In model loading
ggml_backend_buffer_type_t select_buffer_for_weight(
    ggml_tensor* weight,
    ggml_op primary_op,
    const std::vector<ggml_backend_t>& backends) {
    
    // Try each backend in priority order
    for (auto backend : backends) {
        auto dev = ggml_backend_get_device(backend);
        auto buft = ggml_backend_dev_buffer_type(dev);
        
        // Test if this backend supports the operation
        weight->buffer = ggml_backend_buft_alloc_buffer(buft, 0);
        bool supported = ggml_backend_dev_supports_op(dev, create_test_op(weight, primary_op));
        ggml_backend_buffer_free(weight->buffer);
        
        if (supported) {
            return buft;
        }
    }
    
    // Fallback to CPU
    return ggml_backend_cpu_buffer_type();
}
```

---

## 3. Memory Management: Separate Contexts for Weights vs Computation

### Pattern: Multiple Contexts with `no_alloc` Flag

**Location**: `src/whisper.cpp:1687-1859`

```cpp
// Create separate contexts for each buffer type
std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;

auto get_ctx = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
    auto it = ctx_map.find(buft);
    if (it == ctx_map.end()) {
        ggml_init_params params = {
            /*.mem_size   =*/ n_tensors * ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,  // Don't allocate data, just metadata
        };
        
        ggml_context * ctx = ggml_init(params);
        ctx_map[buft] = ctx;
        model.ctxs.emplace_back(ctx);
        
        return ctx;
    }
    return it->second;
};

// Create weight tensors (metadata only, no data allocation yet)
auto create_tensor = [&](asr_tensor type, asr_system system, ggml_tensor * meta, int layer = 0) -> ggml_tensor * {
    ggml_op op = ASR_TENSOR_INFO.at(type);
    ggml_backend_buffer_type_t buft = select_weight_buft(hparams, meta, op, buft_list);
    
    ggml_context * ctx = get_ctx(buft);  // Get or create context for this buffer type
    ggml_tensor * tensor = ggml_dup_tensor(ctx, meta);  // Create tensor metadata
    
    model.tensors[format(ASR_TENSOR_NAMES.at(system).at(type), layer)] = tensor;
    return tensor;
};

// After all tensors are created, allocate buffers
for (auto & p : ctx_map) {
    ggml_backend_buffer_type_t buft = p.first;
    ggml_context * ctx = p.second;
    
    // Allocate backend buffer for all tensors in this context
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (buf) {
        model.buffers.emplace_back(buf);
        
        size_t size_main = ggml_backend_buffer_get_size(buf);
        WHISPER_LOG_INFO("%s: %12s total size = %8.2f MB\n", 
            __func__, ggml_backend_buffer_name(buf), size_main / 1e6);
    }
}
```

**Key Insights**:
- **Lazy allocation**: Create tensor metadata first (`no_alloc = true`), allocate data later
- **Grouped allocation**: All tensors using the same buffer type are allocated together
- **Multiple contexts**: One context per buffer type (e.g., one for Metal, one for CPU)
- **Efficient memory**: Only allocates what's needed for each backend

**Recommendation for segmentation-ggml**:
```cpp
// In model loading
struct SegmentationModel {
    std::vector<ggml_context*> weight_contexts;  // One per buffer type
    std::vector<ggml_backend_buffer_t> weight_buffers;
    std::map<std::string, ggml_tensor*> tensors;
};

void load_weights(SegmentationModel& model, const std::vector<ggml_backend_t>& backends) {
    std::map<ggml_backend_buffer_type_t, ggml_context*> ctx_map;
    
    // Phase 1: Create tensor metadata (no allocation)
    for (auto& [name, shape, op] : weight_specs) {
        auto buft = select_buffer_for_weight(shape, op, backends);
        auto ctx = get_or_create_context(ctx_map, buft);
        
        auto tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, shape[0], shape[1]);
        model.tensors[name] = tensor;
    }
    
    // Phase 2: Allocate buffers
    for (auto& [buft, ctx] : ctx_map) {
        auto buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        model.weight_buffers.push_back(buf);
        model.weight_contexts.push_back(ctx);
    }
    
    // Phase 3: Load weight data
    for (auto& [name, tensor] : model.tensors) {
        load_tensor_data(tensor, name);
    }
}
```

---

## 4. Graph Building: Separate Build from Compute

### Pattern: Pure Graph Construction Functions

**Location**: `src/whisper.cpp:1976-2036`

```cpp
static struct ggml_cgraph * whisper_build_graph_conv(
        whisper_context & wctx,
          whisper_state & wstate) {
    const auto & model   = wctx.model;
    const auto & hparams = model.hparams;
    
    // Use temporary context for graph metadata
    struct ggml_init_params params = {
        /*.mem_size   =*/ wstate.sched_conv.meta.size(),
        /*.mem_buffer =*/ wstate.sched_conv.meta.data(),
        /*.no_alloc   =*/ true,  // Graph building doesn't allocate compute buffers
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    ggml_cgraph * gf = ggml_new_graph(ctx0);
    
    // Create input tensor (metadata only)
    struct ggml_tensor * mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2*n_ctx, n_mels);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);  // Mark as input
    
    // Build computation graph
    struct ggml_tensor * cur = nullptr;
    cur = ggml_conv_1d_ph(ctx0, model.e_conv_1_w, mel, 1, 1);
    cur = ggml_add(ctx0, cur, model.e_conv_1_b);
    cur = ggml_gelu(ctx0, cur);
    
    cur = ggml_conv_1d_ph(ctx0, model.e_conv_2_w, cur, 2, 1);
    cur = ggml_add(ctx0, cur, model.e_conv_2_b);
    cur = ggml_gelu(ctx0, cur);
    
    ggml_set_name(cur, "embd_conv");
    ggml_set_output(cur);  // Mark as output
    
    ggml_build_forward_expand(gf, cur);
    
    ggml_free(ctx0);  // Free temporary context
    
    return gf;
}
```

**Key Insights**:
- **Pure function**: Graph building is separate from execution
- **Temporary context**: Uses a small temporary context just for graph metadata
- **Named tensors**: Uses `ggml_set_name()` for debugging and tensor retrieval
- **Input/output marking**: Explicitly marks input and output tensors
- **Context cleanup**: Frees temporary context after graph is built

**Recommendation for segmentation-ggml**:
```cpp
// In segmentation_model.cpp
ggml_cgraph* build_encoder_graph(
    SegmentationModel& model,
    SegmentationState& state) {
    
    // Temporary context for graph metadata
    ggml_init_params params = {
        .mem_size = state.graph_meta.size(),
        .mem_buffer = state.graph_meta.data(),
        .no_alloc = true
    };
    
    auto ctx = ggml_init(params);
    auto graph = ggml_new_graph(ctx);
    
    // Input: audio features
    auto input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_features, n_frames);
    ggml_set_name(input, "audio_input");
    ggml_set_input(input);
    
    // Build encoder layers
    auto cur = input;
    for (int i = 0; i < model.n_layers; ++i) {
        cur = build_encoder_layer(ctx, model, cur, i);
    }
    
    ggml_set_name(cur, "encoder_output");
    ggml_set_output(cur);
    ggml_build_forward_expand(graph, cur);
    
    ggml_free(ctx);
    return graph;
}
```

---

## 5. Backend Scheduler: Automatic Multi-Backend Orchestration

### Pattern: `ggml_backend_sched` for Automatic Scheduling

**Location**: `src/whisper.cpp:537-573`

```cpp
struct whisper_sched {
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> meta;  // Metadata buffer for graph building
};

// Initialize scheduler with all available backends
static bool whisper_sched_graph_init(
    struct whisper_sched & allocr,
    std::vector<ggml_backend_t> backends,
    std::function<struct ggml_cgraph *()> && get_graph) {
    
    auto & sched = allocr.sched;
    auto & meta  = allocr.meta;
    
    // Create scheduler with all backends
    sched = ggml_backend_sched_new(
        backends.data(),
        nullptr,  // No buffer types (auto-detect)
        backends.size(),
        WHISPER_MAX_NODES,
        false,  // Don't use parallel execution
        true    // Use graph splitting
    );
    
    // Allocate metadata buffer for graph building
    meta.resize(ggml_tensor_overhead()*WHISPER_MAX_NODES + ggml_graph_overhead());
    
    // Pre-allocate compute buffer by running allocation once
    if (!ggml_backend_sched_alloc_graph(sched, get_graph())) {
        WHISPER_LOG_ERROR("%s: failed to allocate the compute buffer\n", __func__);
        return false;
    }
    
    ggml_backend_sched_reset(sched);  // Reset for actual use
    
    return true;
}
```

**Usage in Inference** (`src/whisper.cpp:2358-2418`):

```cpp
static bool whisper_encode_internal(
        whisper_context & wctx,
          whisper_state & wstate,
               const int   mel_offset,
               const int   n_threads) {
    
    // Conv stage
    {
        auto & sched = wstate.sched_conv.sched;
        
        // Build graph
        ggml_cgraph * gf = whisper_build_graph_conv(wctx, wstate);
        
        // Allocate compute buffers for this graph
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            return false;
        }
        
        // Set input data
        struct ggml_tensor * mel = ggml_graph_get_tensor(gf, "mel");
        ggml_backend_tensor_set(mel, wstate.inp_mel.data(), 0, ggml_nelements(mel)*sizeof(float));
        
        // Execute graph (scheduler automatically picks backends)
        if (!ggml_graph_compute_helper(sched, gf, n_threads)) {
            return false;
        }
    }
    
    // Encoder stage
    {
        auto & sched = wstate.sched_encode.sched;
        ggml_cgraph * gf = whisper_build_graph_encoder(wctx, wstate);
        
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            return false;
        }
        
        if (!ggml_graph_compute_helper(sched, gf, n_threads)) {
            return false;
        }
    }
    
    return true;
}
```

**Key Insights**:
- **One scheduler per graph**: Separate schedulers for conv, encoder, cross-attention, decoder
- **Pre-allocation**: Runs allocation once during initialization to determine buffer sizes
- **Automatic backend selection**: Scheduler automatically assigns operations to best backend
- **Graph splitting**: Can split graph across multiple backends if beneficial
- **Reset between runs**: Uses `ggml_backend_sched_reset()` to prepare for next inference

**Recommendation for segmentation-ggml**:
```cpp
// In segmentation_state.cpp
struct SegmentationState {
    ggml_backend_sched_t encoder_sched;
    ggml_backend_sched_t decoder_sched;
    std::vector<uint8_t> encoder_meta;
    std::vector<uint8_t> decoder_meta;
};

bool init_schedulers(
    SegmentationState& state,
    SegmentationModel& model,
    const std::vector<ggml_backend_t>& backends) {
    
    // Initialize encoder scheduler
    state.encoder_sched = ggml_backend_sched_new(
        backends.data(),
        nullptr,
        backends.size(),
        MAX_NODES,
        false,  // No parallel
        true    // Enable splitting
    );
    
    state.encoder_meta.resize(ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead());
    
    // Pre-allocate by building and allocating graph once
    auto encoder_graph = build_encoder_graph(model, state);
    if (!ggml_backend_sched_alloc_graph(state.encoder_sched, encoder_graph)) {
        return false;
    }
    ggml_backend_sched_reset(state.encoder_sched);
    
    // Same for decoder...
    
    return true;
}

// In inference
bool run_encoder(SegmentationState& state, const float* input, size_t input_size) {
    auto graph = build_encoder_graph(state.model, state);
    
    // Allocate compute buffers
    if (!ggml_backend_sched_alloc_graph(state.encoder_sched, graph)) {
        return false;
    }
    
    // Set input
    auto input_tensor = ggml_graph_get_tensor(graph, "audio_input");
    ggml_backend_tensor_set(input_tensor, input, 0, input_size * sizeof(float));
    
    // Compute
    if (ggml_backend_sched_graph_compute(state.encoder_sched, graph) != GGML_STATUS_SUCCESS) {
        return false;
    }
    
    return true;
}
```

---

## 6. KV Cache Management: Efficient Attention State

### Pattern: Backend-Allocated KV Cache

**Location**: `src/whisper.cpp:968-1017`

```cpp
static bool whisper_kv_cache_init(
         struct whisper_kv_cache & cache,
                  ggml_backend_t   backend,
                       ggml_type   wtype,
                         int64_t   n_text_state,
                         int64_t   n_text_layer,
                             int   n_ctx) {
    
    const int64_t n_mem      = n_text_layer * n_ctx;
    const int64_t n_elements = n_text_state * n_mem;
    
    // Small buffer for tensor metadata
    cache.ctx_buf.resize(2 * ggml_tensor_overhead());
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ cache.ctx_buf.size(),
        /*.mem_buffer =*/ cache.ctx_buf.data(),
        /*.no_alloc   =*/ true,  // Don't allocate data
    };
    
    cache.head = 0;
    cache.size = n_ctx;
    cache.cells.clear();
    cache.cells.resize(n_ctx);
    
    struct ggml_context * ctx = ggml_init(params);
    
    // Create K and V tensors (metadata only)
    cache.k = ggml_new_tensor_1d(ctx, wtype, n_elements);
    cache.v = ggml_new_tensor_1d(ctx, wtype, n_elements);
    
    // Allocate actual data on backend
    cache.buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!cache.buffer) {
        WHISPER_LOG_ERROR("%s: failed to allocate memory for the kv cache\n", __func__);
        return false;
    }
    
    // Clear cache to zeros
    ggml_backend_buffer_clear(cache.buffer, 0);
    
    ggml_free(ctx);  // Free metadata context
    
    return true;
}
```

**Key Insights**:
- **Backend allocation**: KV cache is allocated on the same backend as the model (Metal/CPU)
- **Contiguous storage**: K and V are stored as 1D tensors for efficiency
- **Metadata separation**: Small CPU buffer for tensor metadata, large backend buffer for data
- **Zero initialization**: Clears cache to zeros after allocation
- **Context cleanup**: Frees metadata context after allocation

**Recommendation for segmentation-ggml**:
```cpp
// For segmentation, we might not need KV cache (no autoregressive decoding)
// But if we add recurrent processing or temporal modeling:

struct TemporalCache {
    ggml_tensor* hidden_state;
    ggml_backend_buffer_t buffer;
    std::vector<uint8_t> ctx_buf;
};

bool init_temporal_cache(
    TemporalCache& cache,
    ggml_backend_t backend,
    int hidden_size,
    int max_frames) {
    
    cache.ctx_buf.resize(ggml_tensor_overhead());
    
    ggml_init_params params = {
        .mem_size = cache.ctx_buf.size(),
        .mem_buffer = cache.ctx_buf.data(),
        .no_alloc = true
    };
    
    auto ctx = ggml_init(params);
    
    // Create hidden state tensor
    cache.hidden_state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, max_frames);
    
    // Allocate on backend
    cache.buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!cache.buffer) {
        return false;
    }
    
    ggml_backend_buffer_clear(cache.buffer, 0);
    ggml_free(ctx);
    
    return true;
}
```

---

## 7. Data Transfer: Host ↔ Backend

### Pattern: Efficient Data Movement

**Location**: `src/whisper.cpp:1922-1933` (Loading), `src/whisper.cpp:2402` (Input), `src/whisper.cpp:2954` (Output)

```cpp
// Loading weights from file
if (ggml_backend_buffer_is_host(tensor->buffer)) {
    // Direct read for CPU and Metal (Metal uses host-accessible memory)
    loader->read(loader->context, tensor->data, ggml_nbytes(tensor));
    BYTESWAP_TENSOR(tensor);
} else {
    // For non-host backends, read to temp buffer then copy
    read_buf.resize(ggml_nbytes(tensor));
    loader->read(loader->context, read_buf.data(), read_buf.size());
    ggml_backend_tensor_set(tensor, read_buf.data(), 0, ggml_nbytes(tensor));
}

// Setting input data
struct ggml_tensor * mel = ggml_graph_get_tensor(gf, "mel");
ggml_backend_tensor_set(mel, wstate.inp_mel.data(), 0, ggml_nelements(mel)*sizeof(float));

// Getting output data
ggml_backend_tensor_get(logits, logits_out.data() + (n_vocab*i), 
                        sizeof(float)*(n_vocab*i), sizeof(float)*n_vocab);
```

**Key Insights**:
- **Host buffer optimization**: Metal buffers are host-accessible, allowing direct reads
- **Offset support**: `ggml_backend_tensor_set/get` support offset and size parameters
- **Batch operations**: Can read/write partial tensor data efficiently
- **Named tensor lookup**: Uses `ggml_graph_get_tensor()` to find tensors by name

**Recommendation for segmentation-ggml**:
```cpp
// In inference
bool run_inference(
    SegmentationModel& model,
    SegmentationState& state,
    const float* audio_input,
    size_t n_frames,
    float* output_probs) {
    
    // Build graph
    auto graph = build_encoder_graph(model, state);
    
    // Allocate compute buffers
    if (!ggml_backend_sched_alloc_graph(state.encoder_sched, graph)) {
        return false;
    }
    
    // Set input data
    auto input_tensor = ggml_graph_get_tensor(graph, "audio_input");
    ggml_backend_tensor_set(
        input_tensor,
        audio_input,
        0,  // offset
        n_frames * model.n_features * sizeof(float)
    );
    
    // Compute
    if (ggml_backend_sched_graph_compute(state.encoder_sched, graph) != GGML_STATUS_SUCCESS) {
        return false;
    }
    
    // Get output data
    auto output_tensor = ggml_graph_get_tensor(graph, "encoder_output");
    ggml_backend_tensor_get(
        output_tensor,
        output_probs,
        0,  // offset
        n_frames * model.n_classes * sizeof(float)
    );
    
    return true;
}
```

---

## 8. State Management: Separate Model from Inference State

### Pattern: Context (Model) vs State (Inference)

**Location**: `src/whisper.cpp:937-952` (Context), `src/whisper.cpp:3374-3544` (State Init)

```cpp
// Model context (weights, architecture)
struct whisper_context {
    int64_t t_load_us  = 0;
    int64_t t_start_us = 0;
    
    ggml_type wtype = GGML_TYPE_F16;  // weight type
    ggml_type itype = GGML_TYPE_F16;  // intermediate type
    
    whisper_context_params params;
    
    whisper_model model;  // Weights
    whisper_vocab vocab;
    
    whisper_state * state = nullptr;  // Inference state (can be recreated)
    
    std::string path_model;
};

// Inference state (compute buffers, KV cache, schedulers)
struct whisper_state {
    std::vector<ggml_backend_t> backends;
    
    whisper_kv_cache kv_self;   // Self-attention cache
    whisper_kv_cache kv_cross;  // Cross-attention cache
    whisper_kv_cache kv_pad;    // Padding cache
    
    whisper_sched sched_conv;    // Scheduler for conv
    whisper_sched sched_encode;  // Scheduler for encoder
    whisper_sched sched_cross;   // Scheduler for cross-attention
    whisper_sched sched_decode;  // Scheduler for decoder
    
    std::vector<float> logits;   // Output buffer
    std::vector<float> inp_mel;  // Input buffer
    std::vector<float> inp_mask; // Mask buffer
    
    // ... more inference state
};

// Initialization separates model loading from state creation
struct whisper_context * whisper_init_with_params(
    struct whisper_model_loader * loader,
    struct whisper_context_params params) {
    
    // Load model (weights)
    whisper_context * ctx = whisper_init_with_params_no_state(loader, params);
    if (!ctx) {
        return nullptr;
    }
    
    // Create inference state
    ctx->state = whisper_init_state(ctx);
    if (!ctx->state) {
        whisper_free(ctx);
        return nullptr;
    }
    
    return ctx;
}
```

**Key Insights**:
- **Separation of concerns**: Model (immutable weights) vs State (mutable inference data)
- **Multiple states**: Can create multiple states for parallel inference
- **State recreation**: Can free and recreate state without reloading model
- **Resource management**: State owns backends, schedulers, and compute buffers

**Recommendation for segmentation-ggml**:
```cpp
// In segmentation_model.h
struct SegmentationModel {
    // Model architecture and weights (immutable after loading)
    std::vector<ggml_context*> weight_contexts;
    std::vector<ggml_backend_buffer_t> weight_buffers;
    std::map<std::string, ggml_tensor*> weights;
    
    int n_layers;
    int n_features;
    int n_classes;
    
    ggml_type weight_type;
};

struct SegmentationState {
    // Inference state (mutable, can be recreated)
    std::vector<ggml_backend_t> backends;
    
    ggml_backend_sched_t encoder_sched;
    std::vector<uint8_t> encoder_meta;
    
    std::vector<float> input_buffer;
    std::vector<float> output_buffer;
    
    // Optional: temporal cache for recurrent processing
    TemporalCache temporal_cache;
};

// Usage
SegmentationModel* load_model(const char* path, bool use_gpu) {
    auto model = new SegmentationModel();
    
    // Initialize backends
    auto backends = init_backends(use_gpu);
    
    // Load weights
    load_weights(model, path, backends);
    
    // Cleanup temporary backends (weights are now in buffers)
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
    
    return model;
}

SegmentationState* create_state(SegmentationModel* model, bool use_gpu) {
    auto state = new SegmentationState();
    
    // Initialize backends for inference
    state->backends = init_backends(use_gpu);
    
    // Initialize schedulers
    init_schedulers(*state, *model, state->backends);
    
    // Allocate buffers
    state->input_buffer.resize(model->n_features * MAX_FRAMES);
    state->output_buffer.resize(model->n_classes * MAX_FRAMES);
    
    return state;
}

// Can create multiple states for parallel inference
auto state1 = create_state(model, true);
auto state2 = create_state(model, true);
```

---

## 9. Initialization Flow: Complete Lifecycle

### Pattern: Multi-Stage Initialization

**Complete flow from `whisper_init_with_params()`**:

```
1. Load Model (whisper_init_with_params_no_state)
   ├─ Initialize backends list (for weight loading)
   ├─ Create buffer type priority list
   ├─ Create weight contexts (one per buffer type)
   ├─ Create weight tensors (metadata only, no_alloc=true)
   ├─ Allocate backend buffers for weights
   ├─ Load weight data from file
   └─ Mark buffers as GGML_BACKEND_BUFFER_USAGE_WEIGHTS

2. Create Inference State (whisper_init_state)
   ├─ Initialize backends (GPU + CPU)
   ├─ Initialize KV caches on backend[0]
   │  ├─ kv_self (self-attention)
   │  ├─ kv_cross (cross-attention)
   │  └─ kv_pad (padding)
   ├─ Initialize schedulers
   │  ├─ sched_conv: whisper_sched_graph_init(build_graph_conv)
   │  ├─ sched_encode: whisper_sched_graph_init(build_graph_encoder)
   │  ├─ sched_cross: whisper_sched_graph_init(build_graph_cross)
   │  └─ sched_decode: whisper_sched_graph_init(build_graph_decoder)
   └─ Allocate host buffers (logits, input, mask)

3. Inference (whisper_encode_internal, whisper_decode_internal)
   ├─ Build graph (pure function, temporary context)
   ├─ Allocate compute buffers (ggml_backend_sched_alloc_graph)
   ├─ Set input data (ggml_backend_tensor_set)
   ├─ Compute graph (ggml_backend_sched_graph_compute)
   ├─ Get output data (ggml_backend_tensor_get)
   └─ Reset scheduler (ggml_backend_sched_reset)

4. Cleanup (whisper_free)
   ├─ Free state
   │  ├─ Free KV cache buffers
   │  ├─ Free schedulers
   │  └─ Free backends
   ├─ Free model
   │  ├─ Free weight contexts
   │  └─ Free weight buffers
   └─ Delete context
```

**Recommendation for segmentation-ggml**:
```cpp
// Complete initialization flow
SegmentationModel* model = nullptr;
SegmentationState* state = nullptr;

// 1. Load model
model = load_model("model.gguf", use_gpu);
if (!model) {
    fprintf(stderr, "Failed to load model\n");
    return nullptr;
}

// 2. Create inference state
state = create_state(model, use_gpu);
if (!state) {
    fprintf(stderr, "Failed to create state\n");
    free_model(model);
    return nullptr;
}

// 3. Run inference
std::vector<float> audio_input = load_audio("audio.wav");
std::vector<float> output_probs(model->n_classes * n_frames);

if (!run_inference(model, state, audio_input.data(), n_frames, output_probs.data())) {
    fprintf(stderr, "Inference failed\n");
}

// 4. Cleanup
free_state(state);
free_model(model);
```

---

## 10. Best Practices Summary

### Critical Patterns for Metal-Ready Code

1. **Backend Initialization**
   - Always initialize backends in priority order: GPU → ACCEL → CPU
   - Keep CPU as fallback
   - Store backends in state, not model

2. **Memory Management**
   - Use `no_alloc=true` for weight contexts
   - Create separate contexts per buffer type
   - Allocate all tensors in a context together with `ggml_backend_alloc_ctx_tensors_from_buft()`
   - Mark weight buffers with `GGML_BACKEND_BUFFER_USAGE_WEIGHTS`

3. **Graph Building**
   - Keep graph building pure (no side effects)
   - Use temporary contexts for graph metadata
   - Name all input/output tensors
   - Free temporary context after building

4. **Scheduler Usage**
   - One scheduler per graph type
   - Pre-allocate during initialization
   - Reset scheduler between inferences
   - Let scheduler handle backend selection automatically

5. **Data Transfer**
   - Check `ggml_backend_buffer_is_host()` for optimization
   - Use `ggml_backend_tensor_set/get` for all data movement
   - Support offset and size parameters for partial transfers
   - Use named tensor lookup with `ggml_graph_get_tensor()`

6. **State Management**
   - Separate model (weights) from state (inference)
   - Allow multiple states per model
   - Store all mutable data in state
   - Make state recreation cheap

7. **Resource Cleanup**
   - Free in reverse order of creation
   - Free state before model
   - Free schedulers before backends
   - Free contexts after buffers

---

## 11. Code Snippets for segmentation-ggml

### Complete Minimal Example

```cpp
// segmentation_model.h
struct SegmentationModel {
    std::vector<ggml_context*> weight_contexts;
    std::vector<ggml_backend_buffer_t> weight_buffers;
    std::map<std::string, ggml_tensor*> weights;
    int n_layers, n_features, n_classes;
};

struct SegmentationState {
    std::vector<ggml_backend_t> backends;
    ggml_backend_sched_t sched;
    std::vector<uint8_t> meta;
    std::vector<float> input_buffer;
    std::vector<float> output_buffer;
};

// segmentation_model.cpp
SegmentationModel* load_model(const char* path, bool use_gpu) {
    auto model = new SegmentationModel();
    
    // 1. Initialize temporary backends for loading
    auto backends = init_backends(use_gpu);
    
    // 2. Create buffer type list
    auto buft_list = make_buft_list(backends);
    
    // 3. Create weight contexts and tensors
    std::map<ggml_backend_buffer_type_t, ggml_context*> ctx_map;
    
    for (auto& [name, shape, op] : get_weight_specs()) {
        auto buft = select_buffer_for_weight(shape, op, buft_list);
        auto ctx = get_or_create_context(ctx_map, buft);
        
        auto tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, shape[0], shape[1]);
        model->weights[name] = tensor;
    }
    
    // 4. Allocate buffers
    for (auto& [buft, ctx] : ctx_map) {
        auto buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        model->weight_buffers.push_back(buf);
        model->weight_contexts.push_back(ctx);
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    
    // 5. Load weight data
    load_weights_from_file(model, path);
    
    // 6. Cleanup temporary backends
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
    
    return model;
}

SegmentationState* create_state(SegmentationModel* model, bool use_gpu) {
    auto state = new SegmentationState();
    
    // 1. Initialize backends
    state->backends = init_backends(use_gpu);
    
    // 2. Initialize scheduler
    state->meta.resize(ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead());
    
    state->sched = ggml_backend_sched_new(
        state->backends.data(),
        nullptr,
        state->backends.size(),
        MAX_NODES,
        false,
        true
    );
    
    // 3. Pre-allocate by building graph once
    auto graph = build_graph(*model, *state);
    if (!ggml_backend_sched_alloc_graph(state->sched, graph)) {
        free_state(state);
        return nullptr;
    }
    ggml_backend_sched_reset(state->sched);
    
    // 4. Allocate host buffers
    state->input_buffer.resize(model->n_features * MAX_FRAMES);
    state->output_buffer.resize(model->n_classes * MAX_FRAMES);
    
    return state;
}

bool run_inference(
    SegmentationModel* model,
    SegmentationState* state,
    const float* input,
    size_t n_frames,
    float* output) {
    
    // 1. Build graph
    auto graph = build_graph(*model, *state);
    
    // 2. Allocate compute buffers
    if (!ggml_backend_sched_alloc_graph(state->sched, graph)) {
        return false;
    }
    
    // 3. Set input
    auto input_tensor = ggml_graph_get_tensor(graph, "input");
    ggml_backend_tensor_set(input_tensor, input, 0, n_frames * model->n_features * sizeof(float));
    
    // 4. Compute
    if (ggml_backend_sched_graph_compute(state->sched, graph) != GGML_STATUS_SUCCESS) {
        return false;
    }
    
    // 5. Get output
    auto output_tensor = ggml_graph_get_tensor(graph, "output");
    ggml_backend_tensor_get(output_tensor, output, 0, n_frames * model->n_classes * sizeof(float));
    
    return true;
}

ggml_cgraph* build_graph(SegmentationModel& model, SegmentationState& state) {
    ggml_init_params params = {
        .mem_size = state.meta.size(),
        .mem_buffer = state.meta.data(),
        .no_alloc = true
    };
    
    auto ctx = ggml_init(params);
    auto graph = ggml_new_graph(ctx);
    
    // Input
    auto input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, model.n_features, MAX_FRAMES);
    ggml_set_name(input, "input");
    ggml_set_input(input);
    
    // Build layers
    auto cur = input;
    for (int i = 0; i < model.n_layers; ++i) {
        cur = build_layer(ctx, model, cur, i);
    }
    
    // Output
    ggml_set_name(cur, "output");
    ggml_set_output(cur);
    ggml_build_forward_expand(graph, cur);
    
    ggml_free(ctx);
    return graph;
}

void free_state(SegmentationState* state) {
    if (state) {
        ggml_backend_sched_free(state->sched);
        for (auto backend : state->backends) {
            ggml_backend_free(backend);
        }
        delete state;
    }
}

void free_model(SegmentationModel* model) {
    if (model) {
        for (auto ctx : model->weight_contexts) {
            ggml_free(ctx);
        }
        for (auto buf : model->weight_buffers) {
            ggml_backend_buffer_free(buf);
        }
        delete model;
    }
}
```

---

## 12. Key Takeaways

1. **Whisper.cpp is production-ready**: It handles edge cases, provides good error messages, and has clean separation of concerns.

2. **Backend scheduler is powerful**: `ggml_backend_sched` handles all the complexity of multi-backend execution. Use it!

3. **Separate model from state**: This enables parallel inference and makes resource management cleaner.

4. **Pre-allocate everything**: Run allocation once during initialization to determine buffer sizes, then reuse.

5. **Name your tensors**: Makes debugging and tensor lookup much easier.

6. **Metal is first-class**: The code treats Metal as a primary backend, not an afterthought.

7. **Graph building is pure**: Keep graph construction separate from execution for clarity and reusability.

8. **Use buffer types wisely**: Per-tensor buffer type selection ensures optimal placement.

9. **Host buffers are special**: Metal buffers can be host-accessible, enabling direct reads/writes.

10. **Clean up in reverse order**: Free resources in the opposite order of creation to avoid use-after-free.

---

## Next Steps for segmentation-ggml

1. **Implement backend initialization** following the priority order pattern
2. **Create separate model and state structures** as shown above
3. **Implement buffer type selection** for optimal weight placement
4. **Use ggml_backend_sched** for all graph execution
5. **Add proper error handling** and logging like whisper.cpp
6. **Test on Metal** to ensure GPU acceleration works
7. **Profile and optimize** based on actual performance data

This architecture will give us a solid, Metal-ready foundation for the segmentation model!
