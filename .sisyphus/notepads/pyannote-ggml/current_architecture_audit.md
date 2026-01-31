# Current Architecture Audit: segmentation-ggml

This document identifies patterns in the `segmentation-ggml` codebase that differ from modern GGML best practices.

## 1. Model Loading Patterns
- **Legacy Pattern**: `model_load` in `model.cpp` uses `gguf_init_from_file` with `no_alloc = false`.
- **Deviation**: This approach allocates tensor data directly within the GGUF context. Modern GGML best practice is to use `no_alloc = true`, then use `ggml_backend_alloc_ctx_tensors` to allocate weights on a specific backend (CPU, Metal, CUDA).
- **Impact**: Harder to support multiple backends or optimize weight memory placement.

## 2. Graph Computation
- **Legacy Pattern**: `main.cpp` uses `ggml_graph_compute_with_ctx(ctx, graph, n_threads)`.
- **Deviation**: This is a deprecated function. The modern way is to use `ggml_backend_graph_compute(backend, graph)`.
- **Impact**: Computation is locked to the legacy CPU path and cannot easily leverage `ggml-backend` optimizations or hardware acceleration.

## 3. Memory Management
- **Legacy Pattern**: Fixed-size context allocation. In `main.cpp`, the compute context is initialized with a hardcoded size (e.g., 4GB for the full forward pass).
- **Deviation**: Modern implementations use `ggml-alloc` (or `ggml_backend_sched`) to measure the graph and allocate exactly the amount of memory needed for intermediate tensors.
- **Impact**: Extremely high and inefficient memory usage. 4GB for a ~1.5M parameter model is excessive.

## 4. Forward Pass Structure
- **LSTM Unrolling**: `lstm_layer_unidirectional` in `lstm.cpp` uses `ggml_concat` in a loop to build the output sequence.
- **Deviation**: Frequent use of `ggml_concat` creates many nodes in the computation graph and involves many memory copies. 
- **Best Practice**: Pre-allocate the output tensor and use `ggml_view` or `ggml_cpy` to fill it, or use a single `ggml_concat` at the end if necessary.
- **Transposes**: `linear_forward` and `classifier_forward` perform explicit transposes and reshapes for every call. While sometimes necessary, GGML's `ggml_mul_mat` often handles implicit transposes more efficiently if tensors are shaped correctly from the start.

## 5. Backend Abstraction
- **Status**: The code includes `ggml-backend.h` but doesn't actually use any of its abstractions (`ggml_backend_t`, `ggml_backend_buffer_t`, etc.).
- **Impact**: GPU support (Metal) is "reserved" but requires a significant refactor to use the backend-agnostic API.

## Detailed Inventory of Key Calls

### `ggml_init` (Context Creation)
- `model.cpp:103` (via `gguf_init_from_file`): Creates weight context.
- `main.cpp:107`, `218`, `381`, `565`, `695`: Creates compute contexts with manual sizes (256MB to 4GB).

### `ggml_graph_compute*`
- `main.cpp:158`, `289`, `450`, `619`, `720`: All use `ggml_graph_compute_with_ctx`.

### `ggml_new_tensor*` for Input/State
- `main.cpp:118`, `229`, `392`, `577`, `701`: Uses `ggml_new_tensor_3d` for input waveform/features.
- `lstm.cpp:46-47`: Uses `ggml_new_tensor_2d` for hidden/cell state initialization.

### Data Readback
- `main.cpp:168`, `299`, `464`, `630`: Uses `ggml_get_data_f32` to read results directly from compute tensors.

## Summary of Modernization Needs
1. Switch to `ggml_backend_t` for both CPU and Metal.
2. Use `ggml_backend_alloc_ctx_tensors` for weights.
3. Use `ggml_backend_sched` (scheduler) for compute memory management.
4. Replace `ggml_graph_compute_with_ctx` with `ggml_backend_graph_compute`.
5. Optimize LSTM graph construction to avoid excessive `ggml_concat`.
