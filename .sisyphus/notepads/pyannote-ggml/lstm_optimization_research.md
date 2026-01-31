# LSTM Optimization Research for GGML-based Projects

**Research Date:** 2026-01-29  
**Objective:** Understand how GGML-based projects implement LSTM and recurrent operations efficiently without unrolling every timestep into the computation graph.

---

## Executive Summary

**Key Finding:** GGML does NOT have a native LSTM operation. However, GGML-based projects use **three main strategies** to handle recurrent operations efficiently:

1. **Manual LSTM implementation with state management** (whisper.cpp VAD)
2. **Native SSM operations for Mamba** (`ggml_ssm_conv`, `ggml_ssm_scan`)
3. **Native RWKV operations** (`ggml_rwkv_wkv6`, `ggml_rwkv_wkv7`)
4. **Custom operations** (`ggml_map_custom*`, `ggml_custom_4d`)

---

## 1. Does GGML have a native LSTM op?

**Answer: NO**

### Evidence from GGML source code

Searched `/Users/andyye/dev/pyannote-audio/ggml/include/ggml.h` and found:
- No `GGML_OP_LSTM` operation
- No built-in LSTM layer

### What GGML DOES have for recurrent models:

```c
// From ggml.h lines 549-557
GGML_OP_SSM_CONV,      // Mamba convolution
GGML_OP_SSM_SCAN,      // Mamba state space scan
GGML_OP_RWKV_WKV6,     // RWKV v6 operation
GGML_OP_RWKV_WKV7,     // RWKV v7 operation
```

---

## 2. How does whisper.cpp handle LSTM? (Silero VAD)

**Source:** `/tmp/whisper.cpp/src/whisper.cpp` lines 4570-4610

### Strategy: Manual LSTM implementation with persistent state tensors

#### Key Components:

1. **State tensors stored in context:**
```c
struct whisper_vad_context {
    struct ggml_tensor * h_state;  // Hidden state
    struct ggml_tensor * c_state;  // Cell state
};
```

2. **LSTM computation built per timestep:**
```c
static struct ggml_tensor * whisper_vad_build_lstm_layer(
    struct ggml_context * ctx0,
    whisper_vad_context & vctx,
    struct ggml_tensor * cur,
    struct ggml_cgraph * gf) {
    
    const int hdim = vctx.model.hparams.lstm_hidden_size;
    
    // Input-to-hidden transformation
    struct ggml_tensor * inp_gate = ggml_mul_mat(ctx0, model.lstm_ih_weight, x_t);
    inp_gate = ggml_add(ctx0, inp_gate, model.lstm_ih_bias);
    
    // Hidden-to-hidden transformation
    struct ggml_tensor * hid_gate = ggml_mul_mat(ctx0, model.lstm_hh_weight, vctx.h_state);
    hid_gate = ggml_add(ctx0, hid_gate, model.lstm_hh_bias);
    
    // Combined gates
    struct ggml_tensor * out_gate = ggml_add(ctx0, inp_gate, hid_gate);
    
    // Extract gates using views (i, f, g, o)
    struct ggml_tensor * i_t = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, out_gate, hdim, 0 * hdim_size));
    struct ggml_tensor * f_t = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, out_gate, hdim, 1 * hdim_size));
    struct ggml_tensor * g_t = ggml_tanh(ctx0, ggml_view_1d(ctx0, out_gate, hdim, 2 * hdim_size));
    struct ggml_tensor * o_t = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, out_gate, hdim, 3 * hdim_size));
    
    // Update cell state
    struct ggml_tensor * c_out = ggml_add(ctx0,
        ggml_mul(ctx0, f_t, vctx.c_state),
        ggml_mul(ctx0, i_t, g_t));
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, c_out, vctx.c_state));
    
    // Update hidden state
    struct ggml_tensor * out = ggml_mul(ctx0, o_t, ggml_tanh(ctx0, c_out));
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, out, vctx.h_state));
    
    return out;
}
```

#### Key Insight: **State Management Pattern**

The LSTM is NOT unrolled. Instead:
1. **State tensors persist** across graph executions
2. **Graph is built once** for a single timestep
3. **Graph is executed repeatedly** in a loop
4. **States are updated in-place** using `ggml_cpy`

#### Execution Pattern:

```c
// Pseudocode for VAD processing
for (int frame_idx = 0; frame_idx < n_frames; frame_idx++) {
    // Build graph for single timestep (reuses same graph structure)
    ggml_cgraph * gf = whisper_vad_build_graph(vctx);
    
    // Set input for this frame
    ggml_backend_tensor_set(frame_tensor, audio_data[frame_idx], ...);
    
    // Execute graph (updates h_state and c_state in-place)
    ggml_backend_sched_graph_compute(vctx->sched.sched, gf);
    
    // Read output
    float prob = read_output(gf);
}
```

**This avoids unrolling** because:
- Graph represents ONE timestep
- States are persistent tensors (allocated once)
- Loop is in C++ code, not in the graph

---

## 3. How does llama.cpp handle Mamba? (SSM operations)

**Source:** `/tmp/llama.cpp/src/models/graph-context-mamba.cpp`

### Strategy: Native SSM operations with state caching

#### Native GGML Operations:

```c
// From ggml.h lines 2354-2367
GGML_API struct ggml_tensor * ggml_ssm_conv(
    struct ggml_context * ctx,
    struct ggml_tensor  * sx,
    struct ggml_tensor  * c);

GGML_API struct ggml_tensor * ggml_ssm_scan(
    struct ggml_context * ctx,
    struct ggml_tensor  * s,      // state
    struct ggml_tensor  * x,      // input
    struct ggml_tensor  * dt,     // delta time
    struct ggml_tensor  * A,      // state transition
    struct ggml_tensor  * B,      // input projection
    struct ggml_tensor  * C,      // output projection
    struct ggml_tensor  * ids);   // sequence IDs
```

#### Mamba Layer Implementation:

```cpp
// From /tmp/llama.cpp/src/models/graph-context-mamba.cpp lines 50-78
// 1D Convolution with state
x = ggml_ssm_conv(ctx0, conv_x, layer.ssm_conv1d);

// SSM scan operation (parallel associative scan)
ggml_tensor * y_ssm = ggml_ssm_scan(ctx, ssm, x, dt, A, B, C, ids);

// Store updated states back
ggml_build_forward_expand(
    gf, ggml_cpy(ctx0, 
        ggml_view_1d(ctx0, y_ssm, d_state * d_inner * n_seqs, ...),
        ggml_view_1d(ctx0, ssm_states_all, ...)));
```

#### Key Features:

1. **`ggml_ssm_scan` implements parallel associative scan** (Annex D of Mamba paper)
2. **Handles multiple sequences simultaneously** (`n_seqs` dimension)
3. **State management similar to KV cache** in transformers
4. **Optimized CUDA/Metal kernels** for SSM operations

**Files:**
- `/tmp/llama.cpp/ggml/src/ggml-cuda/ssm-scan.cu`
- `/tmp/llama.cpp/ggml/src/ggml-cuda/ssm-conv.cu`
- `/tmp/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/ssm_scan.comp`

---

## 4. How does GGML handle RWKV?

**Source:** `/Users/andyye/dev/pyannote-audio/ggml/include/ggml.h` lines 2419-2445

### Strategy: Native RWKV operations

```c
GGML_API struct ggml_tensor * ggml_rwkv_wkv6(
    struct ggml_context * ctx,
    struct ggml_tensor  * k,
    struct ggml_tensor  * v,
    struct ggml_tensor  * r,
    struct ggml_tensor  * tf,
    struct ggml_tensor  * td,
    struct ggml_tensor  * state);

GGML_API struct ggml_tensor * ggml_rwkv_wkv7(
    struct ggml_context * ctx,
    struct ggml_tensor  * r,
    struct ggml_tensor  * w,
    struct ggml_tensor  * k,
    struct ggml_tensor  * v,
    struct ggml_tensor  * a,
    struct ggml_tensor  * b,
    struct ggml_tensor  * state);
```

**Key Insight:** RWKV operations are **fused into single ops** that handle the recurrent state update internally.

---

## 5. Custom Operation Mechanism

**Source:** `/Users/andyye/dev/pyannote-audio/ggml/include/ggml.h` lines 2470-2547

### GGML provides custom op registration:

```c
typedef void (*ggml_custom1_op_t)(
    struct ggml_tensor * dst,
    const struct ggml_tensor * a,
    int ith, int nth,
    void * userdata);

GGML_API struct ggml_tensor * ggml_map_custom1(
    struct ggml_context   * ctx,
    struct ggml_tensor    * a,
    ggml_custom1_op_t       fun,
    int                     n_tasks,
    void                  * userdata);

// Also: ggml_map_custom2, ggml_map_custom3, ggml_custom_4d
```

**Use case:** You can define a custom operation that internally loops over timesteps.

---

## 6. Imperative Execution Pattern

### Can you build and execute a small graph in a loop?

**YES** - This is exactly what whisper.cpp VAD does.

#### Pattern:

```c
// 1. Allocate persistent state tensors
ggml_tensor * h_state = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
ggml_tensor * c_state = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
ggml_backend_tensor_set(h_state, zeros, ...);  // Initialize to zero
ggml_backend_tensor_set(c_state, zeros, ...);

// 2. Build graph for ONE timestep
ggml_cgraph * build_lstm_step_graph() {
    ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, input_size);
    ggml_set_input(input);
    
    // LSTM computation using h_state, c_state
    ggml_tensor * output = lstm_step(input, h_state, c_state);
    
    // Update states in-place
    ggml_build_forward_expand(gf, ggml_cpy(ctx, new_h, h_state));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, new_c, c_state));
    
    ggml_set_output(output);
    return gf;
}

// 3. Execute in loop
for (int t = 0; t < n_timesteps; t++) {
    ggml_backend_tensor_set(input, data[t], ...);
    ggml_backend_sched_graph_compute(sched, gf);
    ggml_backend_tensor_get(output, result[t], ...);
}
```

#### Does `ggml_backend_sched` support this pattern?

**YES** - The scheduler can handle:
- Persistent tensors that survive across graph executions
- In-place updates via `ggml_cpy`
- Multiple graph executions with different input data

---

## 7. Recommendations for pyannote-ggml LSTM Implementation

### Option A: Follow whisper.cpp VAD pattern (RECOMMENDED)

**Pros:**
- Proven to work in production
- No need for custom GGML ops
- Clear separation of concerns
- Easy to debug

**Implementation:**
```c
// 1. Create persistent state tensors in model context
struct pyannote_lstm_state {
    ggml_tensor * h_state;  // [batch, hidden_size]
    ggml_tensor * c_state;  // [batch, hidden_size]
};

// 2. Build graph for single timestep
ggml_cgraph * build_lstm_step(
    ggml_context * ctx,
    pyannote_lstm_state * state,
    ggml_tensor * input,
    ggml_tensor * weights_ih,
    ggml_tensor * weights_hh,
    ggml_tensor * bias_ih,
    ggml_tensor * bias_hh) {
    
    // Compute gates (i, f, g, o)
    ggml_tensor * gates_i = ggml_mul_mat(ctx, weights_ih, input);
    gates_i = ggml_add(ctx, gates_i, bias_ih);
    
    ggml_tensor * gates_h = ggml_mul_mat(ctx, weights_hh, state->h_state);
    gates_h = ggml_add(ctx, gates_h, bias_hh);
    
    ggml_tensor * gates = ggml_add(ctx, gates_i, gates_h);
    
    // Split gates and apply activations
    // ... (similar to whisper.cpp)
    
    // Update states
    ggml_build_forward_expand(gf, ggml_cpy(ctx, new_c, state->c_state));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, new_h, state->h_state));
    
    return new_h;
}

// 3. Process sequence
for (int t = 0; t < seq_len; t++) {
    ggml_backend_tensor_set(input, sequence[t], ...);
    ggml_backend_sched_graph_compute(sched, gf);
    ggml_backend_tensor_get(output, results[t], ...);
}
```

### Option B: Create custom LSTM operation

**Pros:**
- Single operation in graph
- Potentially faster (fused kernel)
- Cleaner graph visualization

**Cons:**
- Need to implement custom CUDA/Metal kernels
- More complex to maintain
- Harder to debug

**Implementation:**
```c
// Define custom LSTM op
void lstm_forward(
    ggml_tensor * dst,
    const ggml_tensor * input,
    const ggml_tensor * h_prev,
    const ggml_tensor * c_prev,
    int ith, int nth,
    void * userdata) {
    
    // Implement LSTM forward pass
    // This runs on CPU/GPU depending on backend
}

// Use in graph
ggml_tensor * output = ggml_map_custom3(
    ctx, input, h_prev, c_prev,
    lstm_forward,
    GGML_N_TASKS_MAX,
    &lstm_params);
```

### Option C: Batch processing with padding

**For fixed-length sequences:**
```c
// Process entire sequence as batch
// Input: [batch, seq_len, input_size]
// Output: [batch, seq_len, hidden_size]

// Build graph that processes all timesteps
// (This DOES unroll, but only once during graph construction)
for (int t = 0; t < max_seq_len; t++) {
    h[t] = lstm_step(input[t], h[t-1], c[t-1]);
}
```

**Pros:**
- Simple to implement
- Good for fixed-length sequences

**Cons:**
- Graph size grows with sequence length
- Memory usage increases
- Not suitable for variable-length sequences

---

## 8. Key Takeaways

### ✅ What Works:

1. **Persistent state tensors** - Allocate once, update in-place
2. **Graph per timestep** - Build small graph, execute in loop
3. **`ggml_cpy` for state updates** - In-place modification
4. **Backend scheduler handles it** - No special configuration needed

### ❌ What to Avoid:

1. **Unrolling entire sequence** - Graph becomes huge
2. **Recreating graph each step** - Overhead is significant
3. **Copying states unnecessarily** - Use views and in-place ops

### 🎯 Best Practice:

**Follow the whisper.cpp VAD pattern:**
- Persistent state tensors in model context
- Graph represents single timestep
- C++ loop for sequence iteration
- In-place state updates with `ggml_cpy`

---

## 9. References

### Source Code Locations:

1. **whisper.cpp LSTM (Silero VAD):**
   - `/tmp/whisper.cpp/src/whisper.cpp` lines 4570-4720
   - `/tmp/whisper.cpp/src/whisper-arch.h` lines 143-196

2. **llama.cpp Mamba:**
   - `/tmp/llama.cpp/src/models/mamba.cpp`
   - `/tmp/llama.cpp/src/models/graph-context-mamba.cpp`
   - `/tmp/llama.cpp/ggml/src/ggml-cuda/ssm-scan.cu`

3. **GGML Operations:**
   - `/Users/andyye/dev/pyannote-audio/ggml/include/ggml.h`
   - Lines 549-557: SSM/RWKV ops
   - Lines 2354-2367: SSM functions
   - Lines 2419-2445: RWKV functions
   - Lines 2470-2547: Custom ops

### Related Projects:

- **RWKV.cpp:** https://github.com/RWKV/rwkv.cpp
- **whisper.cpp:** https://github.com/ggerganov/whisper.cpp
- **llama.cpp:** https://github.com/ggerganov/llama.cpp

---

## 10. Next Steps for pyannote-ggml

1. **Implement LSTM following whisper.cpp pattern**
   - Create `pyannote_lstm_state` struct
   - Implement `build_lstm_step_graph()`
   - Add state management to model context

2. **Test with small sequence**
   - Verify state persistence
   - Check memory usage
   - Profile performance

3. **Consider batching strategy**
   - How to handle variable-length sequences?
   - Padding vs. dynamic batching

4. **Optimize if needed**
   - Profile bottlenecks
   - Consider custom op if LSTM is critical path
   - Implement CUDA kernel if necessary

---

**Research completed:** 2026-01-29  
**Researcher:** Claude (Librarian agent)
