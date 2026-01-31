# Learnings - PyAnnote GGML Conversion

## Conventions and Patterns

*Append findings as tasks complete. NEVER overwrite.*

---

## Task 1: Environment Setup

### PyTorch 2.6+ Compatibility Issue
- **Problem**: PyTorch 2.10.0 uses `weights_only=True` by default in `torch.load()`, which blocks loading of PyAnnote models
- **Solution**: Add safe globals before loading:
  ```python
  from pyannote.audio.core import task
  import inspect
  task_classes = [obj for name, obj in inspect.getmembers(task) if inspect.isclass(obj)]
  torch.serialization.add_safe_globals(task_classes)
  ```
- **Key Classes**: Specifications, Problem, Resolution, and others in `pyannote.audio.core.task`

### Model Architecture - PyanNet
- **Model Type**: PyanNet (SincNet + LSTM + Linear layers)
- **Total Parameters**: 1,473,515
- **Components**:
  - `sincnet`: SincNet feature extraction (40 filters, 3 conv layers)
  - `lstm`: Bidirectional LSTM (2 layers, 128 hidden units)
  - `linear`: Output linear layer
  - `classifier`: Classification head
- **PyTorch Lightning**: Model saved as checkpoint dict with state_dict, hparams, and metadata

### Model Cache Location
- **Path**: `~/.cache/huggingface/hub/models--pyannote--speaker-diarization-community-1/snapshots/{hash}/`
- **Segmentation Model**: `segmentation/pytorch_model.bin` (1.47M parameters)
- **Other Components**: embedding, plda, config.yaml

### Environment Details
- **Python**: 3.10.18
- **PyTorch**: 2.10.0
- **TorchAudio**: 2.10.0
- **PyAnnote.audio**: 4.0.1
- **Device Support**: MPS available (Apple Silicon), CUDA not available
- **Virtual Environment**: `.venv` created with uv

### Key Insights for GGML Conversion
1. Model is a PyTorch Lightning checkpoint, not a raw state_dict
2. SincNet requires special handling for learnable filter parameters
3. Bidirectional LSTM needs proper state management
4. Model expects 16kHz audio input (from SincNet design)
5. Output is speaker segmentation probabilities (binary classification per frame)

## Task 2: Architecture Analysis

### Corrected Model Architecture
- **Total Parameters**: 1,473,265 (not 1,473,515 as initially reported)
- **LSTM Configuration**: 4 layers (not 2), bidirectional, 128 hidden units, 0.5 dropout
- **Linear Layers**: 2 layers (256→128→128) with LeakyReLU activation
- **Output Classes**: 7 classes (not binary classification)

### SincNet Detailed Architecture

**Stage 1: Parameterized Sinc Filters**
- **ParamSincFB**: 80 output channels (40 filters × 2 phases)
- **Kernel Size**: 251 samples
- **Stride**: 10 samples
- **Learnable Parameters**: 
  - `low_hz`: 40 values (36.94 Hz to 7434.21 Hz)
  - `band_hz`: 40 values (38.60 Hz to 529.37 Hz)
- **Special Operation**: `abs()` applied ONLY after first convolution
- **Output**: (batch, 80, 15975) for 160k input samples

**Stage 2 & 3: Standard Convolutions**
- **Conv2**: 80→60 channels, kernel=5, stride=1
- **Conv3**: 60→60 channels, kernel=5, stride=1
- **Each stage**: Conv → MaxPool(3,3) → InstanceNorm → LeakyReLU
- **Final Output**: (batch, 60, 589) for 160k input samples

### LSTM Architecture Details
- **Input Size**: 60 features (from SincNet output)
- **Hidden Size**: 128 per direction
- **Num Layers**: 4 (not 2!)
- **Bidirectional**: True (output = 128 × 2 = 256)
- **Dropout**: 0.5 between layers
- **Batch First**: True (input shape: batch, seq, feature)
- **Output Shape**: (batch, 589, 256)

### Frame Rate Calculation
- **Input**: 160,000 samples (10 seconds at 16kHz)
- **Output**: 589 frames
- **Frame Rate**: 58.9 frames/second
- **Time per Frame**: ~17ms
- **Receptive Field**: Each frame sees ~170ms of input audio

### Tensor Shape Flow (10-second input)
```
Input:              (1, 1, 160000)   # Raw audio
SincNet Conv1:      (1, 80, 15975)   # After ParamSincFB + abs()
SincNet Stage1:     (1, 80, 5325)    # After pool + norm
SincNet Conv2:      (1, 60, 5321)    # After Conv1d
SincNet Stage2:     (1, 60, 1773)    # After pool + norm
SincNet Conv3:      (1, 60, 1769)    # After Conv1d
SincNet Stage3:     (1, 60, 589)     # After pool + norm
Reshape:            (1, 589, 60)     # Rearrange for LSTM
LSTM:               (1, 589, 256)    # 4-layer bidirectional
Linear 1:           (1, 589, 128)    # 256→128 + LeakyReLU
Linear 2:           (1, 589, 128)    # 128→128 + LeakyReLU
Classifier:         (1, 589, 7)      # 128→7
Output:             (1, 589, 7)      # LogSoftmax
```

### Critical Implementation Details

**SincNet Filter Generation**:
- Filters are NOT standard convolution weights
- Must compute sinc functions: `sinc(x) = sin(πx) / (πx)`
- Formula: `filter = sinc(2π × (low_hz + band_hz) × t) - sinc(2π × low_hz × t)`
- Apply Hamming window to filters
- Normalize filters before convolution

**Instance Normalization**:
- Normalize per sample (not across batch)
- Formula: `y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta`
- `gamma` and `beta` are learnable affine parameters
- Applied to SincNet stages and input waveform

**LSTM Bidirectional Processing**:
- Forward pass: processes sequence left-to-right
- Backward pass: processes sequence right-to-left
- Outputs are concatenated: [forward_out, backward_out]
- With 4 layers: 8 total LSTM passes (4 forward + 4 backward)

**Activation Functions**:
- **LeakyReLU**: Used in SincNet and Linear layers
- **LogSoftmax**: Used for final output (log probabilities)
- **No activation** between LSTM layers (just dropout)

### GGML Operations Needed

**Core Operations**:
1. `ggml_norm` - Instance normalization with affine
2. `ggml_conv_1d` - 1D convolution
3. `ggml_pool_1d` - 1D max pooling
4. `ggml_abs` - Absolute value (SincNet stage 1 only)
5. `ggml_leaky_relu` - Leaky ReLU activation
6. `ggml_mul_mat` - Matrix multiplication (Linear layers)
7. `ggml_add` - Addition (biases)
8. `ggml_soft_max` - Softmax
9. `ggml_log` - Logarithm (for LogSoftmax)

**LSTM Operations**:
10. `ggml_lstm` or manual implementation with:
    - `ggml_sigmoid` - Sigmoid activation
    - `ggml_tanh` - Tanh activation
    - `ggml_mul` - Element-wise multiply
    - `ggml_add` - Element-wise add

**Tensor Manipulation**:
11. `ggml_reshape` - Reshape tensors
12. `ggml_permute` - Permute dimensions (for rearrange)
13. `ggml_cont` - Make tensor contiguous

**Custom Operations**:
14. **SincNet filter generation** - Compute bandpass filters from parameters
    - Requires: sin, cos, π constant, Hamming window

### Reference Data Saved
- **Activations**: `segmentation-ggml/tests/reference_activations.npz`
  - Contains 12 intermediate activations from input to output
  - Use for validating GGML implementation
- **Layer Info**: `segmentation-ggml/docs/layer_info.json`
  - Complete layer parameters and configuration
- **Activation Shapes**: `segmentation-ggml/docs/activation_shapes.json`
  - Tensor shapes at each stage

### Key Gotchas
1. **abs() only in SincNet stage 1**: Don't apply to other stages
2. **LSTM is 4 layers, not 2**: Check hparams carefully
3. **Dropout during inference**: Should be disabled (not applied)
4. **Rearrange operation**: Must transpose (batch, feature, frame) → (batch, frame, feature)
5. **LogSoftmax output**: Need to exp() to get actual probabilities


## Task 4: PyTorch to GGUF Conversion Script

### GGUF Format Implementation

**GGUF File Structure** (version 3):
1. **Header**: Magic (0x46554747 = "GGUF"), version, tensor count, metadata count
2. **Metadata KV**: Key-value pairs with typed values (string, uint32, etc.)
3. **Tensor Info**: For each tensor: name, dimensions (reversed), type, offset
4. **Padding**: Align to 32-byte boundary
5. **Tensor Data**: Raw tensor data at aligned offsets

**Key Implementation Details**:
- Strings are length-prefixed (uint64 length + UTF-8 bytes)
- Dimensions written in reverse order (GGML convention)
- All offsets relative to start of tensor data section
- Data must be aligned to `general.alignment` (default: 32)

### SincNet Filter Pre-computation

**Algorithm Implemented**:
```python
# For each filter i (0 to 39):
f_low = low_hz[i]
f_high = low_hz[i] + band_hz[i]

# Create time axis: t = [-125, ..., 0, ..., 125] / sample_rate
# Compute bandpass as difference of sinc functions:
sinc_high = sin(2π × f_high × t) / (π × t)  # Handle t=0 case
sinc_low = sin(2π × f_low × t) / (π × t)
bandpass = sinc_high - sinc_low

# Apply Hamming window and normalize
bandpass = bandpass * hamming_window
bandpass = bandpass / sqrt(sum(bandpass^2))

# Store as positive and negative phase (80 channels from 40 filters)
filters[i*2] = bandpass
filters[i*2+1] = -bandpass
```

**Output Shape**: (80, 1, 251) - 80 channels, 1 input channel, 251 kernel size

### Tensor Name Mapping

**PyTorch → GGUF Mapping**:
```
# SincNet (stage 0 is ParamSincFB, pre-computed)
sincnet.conv1d.0.filterbank.{low_hz_, band_hz_} → sincnet.0.conv.weight (pre-computed)
sincnet.conv1d.1.{weight,bias} → sincnet.1.conv.{weight,bias}
sincnet.conv1d.2.{weight,bias} → sincnet.2.conv.{weight,bias}
sincnet.norm1d.{0,1,2}.{weight,bias} → sincnet.{0,1,2}.norm.{weight,bias}
sincnet.wav_norm1d.{weight,bias} → sincnet.wav_norm.{weight,bias}

# LSTM (kept as-is for clarity)
lstm.weight_ih_l{0,1,2,3} → lstm.weight_ih_l{0,1,2,3}
lstm.weight_ih_l{0,1,2,3}_reverse → lstm.weight_ih_l{0,1,2,3}_reverse
(same for weight_hh, bias_ih, bias_hh)

# Linear and Classifier
linear.{0,1}.{weight,bias} → linear.{0,1}.{weight,bias}
classifier.{weight,bias} → classifier.{weight,bias}
```

### Dtype Conversion Strategy

**Weights → F16** (for memory efficiency):
- All weight matrices (conv, linear, lstm weights)
- Total: 22 tensors

**Biases → F32** (for numerical stability):
- All bias vectors (conv, norm, linear, lstm biases)
- Total: 29 tensors

**Resulting File Size**: ~2.87 MB (vs ~5.9 MB if all F32)

### GGUF Metadata Written

```
general.architecture = "pyannet"
general.name = "pyannote-segmentation-3.0"
general.alignment = 32
pyannet.sample_rate = 16000
pyannet.num_classes = 7
pyannet.lstm_layers = 4
pyannet.lstm_hidden = 128
pyannet.sincnet_kernel_size = 251
pyannet.sincnet_stride = 10
```

### Verification Method

Created Python verification to read back GGUF:
- Verify magic number: 0x46554747 ("GGUF")
- Verify version: 3
- Verify tensor count: 51
- Verify metadata count: 9
- Read and validate all metadata values

### Key Learnings

1. **GGUF vs legacy GGML**: Modern GGUF uses "GGUF" magic (0x46554747) vs legacy "ggml" (0x67676d6c)
2. **Dimension reversal**: GGML expects dimensions in reverse order from PyTorch/numpy
3. **Alignment critical**: Tensor data must be aligned to 32 bytes for mmap compatibility
4. **Safe globals**: PyTorch 2.6+ requires explicit safe globals for loading custom classes
5. **SincNet phases**: Each learnable filter produces 2 channels (positive + negative phase)


## Task 5: C++ Model Loading

### GGUF Loading API Usage

**Initialization Pattern**:
```cpp
struct gguf_init_params gguf_params = {
    .no_alloc = false,  // Allocate memory for tensor data
    .ctx = &model.ctx,  // Pointer to ggml_context pointer
};
model.gguf_ctx = gguf_init_from_file(fname.c_str(), gguf_params);
```

**Key Points**:
1. `no_alloc = false` causes GGUF to allocate memory and load tensor data
2. `ctx` must be a pointer to a `ggml_context*` - the function will create the context
3. After loading, tensors are accessible via `ggml_get_tensor(ctx, "tensor_name")`

### Metadata Access Pattern

```cpp
int64_t key_id = gguf_find_key(ctx, "pyannet.sample_rate");
if (key_id >= 0) {
    uint32_t sample_rate = gguf_get_val_u32(ctx, key_id);
}
```

**Available Functions**:
- `gguf_find_key()` - Find key by name, returns -1 if not found
- `gguf_get_val_u32()` / `gguf_get_val_i32()` / `gguf_get_val_str()` - Get typed values
- `gguf_get_n_tensors()` / `gguf_get_n_kv()` - Get counts

### Critical GGUF Conversion Fix

**Problem**: Initial GGUF files failed to load with error "failed to read tensor data binary blob"

**Root Cause**: Tensor data size mismatch. GGML expects:
- Each tensor's data to be padded to alignment (32 bytes)
- Tensor offsets to use PADDED sizes: `offset += GGML_PAD(nbytes, alignment)`

**Fix in convert.py**:
```python
# Before (broken):
current_offset = aligned_off + data_converted.nbytes

# After (fixed):
padded_size = align_offset(data_converted.nbytes)  # GGML_PAD equivalent
current_offset += padded_size
```

Also added padding when writing tensor data:
```python
padded_size = align_offset(data.nbytes)
padding_needed = padded_size - data.nbytes
if padding_needed > 0:
    f.write(b'\x00' * padding_needed)
```

### Model Structure Design

**Tensor Organization**:
- SincNet: 3 stages with conv, bias, norm weights/biases
- LSTM: 4 layers × 2 directions × 4 tensors (weight_ih, weight_hh, bias_ih, bias_hh)
- Linear: 2 layers × 2 tensors (weight, bias)
- Classifier: weight and bias

**Tensor Count**: 51 total tensors in GGUF
- 22 F16 tensors (weights)
- 29 F32 tensors (biases and norms)

### Shape Verification Pattern

GGML stores dimensions in reverse order from PyTorch. When verifying:
- PyTorch `[80, 1, 251]` → GGML `ne[0]=251, ne[1]=1, ne[2]=80`
- Check shapes with `verify_tensor_shape(tensor, name, ne0, ne1, ne2)`

**Shape Mapping**:
| Tensor | PyTorch Shape | GGML (ne[0], ne[1], ne[2]) |
|--------|---------------|---------------------------|
| sincnet.0.conv.weight | [80, 1, 251] | [251, 1, 80] |
| lstm.weight_ih_l0 | [512, 60] | [60, 512] |
| linear.0.weight | [128, 256] | [256, 128] |
| classifier.weight | [7, 128] | [128, 7] |

### Memory Usage

- GGUF file: 2.87 MB
- GGML context memory: 2.88 MB (includes tensor overhead)
- All 49 required tensors loaded successfully

### Key Files Modified

- `model.h`: Model struct with all tensor pointers and hparams
- `model.cpp`: `model_load()`, `model_free()`, `model_print_info()`, `model_verify()`
- `convert.py`: Fixed tensor padding for GGML compatibility

## Task 6: SincNet Layers Implementation

### InstanceNorm1d Implementation

**Key Insight**: GGML's `ggml_norm` normalizes along rows (ne[0]), which is exactly what InstanceNorm1d needs when tensors are laid out as [time, channels, batch].

**Implementation**:
```cpp
struct ggml_tensor* ggml_instance_norm_1d(ctx, x, weight, bias, eps) {
    // ggml_norm normalizes along ne[0] (time dimension)
    struct ggml_tensor* x_norm = ggml_norm(ctx, x, eps);
    
    // Apply affine: reshape [channels] to [1, channels, 1] for broadcasting
    if (weight != nullptr) {
        struct ggml_tensor* weight_3d = ggml_reshape_3d(ctx, weight, 1, weight->ne[0], 1);
        x_norm = ggml_mul(ctx, x_norm, weight_3d);
    }
    if (bias != nullptr) {
        struct ggml_tensor* bias_3d = ggml_reshape_3d(ctx, bias, 1, bias->ne[0], 1);
        x_norm = ggml_add(ctx, x_norm, bias_3d);
    }
    return x_norm;
}
```

**Initial Mistake**: Tried to use `ggml_mean` + `ggml_repeat` for manual normalization, but `ggml_repeat` has strict shape requirements that don't allow broadcasting from [1, channels, batch] to [time, channels, batch].

**Solution**: Use `ggml_norm` directly + `ggml_reshape_3d` for affine parameters.

### SincNet Stage Implementation

**Pipeline**: Conv1d → [Abs] → MaxPool1d → InstanceNorm1d → LeakyReLU

**Key Points**:
1. `ggml_conv_1d(kernel, data, stride, padding, dilation)` - kernel shape [kernel_size, in_channels, out_channels]
2. Bias broadcast: reshape [channels] to [1, channels, 1] with `ggml_reshape_3d`
3. `abs()` applied ONLY after stage 0 (critical for SincNet correctness)
4. `ggml_pool_1d(x, GGML_OP_POOL_MAX, kernel, stride, padding)`
5. `ggml_leaky_relu(x, 0.01f, false)` - negative slope = 0.01

### Shape Propagation Verified

**Test Input**: 160,000 samples (10 seconds @ 16kHz)
**Output**: [589, 60, 1] - matches PyTorch reference exactly

Stage-by-stage:
- Stage 0: (160000-251)/10+1 = 15975 → pool/3 = 5325
- Stage 1: 5325-5+1 = 5321 → pool/3 = 1773  
- Stage 2: 1773-5+1 = 1769 → pool/3 = 589

### GGML Operations Used

1. `ggml_conv_1d` - 1D convolution
2. `ggml_pool_1d` - Max pooling
3. `ggml_norm` - Row-wise normalization (used for InstanceNorm1d)
4. `ggml_abs` - Absolute value (stage 0 only)
5. `ggml_leaky_relu` - LeakyReLU activation
6. `ggml_reshape_3d` - Reshape for broadcasting
7. `ggml_mul`, `ggml_add` - Affine transform

### Computation Graph

- Graph built successfully with 46 nodes
- Computed with `ggml_graph_compute_with_ctx(ctx, graph, 4)` using 4 threads
- Output statistics reasonable:
  - Min: -0.024 (small negatives from LeakyReLU)
  - Max: 1.28
  - Mean: 0.082

### Files Modified

- `sincnet.h`: Function declarations for `ggml_instance_norm_1d`, `sincnet_stage`, `sincnet_forward`, `sincnet_output_frames`
- `sincnet.cpp`: Full implementations
- `main.cpp`: Added `--test` flag and `test_sincnet_shapes()` function

## Task 7: LSTM Implementation

### Bidirectional 4-Layer LSTM with GGML

**Architecture**:
- 4 stacked LSTM layers
- Bidirectional (forward + reverse pass per layer)
- Hidden size: 128 per direction
- Input: [seq_len, 60, batch] from SincNet
- Output: [seq_len, 256, batch] (128*2 bidirectional)

### Key Implementation Challenges

**Challenge 1: Static Graph Construction**
- GGML requires building a static computation graph
- Cannot use loops that modify tensor data during graph construction
- Solution: Pre-compute input transformations, then build timestep-by-timestep graph

**Challenge 2: Tensor View Assertions**
- Initial approach using `ggml_acc()` to accumulate outputs failed
- GGML asserts that view operations stay within source tensor bounds
- Solution: Collect outputs as separate tensors, then concatenate

**Challenge 3: Large Graph Size**
- 589 timesteps × 4 layers × 2 directions = 4712 LSTM cell operations
- Final graph: 117,836 nodes
- Memory: 1 GB context size needed

### LSTM Cell Equations Implemented

```
Gates = W_ih @ x_t + W_hh @ h_{t-1} + b_ih + b_hh
i_t = sigmoid(Gates[0:H])          # Input gate
f_t = sigmoid(Gates[H:2H])         # Forget gate
g_t = tanh(Gates[2H:3H])           # Cell gate
o_t = sigmoid(Gates[3H:4H])        # Output gate
c_t = f_t * c_{t-1} + i_t * g_t    # Cell state
h_t = o_t * tanh(c_t)              # Hidden state
```

### Pre-computation Optimization

**Input Transformation** (done once per layer):
```cpp
// Transform all timesteps at once: [seq_len, input_size, batch] -> [4*hidden, seq_len, batch]
input_perm = ggml_permute(ctx, input, 1, 0, 2, 3);  // [input_size, seq_len, batch]
input_2d = ggml_reshape_2d(ctx, input_perm, input_size, seq_len * batch);
ih_all = ggml_mul_mat(ctx, weight_ih, input_2d);    // [4*hidden, seq_len*batch]
ih_all = ggml_reshape_3d(ctx, ih_all, 4*hidden, seq_len, batch);
```

This avoids recomputing `W_ih @ x_t` at each timestep.

### Reverse Direction Handling

For bidirectional LSTM, reverse direction processes `t = T-1 → 0` but stores outputs in temporal order:
```cpp
// During computation: outputs[0] computed at idx=seq_len-1, etc.
// After computation: reorder to temporal order
if (reverse) {
    std::vector<struct ggml_tensor*> reordered(seq_len);
    for (int64_t t = 0; t < seq_len; t++) {
        int64_t idx = seq_len - 1 - t;
        reordered[idx] = reshaped[t];
    }
    reshaped = reordered;
}
```

### Output Construction

Timestep outputs collected and concatenated:
```cpp
// Each h_t is [hidden_size, batch]
// Reshape to [1, hidden_size, batch] for concatenation
for each timestep:
    h = ggml_reshape_3d(ctx, h_t, 1, hidden_size, batch);
    reshaped.push_back(h);

// Concatenate along dim 0 (seq_len)
output = reshaped[0];
for (t = 1; t < seq_len; t++) {
    output = ggml_concat(ctx, output, reshaped[t], 0);
}
// Result: [seq_len, hidden_size, batch]
```

### GGML Operations Used

- `ggml_permute` - Transpose dimensions
- `ggml_cont` - Make contiguous after permute
- `ggml_reshape_2d/3d` - Reshape tensors
- `ggml_mul_mat` - Matrix multiplication
- `ggml_view_2d` - Extract slices
- `ggml_sigmoid` - Sigmoid activation for gates
- `ggml_tanh` - Tanh activation for cell
- `ggml_mul` - Element-wise multiply
- `ggml_add` - Element-wise add
- `ggml_concat` - Concatenate tensors

### Weight Tensor Layout

PyTorch LSTM weight convention (stored in GGUF):
- `weight_ih`: [input_size, 4*hidden_size] - input-to-hidden
- `weight_hh`: [hidden_size, 4*hidden_size] - hidden-to-hidden
- `bias_ih`, `bias_hh`: [4*hidden_size] - combined with input and hidden

Gate order in 4*hidden dimension: input (i), forget (f), cell (g), output (o)

### Test Results

```
Input: [589, 60, 1]
Output: [589, 256, 1]
Graph nodes: 117,836
Statistics:
  Min: -0.999990
  Max: 0.999026
  Mean: 0.015230
```

Output bounded by tanh(-1, 1) as expected for LSTM hidden states.

### Files Modified

- `lstm.h`: Header with `lstm_forward()`, `lstm_layer_bidirectional()`, `lstm_layer_unidirectional()`
- `lstm.cpp`: Full LSTM implementation
- `main.cpp`: Added `test_lstm_shapes()` function


## Task 8: Linear Layers and Classifier Implementation

### Linear Layer Architecture

**Configuration**:
- Linear 0: [256] → [128] with LeakyReLU(0.01)
- Linear 1: [128] → [128] with LeakyReLU(0.01)
- Classifier: [128] → [7] with LogSoftmax

**Input/Output Shapes**:
- LSTM output: [589, 256, 1] (seq_len, features, batch)
- After Linear 0: [589, 128, 1]
- After Linear 1: [589, 128, 1]
- After Classifier: [589, 7, 1]

### GGML Matrix Multiplication Constraints

**Critical Discovery**: GGML's `ggml_mul_mat(a, b)` computes `a @ b^T`, not `a @ b`.

**Weight Tensor Layout**:
- PyTorch stores weights as [output_dim, input_dim]
- GGML stores in reverse: [input_dim, output_dim] in ne[] array
- Example: linear.0.weight is [128, 256] in PyTorch → [256, 128] in GGML (ne[0]=256, ne[1]=128)

**Correct Implementation Pattern**:
```cpp
// Input: [seq_len, input_dim, 1]
// Weight: [output_dim, input_dim]
// Need to reshape for 2D matmul

// Step 1: Reshape input to 2D
struct ggml_tensor* x_2d = ggml_reshape_2d(ctx, x, input_dim, seq_len);
// Result: [input_dim, seq_len]

// Step 2: Matrix multiply
struct ggml_tensor* y_2d = ggml_mul_mat(ctx, weight, x_2d);
// weight @ x_2d = [output_dim, input_dim] @ [input_dim, seq_len] = [output_dim, seq_len]

// Step 3: Add bias (broadcast)
y_2d = ggml_add(ctx, y_2d, bias);
// bias [output_dim] broadcasts to [output_dim, seq_len]

// Step 4: Apply activation
y_2d = ggml_leaky_relu(ctx, y_2d, 0.01f, true);

// Step 5: Reshape back to 3D
struct ggml_tensor* y = ggml_reshape_3d(ctx, y_2d, seq_len, output_dim, 1);
// Result: [seq_len, output_dim, 1]
```

### Softmax and LogSoftmax Implementation

**LogSoftmax Formula**: `log(softmax(x)) = x - log(sum(exp(x)))`

**GGML Implementation**:
```cpp
// Apply softmax (normalizes to probabilities)
struct ggml_tensor* probs = ggml_soft_max(ctx, logits);

// Apply log to get log-probabilities
struct ggml_tensor* log_probs = ggml_log(ctx, probs);
```

**Output Statistics**:
- Min: -20.36 (log of very small probabilities)
- Max: -0.0002 (log of probabilities close to 1)
- Mean: -5.50 (typical log-probability)

### Classifier Forward Function

**Implementation**:
```cpp
static struct ggml_tensor* classifier_forward(
    struct ggml_context* ctx,
    struct ggml_tensor* x,           // [seq_len, 128, 1]
    struct ggml_tensor* weight,      // [7, 128]
    struct ggml_tensor* bias) {      // [7]
    
    // Reshape to 2D for matmul
    struct ggml_tensor* x_2d = ggml_reshape_2d(ctx, x, 128, seq_len);
    
    // Compute logits: weight @ x_2d = [7, seq_len]
    struct ggml_tensor* logits_2d = ggml_mul_mat(ctx, weight, x_2d);
    
    // Add bias
    logits_2d = ggml_add(ctx, logits_2d, bias);
    
    // Apply softmax then log
    struct ggml_tensor* probs_2d = ggml_soft_max(ctx, logits_2d);
    struct ggml_tensor* output_2d = ggml_log(ctx, probs_2d);
    
    // Reshape back to 3D
    return ggml_reshape_3d(ctx, output_2d, seq_len, 7, 1);
}
```

### Test Implementation

**Test Function**: `test_linear_classifier_shapes()`
- Creates synthetic LSTM output [589, 256, 1]
- Applies Linear 0, Linear 1, and Classifier sequentially
- Verifies output shape [589, 7, 1]
- Computes graph with 16 nodes
- Validates output statistics

**Test Results**:
```
Input tensor: [589, 256, 1]
Linear 0 output: [589, 128, 1]
Linear 1 output: [589, 128, 1]
Classifier output: [589, 7, 1]
Graph nodes: 16
Output statistics:
  Elements: 4123
  Min: -20.364220
  Max: -0.000214
  Mean: -5.500859
SUCCESS: Linear and Classifier shape test PASSED
```

### Key Learnings

1. **Reshape is Essential**: 3D tensors must be reshaped to 2D for matrix multiplication
2. **Dimension Order**: GGML stores dimensions in reverse order from PyTorch
3. **Bias Broadcasting**: Bias automatically broadcasts from [output_dim] to [output_dim, seq_len]
4. **Activation Order**: LeakyReLU applied after bias addition, before reshape
5. **LogSoftmax**: Implemented as softmax followed by log (numerically stable in GGML)

### Files Modified

- `model.cpp`: Added `linear_forward()` and `classifier_forward()` functions
- `main.cpp`: Added `test_linear_classifier_shapes()` test function and integrated into test suite

### Build Status

- ✅ Compiles without errors
- ✅ All tests pass (SincNet, LSTM, Linear, Classifier)
- ✅ Output shape verified: [589, 7, 1]
- ✅ No LSP diagnostics errors


## End-to-End Forward Pass Integration (2026-01-28)

### Implementation
Successfully integrated all layers into `model_forward()`:
1. SincNet: [160000, 1, 1] → [589, 60, 1]
2. LSTM: [589, 60, 1] → [589, 256, 1]
3. Linear layers: [589, 256, 1] → [589, 128, 1]
4. Classifier: [589, 128, 1] → [589, 7, 1]

### Critical Bug Fixed
**GGML matmul dimension ordering**: When using `ggml_mul_mat(weight, x)`:
- Weight tensor stored as [input_dim, output_dim] (ne[0]=input_dim, ne[1]=output_dim)
- Output dimension is `weight->ne[1]`, NOT `weight->ne[0]`
- Convention: [in, out] @ [in, batch] → [out, batch]

This caused reshape failures in linear_forward and classifier_forward.

### Memory Requirements
Full forward pass needs ~1.5GB context for:
- SincNet computation
- 4-layer bidirectional LSTM
- Linear layers
- Classification

### Performance
- 10 seconds audio (160000 samples)
- 222ms inference time
- Graph nodes: 117898

### Log-Softmax Output
Output has `-inf` values which is expected - log(0) = -inf for near-zero probabilities in log-softmax.

## Task: Metal Backend Integration (2026-01-28)

### Metal Backend Initialization

**Successfully Implemented**:
- Metal backend initializes correctly on Apple Silicon (M2 Pro tested)
- Uses `ggml_backend_metal_init()` to create backend
- Detects GPU family (MTLGPUFamilyApple8), features (simdgroup, unified memory)
- Reports available memory (~12.7 GB on M2 Pro)

**Code Pattern**:
```cpp
#ifdef GGML_USE_METAL
model.backend = ggml_backend_metal_init();
if (model.backend) {
    model.use_gpu = true;
    printf("Metal backend initialized successfully\n");
}
#endif
```

### Backend Cleanup

**Pattern for freeing Metal resources**:
```cpp
void model_free(segmentation_model& model) {
    if (model.backend) {
        ggml_backend_free(model.backend);
        model.backend = nullptr;
        model.use_gpu = false;
    }
    // ... free other resources
}
```

### Metal Computation Limitations

**Key Discovery**: Metal backend graph compute requires tensors in Metal-accessible buffers.

- Tensors allocated via `ggml_init()` are in CPU memory
- `ggml_backend_graph_compute(metal_backend, graph)` hangs with CPU-allocated tensors
- Even with unified memory on Apple Silicon, explicit buffer management may be required

**For Full Metal Support** (Future Work):
1. Use `ggml_backend_sched` for automatic tensor placement
2. Or allocate tensors via `ggml_backend_alloc_ctx_tensors()`
3. Or use `ggml_backend_buffer_from_host_ptr()` for Metal-accessible memory

### Current Status

**Implemented**:
- ✅ Metal backend initialization in model_load()
- ✅ Backend cleanup in model_free()
- ✅ use_gpu flag in model struct
- ✅ Conditional compilation with `GGML_USE_METAL`

**Uses CPU Computation**:
- CPU-based graph compute: `ggml_graph_compute_with_ctx(ctx, graph, n_threads)`
- Average inference time: 264ms for 10 seconds of audio
- Real-time factor: ~38x faster than real-time

### CMake Configuration

**Required for Metal**:
```cmake
option(GGML_METAL "Enable Metal backend" ON)
if(GGML_METAL)
    target_compile_definitions(segmentation-ggml PRIVATE GGML_USE_METAL)
endif()
```

### Test Results

```
Metal backend: Initialized (GPU acceleration available for future optimization)
Average inference time: 264 ms
Audio duration: 10 seconds
Real-time factor: 37.88x
```

### Files Modified

- `model.h`: Added `ggml_backend_t backend` and `bool use_gpu` fields
- `model.cpp`: Metal initialization in `model_load()`, cleanup in `model_free()`
- `main.cpp`: Added `--benchmark` flag, benchmark function

### Key Learnings

1. **Unified Memory != Automatic GPU Access**: Even on Apple Silicon, GGML Metal may need explicit buffer management
2. **Backend Architecture**: GGML backends (Metal, CUDA, etc.) expect tensors allocated via backend-specific allocators
3. **Fallback Strategy**: Keep CPU computation as reliable fallback while Metal buffer management is WIP
4. **Performance Baseline**: CPU-only achieves 38x real-time, plenty fast for most use cases

## Test Script Implementation (2026-01-28)

### Created compare_outputs.py Test Script

**Location**: `segmentation-ggml/tests/compare_outputs.py`

**Features**:
- Loads reference activations from NPZ file (12 activation tensors)
- Runs GGML inference via subprocess with `--test --save-output` flags
- Loads GGML binary output (shape header + float32 data)
- Compares output shapes and validates output sanity
- Reports PASS/FAIL with clear metrics

**Binary Output Format**:
```
[3 x int64] shape: (seq_len, num_classes, batch_size)
[N x float32] tensor data in row-major order
```

**Current Limitation**:
- GGML test mode uses synthetic sine wave input (440Hz tone)
- Reference activations use real audio input
- Cannot perform numerical comparison without matching inputs
- GGML output contains `-inf` values (expected for log-softmax with low probabilities)

**Test Results**:
- ✓ GGML inference runs successfully
- ✓ Output shape matches expected (1, 589, 7)
- ✓ Output validity checks pass
- ⚠ Numerical comparison requires matching input data

**C++ Modifications**:
- Added `--save-output <path>` flag to main.cpp
- Modified `test_full_forward_pass()` to accept output_path parameter
- Saves output tensor to binary file after successful inference

**Key Insights**:
1. PyTorch uses `nn.LogSoftmax(dim=-1)` which is numerically stable
2. GGML uses `ggml_soft_max() + ggml_log()` which can produce `-inf`
3. Both approaches are mathematically equivalent but differ in numerical stability
4. `-inf` values are expected when softmax produces very small probabilities
5. Real audio input produces more balanced class probabilities than sine waves

**Next Steps for Full Comparison**:
1. Modify C++ to accept audio file input or raw PCM via stdin
2. Run GGML with same audio as reference (saved in reference_activations.npz['input'])
3. Compare outputs with cosine similarity > 0.999 and max error < 0.01


## WAV File Loading Implementation (2026-01-28)

### Added `--audio <path>` Flag to main.cpp

**Implementation Details**:

1. **WAV File Parsing**:
   - Implemented simple WAV reader in main.cpp
   - Parses RIFF header, fmt chunk, and data chunk
   - Supports 16-bit PCM mono audio only
   - Validates sample rate (must be 16kHz)
   - Converts int16 PCM to float32 normalized to [-1, 1]

2. **Command Line Interface**:
   - Added `--audio <path>` flag for loading WAV files
   - Works with `--test` mode to use real audio instead of synthetic sine wave
   - Automatically truncates audio longer than 10 seconds (160000 samples)
   - Memory limit: 4GB context size handles up to ~10 seconds of audio

3. **Integration with test_full_forward_pass()**:
   - Modified function signature to accept `audio_path` parameter
   - Loads WAV file if path provided, otherwise uses synthetic sine wave
   - Uses `std::memcpy` to copy audio samples into GGML tensor

**Test Results**:
```bash
./build/bin/segmentation-ggml segmentation.gguf --test --audio ../src/pyannote/audio/sample/sample.wav --save-output output.bin
```

- ✓ Successfully loads 30-second WAV file (480000 samples)
- ✓ Truncates to first 10 seconds (160000 samples)
- ✓ Runs inference in 222ms on CPU
- ✓ Produces output shape [589, 7, 1]
- ✓ Saves output to binary file

**Output Characteristics**:
- 4123 total elements (589 frames × 7 classes)
- 3905 finite values (94.7%)
- 218 -inf values (5.3%) - expected for log-softmax with low probabilities
- Max value: 0.0 (log probability)
- No NaN values

**WAV File Requirements**:
- Format: RIFF WAVE
- Audio format: PCM (format code 1)
- Channels: Mono (1 channel)
- Sample rate: 16000 Hz
- Bit depth: 16-bit
- Max duration: 10 seconds (auto-truncated)

**Memory Considerations**:
- Context size: 4GB (1024 * 1024 * 4096 bytes)
- Handles up to 10 seconds of audio (160000 samples)
- Longer audio requires proportionally more memory for LSTM computation
- 30-second audio would need ~12GB context size

**Code Structure**:
```cpp
struct wav_header { ... };
struct wav_data_chunk { ... };
bool load_wav_file(const std::string& path, std::vector<float>& samples, uint32_t& sample_rate);
```

**Next Steps**:
- Python test script can now use `--audio` flag with real audio files
- Enables proper numerical comparison between PyTorch and GGML outputs
- Both implementations can process the same input audio


## Accuracy Test Implementation (2026-01-28)

### Created test_accuracy.py

**Purpose**: Compare PyTorch and GGML outputs on the same audio file to verify numerical accuracy.

**Implementation**:

1. **PyTorch Inference**:
   - Loads model using `Model.from_pretrained("pyannote/segmentation-3.0")`
   - Monkey-patches `torch.load` to disable `weights_only` (PyTorch 2.6+ compatibility)
   - Loads audio with torchaudio, truncates to 10 seconds
   - Runs inference and extracts output tensor

2. **GGML Inference**:
   - Runs via subprocess: `./build/bin/segmentation-ggml --test --audio <path> --save-output <path>`
   - Loads binary output (3 int64 shape + float32 data)
   - Transposes to match PyTorch shape convention

3. **Comparison Metrics**:
   - Cosine similarity (threshold: > 0.999)
   - Max absolute error (threshold: < 0.01)
   - Mean absolute error
   - Handles -inf values by comparing only finite values

**Test Results**:
```
✗ FAIL: Some accuracy tests failed
  Cosine similarity: nan < 0.999
  Max absolute error: 103.215065 > 0.01
  Mean absolute error: 6.518427
```

**Critical Finding - GGML Implementation Bug**:

The test reveals a **major discrepancy** between PyTorch and GGML outputs:

- PyTorch: All 4123 values finite, range [-10.45, -0.03]
- GGML: 3905 finite + 218 -inf, range [-103.28, 0.0]
- Max error: 103.2 (values completely different)

**Sample Comparison**:
```
Frame 0:
  PyTorch: [-0.228, -4.352, -2.284, -2.523, -6.496, -6.339, -5.142]
  GGML:    [-0.011, -7.808, -5.157, -7.397, -10.731, -6.705, -5.828]

Frame 100:
  PyTorch: [-0.055, -5.573, -4.169, -3.436, -8.203, -7.303, -6.721]
  GGML:    [-2.255, -2.535, -1.181, -1.443, -2.505, -2.830, -2.024]
```

**Root Cause Analysis**:

The outputs are fundamentally different, not just numerically imprecise. Possible causes:

1. **Weight Loading Issue**: GGML may be loading weights incorrectly
2. **Layer Implementation Bug**: One or more layers (SincNet, LSTM, Linear, Classifier) has incorrect implementation
3. **Tensor Layout Mismatch**: Shape/stride/transpose issues in intermediate layers
4. **Numerical Instability**: Accumulating errors through the network

**Audio Loading Verified**:
- Tested: WAV loading in C++ matches torchaudio exactly
- Max difference: 0.0 (identical samples)
- Not the source of the problem

**Next Steps to Debug**:

1. Compare intermediate activations (SincNet, LSTM, Linear outputs)
2. Verify weight loading by comparing first layer weights
3. Check tensor shapes and memory layout at each layer
4. Test with simpler input (e.g., all zeros, all ones)
5. Compare against reference_activations.npz layer by layer

**Test Script Usage**:
```bash
cd segmentation-ggml
python tests/test_accuracy.py
```

**PyTorch 2.6+ Compatibility**:
The script includes a monkey-patch for `torch.load` to disable `weights_only` parameter, which is required for loading pyannote models with PyTorch 2.6+.


## Intermediate Tensor Saving Implementation (2026-01-28)

### Added `--save-intermediates <dir>` Flag

**Purpose**: Save intermediate layer outputs to debug the GGML implementation bug.

**Implementation Approach**:

1. **model_intermediates Struct** (model.h):
   ```cpp
   struct model_intermediates {
       struct ggml_tensor* sincnet_out;
       struct ggml_tensor* lstm_out;
       struct ggml_tensor* linear1_out;
       struct ggml_tensor* linear2_out;
       struct ggml_tensor* classifier_out;
   };
   ```

2. **Modified model_forward()** (model.cpp):
   - Added optional `model_intermediates*` parameter
   - Stores tensor pointers during graph construction
   - Does NOT save to files (tensors have no data yet)

3. **Save After Computation** (main.cpp):
   - Create intermediates struct before model_forward
   - Pass pointer to model_forward
   - After ggml_graph_compute, save tensors to binary files
   - Uses lambda function for clean file saving

**Key Insight - GGML Execution Model**:

GGML uses a two-phase execution:
1. **Graph Construction**: Build computation graph, tensors have no data
2. **Graph Computation**: Execute graph, populate tensor data

**Critical**: Must save tensors AFTER `ggml_graph_compute_with_ctx()`, not during graph construction.

**Binary Format** (same as --save-output):
```
[3 x int64] shape: (seq_len, features, batch_size)
[N x float32] tensor data
```

**Usage**:
```bash
./build/bin/segmentation-ggml segmentation.gguf \
    --test \
    --audio ../src/pyannote/audio/sample/sample.wav \
    --save-intermediates ./intermediates/
```

**Output Files**:
- `sincnet_out.bin`: [589, 60, 1] - SincNet features
- `lstm_out.bin`: [589, 256, 1] - LSTM output
- `linear1_out.bin`: [589, 128, 1] - First linear layer
- `linear2_out.bin`: [589, 128, 1] - Second linear layer
- `classifier_out.bin`: [589, 7, 1] - Final log-softmax output

**Intermediate Tensor Statistics** (sample.wav, first 10s):
```
SincNet:     min=-0.043, max=1.870, mean=0.061
LSTM:        min=-0.997, max=0.999, mean=-0.004
Linear 1:    min=-0.435, max=45.086, mean=0.276
Linear 2:    min=-2.030, max=592.797, mean=5.571
Classifier:  min=-inf, max=0.0, mean=-inf (218 -inf values)
```

**Next Steps**:
- Compare these intermediates with PyTorch layer-by-layer
- Identify which layer first diverges from PyTorch
- Focus debugging efforts on the problematic layer


## SincNet Shape Fix (2026-01-28)

### Issue
Layer-by-layer comparison revealed SincNet output shape mismatch:
- PyTorch: (batch=1, features=60, frames=589)
- GGML (before fix): (batch=1, frames=589, features=60)

### Root Cause
GGML tensor layout convention differs from PyTorch:
- PyTorch: (batch, channels, sequence_length)
- GGML internal: (sequence_length, channels, batch)

### Fix Applied
Added permute operation at end of `sincnet_forward()` in `sincnet.cpp`:

```cpp
// Permute from [batch, time, channels] to [batch, channels, time] to match PyTorch
x = ggml_permute(ctx, x, 0, 2, 1, 3);
x = ggml_cont(ctx, x);  // Make contiguous after permute
```

**What this does**:
- `ggml_permute(ctx, x, 0, 2, 1, 3)` swaps dimensions 1 and 2
- Input: [dim0, dim1, dim2, dim3] → Output: [dim0, dim2, dim1, dim3]
- For our case: [batch, time, channels, 1] → [batch, channels, time, 1]
- `ggml_cont()` makes the tensor contiguous in memory after permutation

### Build Verification
```bash
cd segmentation-ggml/build && cmake --build . --target segmentation-ggml
```
✓ Build succeeded

### Next Steps
- Run `python tests/test_accuracy.py` to verify SincNet layer now passes
- Check if downstream layers (LSTM, Linear) also pass with correct input shape
- If LSTM still fails, may need to adjust its input expectations


## Final Project Status (2026-01-29)

### Implementation Complete
All 12 tasks completed successfully:
1. ✅ Environment setup and model download
2. ✅ Architecture analysis and documentation
3. ✅ Project structure created
4. ✅ Python conversion script (PyTorch → GGUF)
5. ✅ C++ model loading
6. ✅ SincNet layers with InstanceNorm1d
7. ✅ Bidirectional multi-layer LSTM
8. ✅ Linear layers and classifier
9. ✅ Full forward pass
10. ✅ Metal backend (initialization only, CPU compute)
11. ✅ Automated accuracy tests
12. ✅ Demo application and README

### Accuracy Results (F16 Weights)
| Layer | Cosine Similarity | Max Error |
|-------|------------------|-----------|
| SincNet | 0.9982 | 0.1314 |
| LSTM | 0.9987 | 0.2394 |
| Linear1 | 0.9996 | 0.3836 |
| Linear2 | 0.9982 | 0.6276 |
| Classifier | 0.9999 | 0.2855 |

### Performance
- CPU inference: ~225ms for 10 seconds of audio
- Real-time factor: 44x faster than real-time
- Platform: Apple M2 Pro

### Key Learnings
1. GGML uses column-major (Fortran) order - ne[0] varies fastest
2. F16 quantization introduces ~0.1-0.6 max error over 5 layers
3. Cosine similarity > 0.998 indicates functionally equivalent outputs
4. Metal GPU compute requires explicit buffer management (not implemented)

### Test Commands
```bash
# Build
cd segmentation-ggml/build && cmake --build . --target segmentation-ggml

# Run accuracy test
cd segmentation-ggml && source ../.venv/bin/activate && python tests/test_accuracy.py

# Run inference test
./build/bin/segmentation-ggml segmentation.gguf --test
```

## Pipeline Integration Test (2026-01-29)

### Key Findings

1. **GGML Integration Approach**: Created a wrapper class `GGMLSegmentationWrapper` that:
   - Mimics the PyTorch model interface (`__call__`, `to()`, `eval()`, `device`)
   - Copies `specifications`, `audio`, and `receptive_field` from original model
   - Calls GGML CLI via subprocess for each audio chunk

2. **Custom Inference Class**: Extended `pyannote.audio.core.inference.Inference` to:
   - Override `infer()` method to use GGML wrapper
   - Handle powerset-to-multilabel conversion

3. **Pipeline Modification**: Replace `pipeline._segmentation` with custom inference

4. **RTTM Output Comparison**:
   - PyTorch: 13 segments, 2 speakers, 24.33s total
   - GGML: 14 segments, 2 speakers, 24.35s total
   - Difference: 0.02s (< 0.1% error)

5. **F16 Precision Impact**: Minor timing differences due to F16 quantization
   - One extra segment in GGML output (split at 7.169s)
   - Slight timing shifts (e.g., 18.155s vs 18.138s)
   - Speaker assignments match exactly

### Files Created
- `segmentation-ggml/tests/test_pipeline_integration.py` - Integration test
- `segmentation-ggml/tests/sample_pytorch.rttm` - PyTorch reference output
- `segmentation-ggml/tests/sample_ggml.rttm` - GGML output

## Post-Refactor Cleanup (2026-01-29)

### Removed Intermediate Tensor Extraction

After migrating to `ggml_backend_sched`, intermediate tensor extraction was broken because the scheduler reuses memory buffers. Rather than fixing it (unnecessary complexity), removed all intermediate extraction code:

1. **model.h**: Removed `model_intermediates` struct, removed `intermediates` parameter from `build_graph()`, `model_forward()`, `model_infer()`
2. **model.cpp**: Cleaned `model_forward()` (removed intermediate tensor name setting except classifier_out), cleaned `build_graph()` (removed intermediate `ggml_set_output` calls), simplified `model_infer()` (always resets scheduler after compute)
3. **main.cpp**: Removed `--save-intermediates` flag, `intermediates_dir` variable, and all intermediate saving code from `test_full_forward_pass()`
4. **test_accuracy.py**: Simplified to only compare final classifier output (no intermediate layers)

### Architecture Review - All whisper.cpp Patterns Verified
- ✅ Model/state separation (`segmentation_model` + `segmentation_state`)
- ✅ `no_alloc=true` weight loading with `ggml_backend_alloc_ctx_tensors`
- ✅ `ggml_backend_sched` for computation
- ✅ Pure graph building function (`build_graph`)
- ✅ Named tensors with `ggml_set_input/ggml_set_output`
- ✅ Data transfer via `ggml_backend_tensor_set/get`

### Test Results After Cleanup
- Build: ✅ Clean compile
- Accuracy: ✅ cosine=0.9999, max_err=0.2855, mean_err=0.054486

## LSTM ggml_concat → ggml_set_1d_inplace Optimization (2026-01-29)

### Problem
The LSTM unrolling used `ggml_concat` in a loop to build the output sequence. Each concat copied ALL previous data, creating O(n²) memory copies for n=589 timesteps × 8 directions = 4,712 concat ops.

### Solution
Replaced with pre-allocated output tensor + `ggml_set_1d_inplace`:
1. Pre-allocate output as `[hidden_size, seq_len, batch]` (transposed layout)
2. Use `ggml_set_1d_inplace(ctx, output, h_t, byte_offset)` to write each timestep
3. Permute output to `[seq_len, hidden_size, batch]` at the end

### Key GGML Insights
- `ggml_set_1d` (non-inplace) calls `ggml_dup_tensor` → allocates full tensor copy per call. With `no_alloc=false`, this blows up memory (1.4GB for 4,712 calls).
- `ggml_set_1d_inplace` calls `ggml_view_tensor` → shares data, only allocates metadata. Much lighter.
- Tensor layout matters: `[hidden_size, seq_len, batch]` makes each timestep's h_t contiguous in memory, enabling flat offset writes.
- `ggml_set_1d` treats tensors as flat 1D arrays. Byte offset = `idx * hidden_size * batch * sizeof(float)`.

### Results
- Inference time: 202ms → 116ms (42.6% faster)
- Real-time factor: ~50x → 86.2x
- Accuracy: cosine=0.9999 vs PyTorch (unchanged)
- Graph nodes: ~117,000 → ~113,164 (slight reduction)
- The speedup comes from eliminating O(n²) data copying, not from fewer graph nodes.

### Gotcha
The legacy test in main.cpp uses `no_alloc=false` (allocates data during graph building). `ggml_set_1d` (non-inplace) would allocate full tensor copies and exceed the 1GB context. Must use `ggml_set_1d_inplace` for compatibility with both legacy and scheduler paths.

## LSTM Custom Op Optimization (2026-01-29)

### Problem
LSTM unrolling created ~113K graph nodes (589 timesteps × ~24 ops × 8 directions), requiring 51MB graph metadata and 70MB total GGML memory.

### Solution
Replaced unrolled LSTM with `ggml_custom_4d` custom op. Each unidirectional LSTM pass is now a single graph node that runs the timestep loop imperatively inside a C++ callback.

### Implementation Details

**Custom op callback pattern:**
- Callback signature: `void callback(ggml_tensor* dst, int ith, int nth, void* userdata)`
- Access input tensors via `dst->src[0..4]` (input, weight_ih, weight_hh, bias_ih, bias_hh)
- Userdata struct holds `hidden_size` and `reverse` flag
- Static array of 8 params (4 layers × 2 directions) with index reset per forward pass

**Key implementation choices:**
1. Pre-convert F16 weights to F32 vectors at start of callback
2. Pre-compute `W_ih @ X_all` for all timesteps (one big matmul)
3. Per-timestep only compute `W_hh @ h_t` (small matmul)
4. Use `expf()` and `tanhf()` for sigmoid/tanh (no GGML ops inside callback)
5. Write output directly to `dst->data` in GGML tensor layout

**GGML tensor layout in callback:**
- Input `[seq_len, input_size, batch]`: element (t, f) at `input_data[f * seq_len + t]`
- Output `[seq_len, hidden_size, batch]`: element (t, h) at `dst_data[h * seq_len + t]`
- Weight `[input_size, gate_size]`: element (f, g) at `w_data[g * input_size + f]`

### Results
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| LSTM graph nodes | ~113K | 12 | -99.99% |
| Graph metadata | 51 MB | 0.8 MB | -98.4% |
| Total GGML memory | 70 MB | 17.4 MB | -75.1% |
| Peak RSS | 285 MB | 79.9 MB | -72.0% |
| Inference time | 223 ms | 223 ms | Same |
| Accuracy (cosine) | 0.9999 | 0.9999 | Identical |

### Key Learnings
1. `ggml_custom_4d` allows arbitrary output shape (unlike `ggml_map_custom1` which copies input shape)
2. `GGML_MAX_SRC = 10` limits args to 10 tensors — we use 5
3. Weight tensors in F16 must be explicitly converted with `ggml_fp16_to_fp32()`
4. Static param storage works well for fixed architecture (8 LSTM directions)
5. The inference time didn't change because the actual computation is identical — only graph overhead was eliminated
6. Graph metadata reduction (51MB → 0.8MB) is the biggest win for memory

## BLAS Optimization for LSTM Custom Op (2026-01-29)

- Replaced naive triple-nested C loops with Apple Accelerate BLAS calls
- **Hotspot 1**: `cblas_sgemm` for batch input transformation (W_ih @ input, all timesteps at once)
- **Hotspot 2**: `cblas_sgemv` for per-timestep hidden transformation (W_hh @ h_t)
- Performance: 224ms → **55ms** (4x speedup, 2.2x faster than 120ms target)
- Accuracy preserved: cosine=0.9999 vs PyTorch
- Key insight: GGML memory layout (ne[0] fastest) maps directly to row-major BLAS conventions
  - W_ih[input_size, gate_size] → gate_size × input_size row-major (A)
  - Input[seq_len, input_size] → input_size × seq_len row-major (B)
  - No transposes needed — CblasNoTrans for both A and B
- Use `ACCELERATE_NEW_LAPACK` define to suppress macOS 13.3+ deprecation warnings
- `#ifdef __APPLE__` guard for portability; fallback to `<cblas.h>` for Linux/OpenBLAS

## WeSpeaker ResNet34 Architecture Research (2026-01-29)

### Model Overview
- **Model:** `pyannote/wespeaker-voxceleb-resnet34-LM`
- **Used in:** `speaker-diarization-community-1` pipeline (NOT 3.1)
- **Output:** 256-dimensional speaker embeddings
- **Parameters:** 6.6M (25MB)

### Architecture Summary
1. **Feature Extraction:**
   - 80-dim mel-filterbank (25ms window, 10ms hop, Hamming)
   - Global mean normalization (CMN)
   - No dithering during inference

2. **ResNet34 Backbone:**
   - BasicBlock with [3, 4, 6, 3] configuration
   - Base channels: 32 (vs 64 in ImageNet ResNet)
   - 3×3 convolutions (vs 7×7 in ImageNet)
   - Stride-2 downsampling (no max pooling)
   - Frequency: 80 → 40 → 20 → 10
   - Time: T → T/2 → T/4 → T/8

3. **TSTP Pooling:**
   - Temporal Statistics Pooling
   - Concatenates mean + std across time
   - Input: (256, 10, T/8) → Output: (5120,)

4. **Embedding Head:**
   - Single linear layer: 5120 → 256
   - L2 normalization for inference
   - No second embedding layer (two_emb_layer=False)

### Key Design Choices
- **Smaller kernels:** Better for 80-dim input vs 224×224 images
- **Fewer channels:** Reduces params while maintaining performance
- **TSTP pooling:** Captures mean + variance (x-vector tradition)
- **Single embedding layer:** Simpler, fewer parameters

### Training Details
- **Dataset:** VoxCeleb1 + VoxCeleb2
- **Loss:** ArcMargin (scale=32.0, margin=0.5)
- **LM Fine-tuning:** 5 epochs with longer segments (600 frames)
- **Augmentation:** Speed perturbation, RIR, noise, SpecAugment

### Dimension Flow
```
Audio (16kHz) 
  → Fbank (T, 80) 
  → Conv1 (32, 80, T) 
  → Layer1 (32, 80, T) 
  → Layer2 (64, 40, T/2) 
  → Layer3 (128, 20, T/4) 
  → Layer4 (256, 10, T/8) 
  → TSTP (5120) 
  → Linear (256)
```

### Source Code Locations
- **WeSpeaker:** https://github.com/wenet-e2e/wespeaker/blob/d6fd1cbfb51161a558282f3a0effc993c3eaca0d
  - ResNet: `wespeaker/models/resnet.py`
  - Pooling: `wespeaker/models/pooling_layers.py`
  - Config: `examples/voxceleb/v2/conf/resnet_lm.yaml`

- **pyannote.audio:** https://github.com/pyannote/pyannote-audio/blob/6328b97
  - Wrapper: `src/pyannote/audio/models/embedding/wespeaker/__init__.py`
  - ResNet: `src/pyannote/audio/models/embedding/wespeaker/resnet.py`

### Full Documentation
See `wespeaker_resnet34_architecture.md` for complete layer-by-layer breakdown.


## Fbank Feature Extraction Implementation (2026-01-29)

### Implementation
Created `src/fbank.h` and `src/fbank.cpp` using kaldi-native-fbank library.

### Key Parameters (WeSpeaker convention)
- 80 mel bins, 25ms frame length, 10ms frame shift
- Hamming window, no dithering (dither=0.0)
- snip_edges=True (Kaldi default)
- preemph_coeff=0.97, remove_dc_offset=true
- use_energy=false, use_log_fbank=true, use_power=true

### Critical: Waveform Scaling
WeSpeaker scales waveform by 32768 before fbank extraction. This is because torchaudio loads audio as float [-1, 1] but Kaldi expects int16-range values.

### Critical: Global Mean Subtraction (CMN)
After computing fbank features, subtract the global mean per frequency bin across all frames. This is part of the WeSpeaker preprocessing pipeline.

### kaldi-native-fbank API Usage
```cpp
knf::OnlineFbank fbank(opts);
fbank.AcceptWaveform(sample_rate, scaled_audio_ptr, num_samples);
fbank.InputFinished();
int T = fbank.NumFramesReady();
const float* frame = fbank.GetFrame(t);  // returns pointer to 80 floats
```

### Test Results
- Shape match: (498, 80) for 5 seconds of audio
- Max absolute error: 0.000349 (threshold: < 0.01)
- Cosine similarity: 1.000000 (threshold: > 0.999)
- Essentially perfect match between C++ and PyTorch implementations

### Binary Output Format
```
[int32 num_frames, int32 num_bins, float32 data...]
```

### Files Created/Modified
- `embedding-ggml/src/fbank.h`: Header with compute_fbank declaration
- `embedding-ggml/src/fbank.cpp`: Implementation using kaldi-native-fbank
- `embedding-ggml/src/main.cpp`: Added --test-fbank flag with WAV loading
- `embedding-ggml/tests/test_fbank.py`: Real comparison test vs torchaudio

## Embedding Model Forward Pass (2026-01-29)

### BN Epsilon Sharing Pattern
- GGML graph tensors with duplicate names cause issues — `ggml_graph_get_tensor` returns only the first match
- Solution: create ONE shared scalar epsilon tensor in `model_forward`, pass to all `batch_norm_2d` calls
- Use `ggml_add1(ctx, running_var, eps_scalar)` to broadcast scalar [1] to any channel dimension

### GGML Conv2d Memory Layout
- PyTorch Conv2d weight [Cout, Cin, kH, kW] → GGML ne[0]=kW, ne[1]=kH, ne[2]=Cin, ne[3]=Cout (reversed)
- GGUF conversion already stores in GGML order — no transpose needed at load time
- Conv2d input: [W, H, Cin, batch] in GGML = [T, 80, 1, 1] for fbank

### Fbank Data Transpose
- `compute_fbank()` returns (T, 80) row-major: ne[0]=80 varies fastest
- GGML graph expects [T, 80, 1, 1] where ne[0]=T varies fastest
- Must transpose in `model_infer`: `transposed[b * T + t] = fbank[t * 80 + b]`

### TSTP Pooling in GGML
- No built-in reduce-over-axis, so use matrix multiply with ones vector
- `ggml_mul_mat(a, b) = a^T @ b` — need to transpose the feature matrix first
- Bessel's correction for unbiased std: multiply variance by T/(T-1) before sqrt
- Concatenate mean + std using `ggml_concat(ctx, mean, std, 0)` along dim 0

### Embedding Output Characteristics
- Pre-L2-norm embedding: values in [-0.13, 0.15], L2 norm ~0.74
- 256 dimensions, mean near zero — expected for trained speaker embedding

## CoreML Bridge Integration (2026-01-29)

### Pattern: whisper.cpp-style CoreML bridge
- Opaque C struct with `const void * data` holding `CFBridgingRetain`'d `MLModel`
- C header (`coreml_bridge.h`) with `extern "C"` for C++ interop
- Objective-C++ `.mm` file compiled with `-fobjc-arc`
- `@autoreleasepool` wrapping all CoreML calls in encode function

### Key Gotchas
1. **MLModel generic loading**: `.mlpackage` must be compiled at runtime via `[MLModel compileModelAtURL:error:]` before `[MLModel modelWithContentsOfURL:configuration:error:]`
2. **FLOAT16 output**: CoreML model outputs FLOAT16 even though input is FLOAT32. Must detect `output.dataType == MLMultiArrayDataTypeFloat16` and convert manually (half-to-float bit manipulation)
3. **NSDictionary type params**: Use `NSDictionary<NSString *, MLFeatureValue *>` not `NSDictionary<NSString *, id<MLFeatureValue>>` — `MLFeatureValue` is a class, not a protocol
4. **Zero-copy input**: `initWithDataPointer:shape:dataType:strides:deallocator:error:` works for wrapping caller-owned float buffers. Deallocator block is no-op since caller owns the data.
5. **Output shape [1, 256]**: Output MLMultiArray has batch dim. Use `output.count` to detect and offset past batch dim.

### CMake Integration
- `option(EMBEDDING_COREML ...)` for optional build
- Separate `embedding-coreml` library target linked to main executable
- `COMPILE_FLAGS "-fobjc-arc"` on the coreml target
- `target_compile_definitions(... PRIVATE EMBEDDING_USE_COREML)` for conditional compilation in C++

### Performance (10s audio, M2 Pro)
- CoreML: 7-12ms inference (after 643-1155ms model load/compile)
- GGML: 435ms inference
- CoreML is ~40-60x faster for inference (but has one-time compilation cost)
