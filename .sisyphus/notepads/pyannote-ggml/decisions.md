# Architectural Decisions

## Key Choices

*Append decisions as they are made. NEVER overwrite.*

---

## Initial Decisions (from planning)

### SincNet Conversion Strategy
- **Decision**: Pre-compute parametric sinc filters as static conv weights
- **Rationale**: Simpler than implementing sinc math in C++, sufficient for inference-only use case
- **Date**: 2026-01-28

### InstanceNorm1d Implementation
- **Decision**: Implement manually using GGML primitives (not LayerNorm substitution)
- **Rationale**: User wants accuracy, willing to accept implementation complexity for correctness
- **Date**: 2026-01-28

### Bidirectional LSTM Strategy
- **Decision**: Process forward pass → backward pass → concatenate outputs
- **Rationale**: Standard approach, matches PyTorch semantics exactly
- **Date**: 2026-01-28

### LSTM State Precision
- **Decision**: Use F32 for hidden/cell state accumulation
- **Rationale**: Numerical stability over long sequences, weights can be F16
- **Date**: 2026-01-28

---

## Task 2: Architecture Analysis

### Documentation Structure
- **Decision**: Create comprehensive `architecture.md` with layer-by-layer breakdown
- **Rationale**: 
  - GGML implementation requires exact understanding of every operation
  - Future tasks (4-8) will reference this documentation
  - Serves as single source of truth for tensor shapes and operations
- **Format**: Markdown with clear sections, code blocks, and tables

### Reference Data Format
- **Decision**: Save activations as compressed NumPy `.npz` file
- **Rationale**:
  - NumPy is standard for numerical data in Python
  - Compressed format saves disk space
  - Easy to load and inspect with numpy
  - Compatible with both Python and C++ (via cnpy or similar)
- **Alternative Considered**: JSON (rejected - too large for numerical arrays)

### Analysis Script Design
- **Decision**: Single Python script that loads model, extracts info, and saves results
- **Rationale**:
  - Reproducible analysis
  - Can be re-run if model changes
  - Captures all information in one pass
- **Output Files**:
  - `architecture.md` - Human-readable documentation
  - `layer_info.json` - Machine-readable layer parameters
  - `activation_shapes.json` - Quick reference for shapes
  - `reference_activations.npz` - Numerical validation data

### SincNet Filter Documentation
- **Decision**: Document both configuration and learned parameter values
- **Rationale**:
  - Configuration (kernel_size, stride, etc.) needed for GGML layer setup
  - Learned parameters (low_hz, band_hz) needed for filter generation
  - Ranges documented to understand frequency coverage (36 Hz - 7.4 kHz)
- **Key Finding**: Filters cover speech-relevant frequencies (50 Hz - 8 kHz)

### LSTM Layer Count Correction
- **Decision**: Document actual layer count (4) vs. initial assumption (2)
- **Rationale**:
  - Critical for correct GGML implementation
  - Affects parameter count and computation time
  - Shows importance of extracting actual model vs. relying on defaults
- **Lesson**: Always verify hyperparameters from loaded model, not documentation


## Task 4: Conversion Script Implementation

### GGUF Format Choice
- **Decision**: Use modern GGUF format (version 3) instead of legacy GGML binary format
- **Rationale**:
  - GGUF is the current standard for ggml models
  - Better extensibility via key-value metadata
  - mmap support for fast loading
  - Self-documenting (architecture info embedded)
- **Date**: 2026-01-28

### SincNet Filter Pre-computation
- **Decision**: Pre-compute SincNet filters at conversion time, not runtime
- **Rationale**:
  - C++ implementation doesn't need sinc function math
  - Filters are deterministic given learned parameters
  - Faster inference (no filter generation per forward pass)
  - Output verified: 80 channels, 251 kernel, matches PyTorch
- **Alternative Considered**: Store low_hz/band_hz and compute at runtime (rejected - adds complexity)
- **Date**: 2026-01-28

### Filter Normalization Method
- **Decision**: Use L2 normalization (sqrt of sum of squares) for filter weights
- **Rationale**:
  - Matches PyTorch ParamSincFB behavior
  - Energy-preserving normalization
  - Validated against reference activations
- **Date**: 2026-01-28

### Dtype Strategy
- **Decision**: Weights → F16, Biases → F32
- **Rationale**:
  - F16 weights reduce memory by ~50% with minimal accuracy loss
  - F32 biases maintain numerical stability (small tensors, big impact)
  - LSTM states computed in F32 during inference (decision from planning)
- **Result**: 2.87 MB file size (vs ~5.9 MB all-F32)
- **Date**: 2026-01-28

### Tensor Name Convention
- **Decision**: Use dot-notation with layer indices: `sincnet.{stage}.{layer}.{param}`
- **Rationale**:
  - Clear hierarchy
  - Matches C++ model structure expectations
  - LSTM names kept as-is for clarity (standard PyTorch naming)
- **Date**: 2026-01-28

### Metadata Schema
- **Decision**: Include both general GGUF metadata and architecture-specific metadata
- **Rationale**:
  - `general.*` keys follow GGUF standard
  - `pyannet.*` keys provide model-specific info for C++ loader
  - Sample rate critical for audio processing
  - LSTM dimensions needed for state allocation
- **Date**: 2026-01-28


## Task 6: SincNet Layers Implementation

### InstanceNorm1d Strategy
- **Decision**: Use GGML's `ggml_norm` for normalization + reshape for affine parameters
- **Rationale**:
  - `ggml_norm` normalizes along rows (ne[0]) which is the time dimension in GGML layout
  - This matches InstanceNorm1d semantics exactly
  - Manual implementation with `ggml_mean` + `ggml_repeat` failed due to broadcast constraints
- **Alternative Rejected**: Manual mean/var computation - `ggml_repeat` doesn't support needed broadcast shapes
- **Date**: 2026-01-28

### Affine Parameter Broadcasting
- **Decision**: Use `ggml_reshape_3d` to reshape [channels] to [1, channels, 1] for broadcasting
- **Rationale**:
  - GGML's mul/add operations require compatible shapes
  - Reshape is a zero-copy view operation (fast)
  - Broadcasting then works correctly with [time, channels, batch] tensors
- **Date**: 2026-01-28

### Bias Addition in Conv1d
- **Decision**: Reshape bias from [channels] to [1, channels, 1] instead of using ggml_repeat
- **Rationale**:
  - `ggml_repeat` has strict constraints that don't match our broadcasting needs
  - Reshape + add is simpler and more efficient
- **Date**: 2026-01-28

### Test Implementation
- **Decision**: Add `--test` flag to main.cpp for SincNet shape verification
- **Rationale**:
  - Shape propagation test validates all layer connections
  - Uses actual model weights from GGUF
  - Quick feedback loop for development
- **Date**: 2026-01-28
