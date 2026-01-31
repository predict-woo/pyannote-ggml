# Issues and Gotchas

## Problems Encountered

*Append issues as they arise. NEVER overwrite.*

---

## Task 4: Conversion Script

### PyTorch Safe Globals Warning
- **Issue**: PyTorch 2.6+ requires explicit safe globals registration for custom classes
- **Symptom**: `_pickle.UnpicklingError: Weights only load failed` when loading model
- **Solution**: 
  ```python
  from pyannote.audio.core import task
  import inspect
  task_classes = [obj for name, obj in inspect.getmembers(task) if inspect.isclass(obj)]
  torch.serialization.add_safe_globals(task_classes)
  ```
- **Status**: Resolved
- **Date**: 2026-01-28

### Type Annotation Strictness
- **Issue**: LSP reports many type annotation warnings in convert.py
- **Symptom**: Warnings about `Unknown` types, deprecated `Dict`/`List` usage
- **Impact**: Cosmetic only - script functions correctly
- **Future Fix**: Could add proper type hints for numpy arrays and use Python 3.9+ generics
- **Status**: Deferred (non-blocking)
- **Date**: 2026-01-28

### File Size Discrepancy
- **Issue**: Expected ~6MB file size, got 2.87MB
- **Reason**: Original estimate was for F32 weights; using F16 weights reduces size ~50%
- **Math**: 1.47M params × 2 bytes (F16) ≈ 2.9MB
- **Status**: Expected behavior (not an issue)
- **Date**: 2026-01-28


## Task 5: GGUF Loading Issues

### Issue: "failed to read tensor data binary blob"

**Symptom**: C++ GGUF loader failed with error message from gguf.cpp:687

**Root Cause**: File size mismatch
- Expected: 3,007,872 bytes
- Actual: 3,007,868 bytes (4 bytes short)
- Difference: Last tensor (classifier.bias, 28 bytes) not padded to 32 bytes

**Investigation Steps**:
1. Verified GGUF header was correct (magic, version, counts)
2. Verified metadata was readable
3. Traced error to ggml/src/gguf.cpp line 684: `gr.read(data->data, ctx->size)`
4. Discovered ctx->size calculation uses GGML_PAD for each tensor
5. Python conversion script wasn't padding tensor data to alignment

**Fix**: Modified convert.py to:
1. Use padded offsets: `current_offset += align_offset(data.nbytes)`
2. Write padding after each tensor: `f.write(b'\x00' * padding_needed)`

**Lesson**: GGUF tensor data must be padded to `general.alignment` (default 32 bytes) for EACH tensor, not just overall alignment.

## Critical Bug: GGML Output Mismatch (2026-01-28)

### Issue
GGML implementation produces completely different outputs compared to PyTorch reference.

### Severity
**CRITICAL** - The model is not functionally equivalent to PyTorch.

### Evidence
- Max absolute error: 103.2 (expected < 0.01)
- Cosine similarity: NaN (expected > 0.999)
- Output value ranges completely different
- 218 -inf values in GGML vs 0 in PyTorch

### Impact
- GGML model cannot be used for production
- Numerical accuracy test fails completely
- Outputs are not just imprecise, they're fundamentally wrong

### Reproduction
```bash
cd segmentation-ggml
python tests/test_accuracy.py
```

### Investigation Needed
1. Compare intermediate layer outputs (use reference_activations.npz)
2. Verify weight loading correctness
3. Check tensor layout/transpose operations
4. Test individual layers in isolation

### Blocked Tasks
- Cannot proceed with accuracy validation
- Cannot deploy GGML model
- Need to fix implementation before further testing


## Root Cause Identified: SincNet Shape Mismatch (2026-01-28)

### Layer-by-Layer Comparison Results

```
SincNet      ✗ FAIL  shape mismatch: PyTorch=(1, 60, 589) GGML=(1, 589, 60)
LSTM         ✗ FAIL  cosine=0.0001 max_err=1.9601 mean_err=0.104696
Linear1      ✗ FAIL  cosine=-0.1214 max_err=49.7484 mean_err=4.065284
Linear2      ✗ FAIL  cosine=-0.1016 max_err=595.3721 mean_err=11.956669
Classifier   ✗ FAIL  cosine=0.1552 max_err=107.1391 mean_err=6.370457
```

### Root Cause

**DIVERGENCE STARTS AT: SincNet**

The SincNet layer outputs have a **shape mismatch**:
- **PyTorch**: (batch=1, features=60, frames=589)
- **GGML**: (batch=1, frames=589, features=60)

This is a **transpose issue** - the features and frames dimensions are swapped.

### Impact

All subsequent layers fail because they receive incorrectly shaped input:
- LSTM expects (batch, frames, features) but gets transposed data
- This causes the entire forward pass to produce wrong outputs
- The error propagates and amplifies through each layer

### Technical Details

**PyTorch Convention**: (batch, channels/features, sequence_length)
**GGML Implementation**: (sequence_length, channels/features, batch)

The GGML implementation is using a different tensor layout convention than PyTorch.

### Fix Required

The SincNet implementation in GGML needs to:
1. Either output tensors in PyTorch's (batch, features, frames) format
2. Or ensure all subsequent layers expect (batch, frames, features) format
3. Check the sincnet_forward() function in sincnet.cpp

### Files to Investigate

- `segmentation-ggml/src/sincnet.cpp` - SincNet implementation
- `segmentation-ggml/src/model.cpp` - Check tensor layout assumptions
- `segmentation-ggml/src/lstm.cpp` - Verify expected input shape

### Next Steps

1. Review sincnet_forward() output tensor construction
2. Check if transpose is needed after SincNet
3. Verify LSTM input shape expectations
4. Fix the shape mismatch and re-run test

