# CoreML Conversion Learnings

## 2026-01-29: convert_coreml.py created successfully

### Key Findings

- **einops.rearrange traces cleanly** when replaced with native `reshape/flatten` — `torch.jit.trace` cannot handle einops
- **Bessel-corrected std** (`correction=1`) does NOT need special handling for tracing — but we implemented it manually anyway for clarity: `std = sqrt(sum((x-mean)^2) / (N-1))` with eps=1e-8
- **RangeDim works** with coremltools 9.0 for variable-length time dimension: `ct.RangeDim(lower_bound=100, upper_bound=2000, default=998)`
- **FLOAT16 precision** yields cosine similarity of 0.999967 vs PyTorch FP32 — more than sufficient
- **torch.load monkey-patch** must use `kwargs["weights_only"] = False` (force override), NOT `kwargs.setdefault(...)`, because Lightning explicitly passes `weights_only=True`
- **coremltools 9.0 + PyTorch 2.10** works despite warning about untested version (2.7.0 is latest tested)
- **overflow warning** in `optimize_repeat_ops.py` during MIL pipeline is benign — conversion succeeds

### Architecture Details Confirmed

- ResNet34 output before TSTP: `(B, 256, 10, T/8)` — the 10 comes from 80/8 (feat_dim / 3 stride-2 layers)
- TSTP flatten: `(B, 256, 10, T/8)` → `(B, 2560, T/8)`
- Stats pool: `(B, 2560, T/8)` → `(B, 5120)` (mean + std concatenated)
- seg_1 linear: `(B, 5120)` → `(B, 256)`
- two_emb_layer=False means seg_2 and seg_bn_1 are nn.Identity (unused)

### Wrapper Design

- Copy individual submodules (conv1, bn1, layer1-4, seg_1) rather than wrapping the whole ResNet
- This avoids tracing the TSTP/StatsPool modules which use einops
- The wrapper's forward() reimplements TSTP pooling with native PyTorch ops
