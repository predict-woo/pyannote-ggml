#!/usr/bin/env python3
"""Convert WeSpeaker ResNet34 embedding model to CoreML format.

Converts ONLY the neural network portion (ResNet34 → TSTP → Linear),
NOT the fbank feature extraction (which stays in C++).

Input:  fbank features (1, T, 80) where T is variable (100-2000 frames)
Output: 256-dim speaker embedding (1, 256)

Usage:
    cd embedding-ggml && python convert_coreml.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class ResNetEmbeddingWrapper(nn.Module):
    """Wrapper that takes fbank features and returns the speaker embedding.

    This replaces einops.rearrange and std(correction=1) with trace-friendly
    native PyTorch ops so torch.jit.trace works cleanly.
    """

    def __init__(self, resnet):
        super().__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.seg_1 = resnet.seg_1

    def forward(self, fbank: torch.Tensor) -> torch.Tensor:
        """Forward pass from fbank features to embedding.

        Parameters
        ----------
        fbank : (1, T, 80) torch.Tensor
            Fbank features (batch=1, frames=T, features=80)

        Returns
        -------
        embedding : (1, 256) torch.Tensor
            Speaker embedding
        """
        x = fbank.permute(0, 2, 1)     # (B,T,F) -> (B,F,T)
        x = x.unsqueeze(1)             # (B,F,T) -> (B,1,F,T)

        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)             # (B, 256, 10, T/8)

        # TSTP pooling — native ops replace einops.rearrange for tracing
        batch, dim, chan, frames = x.shape
        x = x.reshape(batch, dim * chan, frames)  # (B, 2560, T/8)

        # Bessel-corrected stats: std = sqrt(sum((x-mean)^2) / (N-1))
        mean = x.mean(dim=-1)
        n = x.shape[-1]
        diff = x - mean.unsqueeze(-1)
        var = (diff * diff).sum(dim=-1) / (n - 1)
        std = torch.sqrt(var + 1e-8)
        stats = torch.cat([mean, std], dim=-1)  # (B, 5120)

        return self.seg_1(stats)  # (B, 256)


def main():
    import coremltools as ct

    print("=" * 60)
    print("WeSpeaker ResNet34 → CoreML Conversion")
    print("=" * 60)

    print("\n[1/6] Loading PyTorch model...")

    # Monkey-patch torch.load — force weights_only=False for PyTorch 2.6+
    # Lightning passes weights_only=True explicitly, so we must override it
    _original_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)

    torch.load = _patched_load

    try:
        from pyannote.audio import Model

        model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    finally:
        torch.load = _original_load

    resnet = model.resnet
    resnet.eval()
    print(f"  Loaded ResNet34: feat_dim=80, embed_dim=256, two_emb_layer=False")
    print(f"  Stats dim: {resnet.stats_dim}, Pool out dim: {resnet.pool_out_dim}")

    print("\n[2/6] Creating trace-friendly wrapper module...")
    wrapper = ResNetEmbeddingWrapper(resnet)
    wrapper.eval()
    print("  Wrapper created (replaces einops + Bessel std with native ops)")

    print("\n[3/6] Tracing model with torch.jit.trace...")
    example_input = torch.randn(1, 998, 80)
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapper, example_input)
    print("  Traced successfully with input shape (1, 998, 80)")

    with torch.no_grad():
        pt_out = wrapper(example_input)
        traced_out = traced_model(example_input)
    trace_diff = (pt_out - traced_out).abs().max().item()
    print(f"  Trace verification: max abs diff = {trace_diff:.2e}")

    print("\n[4/6] Converting to CoreML (.mlpackage)...")

    input_shape = ct.Shape(
        shape=(
            1,
            ct.RangeDim(lower_bound=100, upper_bound=2000, default=998),
            80,
        )
    )

    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="fbank_features", shape=input_shape)],
        outputs=[ct.TensorType(name="embedding")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
    )
    print("  CoreML conversion successful")

    output_path = Path("embedding.mlpackage")
    print(f"\n[5/6] Saving to {output_path}...")
    mlmodel.save(str(output_path))
    print(f"  Saved: {output_path}")

    print("\n[6/6] Sanity check: PyTorch vs CoreML...")
    torch.manual_seed(42)
    test_input = torch.randn(1, 998, 80)

    with torch.no_grad():
        pt_embedding = wrapper(test_input).numpy().flatten()

    coreml_input = {"fbank_features": test_input.numpy()}
    coreml_output = mlmodel.predict(coreml_input)
    coreml_embedding = coreml_output["embedding"].flatten()

    cos_sim = np.dot(pt_embedding, coreml_embedding) / (
        np.linalg.norm(pt_embedding) * np.linalg.norm(coreml_embedding)
    )
    max_diff = np.abs(pt_embedding - coreml_embedding).max()
    mean_diff = np.abs(pt_embedding - coreml_embedding).mean()

    print(f"  PyTorch embedding norm:  {np.linalg.norm(pt_embedding):.4f}")
    print(f"  CoreML embedding norm:   {np.linalg.norm(coreml_embedding):.4f}")
    print(f"  Cosine similarity:       {cos_sim:.6f}")
    print(f"  Max absolute diff:       {max_diff:.6f}")
    print(f"  Mean absolute diff:      {mean_diff:.6f}")

    if cos_sim > 0.99:
        print(f"\n  ✓ PASS: Cosine similarity {cos_sim:.6f} > 0.99")
    else:
        print(f"\n  ✗ FAIL: Cosine similarity {cos_sim:.6f} <= 0.99")
        print("    The CoreML model may not match PyTorch output accurately.")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("Conversion Summary")
    print(f"{'=' * 60}")
    print(f"  Model:       WeSpeaker ResNet34 (pyannote/wespeaker-voxceleb-resnet34-LM)")
    print(f"  Input:       fbank_features (1, T, 80), T ∈ [100, 2000]")
    print(f"  Output:      embedding (1, 256)")
    print(f"  Format:      CoreML ML Program (.mlpackage)")
    print(f"  Precision:   FLOAT16")
    print(f"  Target:      macOS 13+")
    print(f"  Output file: {output_path.resolve()}")
    print(f"  Cosine sim:  {cos_sim:.6f}")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
