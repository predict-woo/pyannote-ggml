#!/usr/bin/env python3
"""Convert PyAnnote PyanNet segmentation model to CoreML format.

Converts the full PyanNet pipeline (SincNet → LSTM → Linear → Classifier)
to a CoreML .mlpackage for inference on Apple Neural Engine.

Input:  raw waveform (1, 1, 160000) — 10s at 16kHz mono
Output: log-probabilities (1, 589, 7) — powerset segmentation

Usage:
    cd segmentation-ggml && python convert_coreml.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PyanNetWrapper(nn.Module):
    """Trace-friendly wrapper for PyanNet segmentation model.

    Replaces einops.rearrange with .permute(), freezes SincNet parametric
    filters into regular Conv1d weights, and includes log_softmax activation.
    """

    def __init__(self, model):
        super().__init__()

        # --- SincNet ---
        sincnet = model.sincnet

        # Instance norm on raw waveform
        self.wav_norm = sincnet.wav_norm1d

        # Stage 0: ParamSincFB → frozen Conv1d (abs + pool + norm + leaky_relu)
        filterbank = sincnet.conv1d[0].filterbank
        with torch.no_grad():
            filters = filterbank.filters()  # (80, 1, 251)
        self.sinc_conv = nn.Conv1d(
            in_channels=1,
            out_channels=80,
            kernel_size=251,
            stride=filterbank.stride,
            padding=0,
            bias=False,
        )
        with torch.no_grad():
            self.sinc_conv.weight.copy_(filters)

        self.pool0 = sincnet.pool1d[0]
        self.norm0 = sincnet.norm1d[0]

        # Stage 1: Conv1d(80, 60, 5) + pool + norm
        self.conv1 = sincnet.conv1d[1]
        self.pool1 = sincnet.pool1d[1]
        self.norm1 = sincnet.norm1d[1]

        # Stage 2: Conv1d(60, 60, 5) + pool + norm
        self.conv2 = sincnet.conv1d[2]
        self.pool2 = sincnet.pool1d[2]
        self.norm2 = sincnet.norm1d[2]

        # --- LSTM ---
        self.lstm = model.lstm

        # --- Linear layers ---
        self.linear = model.linear

        # --- Classifier ---
        self.classifier = model.classifier

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        waveform : (1, 1, 160000) torch.Tensor
            Raw audio waveform (batch=1, channels=1, samples=160000)

        Returns
        -------
        output : (1, 589, 7) torch.Tensor
            Log-probabilities (powerset segmentation)
        """
        # SincNet stage 0: parametric sinc → abs → pool → norm → leaky_relu
        x = self.wav_norm(waveform)
        x = self.sinc_conv(x)
        x = torch.abs(x)
        x = F.leaky_relu(self.norm0(self.pool0(x)))

        # SincNet stage 1
        x = self.conv1(x)
        x = F.leaky_relu(self.norm1(self.pool1(x)))

        # SincNet stage 2
        x = self.conv2(x)
        x = F.leaky_relu(self.norm2(self.pool2(x)))

        # Permute: (B, feat, time) → (B, time, feat) — replaces einops.rearrange
        x = x.permute(0, 2, 1)

        # LSTM
        x, _ = self.lstm(x)

        # Linear layers with leaky_relu
        for linear in self.linear:
            x = F.leaky_relu(linear(x))

        # Classifier + log_softmax
        x = self.classifier(x)
        x = F.log_softmax(x, dim=-1)

        return x


def main():
    import coremltools as ct

    print("=" * 60)
    print("PyanNet Segmentation → CoreML Conversion")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load PyTorch model
    # ------------------------------------------------------------------
    print("\n[1/6] Loading PyTorch model...")

    _original_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)

    torch.load = _patched_load

    try:
        from pyannote.audio import Model

        model = Model.from_pretrained("pyannote/segmentation-3.0")
    finally:
        torch.load = _original_load

    model.eval()
    print("  Loaded PyanNet segmentation-3.0")

    # ------------------------------------------------------------------
    # Step 2: Create trace-friendly wrapper
    # ------------------------------------------------------------------
    print("\n[2/6] Creating trace-friendly wrapper...")
    wrapper = PyanNetWrapper(model)
    wrapper.eval()
    print("  Wrapper created (frozen SincNet filters, native permute, log_softmax)")

    # ------------------------------------------------------------------
    # Step 3: Trace with torch.jit.trace
    # ------------------------------------------------------------------
    print("\n[3/6] Tracing model with torch.jit.trace...")
    example_input = torch.randn(1, 1, 160000)
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapper, example_input)
    print("  Traced successfully with input shape (1, 1, 160000)")

    # Verify trace matches wrapper
    with torch.no_grad():
        pt_out = wrapper(example_input)
        traced_out = traced_model(example_input)
    trace_diff = (pt_out - traced_out).abs().max().item()
    print(f"  Trace verification: max abs diff = {trace_diff:.2e}")
    print(f"  Output shape: {pt_out.shape}")

    # Also verify wrapper matches original model
    with torch.no_grad():
        orig_out = model(example_input)
    wrapper_diff = (pt_out - orig_out).abs().max().item()
    print(f"  Wrapper vs original: max abs diff = {wrapper_diff:.2e}")

    if wrapper_diff > 1e-4:
        print(f"  ⚠ WARNING: wrapper differs from original by {wrapper_diff:.2e}")
        print("  This may indicate a bug in the wrapper.")

    # ------------------------------------------------------------------
    # Step 4: Convert to CoreML
    # ------------------------------------------------------------------
    print("\n[4/6] Converting to CoreML (.mlpackage)...")

    # Fixed input shape: (1, 1, 160000) — always 10s chunks
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="waveform", shape=(1, 1, 160000))],
        outputs=[ct.TensorType(name="log_probabilities")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.ALL,
    )
    print("  CoreML conversion successful")

    # ------------------------------------------------------------------
    # Step 5: Save .mlpackage
    # ------------------------------------------------------------------
    output_path = Path(__file__).parent / "segmentation.mlpackage"
    print(f"\n[5/6] Saving to {output_path}...")
    mlmodel.save(str(output_path))
    print(f"  Saved: {output_path}")

    # ------------------------------------------------------------------
    # Step 6: Sanity check — PyTorch vs CoreML
    # ------------------------------------------------------------------
    print("\n[6/6] Sanity check: PyTorch vs CoreML...")
    torch.manual_seed(42)
    test_input = torch.randn(1, 1, 160000)

    with torch.no_grad():
        pt_output = wrapper(test_input).numpy().flatten()

    coreml_input = {"waveform": test_input.numpy()}
    coreml_result = mlmodel.predict(coreml_input)
    coreml_output = coreml_result["log_probabilities"].flatten()

    cos_sim = np.dot(pt_output, coreml_output) / (
        np.linalg.norm(pt_output) * np.linalg.norm(coreml_output)
    )
    max_diff = np.abs(pt_output - coreml_output).max()
    mean_diff = np.abs(pt_output - coreml_output).mean()

    print(f"  Output size:           {len(pt_output)} (589 × 7 = 4123)")
    print(f"  PyTorch output norm:   {np.linalg.norm(pt_output):.4f}")
    print(f"  CoreML output norm:    {np.linalg.norm(coreml_output):.4f}")
    print(f"  Cosine similarity:     {cos_sim:.6f}")
    print(f"  Max absolute diff:     {max_diff:.6f}")
    print(f"  Mean absolute diff:    {mean_diff:.6f}")

    if cos_sim > 0.99:
        print(f"\n  ✓ PASS: Cosine similarity {cos_sim:.6f} > 0.99")
    else:
        print(f"\n  ✗ FAIL: Cosine similarity {cos_sim:.6f} <= 0.99")
        print("    The CoreML model may not match PyTorch output accurately.")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("Conversion Summary")
    print(f"{'=' * 60}")
    print(f"  Model:       PyanNet (pyannote/segmentation-3.0)")
    print(f"  Input:       waveform (1, 1, 160000) — 10s at 16kHz mono")
    print(f"  Output:      log_probabilities (1, 589, 7)")
    print(f"  Format:      CoreML ML Program (.mlpackage)")
    print(f"  Precision:   FLOAT32")
    print(f"  Target:      macOS 13+")
    print(f"  Output file: {output_path.resolve()}")
    print(f"  Cosine sim:  {cos_sim:.6f}")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
