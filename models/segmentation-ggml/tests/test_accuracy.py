#!/usr/bin/env python3
"""
PyTorch vs GGML Accuracy Test

Compares final classifier output between PyTorch and GGML to verify
numerical accuracy of the GGML implementation.

Test Criteria (F16 weights):
- Cosine similarity > 0.995
- Max absolute error < 1.0
"""

import numpy as np
import subprocess
import sys
from pathlib import Path
import torch
import torchaudio
from pyannote.audio import Model

COSINE_SIMILARITY_THRESHOLD = 0.995
MAX_ERROR_THRESHOLD = 1.0


def patch_torch_load():
    """Monkey-patch torch.load to disable weights_only for PyTorch 2.6+"""
    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)

    torch.load = patched_load


def cosine_similarity(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    dot_product = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    return dot_product / (norm_a * norm_b)


def run_pytorch_inference(model_name, audio_path, max_samples=160000):
    print("Running PyTorch inference...")

    patch_torch_load()

    model = Model.from_pretrained(model_name, use_auth_token=None)
    model.eval()

    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]

    print(f"  Audio shape: {waveform.shape}")
    print(f"  Sample rate: {sample_rate} Hz")

    with torch.no_grad():
        output = model(waveform.unsqueeze(0))

    print(f"  Output shape: {output.shape}")
    return output.cpu().numpy()


def run_ggml_inference(executable_path, model_path, audio_path, output_path):
    print("\nRunning GGML inference...")

    cmd = [
        str(executable_path),
        str(model_path),
        "--test",
        "--audio", str(audio_path),
        "--save-output", str(output_path)
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60
    )

    if result.returncode != 0:
        print(f"  ERROR: GGML inference failed")
        print(result.stderr)
        return None

    if "SUCCESS: End-to-End Forward Pass PASSED" not in result.stdout:
        print("  ERROR: GGML test did not pass")
        return None

    print("  ✓ GGML inference completed")

    with open(output_path, 'rb') as f:
        shape = np.fromfile(f, dtype=np.int64, count=3)
        ne0, ne1, ne2 = shape
        total_elements = ne0 * ne1 * ne2
        data = np.fromfile(f, dtype=np.float32, count=total_elements)

        # GGML column-major -> reshape with Fortran order
        output = data.reshape((ne0, ne1, ne2), order='F')
        # Transpose to PyTorch layout: (batch, seq_len, num_classes)
        output = np.transpose(output, (2, 0, 1))

    print(f"  Output shape: {output.shape}")
    return output


def main():
    print("=" * 60)
    print("PyTorch vs GGML Accuracy Test")
    print("=" * 60)

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    audio_path = project_root.parent / "samples" / "sample.wav"
    model_path = project_root / "segmentation.gguf"
    executable_path = project_root / "build" / "bin" / "segmentation-ggml"
    output_path = script_dir / "ggml_accuracy_output.bin"

    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        return 1

    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        return 1

    if not executable_path.exists():
        print(f"ERROR: Executable not found: {executable_path}")
        return 1

    print(f"\nAudio: {audio_path.name}")
    print(f"Model: {model_path.name}")
    print()

    pytorch_output = run_pytorch_inference(
        "pyannote/segmentation-3.0",
        str(audio_path),
        max_samples=160000
    )

    ggml_output = run_ggml_inference(
        executable_path,
        model_path,
        audio_path,
        output_path
    )

    if ggml_output is None:
        print("\nERROR: GGML inference failed")
        return 1

    print("\nComparing final classifier output...")

    finite_mask = np.isfinite(pytorch_output) & np.isfinite(ggml_output)

    if not finite_mask.any():
        print("  ✗ FAIL  no finite values to compare")
        return 1

    cos_sim = cosine_similarity(pytorch_output[finite_mask], ggml_output[finite_mask])
    max_error = np.max(np.abs(pytorch_output[finite_mask] - ggml_output[finite_mask]))
    mean_error = np.mean(np.abs(pytorch_output[finite_mask] - ggml_output[finite_mask]))

    passed = cos_sim >= COSINE_SIMILARITY_THRESHOLD and max_error <= MAX_ERROR_THRESHOLD

    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  Classifier  {status}  cosine={cos_sim:.4f} max_err={max_error:.4f} mean_err={mean_error:.6f}")

    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)

    if passed:
        print("\n✓ PASS: GGML output matches PyTorch reference")
    else:
        print("\n✗ FAIL: Output divergence detected")
        print(f"  Cosine similarity: {cos_sim:.4f} (threshold: {COSINE_SIMILARITY_THRESHOLD})")
        print(f"  Max error: {max_error:.4f} (threshold: {MAX_ERROR_THRESHOLD})")

    print("=" * 60)

    if output_path.exists():
        output_path.unlink()

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
