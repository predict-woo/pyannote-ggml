#!/usr/bin/env python3
"""Compare GGML vs PyTorch embedding model output.

Runs both implementations on the same audio and compares the 256-dim
speaker embedding vectors.
"""

import numpy as np
import struct
import subprocess
import sys
import os
import torch


def patch_torch_load():
    """Monkey-patch torch.load to disable weights_only for PyTorch 2.6+"""
    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)

    torch.load = patched_load


def compute_pytorch_embedding(audio_path: str, duration_s: float = 10.0):
    import torchaudio
    from pyannote.audio import Model

    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000

    max_samples = int(duration_s * sr)
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    print(f"  Audio shape: {waveform.shape} ({waveform.shape[1] / sr:.2f}s)")

    patch_torch_load()
    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    model.eval()

    with torch.no_grad():
        # forward() expects (batch, channel, sample) -> add batch dim
        embedding = model(waveform.unsqueeze(0))

    return embedding.squeeze().numpy()


def compute_ggml_embedding(audio_path: str, model_path: str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    binary = os.path.join(script_dir, '..', 'build', 'bin', 'embedding-ggml')
    binary = os.path.abspath(binary)
    cwd = os.path.abspath(os.path.join(script_dir, '..'))
    output_file = os.path.join(cwd, 'embedding_output.bin')

    if not os.path.exists(binary):
        raise FileNotFoundError(f"GGML binary not found: {binary}")

    if os.path.exists(output_file):
        os.remove(output_file)

    cmd = [binary, model_path, '--test-inference', '--audio', audio_path]
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=60)

    if result.returncode != 0:
        print("GGML stdout:", result.stdout)
        print("GGML stderr:", result.stderr)
        raise RuntimeError(f"GGML inference failed with return code {result.returncode}")

    if not os.path.exists(output_file):
        print("GGML stdout:", result.stdout)
        raise FileNotFoundError(f"Output file not found: {output_file}")

    # Binary format: [int32 dim, float32 data[dim]]
    with open(output_file, 'rb') as f:
        dim = struct.unpack('<i', f.read(4))[0]
        data = np.frombuffer(f.read(dim * 4), dtype=np.float32).copy()

    os.remove(output_file)

    return data


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    print("=" * 60)
    print("Embedding GGML vs PyTorch Accuracy Test")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_path = os.path.abspath(
        os.path.join(script_dir, '..', '..', 'src', 'pyannote', 'audio', 'sample', 'sample.wav')
    )
    model_path = os.path.abspath(
        os.path.join(script_dir, '..', 'embedding.gguf')
    )

    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)

    binary_path = os.path.abspath(
        os.path.join(script_dir, '..', 'build', 'bin', 'embedding-ggml')
    )
    if not os.path.exists(binary_path):
        print(f"ERROR: GGML binary not found: {binary_path}")
        print("  Build with: cmake -B build && cmake --build build")
        sys.exit(1)

    print(f"\nAudio: {os.path.basename(audio_path)}")
    print(f"Model: {os.path.basename(model_path)}")

    print("\nComputing PyTorch embedding...")
    pytorch_emb = compute_pytorch_embedding(audio_path)
    print(f"  Shape: {pytorch_emb.shape}")
    print(f"  L2 norm: {np.linalg.norm(pytorch_emb):.6f}")
    print(f"  Range: [{pytorch_emb.min():.6f}, {pytorch_emb.max():.6f}]")
    print(f"  First 5: {pytorch_emb[:5]}")

    print("\nComputing GGML embedding...")
    ggml_emb = compute_ggml_embedding(audio_path, model_path)
    print(f"  Shape: {ggml_emb.shape}")
    print(f"  L2 norm: {np.linalg.norm(ggml_emb):.6f}")
    print(f"  Range: [{ggml_emb.min():.6f}, {ggml_emb.max():.6f}]")
    print(f"  First 5: {ggml_emb[:5]}")

    print("\n" + "-" * 60)
    print("Comparing results...")
    print("-" * 60)

    assert pytorch_emb.shape == ggml_emb.shape, (
        f"Shape mismatch: PyTorch {pytorch_emb.shape} vs GGML {ggml_emb.shape}"
    )

    cos_sim = cosine_similarity(pytorch_emb, ggml_emb)
    print(f"  Cosine similarity: {cos_sim:.6f}")

    abs_diff = np.abs(pytorch_emb - ggml_emb)
    max_err = float(abs_diff.max())
    mean_err = float(abs_diff.mean())
    print(f"  Max absolute error: {max_err:.6f}")
    print(f"  Mean absolute error: {mean_err:.6f}")

    print()
    COSINE_THRESHOLD = 0.99
    MAX_ERR_THRESHOLD = 1.0

    cos_pass = cos_sim > COSINE_THRESHOLD
    err_pass = max_err < MAX_ERR_THRESHOLD

    print(f"{'PASS' if cos_pass else 'FAIL'}: Cosine similarity {cos_sim:.6f} {'>' if cos_pass else '<='} {COSINE_THRESHOLD}")
    print(f"{'PASS' if err_pass else 'FAIL'}: Max absolute error {max_err:.6f} {'<' if err_pass else '>='} {MAX_ERR_THRESHOLD}")

    print()
    if cos_pass and err_pass:
        print("=" * 60)
        print("All checks passed!")
        print("=" * 60)
        return 0
    else:
        print("=" * 60)
        print("SOME CHECKS FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
