#!/usr/bin/env python3
"""Compare C++ kaldi-native-fbank output against torchaudio.compliance.kaldi.fbank."""

import numpy as np
import struct
import subprocess
import sys
import os

import torch
import torchaudio


SAMPLE_WAV = os.path.join(os.path.dirname(__file__), "..", "..", "samples", "sample.wav")
CPP_BINARY = os.path.join(os.path.dirname(__file__), "..", "build", "bin", "embedding-ggml")
FBANK_OUTPUT = "fbank_output.bin"
SAMPLE_RATE = 16000
MAX_SECONDS = 5


def compute_pytorch_fbank(wav_path: str) -> np.ndarray:
    waveform, sr = torchaudio.load(wav_path)
    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE}Hz, got {sr}Hz"

    waveform = waveform[:, :SAMPLE_RATE * MAX_SECONDS]

    waveform_scaled = waveform * 32768.0

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform_scaled,
        sample_frequency=SAMPLE_RATE,
        num_mel_bins=80,
        frame_length=25.0,
        frame_shift=10.0,
        dither=0.0,
        snip_edges=True,
        window_type="hamming",
        preemphasis_coefficient=0.97,
        remove_dc_offset=True,
        use_energy=False,
        use_log_fbank=True,
        use_power=True,
    )

    fbank = fbank - fbank.mean(dim=0, keepdim=True)

    return fbank.numpy()


def run_cpp_fbank(wav_path: str) -> np.ndarray:
    assert os.path.exists(CPP_BINARY), f"C++ binary not found: {CPP_BINARY}"

    if os.path.exists(FBANK_OUTPUT):
        os.remove(FBANK_OUTPUT)

    cmd = [CPP_BINARY, "dummy.gguf", "--test-fbank", "--audio", wav_path]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"C++ binary failed with return code {result.returncode}")

    assert os.path.exists(FBANK_OUTPUT), f"Output file not found: {FBANK_OUTPUT}"

    with open(FBANK_OUTPUT, "rb") as f:
        num_frames = struct.unpack("i", f.read(4))[0]
        num_bins = struct.unpack("i", f.read(4))[0]
        data = np.frombuffer(f.read(num_frames * num_bins * 4), dtype=np.float32)
        data = data.reshape(num_frames, num_bins)

    os.remove(FBANK_OUTPUT)
    return data


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten()
    b_flat = b.flatten()
    return float(np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat)))


def test_fbank_matches_pytorch():
    wav_path = os.path.abspath(SAMPLE_WAV)
    assert os.path.exists(wav_path), f"Sample WAV not found: {wav_path}"

    print("=" * 60)
    print("Fbank C++ vs PyTorch Comparison Test")
    print("=" * 60)

    print("\nComputing PyTorch fbank...")
    pytorch_fbank = compute_pytorch_fbank(wav_path)
    print(f"  Shape: {pytorch_fbank.shape}")
    print(f"  Range: [{pytorch_fbank.min():.4f}, {pytorch_fbank.max():.4f}]")
    print(f"  Mean: {pytorch_fbank.mean():.4f}")

    print("\nComputing C++ fbank...")
    cpp_fbank = run_cpp_fbank(wav_path)
    print(f"  Shape: {cpp_fbank.shape}")
    print(f"  Range: [{cpp_fbank.min():.4f}, {cpp_fbank.max():.4f}]")
    print(f"  Mean: {cpp_fbank.mean():.4f}")

    print("\nComparing results...")

    assert pytorch_fbank.shape == cpp_fbank.shape, (
        f"Shape mismatch: PyTorch {pytorch_fbank.shape} vs C++ {cpp_fbank.shape}"
    )
    print(f"  Shape match: {pytorch_fbank.shape}")

    max_abs_err = float(np.max(np.abs(pytorch_fbank - cpp_fbank)))
    mean_abs_err = float(np.mean(np.abs(pytorch_fbank - cpp_fbank)))
    print(f"  Max absolute error: {max_abs_err:.6f}")
    print(f"  Mean absolute error: {mean_abs_err:.6f}")

    cos_sim = cosine_similarity(pytorch_fbank, cpp_fbank)
    print(f"  Cosine similarity: {cos_sim:.6f}")

    print("\n" + "=" * 60)

    assert max_abs_err < 0.01, f"Max absolute error {max_abs_err:.6f} >= 0.01"
    print("PASS: Max absolute error < 0.01")

    assert cos_sim > 0.999, f"Cosine similarity {cos_sim:.6f} <= 0.999"
    print("PASS: Cosine similarity > 0.999")

    print("\nAll checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_fbank_matches_pytorch()
