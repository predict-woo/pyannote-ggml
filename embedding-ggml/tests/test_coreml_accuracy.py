#!/usr/bin/env python3
"""Compare CoreML vs PyTorch embedding model output on REAL audio.

Loads a real audio file, computes speaker embeddings via both PyTorch and
CoreML, and compares the 256-dim vectors.  Optionally also compares GGML
output if the binary is available.
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


def compute_pytorch_embedding(audio_path: str, duration_s: float = 5.0):
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
        embedding = model(waveform.unsqueeze(0))  # (batch, channel, samples)

    return embedding.squeeze().numpy(), model, waveform


def compute_coreml_embedding(model, waveform: torch.Tensor, coreml_path: str):
    import coremltools as ct

    mlmodel = ct.models.MLModel(coreml_path)

    with torch.no_grad():
        fbank = model.compute_fbank(waveform.unsqueeze(0))  # (1, T, 80)

    print(f"  Fbank shape: {fbank.shape}")

    coreml_input = {"fbank_features": fbank.numpy()}
    coreml_output = mlmodel.predict(coreml_input)
    coreml_embedding = coreml_output["embedding"].flatten()

    return coreml_embedding


def compute_ggml_embedding(audio_path: str, model_path: str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    binary = os.path.join(script_dir, '..', 'build', 'bin', 'embedding-ggml')
    binary = os.path.abspath(binary)
    cwd = os.path.abspath(os.path.join(script_dir, '..'))
    output_file = os.path.join(cwd, 'embedding_output.bin')

    if not os.path.exists(binary):
        return None

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

    with open(output_file, 'rb') as f:
        dim = struct.unpack('<i', f.read(4))[0]
        data = np.frombuffer(f.read(dim * 4), dtype=np.float32).copy()

    os.remove(output_file)

    return data


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def print_embedding_stats(name: str, emb: np.ndarray):
    print(f"  Shape: {emb.shape}")
    print(f"  L2 norm: {np.linalg.norm(emb):.6f}")
    print(f"  Range: [{emb.min():.6f}, {emb.max():.6f}]")
    print(f"  First 5: {emb[:5]}")


def compare_embeddings(name_a: str, emb_a: np.ndarray, name_b: str, emb_b: np.ndarray,
                       cosine_threshold: float, max_err_threshold: float):
    print(f"\n{'-' * 60}")
    print(f"Comparing {name_a} vs {name_b}...")
    print(f"{'-' * 60}")

    assert emb_a.shape == emb_b.shape, (
        f"Shape mismatch: {name_a} {emb_a.shape} vs {name_b} {emb_b.shape}"
    )

    cos_sim = cosine_similarity(emb_a, emb_b)
    print(f"  Cosine similarity: {cos_sim:.6f}")

    abs_diff = np.abs(emb_a - emb_b)
    max_err = float(abs_diff.max())
    mean_err = float(abs_diff.mean())
    print(f"  Max absolute error: {max_err:.6f}")
    print(f"  Mean absolute error: {mean_err:.6f}")

    print()
    cos_pass = cos_sim > cosine_threshold
    err_pass = max_err < max_err_threshold

    print(f"{'PASS' if cos_pass else 'FAIL'}: Cosine similarity {cos_sim:.6f} {'>' if cos_pass else '<='} {cosine_threshold}")
    print(f"{'PASS' if err_pass else 'FAIL'}: Max absolute error {max_err:.6f} {'<' if err_pass else '>='} {max_err_threshold}")

    return cos_pass, err_pass


def main():
    print("=" * 60)
    print("CoreML vs PyTorch Accuracy Test (Real Audio)")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_path = os.path.abspath(
        os.path.join(script_dir, '..', '..', 'src', 'pyannote', 'audio', 'sample', 'sample.wav')
    )
    coreml_path = os.path.abspath(
        os.path.join(script_dir, '..', 'embedding.mlpackage')
    )
    ggml_model_path = os.path.abspath(
        os.path.join(script_dir, '..', 'embedding.gguf')
    )

    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)

    if not os.path.exists(coreml_path):
        print(f"ERROR: CoreML model not found: {coreml_path}")
        print("  Convert with: cd embedding-ggml && python convert_coreml.py")
        sys.exit(1)

    print(f"\nAudio:  {os.path.basename(audio_path)}")
    print(f"CoreML: {os.path.basename(coreml_path)}")

    print("\nComputing PyTorch embedding...")
    pytorch_emb, model, waveform = compute_pytorch_embedding(audio_path)
    print_embedding_stats("PyTorch", pytorch_emb)

    print("\nComputing CoreML embedding...")
    coreml_emb = compute_coreml_embedding(model, waveform, coreml_path)
    print_embedding_stats("CoreML", coreml_emb)

    COSINE_THRESHOLD = 0.999
    MAX_ERR_THRESHOLD = 1.0

    cos_pass, err_pass = compare_embeddings(
        "PyTorch", pytorch_emb, "CoreML", coreml_emb,
        COSINE_THRESHOLD, MAX_ERR_THRESHOLD,
    )

    all_pass = cos_pass and err_pass

    ggml_binary = os.path.abspath(
        os.path.join(script_dir, '..', 'build', 'bin', 'embedding-ggml')
    )
    if os.path.exists(ggml_binary) and os.path.exists(ggml_model_path):
        print("\nComputing GGML embedding...")
        try:
            ggml_emb = compute_ggml_embedding(audio_path, ggml_model_path)
            if ggml_emb is not None:
                print_embedding_stats("GGML", ggml_emb)

                ggml_cos_pass, ggml_err_pass = compare_embeddings(
                    "GGML", ggml_emb, "CoreML", coreml_emb,
                    0.99, MAX_ERR_THRESHOLD,
                )
                print("  (GGML vs CoreML comparison is informational only)")
        except Exception as e:
            print(f"  GGML comparison skipped: {e}")
    else:
        print("\n(Skipping GGML comparison — binary or model not found)")

    print()
    if all_pass:
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
