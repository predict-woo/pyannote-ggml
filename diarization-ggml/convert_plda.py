#!/usr/bin/env python3
"""Convert PLDA .npz files to flat binary format for C++ inference.

Loads xvec_transform.npz and plda.npz from the pyannote HuggingFace hub,
pre-computes the eigendecomposition (avoiding LAPACK in C++), and writes
a flat binary file that C++ can load with simple fread() calls.

Binary format (plda.bin):
    Magic:    b"PLDA"           (4 bytes)
    Version:  uint32 = 1        (4 bytes)
    mean1:    float64[256]      (2048 bytes)
    mean2:    float64[128]      (1024 bytes)
    lda:      float64[256*128]  (262144 bytes, row-major)
    plda_mu:  float64[128]      (1024 bytes)
    plda_tr:  float64[128*128]  (131072 bytes, row-major, post-eigendecomposition)
    plda_psi: float64[128]      (1024 bytes, post-eigendecomposition)

Usage:
    python convert_plda.py \\
        --transform-npz path/to/xvec_transform.npz \\
        --plda-npz path/to/plda.npz \\
        --output plda.bin
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

PLDA_MAGIC = b"PLDA"
PLDA_VERSION = 1

XVEC_DIM = 256
LDA_DIM = 128


def l2_norm(vec_or_matrix):
    """L2 normalization of vector array (matches vbx.py implementation).

    Args:
        vec_or_matrix: one vector or array of vectors

    Returns:
        normalized vector or array of normalized vectors
    """
    if len(vec_or_matrix.shape) == 1:
        return vec_or_matrix / np.linalg.norm(vec_or_matrix)
    elif len(vec_or_matrix.shape) == 2:
        return vec_or_matrix / np.linalg.norm(vec_or_matrix, axis=1, ord=2)[:, np.newaxis]
    else:
        raise ValueError(f"Expected 1D or 2D array, got {len(vec_or_matrix.shape)}D")


def print_array_stats(name, arr):
    """Print shape and statistics for an array."""
    print(f"  {name:20s}  shape={str(arr.shape):16s}  dtype={str(arr.dtype):10s}  "
          f"min={arr.min():12.6f}  max={arr.max():12.6f}  mean={arr.mean():12.6f}")


def load_and_validate_npz(transform_path, plda_path):
    """Load and validate the .npz files.

    Args:
        transform_path: Path to xvec_transform.npz
        plda_path: Path to plda.npz

    Returns:
        Tuple of (mean1, mean2, lda, plda_mu, plda_tr, plda_psi)
    """
    print(f"Loading transform: {transform_path}")
    x = np.load(transform_path)
    mean1, mean2, lda = x["mean1"], x["mean2"], x["lda"]

    print(f"Loading PLDA: {plda_path}")
    p = np.load(plda_path)
    plda_mu, plda_tr, plda_psi = p["mu"], p["tr"], p["psi"]

    print("\nOriginal arrays (before conversion):")
    print_array_stats("mean1", mean1)
    print_array_stats("mean2", mean2)
    print_array_stats("lda", lda)
    print_array_stats("plda_mu", plda_mu)
    print_array_stats("plda_tr", plda_tr)
    print_array_stats("plda_psi", plda_psi)

    errors = []
    if mean1.shape != (XVEC_DIM,):
        errors.append(f"mean1: expected ({XVEC_DIM},), got {mean1.shape}")
    if mean2.shape != (LDA_DIM,):
        errors.append(f"mean2: expected ({LDA_DIM},), got {mean2.shape}")
    if lda.shape != (XVEC_DIM, LDA_DIM):
        errors.append(f"lda: expected ({XVEC_DIM}, {LDA_DIM}), got {lda.shape}")
    if plda_mu.shape != (LDA_DIM,):
        errors.append(f"plda_mu: expected ({LDA_DIM},), got {plda_mu.shape}")
    if plda_tr.shape != (LDA_DIM, LDA_DIM):
        errors.append(f"plda_tr: expected ({LDA_DIM}, {LDA_DIM}), got {plda_tr.shape}")
    if plda_psi.shape != (LDA_DIM,):
        errors.append(f"plda_psi: expected ({LDA_DIM},), got {plda_psi.shape}")

    if errors:
        print("\nShape validation errors:")
        for e in errors:
            print(f"  ERROR: {e}")
        sys.exit(1)

    print("\n  All shapes validated OK")
    return mean1, mean2, lda, plda_mu, plda_tr, plda_psi


def compute_eigendecomposition(plda_tr, plda_psi):
    """Pre-compute the eigendecomposition from vbx_setup().

    This performs the exact computation from vbx.py:201-208:
        W = inv(plda_tr.T @ plda_tr)
        B = inv((plda_tr.T / plda_psi) @ plda_tr)
        acvar, wccn = eigh(B, W)
        plda_psi = acvar[::-1]
        plda_tr = wccn.T[::-1]

    Args:
        plda_tr: Original PLDA transform matrix (128, 128)
        plda_psi: Original PLDA eigenvalues (128,)

    Returns:
        Tuple of (new_plda_tr, new_plda_psi) after eigendecomposition
    """
    print("\nComputing eigendecomposition (vbx_setup)...")

    # Within-class covariance matrix
    W = np.linalg.inv(plda_tr.T.dot(plda_tr))
    print(f"  W = inv(plda_tr.T @ plda_tr): shape={W.shape}")

    # Between-class covariance matrix
    B = np.linalg.inv((plda_tr.T / plda_psi).dot(plda_tr))
    print(f"  B = inv((plda_tr.T / plda_psi) @ plda_tr): shape={B.shape}")

    # Generalized eigenvalue problem
    acvar, wccn = eigh(B, W)
    print(f"  eigh(B, W): acvar shape={acvar.shape}, wccn shape={wccn.shape}")

    # Reverse order (largest eigenvalues first)
    new_plda_psi = acvar[::-1]
    new_plda_tr = wccn.T[::-1]

    print(f"  Post-eigendecomposition plda_psi: min={new_plda_psi.min():.6f}, "
          f"max={new_plda_psi.max():.6f}")
    print(f"  Post-eigendecomposition plda_tr: shape={new_plda_tr.shape}")

    return new_plda_tr, new_plda_psi


def write_binary(output_path, mean1, mean2, lda, plda_mu, plda_tr, plda_psi):
    """Write all arrays to flat binary file.

    All arrays are converted to float64 and written in row-major order.

    Args:
        output_path: Path for output binary file
        mean1: float64[256]
        mean2: float64[128]
        lda: float64[256, 128]
        plda_mu: float64[128]
        plda_tr: float64[128, 128] (post-eigendecomposition)
        plda_psi: float64[128] (post-eigendecomposition)
    """
    print(f"\nWriting binary file: {output_path}")

    mean1 = np.ascontiguousarray(mean1, dtype=np.float64)
    mean2 = np.ascontiguousarray(mean2, dtype=np.float64)
    lda = np.ascontiguousarray(lda, dtype=np.float64)
    plda_mu = np.ascontiguousarray(plda_mu, dtype=np.float64)
    plda_tr = np.ascontiguousarray(plda_tr, dtype=np.float64)
    plda_psi = np.ascontiguousarray(plda_psi, dtype=np.float64)

    with open(output_path, "wb") as f:
        f.write(PLDA_MAGIC)
        f.write(struct.pack("<I", PLDA_VERSION))
        f.write(mean1.tobytes())
        f.write(mean2.tobytes())
        f.write(lda.tobytes())
        f.write(plda_mu.tobytes())
        f.write(plda_tr.tobytes())
        f.write(plda_psi.tobytes())

        file_size = f.tell()

    expected_size = (
        4                           # magic
        + 4                         # version
        + XVEC_DIM * 8             # mean1
        + LDA_DIM * 8              # mean2
        + XVEC_DIM * LDA_DIM * 8  # lda
        + LDA_DIM * 8              # plda_mu
        + LDA_DIM * LDA_DIM * 8   # plda_tr
        + LDA_DIM * 8              # plda_psi
    )

    print(f"  File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
    print(f"  Expected:  {expected_size:,} bytes")

    if file_size != expected_size:
        print(f"  ERROR: Size mismatch! Expected {expected_size}, got {file_size}")
        sys.exit(1)
    else:
        print("  Size verified OK")


def generate_validation(output_path, mean1, mean2, lda, plda_mu, plda_tr, plda_psi):
    """Generate validation data for C++ implementation testing.

    Creates a .npz file with:
    - test_embeddings: random (10, 256) float32 embeddings
    - xvec_transform_output: expected output after xvec transform
    - plda_transform_output: expected output after plda transform

    Args:
        output_path: Path for validation .npz file
        mean1, mean2, lda: xvec transform parameters (float64)
        plda_mu, plda_tr, plda_psi: PLDA parameters (float64, post-eigendecomposition)
    """
    print(f"\nGenerating validation data: {output_path}")

    np.random.seed(42)
    test_embeddings = np.random.randn(10, XVEC_DIM).astype(np.float32)
    print(f"  Test embeddings: shape={test_embeddings.shape}, dtype={test_embeddings.dtype}")

    # xvec transform (vbx.py:211-213):
    # center(mean1) -> L2norm -> scale(sqrt(256)) -> LDA.T @ x -> center(mean2) -> scale(sqrt(128)) -> L2norm
    centered = test_embeddings - mean1
    normed = l2_norm(centered)
    scaled = np.sqrt(lda.shape[0]) * normed
    projected = lda.T.dot(scaled.T).T
    recentered = projected - mean2
    xvec_output = np.sqrt(lda.shape[1]) * l2_norm(recentered)

    print(f"  xvec_transform output: shape={xvec_output.shape}")
    print_array_stats("    xvec_output", xvec_output)

    # plda transform (vbx.py:215-217): center(plda_mu) -> plda_tr.T @ x -> truncate to lda_dim
    plda_centered = xvec_output - plda_mu
    plda_output = plda_centered.dot(plda_tr.T)[:, :LDA_DIM]

    print(f"  plda_transform output: shape={plda_output.shape}")
    print_array_stats("    plda_output", plda_output)

    np.savez(
        output_path,
        test_embeddings=test_embeddings,
        xvec_transform_output=xvec_output.astype(np.float64),
        plda_transform_output=plda_output.astype(np.float64),
    )

    print(f"  Saved validation file: {output_path}")
    return test_embeddings, xvec_output, plda_output


def validate_binary(binary_path, mean1, mean2, lda, plda_mu, plda_tr, plda_psi):
    """Read back the binary file and verify it matches the original arrays.

    Args:
        binary_path: Path to the binary file
        mean1, mean2, lda, plda_mu, plda_tr, plda_psi: Expected arrays (float64)

    Returns:
        True if validation passes
    """
    print(f"\nValidating binary file: {binary_path}")

    with open(binary_path, "rb") as f:
        magic = f.read(4)
        assert magic == PLDA_MAGIC, f"Bad magic: {magic!r}"

        version = struct.unpack("<I", f.read(4))[0]
        assert version == PLDA_VERSION, f"Bad version: {version}"

        r_mean1 = np.frombuffer(f.read(XVEC_DIM * 8), dtype=np.float64)
        r_mean2 = np.frombuffer(f.read(LDA_DIM * 8), dtype=np.float64)
        r_lda = np.frombuffer(f.read(XVEC_DIM * LDA_DIM * 8), dtype=np.float64).reshape(XVEC_DIM, LDA_DIM)
        r_plda_mu = np.frombuffer(f.read(LDA_DIM * 8), dtype=np.float64)
        r_plda_tr = np.frombuffer(f.read(LDA_DIM * LDA_DIM * 8), dtype=np.float64).reshape(LDA_DIM, LDA_DIM)
        r_plda_psi = np.frombuffer(f.read(LDA_DIM * 8), dtype=np.float64)

        remaining = f.read()
        assert len(remaining) == 0, f"Unexpected trailing bytes: {len(remaining)}"

    checks = [
        ("mean1", mean1, r_mean1),
        ("mean2", mean2, r_mean2),
        ("lda", lda, r_lda),
        ("plda_mu", plda_mu, r_plda_mu),
        ("plda_tr", plda_tr, r_plda_tr),
        ("plda_psi", plda_psi, r_plda_psi),
    ]

    all_ok = True
    for name, expected, actual in checks:
        expected_f64 = expected.astype(np.float64)
        max_err = np.max(np.abs(expected_f64.ravel() - actual.ravel()))
        if max_err > 1e-15:
            print(f"  FAIL: {name} max error = {max_err:.2e}")
            all_ok = False
        else:
            print(f"  OK:   {name} (exact match)")

    if all_ok:
        print("  All arrays validated OK — binary matches source data exactly")
    else:
        print("  WARNING: Some arrays have non-zero error (may be dtype conversion)")

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Convert PLDA .npz files to flat binary format for C++ inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert from HuggingFace cache
    python convert_plda.py \\
        --transform-npz ~/.cache/huggingface/hub/models--pyannote--speaker-diarization-community-1/snapshots/*/plda/xvec_transform.npz \\
        --plda-npz ~/.cache/huggingface/hub/models--pyannote--speaker-diarization-community-1/snapshots/*/plda/plda.npz \\
        --output plda.bin

    # Convert from local files
    python convert_plda.py \\
        --transform-npz ./xvec_transform.npz \\
        --plda-npz ./plda.npz \\
        --output plda.bin

Binary format:
    Magic:    b"PLDA"           (4 bytes)
    Version:  uint32 = 1        (4 bytes)
    mean1:    float64[256]      (2048 bytes)
    mean2:    float64[128]      (1024 bytes)
    lda:      float64[256*128]  (262144 bytes, row-major)
    plda_mu:  float64[128]      (1024 bytes)
    plda_tr:  float64[128*128]  (131072 bytes, row-major, post-eigendecomposition)
    plda_psi: float64[128]      (1024 bytes, post-eigendecomposition)
    Total:    397,320 bytes (~388 KB)
""",
    )
    parser.add_argument(
        "--transform-npz",
        type=str,
        required=True,
        help="Path to xvec_transform.npz (contains mean1, mean2, lda)",
    )
    parser.add_argument(
        "--plda-npz",
        type=str,
        required=True,
        help="Path to plda.npz (contains mu, tr, psi)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="plda.bin",
        help="Output binary file path (default: plda.bin)",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip generating validation .npz file",
    )

    args = parser.parse_args()

    import glob

    transform_matches = glob.glob(args.transform_npz)
    if transform_matches:
        transform_path = Path(transform_matches[0])
        if len(transform_matches) > 1:
            print(f"Warning: Multiple matches for transform-npz, using: {transform_path}")
    else:
        transform_path = Path(args.transform_npz)

    plda_matches = glob.glob(args.plda_npz)
    if plda_matches:
        plda_path = Path(plda_matches[0])
        if len(plda_matches) > 1:
            print(f"Warning: Multiple matches for plda-npz, using: {plda_path}")
    else:
        plda_path = Path(args.plda_npz)

    if not transform_path.exists():
        print(f"Error: Transform file not found: {transform_path}")
        sys.exit(1)
    if not plda_path.exists():
        print(f"Error: PLDA file not found: {plda_path}")
        sys.exit(1)

    output_path = Path(args.output)

    print("=" * 60)
    print("PLDA Converter — .npz to flat binary")
    print("=" * 60)

    mean1, mean2, lda, plda_mu, plda_tr_orig, plda_psi_orig = load_and_validate_npz(
        transform_path, plda_path
    )

    plda_tr, plda_psi = compute_eigendecomposition(plda_tr_orig, plda_psi_orig)

    print("\nConverting all arrays to float64...")
    mean1 = mean1.astype(np.float64)
    mean2 = mean2.astype(np.float64)
    lda = lda.astype(np.float64)
    plda_mu = plda_mu.astype(np.float64)
    plda_tr = plda_tr.astype(np.float64)
    plda_psi = plda_psi.astype(np.float64)

    print("\nFinal arrays (float64):")
    print_array_stats("mean1", mean1)
    print_array_stats("mean2", mean2)
    print_array_stats("lda", lda)
    print_array_stats("plda_mu", plda_mu)
    print_array_stats("plda_tr", plda_tr)
    print_array_stats("plda_psi", plda_psi)

    write_binary(output_path, mean1, mean2, lda, plda_mu, plda_tr, plda_psi)
    validate_binary(output_path, mean1, mean2, lda, plda_mu, plda_tr, plda_psi)

    validation_path = None
    if not args.no_validation:
        validation_path = output_path.with_name("plda_validation.npz")
        generate_validation(
            validation_path, mean1, mean2, lda, plda_mu, plda_tr, plda_psi
        )

    print("\n" + "=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print(f"  Input transform: {transform_path}")
    print(f"  Input PLDA:      {plda_path}")
    print(f"  Output binary:   {output_path}")
    if validation_path:
        print(f"  Validation:      {validation_path}")
    print(f"  Binary size:     {output_path.stat().st_size:,} bytes")
    print(f"\nConversion complete!")


if __name__ == "__main__":
    main()
