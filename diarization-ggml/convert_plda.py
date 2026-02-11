#!/usr/bin/env python3
"""Convert PLDA weights to GGUF format for C++ inference.

Two modes:
  1. From .npz source files (pre-computes eigendecomposition):
     python convert_plda.py \
         --transform-npz path/to/xvec_transform.npz \
         --plda-npz path/to/plda.npz

  2. From existing plda.bin (already post-eigendecomposition):
     python convert_plda.py --from-bin plda.bin
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType, GGUFReader

XVEC_DIM = 256
LDA_DIM = 128

PLDA_BIN_MAGIC = b"PLDA"
PLDA_BIN_VERSION = 1


def load_from_npz(transform_path, plda_path):
    from scipy.linalg import eigh

    print(f"Loading transform: {transform_path}")
    x = np.load(transform_path)
    mean1, mean2, lda = x["mean1"], x["mean2"], x["lda"]

    print(f"Loading PLDA: {plda_path}")
    p = np.load(plda_path)
    plda_mu, plda_tr, plda_psi = p["mu"], p["tr"], p["psi"]

    # vbx_setup() eigendecomposition — must happen before saving
    # W = inv(plda_tr.T @ plda_tr), B = inv((plda_tr.T / plda_psi) @ plda_tr)
    # acvar, wccn = eigh(B, W); psi = acvar[::-1]; tr = wccn.T[::-1]
    W = np.linalg.inv(plda_tr.T.dot(plda_tr))
    B = np.linalg.inv((plda_tr.T / plda_psi).dot(plda_tr))
    acvar, wccn = eigh(B, W)
    plda_psi = acvar[::-1]
    plda_tr = wccn.T[::-1]

    return (
        mean1.astype(np.float64),
        mean2.astype(np.float64),
        lda.astype(np.float64),
        plda_mu.astype(np.float64),
        plda_tr.astype(np.float64),
        plda_psi.astype(np.float64),
    )


def load_from_bin(bin_path):
    print(f"Loading existing plda.bin: {bin_path}")
    with open(bin_path, "rb") as f:
        magic = f.read(4)
        assert magic == PLDA_BIN_MAGIC, f"Bad magic: {magic!r}"
        version = struct.unpack("<I", f.read(4))[0]
        assert version == PLDA_BIN_VERSION, f"Bad version: {version}"
        mean1 = np.frombuffer(f.read(XVEC_DIM * 8), dtype=np.float64).copy()
        mean2 = np.frombuffer(f.read(LDA_DIM * 8), dtype=np.float64).copy()
        lda = np.frombuffer(f.read(XVEC_DIM * LDA_DIM * 8), dtype=np.float64).copy().reshape(XVEC_DIM, LDA_DIM)
        plda_mu = np.frombuffer(f.read(LDA_DIM * 8), dtype=np.float64).copy()
        plda_tr = np.frombuffer(f.read(LDA_DIM * LDA_DIM * 8), dtype=np.float64).copy().reshape(LDA_DIM, LDA_DIM)
        plda_psi = np.frombuffer(f.read(LDA_DIM * 8), dtype=np.float64).copy()
        assert len(f.read()) == 0, "Unexpected trailing bytes"
    print(f"  Loaded 6 arrays (already post-eigendecomposition)")
    return mean1, mean2, lda, plda_mu, plda_tr, plda_psi


def write_plda_gguf(output_path, mean1, mean2, lda, plda_mu, plda_tr, plda_psi):
    writer = GGUFWriter(output_path, "plda")
    writer.add_name("pyannote-plda-vbx")
    writer.add_uint32("plda.xvec_dim", XVEC_DIM)
    writer.add_uint32("plda.lda_dim", LDA_DIM)

    tensors = [
        ("plda.mean1", np.ascontiguousarray(mean1, dtype=np.float64)),
        ("plda.mean2", np.ascontiguousarray(mean2, dtype=np.float64)),
        ("plda.lda",   np.ascontiguousarray(lda, dtype=np.float64)),
        ("plda.mu",    np.ascontiguousarray(plda_mu, dtype=np.float64)),
        ("plda.tr",    np.ascontiguousarray(plda_tr, dtype=np.float64)),
        ("plda.psi",   np.ascontiguousarray(plda_psi, dtype=np.float64)),
    ]

    for name, data in tensors:
        writer.add_tensor(name, data, raw_dtype=GGMLQuantizationType.F64)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = Path(output_path).stat().st_size
    print(f"\nWrote {output_path} ({file_size:,} bytes)")
    for name, data in tensors:
        print(f"  {name:16s}  {str(data.shape):16s}  f64")


def validate_gguf(gguf_path, mean1, mean2, lda, plda_mu, plda_tr, plda_psi):
    print(f"\nValidating: {gguf_path}")
    reader = GGUFReader(str(gguf_path))
    tensor_map = {t.name: t.data for t in reader.tensors}

    checks = [
        ("plda.mean1", mean1), ("plda.mean2", mean2), ("plda.lda", lda),
        ("plda.mu", plda_mu), ("plda.tr", plda_tr), ("plda.psi", plda_psi),
    ]

    all_ok = True
    for name, expected in checks:
        actual = tensor_map[name].ravel()
        max_err = np.max(np.abs(expected.astype(np.float64).ravel() - actual))
        status = "OK" if max_err < 1e-15 else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  {status}: {name} (max_err={max_err:.2e})")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Convert PLDA weights to GGUF format")
    parser.add_argument("--from-bin", type=str, help="Convert existing plda.bin to GGUF")
    parser.add_argument("--transform-npz", type=str, help="Path to xvec_transform.npz")
    parser.add_argument("--plda-npz", type=str, help="Path to plda.npz")
    parser.add_argument("-o", "--output", type=str, default="plda.gguf")
    args = parser.parse_args()

    if args.from_bin:
        mean1, mean2, lda, plda_mu, plda_tr, plda_psi = load_from_bin(args.from_bin)
    elif args.transform_npz and args.plda_npz:
        import glob
        transform_matches = glob.glob(args.transform_npz)
        plda_matches = glob.glob(args.plda_npz)
        if not transform_matches:
            sys.exit(f"Error: no match for {args.transform_npz}")
        if not plda_matches:
            sys.exit(f"Error: no match for {args.plda_npz}")
        mean1, mean2, lda, plda_mu, plda_tr, plda_psi = load_from_npz(
            transform_matches[0], plda_matches[0]
        )
    else:
        parser.error("Provide either --from-bin or both --transform-npz and --plda-npz")

    write_plda_gguf(args.output, mean1, mean2, lda, plda_mu, plda_tr, plda_psi)
    validate_gguf(args.output, mean1, mean2, lda, plda_mu, plda_tr, plda_psi)
    print("\nDone.")


if __name__ == "__main__":
    main()
