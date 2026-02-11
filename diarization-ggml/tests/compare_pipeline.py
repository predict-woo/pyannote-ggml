#!/usr/bin/env python3
"""Stage-by-stage comparison of Python pyannote and C++ diarization pipelines.

Runs both implementations on the same audio file, captures intermediate
outputs at each pipeline stage, and compares them using per-stage tolerance
thresholds derived from the verification budget.

Currently compares the final RTTM output (end-to-end). Structured to support
per-stage comparison when the C++ binary gains --dump-stage support.

Usage:
    python compare_pipeline.py --audio audio.wav --seg-model seg.gguf --emb-model emb.gguf --plda plda.gguf
    python compare_pipeline.py --help
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


TOLERANCES = {
    "segmentation": {"metric": "cosine", "threshold": 0.998},
    "powerset": {"metric": "exact", "threshold": 0},
    "count": {"metric": "exact", "threshold": 0},
    "embeddings": {"metric": "cosine", "threshold": 0.99},
    "plda": {"metric": "max_abs_diff", "threshold": 1e-5},
    "ahc": {"metric": "exact", "threshold": 0},
    "vbx": {"metric": "max_abs_diff", "threshold": 0.01},
    "rttm": {"metric": "der", "threshold": 1.0},
}


def patch_torch_load():
    """Monkey-patch torch.load to disable weights_only for PyTorch 2.6+."""
    import torch

    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = patched_load


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def compare_stage(
    stage_name: str,
    python_data: np.ndarray,
    cpp_data: np.ndarray,
) -> dict:
    """Compare a single stage's outputs using the tolerance budget.

    Returns dict with keys: stage, metric, value, threshold, passed.
    """
    tol = TOLERANCES[stage_name]
    metric_name = tol["metric"]
    threshold = tol["threshold"]

    if metric_name == "cosine":
        finite_mask = np.isfinite(python_data) & np.isfinite(cpp_data)
        if not finite_mask.any():
            return {
                "stage": stage_name,
                "metric": metric_name,
                "value": 0.0,
                "threshold": threshold,
                "passed": False,
            }
        value = cosine_similarity(python_data[finite_mask], cpp_data[finite_mask])
        passed = value >= threshold

    elif metric_name == "exact":
        value = float(np.sum(python_data != cpp_data))
        passed = value == threshold

    elif metric_name == "max_abs_diff":
        finite_mask = np.isfinite(python_data) & np.isfinite(cpp_data)
        if not finite_mask.any():
            return {
                "stage": stage_name,
                "metric": metric_name,
                "value": float("inf"),
                "threshold": threshold,
                "passed": False,
            }
        value = max_abs_diff(python_data[finite_mask], cpp_data[finite_mask])
        passed = value < threshold

    else:
        raise ValueError(f"Unknown metric: {metric_name}")

    return {
        "stage": stage_name,
        "metric": metric_name,
        "value": value,
        "threshold": threshold,
        "passed": passed,
    }


def compare_der(hypothesis_rttm: str, reference_rttm: str) -> dict:
    """Compare two RTTM files via DER using pyannote.metrics."""
    from compare_rttm import compute_der, load_rttm

    hypothesis = load_rttm(hypothesis_rttm)
    reference = load_rttm(reference_rttm)
    results = compute_der(hypothesis, reference)
    der_pct = results["der"] * 100.0

    return {
        "stage": "rttm",
        "metric": "der",
        "value": der_pct,
        "threshold": TOLERANCES["rttm"]["threshold"],
        "passed": der_pct <= TOLERANCES["rttm"]["threshold"],
        "details": results,
    }


def run_python_pipeline(audio_path: str, output_dir: str) -> dict:
    """Run the pyannote Python pipeline, capturing intermediates via hook.

    Returns dict mapping stage_name -> numpy array or file path.
    """
    import torch
    from pyannote.audio import Pipeline

    patch_torch_load()

    print("Loading Python pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1"
    )

    intermediates = {}

    def capture_hook(step_name, step_artifact, **kwargs):
        if step_artifact is None:
            return
        if hasattr(step_artifact, "data"):
            intermediates[step_name] = np.array(step_artifact.data)
        elif isinstance(step_artifact, np.ndarray):
            intermediates[step_name] = step_artifact
        elif isinstance(step_artifact, torch.Tensor):
            intermediates[step_name] = step_artifact.cpu().numpy()

    print(f"Running Python pipeline on {audio_path}...")
    output = pipeline(audio_path, hook=capture_hook)

    if hasattr(output, "speaker_diarization"):
        diarization = output.speaker_diarization
    else:
        diarization = output

    python_rttm = os.path.join(output_dir, "python.rttm")
    with open(python_rttm, "w") as f:
        uri = Path(audio_path).stem
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            f.write(
                f"SPEAKER {uri} 1 {turn.start:.3f} {turn.end - turn.start:.3f} "
                f"<NA> <NA> {speaker} <NA> <NA>\n"
            )

    intermediates["rttm_path"] = python_rttm

    for name, data in intermediates.items():
        if isinstance(data, np.ndarray):
            np.save(os.path.join(output_dir, f"python_{name}.npy"), data)
            print(f"  [{name}] shape={data.shape} dtype={data.dtype}")

    print(f"  Python RTTM: {python_rttm}")
    return intermediates


def run_cpp_pipeline(
    audio_path: str,
    cpp_binary: str,
    seg_model: str,
    emb_model: str,
    plda_path: str,
    output_dir: str,
    coreml_model: str | None = None,
) -> dict:
    """Run the C++ diarization binary.

    Returns dict mapping stage_name -> numpy array or file path.
    """
    cpp_rttm = os.path.join(output_dir, "cpp.rttm")

    cmd = [
        cpp_binary,
        seg_model,
        emb_model,
        audio_path,
        "--plda", plda_path,
        "-o", cpp_rttm,
    ]

    if coreml_model:
        cmd.extend(["--coreml", coreml_model])

    print(f"Running C++ pipeline: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        print(f"C++ stderr:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(
            f"C++ binary exited with code {result.returncode}"
        )

    if result.stderr:
        print(f"C++ log:\n{result.stderr}")

    outputs = {"rttm_path": cpp_rttm}

    stage_files = {
        "segmentation": os.path.join(output_dir, "cpp_segmentation.npy"),
        "powerset": os.path.join(output_dir, "cpp_powerset.npy"),
        "count": os.path.join(output_dir, "cpp_count.npy"),
        "embeddings": os.path.join(output_dir, "cpp_embeddings.npy"),
        "plda": os.path.join(output_dir, "cpp_plda.npy"),
        "ahc": os.path.join(output_dir, "cpp_ahc.npy"),
        "vbx": os.path.join(output_dir, "cpp_vbx.npy"),
    }
    for stage_name, fpath in stage_files.items():
        if os.path.exists(fpath):
            outputs[stage_name] = np.load(fpath)
            print(f"  [{stage_name}] shape={outputs[stage_name].shape}")

    print(f"  C++ RTTM: {cpp_rttm}")
    return outputs


def print_result(r: dict):
    symbol = "+" if r["passed"] else "x"
    if r["metric"] == "cosine":
        detail = f"cosine={r['value']:.6f} (>= {r['threshold']})"
    elif r["metric"] == "exact":
        detail = f"mismatches={int(r['value'])} (== {int(r['threshold'])})"
    elif r["metric"] == "max_abs_diff":
        detail = f"max_diff={r['value']:.6e} (< {r['threshold']})"
    elif r["metric"] == "der":
        detail = f"DER={r['value']:.2f}% (<= {r['threshold']}%)"
    else:
        detail = f"value={r['value']}"

    status = "PASS" if r["passed"] else "FAIL"
    print(f"  [{symbol}] {r['stage']:20s} {status}  {detail}")


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    repo_root = project_root.parent

    default_audio = str(
        repo_root / "samples" / "sample.wav"
    )
    default_binary = str(
        project_root / "build" / "bin" / "diarization-ggml"
    )

    parser = argparse.ArgumentParser(
        description="Stage-by-stage comparison of Python and C++ diarization pipelines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Tolerance budget:
  segmentation   cosine >= 0.998
  powerset       exact match (binary identical)
  count          exact match (integer identical)
  embeddings     cosine >= 0.99
  plda           max abs diff < 1e-5
  ahc            exact match (cluster assignments)
  vbx            max abs diff < 0.01
  rttm           DER <= 1.0%%

Currently only the RTTM end-to-end comparison is active.
Stage-by-stage comparison activates when C++ --dump-stage is implemented.
""",
    )
    parser.add_argument(
        "--audio",
        default=default_audio,
        help=f"Path to test audio WAV file (default: {default_audio})",
    )
    parser.add_argument(
        "--cpp-binary",
        default=default_binary,
        help=f"Path to C++ diarization binary (default: {default_binary})",
    )
    parser.add_argument(
        "--seg-model",
        required=True,
        help="Path to segmentation GGUF model",
    )
    parser.add_argument(
        "--emb-model",
        required=True,
        help="Path to embedding GGUF model",
    )
    parser.add_argument(
        "--coreml-model",
        default=None,
        help="Path to CoreML embedding model (.mlpackage)",
    )
    parser.add_argument(
        "--plda",
        required=True,
        help="Path to PLDA binary file",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for intermediate outputs (default: auto tmpdir)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"ERROR: Audio file not found: {args.audio}", file=sys.stderr)
        return 1

    if not os.path.exists(args.cpp_binary):
        print(
            f"ERROR: C++ binary not found: {args.cpp_binary}", file=sys.stderr
        )
        return 1

    for label, path in [
        ("seg-model", args.seg_model),
        ("emb-model", args.emb_model),
        ("plda", args.plda),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            return 1

    if args.coreml_model and not os.path.exists(args.coreml_model):
        print(
            f"ERROR: CoreML model not found: {args.coreml_model}",
            file=sys.stderr,
        )
        return 1

    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = tempfile.mkdtemp(prefix="diarization_compare_")

    print("=" * 60)
    print("Diarization Pipeline Comparison")
    print("=" * 60)
    print(f"  Audio:      {args.audio}")
    print(f"  C++ binary: {args.cpp_binary}")
    print(f"  Seg model:  {args.seg_model}")
    print(f"  Emb model:  {args.emb_model}")
    print(f"  PLDA:       {args.plda}")
    if args.coreml_model:
        print(f"  CoreML:     {args.coreml_model}")
    print(f"  Output dir: {output_dir}")
    print()

    print("-" * 60)
    print("Step 1: Python pipeline")
    print("-" * 60)
    python_outputs = run_python_pipeline(args.audio, output_dir)
    print()

    print("-" * 60)
    print("Step 2: C++ pipeline")
    print("-" * 60)
    cpp_outputs = run_cpp_pipeline(
        args.audio,
        args.cpp_binary,
        args.seg_model,
        args.emb_model,
        args.plda,
        output_dir,
        coreml_model=args.coreml_model,
    )
    print()

    print("-" * 60)
    print("Step 3: Comparison")
    print("-" * 60)

    results = []
    all_passed = True

    comparable_stages = [
        "segmentation",
        "powerset",
        "count",
        "embeddings",
        "plda",
        "ahc",
        "vbx",
    ]
    for stage in comparable_stages:
        if stage in python_outputs and stage in cpp_outputs:
            r = compare_stage(stage, python_outputs[stage], cpp_outputs[stage])
            results.append(r)
            print_result(r)
            if not r["passed"]:
                all_passed = False
        else:
            print(f"  [~] {stage:20s} SKIP  (dump not available)")

    python_rttm = python_outputs.get("rttm_path")
    cpp_rttm = cpp_outputs.get("rttm_path")

    if python_rttm and cpp_rttm and os.path.exists(cpp_rttm):
        sys.path.insert(0, str(script_dir))
        r = compare_der(cpp_rttm, python_rttm)
        results.append(r)
        print_result(r)
        if not r["passed"]:
            all_passed = False
    else:
        print(f"  [~] {'rttm':20s} SKIP  (RTTM file missing)")

    print()
    print("=" * 60)
    if all_passed:
        print("ALL CHECKS PASSED")
    else:
        failed = [r["stage"] for r in results if not r["passed"]]
        print(f"FAILED STAGES: {', '.join(failed)}")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
