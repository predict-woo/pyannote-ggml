#!/usr/bin/env python3
"""Compare two RTTM files and compute Diarization Error Rate (DER).

Usage:
    python compare_rttm.py <hypothesis.rttm> <reference.rttm> [--collar COLLAR] [--threshold THRESHOLD]

Parses both RTTM files into pyannote.core.Annotation objects, computes DER
using pyannote.metrics, and prints detailed error breakdown.

Exit codes:
    0 — DER delta is within threshold
    1 — DER delta exceeds threshold (or error)
"""

import argparse
import sys
from pathlib import Path

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate


def load_rttm(path: str) -> Annotation:
    """Parse an RTTM file into a pyannote Annotation.

    RTTM format per line:
        SPEAKER <uri> <channel> <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>

    Parameters
    ----------
    path : str
        Path to RTTM file.

    Returns
    -------
    Annotation
        Speaker diarization annotation.
    """
    annotation = Annotation()
    rttm_path = Path(path)

    if not rttm_path.exists():
        raise FileNotFoundError(f"RTTM file not found: {path}")

    with open(rttm_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith(";"):
                continue

            parts = line.split()
            if len(parts) < 8:
                print(
                    f"WARNING: Skipping malformed line {line_num}: {line}",
                    file=sys.stderr,
                )
                continue

            if parts[0] != "SPEAKER":
                continue

            try:
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
            except (ValueError, IndexError) as e:
                print(
                    f"WARNING: Skipping line {line_num} ({e}): {line}",
                    file=sys.stderr,
                )
                continue

            if duration <= 0:
                continue

            annotation[Segment(start, start + duration)] = speaker

    return annotation


def compute_der(
    hypothesis: Annotation,
    reference: Annotation,
    collar: float = 0.0,
    skip_overlap: bool = False,
) -> dict:
    """Compute DER and its components.

    Parameters
    ----------
    hypothesis : Annotation
        Hypothesized diarization.
    reference : Annotation
        Reference diarization.
    collar : float
        Forgiveness collar in seconds around reference boundaries.
    skip_overlap : bool
        Whether to skip overlapping speech regions.

    Returns
    -------
    dict
        Dictionary with keys: der, missed_speech, false_alarm, confusion, total.
    """
    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    der = metric(reference, hypothesis, detailed=True)

    return {
        "der": der["diarization error rate"],
        "missed_speech": der["missed detection"],
        "false_alarm": der["false alarm"],
        "confusion": der["confusion"],
        "total": der["total"],
    }


def format_percentage(value: float) -> str:
    """Format a ratio as percentage string."""
    return f"{value * 100:.2f}%"


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    return f"{seconds:.3f}s"


def main():
    parser = argparse.ArgumentParser(
        description="Compare two RTTM files and compute Diarization Error Rate (DER).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s hypothesis.rttm reference.rttm
  %(prog)s output.rttm ground_truth.rttm --collar 0.25
  %(prog)s cpp_output.rttm python_output.rttm --threshold 2.0
""",
    )
    parser.add_argument(
        "hypothesis",
        help="Path to hypothesis RTTM file",
    )
    parser.add_argument(
        "reference",
        help="Path to reference RTTM file",
    )
    parser.add_argument(
        "--collar",
        type=float,
        default=0.0,
        help="Forgiveness collar in seconds (default: 0.0)",
    )
    parser.add_argument(
        "--skip-overlap",
        action="store_true",
        default=False,
        help="Skip overlapping speech regions when computing DER",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Maximum acceptable DER in %% (default: 1.0)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("RTTM Comparison — Diarization Error Rate")
    print("=" * 60)

    try:
        print(f"\nHypothesis: {args.hypothesis}")
        hypothesis = load_rttm(args.hypothesis)
        hyp_labels = hypothesis.labels()
        hyp_segments = len(list(hypothesis.itertracks()))
        print(f"  Speakers: {len(hyp_labels)} ({', '.join(sorted(hyp_labels))})")
        print(f"  Segments: {hyp_segments}")

        print(f"\nReference:  {args.reference}")
        reference = load_rttm(args.reference)
        ref_labels = reference.labels()
        ref_segments = len(list(reference.itertracks()))
        print(f"  Speakers: {len(ref_labels)} ({', '.join(sorted(ref_labels))})")
        print(f"  Segments: {ref_segments}")

    except FileNotFoundError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1

    print(f"\nSettings: collar={args.collar}s, skip_overlap={args.skip_overlap}")
    print("-" * 60)

    results = compute_der(
        hypothesis,
        reference,
        collar=args.collar,
        skip_overlap=args.skip_overlap,
    )

    der_pct = results["der"] * 100
    total_s = results["total"]

    print(f"\n  Total reference duration: {format_duration(total_s)}")
    print(f"  Missed speech:           {format_percentage(results['missed_speech'] / total_s if total_s > 0 else 0)} ({format_duration(results['missed_speech'])})")
    print(f"  False alarm:             {format_percentage(results['false_alarm'] / total_s if total_s > 0 else 0)} ({format_duration(results['false_alarm'])})")
    print(f"  Speaker confusion:       {format_percentage(results['confusion'] / total_s if total_s > 0 else 0)} ({format_duration(results['confusion'])})")
    print(f"  ----------------------------------------")
    print(f"  Diarization Error Rate:  {der_pct:.2f}%")

    print()
    passed = der_pct <= args.threshold
    status = "PASS" if passed else "FAIL"
    symbol = "+" if passed else "x"

    print(f"  [{symbol}] {status}: DER {der_pct:.2f}% {'<=' if passed else '>'} {args.threshold:.1f}% threshold")
    print()
    print("=" * 60)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
