#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import tempfile


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class CheckRunner:
    def __init__(self):
        self.total = 0
        self.passed = 0

    def check(self, name, ok, fail_detail=""):
        self.total += 1
        if ok:
            self.passed += 1
            print(f"[PASS] {name}")
            return True
        detail = f" ({fail_detail})" if fail_detail else ""
        print(f"[FAIL] {name}{detail}")
        return False

    def summary(self):
        print(f"\n{self.passed}/{self.total} checks passed")
        return self.passed == self.total


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end integration test for transcribe CLI",
    )
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--seg-model")
    parser.add_argument("--emb-model")
    parser.add_argument("--whisper-model")
    parser.add_argument("--plda")
    parser.add_argument("--seg-coreml")
    parser.add_argument("--emb-coreml")
    parser.add_argument("--vad-model")
    parser.add_argument("--audio", default="../samples/sample.wav")
    parser.add_argument("--reference-rttm")
    args = parser.parse_args()

    env_map = {
        "seg_model": "SEG_MODEL",
        "emb_model": "EMB_MODEL",
        "whisper_model": "WHISPER_MODEL",
        "plda": "PLDA",
        "seg_coreml": "SEG_COREML",
        "emb_coreml": "EMB_COREML",
        "vad_model": "VAD_MODEL",
        "reference_rttm": "REFERENCE_RTTM",
    }

    for field, env_name in env_map.items():
        if getattr(args, field) is None:
            env_value = os.environ.get(env_name)
            if env_value:
                setattr(args, field, env_value)

    missing = []
    for required in ["seg_model", "emb_model", "whisper_model", "plda"]:
        if not getattr(args, required):
            missing.append(required)

    if missing:
        parser.error(
            "missing required model path(s): "
            + ", ".join(missing)
            + " (pass args or set env vars SEG_MODEL/EMB_MODEL/WHISPER_MODEL/PLDA)"
        )

    return args


def run_command(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def validate_transcribe_json(data, checks):
    ok = True
    segments = data.get("segments") if isinstance(data, dict) else None
    ok &= checks.check("JSON has 'segments' list", isinstance(segments, list))
    if not isinstance(segments, list):
        return False

    ok &= checks.check("At least 1 segment exists", len(segments) >= 1)

    speakers = set()
    total_words = 0
    prev_segment_start = None

    segments_sorted = True
    segment_start_non_negative = True
    word_time_non_negative = True
    segment_fields_valid = True
    word_fields_valid = True
    word_time_order_valid = True
    monotonic_within_segment = True

    for seg in segments:
        if not isinstance(seg, dict):
            segment_fields_valid = False
            continue

        speaker = seg.get("speaker")
        start = seg.get("start")
        duration = seg.get("duration")
        words = seg.get("words")

        if (
            not isinstance(speaker, str)
            or not isinstance(start, (int, float))
            or isinstance(start, bool)
            or not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not isinstance(words, list)
        ):
            segment_fields_valid = False
            continue

        start_f = float(start)
        words_list = words
        speakers.add(speaker)

        if start_f < 0:
            segment_start_non_negative = False

        if prev_segment_start is not None and start_f < prev_segment_start:
            segments_sorted = False
        prev_segment_start = start_f

        prev_word_start = None
        prev_word_end = None
        for word in words_list:
            total_words += 1
            if not isinstance(word, dict):
                word_fields_valid = False
                continue

            text = word.get("text")
            w_start = word.get("start")
            w_end = word.get("end")

            if (
                not isinstance(text, str)
                or text.strip() == ""
                or not isinstance(w_start, (int, float))
                or isinstance(w_start, bool)
                or not isinstance(w_end, (int, float))
                or isinstance(w_end, bool)
            ):
                word_fields_valid = False
                continue

            w_start_f = float(w_start)
            w_end_f = float(w_end)

            if w_start_f < 0 or w_end_f < 0:
                word_time_non_negative = False
            if w_start_f > w_end_f:
                word_time_order_valid = False

            if prev_word_start is not None and w_start_f < prev_word_start:
                monotonic_within_segment = False
            if prev_word_end is not None and w_end_f < prev_word_end:
                monotonic_within_segment = False

            prev_word_start = w_start_f
            prev_word_end = w_end_f

    ok &= checks.check("At least 2 distinct speakers", len(speakers) >= 2)
    ok &= checks.check("Each segment has required types", segment_fields_valid)
    ok &= checks.check("Each word has required types", word_fields_valid)
    ok &= checks.check("Word timestamps satisfy start <= end", word_time_order_valid)
    ok &= checks.check(
        "Word timestamps in each segment are non-decreasing",
        monotonic_within_segment,
    )
    ok &= checks.check("Total word count >= 10", total_words >= 10)
    ok &= checks.check("All word timestamps are >= 0", word_time_non_negative)
    ok &= checks.check("All segment starts are >= 0", segment_start_non_negative)
    ok &= checks.check("Segments are sorted by start", segments_sorted)

    return ok


def maybe_run_diarization_regression(args, checks):
    diar_bin = os.path.join(args.build_dir, "bin", "diarization-ggml")
    compare_script = os.path.join("tests", "compare_rttm.py")

    required_for_regression = [args.seg_model, args.emb_model, args.plda, args.audio]
    if not os.path.exists(diar_bin):
        print("[SKIP] Diarization regression: diarization-ggml binary not found")
        return
    if not all(os.path.exists(path) for path in required_for_regression):
        print("[SKIP] Diarization regression: model/audio path missing")
        return

    if not args.reference_rttm:
        print("[SKIP] Diarization regression: --reference-rttm not provided")
        return
    if not os.path.exists(args.reference_rttm):
        print("[SKIP] Diarization regression: reference RTTM not found")
        return

    with tempfile.NamedTemporaryFile(suffix=".rttm", delete=False) as tmp:
        out_rttm = tmp.name

    try:
        diar_cmd = [
            diar_bin,
            args.seg_model,
            args.emb_model,
            args.audio,
            "--plda",
            args.plda,
            "-o",
            out_rttm,
        ]
        if args.emb_coreml:
            diar_cmd.extend(["--coreml", args.emb_coreml])
        if args.seg_coreml:
            diar_cmd.extend(["--seg-coreml", args.seg_coreml])

        diar_proc = run_command(diar_cmd)
        checks.check(
            "Diarization CLI exits successfully",
            diar_proc.returncode == 0,
            diar_proc.stderr.strip() or diar_proc.stdout.strip(),
        )

        if diar_proc.returncode != 0:
            return

        compare_cmd = [
            sys.executable,
            compare_script,
            out_rttm,
            args.reference_rttm,
            "--threshold",
            "1.0",
        ]
        compare_proc = run_command(compare_cmd)
        checks.check(
            "DER regression (<= 1.0%) passes",
            compare_proc.returncode == 0,
            compare_proc.stdout.strip() or compare_proc.stderr.strip(),
        )
    finally:
        try:
            os.unlink(out_rttm)
        except OSError:
            pass


def main():
    args = parse_args()
    checks = CheckRunner()

    transcribe_bin = os.path.join(args.build_dir, "bin", "transcribe")
    checks.check("Transcribe binary exists", os.path.exists(transcribe_bin))

    transcribe_cmd = [
        transcribe_bin,
        args.audio,
        "--seg-model",
        args.seg_model,
        "--emb-model",
        args.emb_model,
        "--whisper-model",
        args.whisper_model,
        "--plda",
        args.plda,
    ]
    if args.seg_coreml:
        transcribe_cmd.extend(["--seg-coreml", args.seg_coreml])
    if args.emb_coreml:
        transcribe_cmd.extend(["--emb-coreml", args.emb_coreml])
    if args.vad_model:
        transcribe_cmd.extend(["--vad-model", args.vad_model])

    proc = run_command(transcribe_cmd)
    checks.check(
        "Transcribe CLI exits successfully",
        proc.returncode == 0,
        proc.stderr.strip() or proc.stdout.strip(),
    )

    transcribe_json_ok = False
    if proc.returncode == 0:
        try:
            payload = json.loads(proc.stdout)
            transcribe_json_ok = checks.check("Transcribe stdout is valid JSON", True)
        except json.JSONDecodeError as exc:
            payload = None
            checks.check("Transcribe stdout is valid JSON", False, str(exc))

        if transcribe_json_ok:
            validate_transcribe_json(payload, checks)

    maybe_run_diarization_regression(args, checks)

    all_passed = checks.summary()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
