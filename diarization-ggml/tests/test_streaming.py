#!/usr/bin/env python3
"""
Test streaming diarization output matches offline diarization output.

Usage:
    python test_streaming.py [--audio path/to/audio.wav]
    
Tests:
1. Streaming/Offline equivalence: streaming_finalize() output == diarize() output
2. DER sanity check: Compare against Python reference (if available)
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Get the repository root and diarization-ggml directory
SCRIPT_DIR = Path(__file__).parent.absolute()
DIARIZATION_DIR = SCRIPT_DIR.parent
REPO_ROOT = DIARIZATION_DIR.parent


def run_command(cmd, cwd=None):
    """Run command and return (success, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def test_streaming_offline_equivalence(audio_path: Path, verbose: bool = False):
    """Test that streaming_finalize() output matches diarize() output."""
    
    print("\n=== Test 1: Streaming/Offline Equivalence ===")
    
    # Paths
    build_dir = DIARIZATION_DIR / "build" / "bin"
    streaming_test = build_dir / "streaming_test"
    diarization_ggml = build_dir / "diarization-ggml"
    
    # Check executables exist
    if not streaming_test.exists():
        print(f"ERROR: streaming_test not found at {streaming_test}")
        return False
    if not diarization_ggml.exists():
        print(f"ERROR: diarization-ggml not found at {diarization_ggml}")
        return False
    
    # Model paths (relative to build dir when running from diarization-ggml)
    seg_gguf = REPO_ROOT / "models" / "segmentation-ggml" / "segmentation.gguf"
    emb_gguf = REPO_ROOT / "models" / "embedding-ggml" / "embedding.gguf"
    plda_bin = DIARIZATION_DIR / "plda.gguf"
    emb_coreml = REPO_ROOT / "models" / "embedding-ggml" / "embedding.mlpackage"
    seg_coreml = REPO_ROOT / "models" / "segmentation-ggml" / "segmentation.mlpackage"
    
    # Check required files
    for path, name in [(plda_bin, "PLDA model"), (emb_coreml, "Embedding CoreML"), (seg_coreml, "Segmentation CoreML")]:
        if not path.exists():
            print(f"WARNING: {name} not found at {path}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        streaming_rttm = Path(tmpdir) / "streaming.rttm"
        offline_rttm = Path(tmpdir) / "offline.rttm"
        
        # Run streaming test
        print(f"Running streaming_test on {audio_path}...")
        cmd_streaming = [
            str(streaming_test),
            str(audio_path),
            "--plda", str(plda_bin),
            "--coreml", str(emb_coreml),
            "--seg-coreml", str(seg_coreml),
            "-o", str(streaming_rttm)
        ]
        success, stdout, stderr = run_command(cmd_streaming, cwd=str(DIARIZATION_DIR))
        if verbose:
            print(f"stdout: {stdout}")
            print(f"stderr: {stderr}")
        if not success:
            print(f"FAILED: streaming_test failed")
            print(f"stderr: {stderr}")
            return False
        print("  streaming_test: OK")
        
        # Run offline diarization
        print(f"Running diarization-ggml on {audio_path}...")
        cmd_offline = [
            str(diarization_ggml),
            str(seg_gguf),
            str(emb_gguf),
            str(audio_path),
            "--plda", str(plda_bin),
            "--coreml", str(emb_coreml),
            "--seg-coreml", str(seg_coreml),
            "-o", str(offline_rttm)
        ]
        success, stdout, stderr = run_command(cmd_offline, cwd=str(DIARIZATION_DIR))
        if verbose:
            print(f"stdout: {stdout}")
            print(f"stderr: {stderr}")
        if not success:
            print(f"FAILED: diarization-ggml failed")
            print(f"stderr: {stderr}")
            return False
        print("  diarization-ggml: OK")
        
        # Compare outputs
        print("Comparing RTTM outputs...")
        if not streaming_rttm.exists():
            print(f"FAILED: streaming output not created")
            return False
        if not offline_rttm.exists():
            print(f"FAILED: offline output not created")
            return False
        
        with open(streaming_rttm) as f:
            streaming_content = f.read()
        with open(offline_rttm) as f:
            offline_content = f.read()
        
        if streaming_content == offline_content:
            print("  PASS: Streaming output matches offline output (byte-identical)")
            return True
        else:
            print("  FAIL: Outputs differ!")
            print(f"  Streaming ({len(streaming_content)} bytes):")
            for line in streaming_content.strip().split('\n')[:5]:
                print(f"    {line}")
            print(f"  Offline ({len(offline_content)} bytes):")
            for line in offline_content.strip().split('\n')[:5]:
                print(f"    {line}")
            return False


def test_der_sanity(streaming_rttm: Path, reference_rttm: Path, threshold: float = 1.0):
    """Test DER against Python reference is within threshold."""
    
    print("\n=== Test 2: DER Sanity Check ===")
    
    compare_script = DIARIZATION_DIR / "tests" / "compare_rttm.py"
    if not compare_script.exists():
        print(f"SKIP: compare_rttm.py not found at {compare_script}")
        return True  # Not a failure, just skip
    
    if not reference_rttm.exists():
        print(f"SKIP: Reference RTTM not found at {reference_rttm}")
        return True
    
    # Run DER comparison
    cmd = [
        sys.executable,
        str(compare_script),
        str(streaming_rttm),
        str(reference_rttm),
        "--threshold", str(threshold)
    ]
    success, stdout, stderr = run_command(cmd)
    print(f"  {stdout.strip()}")
    
    if success:
        print(f"  PASS: DER <= {threshold}%")
        return True
    else:
        print(f"  FAIL: DER > {threshold}%")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test streaming diarization")
    parser.add_argument("--audio", type=str, default=None,
                        help="Audio file to test (default: samples/sample.wav)")
    parser.add_argument("--reference", type=str, default=None,
                        help="Reference RTTM for DER comparison")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()
    
    # Default audio path
    if args.audio:
        audio_path = Path(args.audio)
    else:
        audio_path = REPO_ROOT / "samples" / "sample.wav"
    
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        return 1
    
    print(f"Testing with audio: {audio_path}")
    
    # Run tests
    all_passed = True
    
    # Test 1: Streaming/Offline equivalence
    if not test_streaming_offline_equivalence(audio_path, args.verbose):
        all_passed = False
    
    # Test 2: DER sanity (optional)
    if args.reference:
        reference_path = Path(args.reference)
        # This would need the streaming RTTM path, which we'd need to persist
        # For now, skip this test
        print("\n=== Test 2: DER Sanity Check ===")
        print("  SKIP: DER test requires --reference flag implementation")
    
    # Final result
    print("\n" + "=" * 50)
    if all_passed:
        print("ALL TESTS PASSED")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
