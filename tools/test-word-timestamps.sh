#!/usr/bin/env bash
# Quick script to test whisper.cpp word-level timestamps on long_sample.wav
# Usage: ./tools/test-word-timestamps.sh [model-path] [audio-path]
#
# Runs whisper-cli with --output-json-full to get word-level timestamps,
# optionally with --dtw for more accurate token-level timing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

WHISPER_CLI="$REPO_ROOT/whisper.cpp/build/bin/whisper-cli"
MODEL="${1:-$REPO_ROOT/whisper.cpp/models/ggml-large-v3-turbo-q5_0.bin}"
AUDIO="${2:-$REPO_ROOT/samples/long_sample.wav}"
OUTDIR="/tmp/whisper-word-timestamps"

mkdir -p "$OUTDIR"

if [ ! -f "$WHISPER_CLI" ]; then
    echo "ERROR: whisper-cli not found at $WHISPER_CLI"
    echo "Build whisper.cpp first: cd whisper.cpp && cmake -B build && cmake --build build"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found at $MODEL"
    echo "Available models:"
    ls "$REPO_ROOT/whisper.cpp/models/"*.bin 2>/dev/null || echo "  (none)"
    exit 1
fi

if [ ! -f "$AUDIO" ]; then
    echo "ERROR: Audio file not found at $AUDIO"
    exit 1
fi

echo "=== Whisper Word-Level Timestamp Test ==="
echo "Model:  $MODEL"
echo "Audio:  $AUDIO"
echo "Output: $OUTDIR"
echo ""

# Run 1: Default word timestamps (no DTW)
echo "--- Run 1: Default word timestamps ---"
"$WHISPER_CLI" \
    -m "$MODEL" \
    -f "$AUDIO" \
    -l ko \
    --output-json-full \
    -of "$OUTDIR/default" \
    --print-progress \
    -pp \
    2>&1

echo ""
echo "Output: $OUTDIR/default.json"
echo ""

# Run 2: With DTW token-level timestamps (large-v3-turbo)
echo "--- Run 2: DTW token-level timestamps ---"
"$WHISPER_CLI" \
    -m "$MODEL" \
    -f "$AUDIO" \
    -l ko \
    --dtw large.v3.turbo \
    --no-flash-attn \
    --output-json-full \
    -of "$OUTDIR/dtw" \
    --print-progress \
    -pp \
    2>&1

echo ""
echo "Output: $OUTDIR/dtw.json"
echo ""

echo "--- Run 3: DTW top-2 average (top 2 layers, plain averaging) ---"
"$WHISPER_CLI" \
    -m "$MODEL" \
    -f "$AUDIO" \
    -l ko \
    --dtw top-2 \
    --no-flash-attn \
    --output-json-full \
    -of "$OUTDIR/dtw-top2-avg" \
    --print-progress \
    -pp \
    2>&1

echo ""
echo "Output: $OUTDIR/dtw-top2-avg.json"
echo ""

echo "--- Run 4: DTW top-2 L2 norm (top 2 layers + L2 norm head filtering) ---"
"$WHISPER_CLI" \
    -m "$MODEL" \
    -f "$AUDIO" \
    -l ko \
    --dtw top-2-norm \
    --no-flash-attn \
    --output-json-full \
    -of "$OUTDIR/dtw-top2-norm" \
    --print-progress \
    -pp \
    2>&1

echo ""
echo "Output: $OUTDIR/dtw-top2-norm.json"
echo ""

echo "=== Done ==="
echo ""
echo "Compare the four outputs:"
echo "  Default:       $OUTDIR/default.json"
echo "  DTW preset:    $OUTDIR/dtw.json              (--dtw large-v3-turbo, hardcoded heads)"
echo "  DTW top-2 avg: $OUTDIR/dtw-top2-avg.json     (--dtw top-2, plain averaging)"
echo "  DTW top-2 L2:  $OUTDIR/dtw-top2-norm.json    (--dtw top-2-norm, L2 norm filtering)"
echo ""
echo "Quick peek at first segment tokens:"
python3 -c "
import json, sys

for name in ['default', 'dtw', 'dtw-top2-avg', 'dtw-top2-norm']:
    path = f'$OUTDIR/{name}.json'
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        print(f'{name}: ERROR reading: {e}')
        continue

    print(f'\n=== {name.upper()} ===')
    segs = data.get('transcription', [])
    print(f'Total segments: {len(segs)}')

    # Show first 3 segments with word-level timestamps
    for seg in segs[:3]:
        t0 = seg.get('offsets', {}).get('from', 0) / 1000
        t1 = seg.get('offsets', {}).get('to', 0) / 1000
        text = seg.get('text', '').strip()
        print(f'\n  [{t0:.2f}s - {t1:.2f}s] {text}')

        tokens = seg.get('tokens', [])
        for tok in tokens[:12]:
            tt0 = tok.get('offsets', {}).get('from', 0) / 1000
            tt1 = tok.get('offsets', {}).get('to', 0) / 1000
            p = tok.get('p', 0)
            t_dtw = tok.get('t_dtw', -1)
            tok_text = tok.get('text', '')
            dtw_info = f'  dtw={t_dtw}' if t_dtw >= 0 else ''
            print(f'    [{tt0:7.3f}s - {tt1:7.3f}s] p={p:.3f}{dtw_info}  \"{tok_text}\"')
        if len(tokens) > 12:
            print(f'    ... ({len(tokens) - 12} more tokens)')
" 2>&1 || echo "(python3 summary failed, check JSON files manually)"
