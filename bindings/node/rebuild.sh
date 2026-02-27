#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIARIZATION_DIR="$PROJECT_ROOT/diarization-ggml"
ADDON_DIR="$SCRIPT_DIR/packages/darwin-arm64"
NODE_MODULES_ADDON="$SCRIPT_DIR/node_modules/@pyannote-cpp-node/darwin-arm64"

echo "=== Project root: $PROJECT_ROOT ==="
echo "=== Rebuilding C++ pipeline (build/) ==="
cmake --build "$DIARIZATION_DIR/build" --parallel

echo ""
echo "=== Rebuilding C++ pipeline (build-static/) ==="
cmake --build "$DIARIZATION_DIR/build-static" --parallel

echo ""
echo "=== Rebuilding native addon (cmake-js) ==="
cd "$ADDON_DIR"
npx cmake-js clean 2>/dev/null || true
npx cmake-js build --CDEMBEDDING_COREML=ON --CDSEGMENTATION_COREML=ON --CDWHISPER_COREML=ON

echo ""
echo "=== Copying fresh addon to node_modules ==="
FRESH="$ADDON_DIR/build/Release/pyannote-addon.node"
STALE="$NODE_MODULES_ADDON/build/Release/pyannote-addon.node"

if [ -f "$FRESH" ]; then
    mkdir -p "$(dirname "$STALE")"
    cp "$FRESH" "$STALE"
    echo "Copied: $(stat -f '%z bytes' "$FRESH")"
    echo "  From: $FRESH"
    echo "  To:   $STALE"
else
    echo "ERROR: Fresh addon not found at $FRESH"
    exit 1
fi

echo ""
echo "=== Building TypeScript ==="
cd "$SCRIPT_DIR/packages/pyannote-cpp-node"
npx tsc

echo ""
echo "=== Done ==="
