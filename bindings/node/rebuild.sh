#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIARIZATION_DIR="$PROJECT_ROOT/diarization-ggml"
ADDON_DIR="$SCRIPT_DIR/packages/darwin-arm64"

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
echo "=== Staging native addon ==="
cd "$SCRIPT_DIR"
node ./scripts/stage-native-addon.js --platform darwin-arm64 --copy-node-modules true

echo ""
echo "=== Building TypeScript ==="
cd "$SCRIPT_DIR/packages/pyannote-cpp-node"
npx tsc

echo ""
echo "=== Done ==="
