#!/usr/bin/env bash
# Run the GLiNER unit and optional real-checkpoint tests without building unrelated model suites.
set -euo pipefail
GLINER_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GLINER_HARNESS="$(mktemp -d /tmp/gliner-validation.XXXXXX)"
trap 'rm -rf "$GLINER_HARNESS"' EXIT
mkdir -p "$GLINER_HARNESS/Sources" "$GLINER_HARNESS/Tests"
ln -s "$GLINER_ROOT/Sources/AudioCommon" "$GLINER_HARNESS/Sources/AudioCommon"
ln -s "$GLINER_ROOT/Sources/GLiNER" "$GLINER_HARNESS/Sources/GLiNER"
ln -s "$GLINER_ROOT/Sources/GLiNERBenchmark" "$GLINER_HARNESS/Sources/GLiNERBenchmark"
ln -s "$GLINER_ROOT/Tests/GLiNERTests" "$GLINER_HARNESS/Tests/GLiNERTests"
python3 - "$GLINER_ROOT" "$GLINER_HARNESS" <<'PY'
import json,sys
from pathlib import Path
root,harness=map(Path,sys.argv[1:])
pins=json.loads((root/'Package.resolved').read_text())['pins']
versions={p['identity']:p['state']['version'] for p in pins}
(harness/'Package.swift').write_text('''// swift-tools-version: 5.10
import PackageDescription
let package = Package(name: "GLiNERValidation", platforms: [.macOS("15.0")], dependencies: [
 .package(url:"https://github.com/ml-explore/mlx-swift",exact:"%s"),
 .package(url:"https://github.com/huggingface/swift-transformers",exact:"%s")
], targets: [
 .target(name:"AudioCommon",dependencies:[.product(name:"Hub",package:"swift-transformers")]),
 .target(name:"GLiNER",dependencies:["AudioCommon",.product(name:"MLX",package:"mlx-swift"),.product(name:"MLXNN",package:"mlx-swift"),.product(name:"Hub",package:"swift-transformers"),.product(name:"Tokenizers",package:"swift-transformers")],resources:[.copy("LICENSE-reference")]),
 .executableTarget(name:"GLiNERBenchmark",dependencies:["GLiNER",.product(name:"MLX",package:"mlx-swift")]),
 .testTarget(name:"GLiNERTests",dependencies:["GLiNER","GLiNERBenchmark",.product(name:"MLX",package:"mlx-swift"),.product(name:"MLXNN",package:"mlx-swift")])
])
''' % (versions['mlx-swift'],versions['swift-transformers']))
PY
swift build --package-path "$GLINER_HARNESS" --scratch-path "$GLINER_ROOT/.build" --build-tests -c release --disable-sandbox -Xswiftc -enable-testing -j 4
for GLINER_TEST_NAME in GLiNERTests GLiNERValidationPackageTests; do
    GLINER_BUNDLE="$GLINER_ROOT/.build/release/$GLINER_TEST_NAME.xctest/Contents/MacOS"
    if [[ -d "$GLINER_BUNDLE" ]]; then
        cp "$GLINER_ROOT/.build/release/mlx.metallib" "$GLINER_BUNDLE/mlx.metallib"
    fi
done
swift test --package-path "$GLINER_HARNESS" --scratch-path "$GLINER_ROOT/.build" -c release --skip-build --disable-sandbox -Xswiftc -enable-testing --no-parallel --filter GLiNER
