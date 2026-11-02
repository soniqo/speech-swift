#!/usr/bin/env python3
"""Resolve, and optionally build, a Qwen3TTS consumer with an explicit MLX version."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-version", default="0.33.0")
    parser.add_argument("--scratch-path", type=Path)
    parser.add_argument("--build", action="store_true", help="Also compile the consumer in release mode")
    parser.add_argument("--speech-url", help="Test a remote repository instead of this local checkout")
    parser.add_argument("--speech-revision", help="Exact remote commit to test (requires --speech-url)")
    args = parser.parse_args()
    if bool(args.speech_url) != bool(args.speech_revision):
        parser.error("--speech-url and --speech-revision must be provided together")
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="speech-mlx-consumer-") as folder:
        consumer = Path(folder)
        (consumer / "Sources/Probe").mkdir(parents=True)
        (consumer / "Sources/Probe/main.swift").write_text("import Qwen3TTS\n")
        manifest = '''// swift-tools-version: 6.2
import PackageDescription
let package = Package(name: "Consumer", platforms: [.macOS(.v15)], dependencies: [
    SPEECH_DEPENDENCY,
    .package(url: "https://github.com/ml-explore/mlx-swift", exact: MLX_VERSION)
], targets: [.executableTarget(name: "Probe", dependencies: [
    .product(name: "Qwen3TTS", package: "speech-swift"),
    .product(name: "MLX", package: "mlx-swift")
])])
'''
        if args.speech_url:
            dependency = '.package(name: "speech-swift", url: URL, revision: REVISION)'
            dependency = dependency.replace("URL", json.dumps(args.speech_url, ensure_ascii=False))
            dependency = dependency.replace("REVISION", json.dumps(args.speech_revision, ensure_ascii=False))
        else:
            dependency = '.package(name: "speech-swift", path: PATH)'
            dependency = dependency.replace("PATH", json.dumps(str(root), ensure_ascii=False))
        manifest = manifest.replace("SPEECH_DEPENDENCY", dependency)
        manifest = manifest.replace("MLX_VERSION", json.dumps(args.mlx_version))
        (consumer / "Package.swift").write_text(manifest)
        command = ["swift", "package", "--package-path", str(consumer)]
        if args.scratch_path:
            command += ["--scratch-path", str(args.scratch_path.resolve())]
        subprocess.run(command + ["resolve"], check=True)
        pins = json.loads((consumer / "Package.resolved").read_text())["pins"]
        selected = next((pin.get("state", {}).get("version") for pin in pins if pin["identity"] == "mlx-swift"), None)
        if selected != args.mlx_version:
            raise RuntimeError(f"Expected MLX {args.mlx_version}, resolved {selected}")
        print(f"Qwen3TTS consumer resolves MLX {selected}")
        if args.build:
            build = ["swift", "build", "--package-path", str(consumer)]
            if args.scratch_path:
                build += ["--scratch-path", str(args.scratch_path.resolve())]
            subprocess.run(build + ["-c", "release", "-j", "4", "--product", "Probe"], check=True)
            print("Qwen3TTS consumer builds successfully")


if __name__ == "__main__":
    main()
