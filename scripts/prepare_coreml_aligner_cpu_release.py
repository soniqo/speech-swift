#!/usr/bin/env python3
"""Prepare additive CPU-compatible model assets locally; never upload them.

The original GPU decoder is preserved. Versioned CPU bundles reuse its weights
and promote only RMS normalization to Float32. Requires coremltools==9.0.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from urllib.request import urlopen

import coremltools as ct

from prepare_coreml_aligner_cpu_probe import link_copy, sha256, transform

BUNDLE_NAME = "text_decoder_cpu_v1"
VARIANTS = {
    "int8": {
        "repo": "aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT8",
        "revision": "ac27c13a9dae935e7d478b996d89d94ff617e3ca",
        "weight_sha256": "083cfeabb070794b7745694564f0154aa4329020c7eb344aff3406f348b25e3f",
    },
    "fp16": {
        "repo": "aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-FP16",
        "revision": "04ca8c28f641b9e32a0ac8d1edd3cdec83f22159",
        "weight_sha256": "7b76d348eeade1d225514012c3b727fce16d2379b92b0fbf1df3e175f852619f",
    },
}


def pinned_model_card(metadata):
    repo, revision = metadata["repo"], metadata["revision"]
    with urlopen(f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true", timeout=90) as response:
        tree = json.load(response)
    entry = next(item for item in tree if item["path"] == "README.md")
    with urlopen(f"https://huggingface.co/{repo}/resolve/{revision}/README.md", timeout=90) as response:
        data = response.read(2 * 1024 * 1024)
    if len(data) != entry["size"]:
        raise ValueError("The pinned model card has an unexpected size")
    lfs = entry.get("lfs")
    digest = (hashlib.sha256(data).hexdigest() if lfs else
              hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest())
    if digest != (lfs["oid"] if lfs else entry["oid"]):
        raise ValueError("The pinned model card has an unexpected hash")
    return data.decode("utf-8")


def cpu_model_card(original):
    # Preserve only repository links whose public visibility was verified.
    # Dropping the entire line also removes an unverified link's label.
    allowed_repositories = {"soniqo/speech-swift"}
    original = "\n".join(line for line in original.splitlines()
        if all(repo in allowed_repositories for repo in re.findall(
            r"https://github\.com/([^/\s)>]+/[^/\s)>]+)", line)))
    original = original.replace("`<ts>` word `<ts>` slots", "`word<timestamp><timestamp>` slots")
    original = re.sub(
        r"^The decoder uses `MLComputeUnits = \.all`.*$",
        "The Swift decoder defaults to `.cpuAndGPU`; the encoder defaults to "
        "`.cpuAndNeuralEngine`. Explicit `.cpuOnly` uses the separate CPU-compatible bundle below.",
        original, flags=re.MULTILINE)
    section = f"""

## CPU-compatible decoder

`{BUNDLE_NAME}.mlmodelc` and `{BUNDLE_NAME}.mlpackage` are additional decoder
bundles for CPU-only execution. Their 113 RMS-normalization calculations use
Float32; learned weights and the existing model input/output types are preserved.
The original `text_decoder` bundles remain available for GPU execution.

The legacy decoder returns all-zero logits with CPU-only execution on
macOS 15.8.2. The Float32-normalization INT8 graph matches all 22 corrected
word boundaries in the reference fixture on that runtime. The release validation
manifest records the supported checks; validation on other devices is separate.

Updated speech-swift loaders select the versioned bundle for explicit `.cpuOnly`
execution. Run one online model load to populate it; an existing legacy cache
does not contain the new CPU bundle. Local/custom exports can provide the same
bundle name to select it. The versioned bundle must not be changed in place;
future graph changes require a new bundle version.
"""
    return original.rstrip() + section


def prepare_variant(variant, source_root, cache_root, output_root):
    metadata = VARIANTS[variant]
    base = cache_root / variant
    source = source_root / variant / "text_decoder.mlpackage"
    source_weights = source / "Data/com.apple.CoreML/weights/weight.bin"
    if sha256(source_weights) != metadata["weight_sha256"]:
        raise ValueError(f"{variant}: unexpected source decoder weights")
    original_weights = base / "text_decoder.mlmodelc/weights/weight.bin"
    if sha256(original_weights) != metadata["weight_sha256"]:
        raise ValueError(f"{variant}: unexpected compiled decoder weights")
    destination = output_root / variant
    if destination.exists():
        raise ValueError(f"{variant}: output already exists")
    upload = destination / "upload"
    upload.mkdir(parents=True)
    package = upload / f"{BUNDLE_NAME}.mlpackage"
    shutil.copytree(source, package, copy_function=link_copy)
    model = package / "Data/com.apple.CoreML/model.mlmodel"
    specification = ct.utils.load_spec(str(model))
    repaired = transform(specification, "norm32")
    repaired.description.metadata.userDefined["cpu_normalization_precision"] = "float32"
    repaired.description.metadata.userDefined["cpu_decoder_bundle_version"] = "1"
    repaired.description.metadata.userDefined["source_model_revision"] = metadata["revision"]
    model.unlink()  # Never modify a hard link to the original specification.
    ct.utils.save_spec(repaired, str(model))
    compiled = Path(ct.models.utils.compile_model(str(package)))
    final = upload / f"{BUNDLE_NAME}.mlmodelc"
    try:
        shutil.copytree(compiled, final, copy_function=link_copy)
    finally:
        shutil.rmtree(compiled)
    compiled_weights = final / "weights/weight.bin"
    if sha256(compiled_weights) != metadata["weight_sha256"]:
        raise ValueError(f"{variant}: the CPU repair changed learned weights")
    compiled_weights.unlink()
    os.link(original_weights, compiled_weights)
    (upload / "README.md").write_text(cpu_model_card(pinned_model_card(metadata)))
    manifest = {
        "version": 1,
        "variant": variant,
        "repo": metadata["repo"],
        "source_revision": metadata["revision"],
        "decoder_bundle": BUNDLE_NAME,
        "normalization_precision": "float32",
        "normalization_chains": 113,
        "weight_sha256": metadata["weight_sha256"],
        "legacy_gpu_decoder_modified": False,
        "coremltools_version": ct.__version__,
        "files": [],
    }
    for path in sorted(upload.rglob("*")):
        if path.is_file():
            manifest["files"].append({
                "path": path.relative_to(upload).as_posix(),
                "size": path.stat().st_size,
                "sha256": (metadata["weight_sha256"] if path.name == "weight.bin" else sha256(path)),
            })
    (upload / "cpu_decoder_v1.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Keep local inference inputs outside the directory prepared for upload.
    validation = destination / "validation"
    shutil.copytree(base, validation, copy_function=link_copy)
    for item in upload.iterdir():
        if item.is_dir():
            shutil.copytree(item, validation / item.name, copy_function=link_copy)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for variant in VARIANTS:
        result = prepare_variant(variant, arguments.source_root.resolve(),
                                 arguments.cache_root.resolve(), arguments.output_root.resolve())
        results.append(result)
        print(json.dumps({"variant": variant, "repo": result["repo"],
                          "decoder_bundle": BUNDLE_NAME, "files": len(result["files"])}), flush=True)
    (arguments.output_root / "release-manifest.json").write_text(json.dumps(results, indent=2) + "\n")
