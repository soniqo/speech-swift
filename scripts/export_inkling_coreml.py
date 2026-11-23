#!/usr/bin/env python3
"""Export Inkling 2 Mini to shared Core ML partitions with INT8 text weights.

Requires an Apple Silicon Mac, torch, coremltools==9.0, numpy and huggingface_hub.
The remote source reader streams pinned tensor ranges without storing the full
source checkpoint. Expert weights stay INT8 through routing; vision uses FP16,
and convolution histories, routing and sensitive parameters retain FP32.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from export_inkling_mlx import SOURCE_MODEL, SOURCE_REVISION, ASSETS, resolve_source, sha256, write_json
from inkling_coreml import (Embedding, DecoderLayer, OutputHead, Vision, convert_part,
                            conversion_inputs, layer_visits)
from inkling_coreml_source import Checkpoint
from prepare_inkling_coreml_bridge import installed_bridge_receipt

FORMAT = "speech-swift-inkling2-coreml-v1"


def copy_verified_asset(source, destination):
    """Keep matching metadata on resume, including read-only hub files."""
    if destination.exists():
        if not destination.is_file() or sha256(source) != sha256(destination):
            raise ValueError(f"Existing export metadata differs from the pinned source: {destination.name}")
        return
    shutil.copy2(source, destination)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--context-limit", type=int, default=8192)
    parser.add_argument("--prefill-size", type=int, choices=[1], default=1,
                        help="Tokens per partition call; one bounds routed-weight workspace on 48 GB Macs.")
    parser.add_argument("--image-patch-limit", type=int, default=1024)
    parser.add_argument("--group-size", type=int, choices=[32, 64, 128], default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-compile", action="store_true", help="Save portable packages without local compilation and parity checks.")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
        parser.error("Source revision must be an immutable 40-character commit")
    if not 1 <= args.context_limit <= 1048576:
        parser.error("Use a context in 1...1048576")
    if not 1 <= args.image_patch_limit <= 1024:
        parser.error("Image patch limit must be in 1...1024")
    return args


def copy_cow(source, destination):
    """Copy with independent APFS copy-on-write storage when available."""
    result = subprocess.run(["/bin/cp", "-c", "-p", str(source), str(destination)], capture_output=True)
    if result.returncode:
        shutil.copy2(source, destination)
    return str(destination)


def supports_copy_on_write(directory):
    with tempfile.TemporaryDirectory(prefix=".cow-check-", dir=directory) as temporary:
        source = Path(temporary) / "source"
        source.write_bytes(b"Core ML export storage probe\n")
        result = subprocess.run(["/bin/cp", "-c", str(source), str(Path(temporary) / "copy")], capture_output=True)
        return result.returncode == 0


def compile_and_check(package, module, config, kind, options, compiled):
    import coremltools as ct
    def resident_bytes():
        return int(subprocess.check_output(["/bin/ps", "-o", "rss=", "-p", str(os.getpid())]).strip()) * 1024
    specs, sample, outputs = conversion_inputs(config, kind, **options)
    with torch.no_grad():
        expected = module(*[torch.from_numpy(value) for value in sample.values()])
    if not isinstance(expected, tuple):
        expected = (expected,)
    before_load = resident_bytes()
    model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
    after_load = resident_bytes()
    actual = model.predict({name: value for name, value in sample.items() if name in model.input_description})
    after_predict = resident_bytes()
    if kind == "layer" and after_predict - before_load > 3 * 1024**3:
        raise MemoryError("Core ML layer exceeded the 3 GiB additional runtime budget; inspect expert expansion before continuing")
    errors = {}
    for name, tensor in zip(outputs, expected):
        target = tensor.numpy()
        errors[name] = {"max_absolute": float(np.max(np.abs(actual[name].astype(np.float32) - target.astype(np.float32))))}
        np.testing.assert_allclose(actual[name], target, rtol=.02, atol=.02)
        if name.startswith("history_") and actual[name].dtype != np.float32:
            raise ValueError("Core ML changed convolution-history precision")
    temporary_compiled = Path(model.get_compiled_model_path())
    compiled.parent.mkdir(parents=True, exist_ok=True)
    if compiled.exists():
        raise ValueError("Refusing to overwrite an existing compiled partition")
    shutil.copytree(temporary_compiled, compiled, copy_function=copy_cow)
    package_weights = list(package.rglob("weight.bin"))
    compiled_weights = list(compiled.rglob("weight.bin"))
    shared = False
    if len(package_weights) == len(compiled_weights) == 1:
        original, duplicate = package_weights[0], compiled_weights[0]
        if original.stat().st_size == duplicate.stat().st_size and sha256(original) == sha256(duplicate):
            replacement = duplicate.with_name("weight.cow")
            copy_cow(original, replacement)
            os.replace(replacement, duplicate)
            shared = True
    del model
    gc.collect()
    # Core ML leaves its compiler output in the process's temporary directory.
    # The checked copy above is complete and has independent file ownership.
    temporary_root = Path(tempfile.gettempdir()).resolve()
    if temporary_root in temporary_compiled.resolve().parents:
        shutil.rmtree(temporary_compiled, ignore_errors=True)
    return {"passed": True, "output_errors": errors, "history_dtype": "float32",
            "resident_bytes": {"before_load": before_load, "after_load": after_load, "after_predict": after_predict},
            "compiled_weight_bytes_match_package": shared}


def export(args):
    torch.set_num_threads(2)
    if importlib.metadata.version('coremltools') != '9.0':
        raise ValueError('This exporter is validated with coremltools==9.0')
    bridge = installed_bridge_receipt() if not args.no_compile else None
    output = args.output.expanduser().resolve()
    source = resolve_source(SimpleNamespace(source_dir=args.source_dir, source_revision=args.source_revision,
                                             stream_source=args.source_dir is None, include_mtp=False))
    source = source.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Source and output directories must be separate")
    checkpoint = Checkpoint(source, stream=args.source_dir is None, revision=args.source_revision)
    text = checkpoint.config["text_config"]
    vision = copy.deepcopy(checkpoint.config["vision_config"])
    vision.setdefault("layer_norm_eps", 1e-5)
    request = {"format": FORMAT, "source_model": SOURCE_MODEL, "source_revision": args.source_revision,
        "context_limit": args.context_limit, "prefill_size": args.prefill_size,
        "image_patch_limit": args.image_patch_limit, "expert_group_size": args.group_size,
        "compiled_partitions": not args.no_compile,
        "weight_bits": 8, "activation_dtype": "float32", "convolution_history_dtype": "float32",
        "physical_layers": text["num_hidden_layers"], "layer_visits": layer_visits(text),
        "weight_precision": {"experts": "int8", "text_matrices": "int8", "vision": "float16",
                             "routers_norms_controllers": "float32"}}
    scripts = Path(__file__).resolve().parent
    request["exporter_sha256"] = {name: sha256(scripts / name) for name in
        ["export_inkling_coreml.py", "inkling_coreml.py", "inkling_coreml_source.py", "export_inkling_mlx.py",
         "prepare_inkling_coreml_bridge.py"]}
    manifest_path = output / "coreml-manifest.json"
    if output.exists() and any(output.iterdir()):
        if not args.resume or not manifest_path.is_file():
            raise ValueError("Use an empty output directory, or --resume with its matching manifest")
        manifest = json.loads(manifest_path.read_text())
        if any(manifest.get(key) != value for key, value in request.items()):
            raise ValueError("Resume arguments do not match the existing Core ML export")
    else:
        output.mkdir(parents=True, exist_ok=True)
        manifest = {**request, "status": "in_progress", "parts": {}, "validation_status": "not_run",
            "toolchain": {"python": sys.version.split()[0], **{name: importlib.metadata.version(name)
                for name in ["torch", "coremltools", "numpy", "huggingface_hub"]}}}
    estimate = checkpoint.estimate_bytes(args.group_size)
    cow = supports_copy_on_write(output)
    required = estimate * (2 if not args.no_compile and not cow else 1) + 10 * 1024**3
    if not manifest["parts"] and shutil.disk_usage(output).free < required:
        raise ValueError(f"Need at least {required / 1024**3:.1f} GiB free for streamed conversion")
    manifest["storage"] = {"copy_on_write_supported": cow, "estimated_weight_bytes": estimate,
                           "required_initial_free_bytes": required}
    write_json(manifest_path, manifest)
    if bridge is not None:
        write_json(output / 'coremltools-python-bridge.json', bridge)
    architecture = copy.deepcopy(checkpoint.config)
    architecture.pop("quantization_config", None)
    architecture["vision_config"] = vision
    architecture["coreml_format"] = FORMAT
    architecture["quantization"] = {"bits": 8, "group_size": args.group_size, "mode": "affine"}
    write_json(output / "config.json", architecture)
    for name in [*ASSETS, "README.md", "LICENSE", "LICENSE.txt", "NOTICE", "NOTICE.txt", "hf_quant_config.json"]:
        if (source / name).is_file():
            target = "source_README.md" if name == "README.md" else "source_quant_config.json" if name == "hf_quant_config.json" else name
            copy_verified_asset(source / name, output / target)
    copy_verified_asset(source / "config.json", output / "source_config.json")
    if (source / "tiktoken").is_dir() and not (output / "tiktoken").exists():
        shutil.copytree(source / "tiktoken", output / "tiktoken")
    notice = (Path(__file__).resolve().parents[1] / 'ThirdPartyNotices/Inkling2Mini.txt').read_text()
    (output / 'SPEECH_SWIFT_NOTICE.txt').write_text(notice)
    if not any((output / name).is_file() for name in ['LICENSE', 'LICENSE.txt']):
        (output / 'LICENSE').write_text(notice[notice.index('                                 Apache License'):])
    sequence = [("embedding", "embedding", None)] + [(f"layer-{i}", "layer", i) for i in range(text["num_hidden_layers"])]
    sequence += [("head", "head", None), ("vision", "vision", None)]
    for name, kind, index in sequence:
        if name in manifest["parts"]:
            if not (output / manifest["parts"][name]["package"]).is_dir():
                raise ValueError(f"Completed partition is missing: {name}")
            if not args.no_compile and not (output / manifest["parts"][name].get("compiled", "missing")).is_dir():
                raise ValueError(f"Completed compiled partition is missing: {name}")
            print(f"Already exported {name}", flush=True)
            continue
        print(f"Reading {name}", flush=True)
        if shutil.disk_usage(output).free < 4 * 1024**3:
            raise OSError("Less than 4 GiB free; reclaim space and resume the completed partitions")
        common = {"group_size": args.group_size, "dtype": torch.float32}
        if kind == "embedding":
            module = Embedding(text, checkpoint.embedding(), **common)
        elif kind == "layer":
            module = DecoderLayer(text, index, checkpoint.layer(index), **common)
        elif kind == "head":
            module = OutputHead(text, checkpoint.head(), **common)
        else:
            module = Vision(vision, checkpoint.vision(), dtype=torch.float16)
        options = {"context_limit": args.context_limit, "prefill_size": args.prefill_size,
                   "max_patches": args.image_patch_limit,
                   "dtype": torch.float16 if kind == "vision" else torch.float32}
        if index is not None:
            options["layer"] = index
        config = vision if kind == "vision" else text
        work = output / ".work"
        work.mkdir(exist_ok=True)
        stage = work / f"{name}.mlpackage"
        if stage.exists():
            shutil.rmtree(stage)
        print(f"Converting {name}", flush=True)
        report = convert_part(module, config, kind, stage, **options)
        package = output / f"{name}.mlpackage"
        if package.exists():
            raise ValueError(f"Unrecorded partition exists: {name}; inspect it before resuming")
        report["package"] = package.name
        if not args.no_compile:
            compiled = output / "compiled" / f"{name}.mlmodelc"
            compiled_stage = work / f"{name}.mlmodelc"
            if compiled_stage.exists():
                shutil.rmtree(compiled_stage)
            report["compile_check"] = compile_and_check(stage, module, config, kind, options, compiled_stage)
            report["compiled"] = str(compiled.relative_to(output))
            compiled.parent.mkdir(exist_ok=True)
            if compiled.exists():
                raise ValueError(f"Unrecorded compiled partition exists: {name}")
            compiled_stage.rename(compiled)
        stage.rename(package)
        manifest["parts"][name] = report
        manifest.setdefault("source_tensor_sha256", {}).update(getattr(checkpoint.tensors, "tensor_hashes", {}))
        write_json(manifest_path, manifest)
        del module
        gc.collect()
        print(f"Finished {name}; free {shutil.disk_usage(output).free / 1024**3:.1f} GiB", flush=True)
    manifest["status"] = "converted"
    manifest["validation_status"] = "partition_parity_passed" if not args.no_compile else "not_run"
    manifest.setdefault("source_tensor_sha256", {}).update(getattr(checkpoint.tensors, "tensor_hashes", {}))
    manifest["files"] = {str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                         for path in sorted(output.rglob("*")) if path.is_file() and path != manifest_path}
    write_json(manifest_path, manifest)
    return manifest


if __name__ == "__main__":
    export(parse_args())
