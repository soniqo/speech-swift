#!/usr/bin/env python3
"""Prepare diagnostic decoder graphs; does not modify published models.

Requires coremltools==9.0 on macOS. Reuses the pinned INT8 cache from the
manual CPU compatibility job. Every candidate keeps the original weights.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
from urllib.request import urlopen

import coremltools as ct
import numpy as np
from coremltools.proto import MIL_pb2 as mil
from coremltools.proto.FeatureTypes_pb2 import ArrayFeatureType

REPO = "aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT8"
REVISION = "ac27c13a9dae935e7d478b996d89d94ff617e3ca"
WEIGHT_SHA256 = "083cfeabb070794b7745694564f0154aa4329020c7eb344aff3406f348b25e3f"
MODES = ("recompile", "output32", "norm32")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def renamed(operation, name):
    result = deepcopy(operation)
    if "name" in result.attributes:
        result.attributes["name"].immediateValue.tensor.strings.values[:] = [name]
    result.outputs[0].name = name
    return result


def reference(operation, key):
    bindings = operation.inputs[key].arguments
    if len(bindings) != 1 or bindings[0].WhichOneof("binding") != "name":
        raise ValueError(f"Expected a single named {key} input for {operation.type}")
    return bindings[0].name


def transform(specification, mode, expected_norms=113):
    """Strict, graph-only experiment. Leave the input specification untouched."""
    if mode not in MODES:
        raise ValueError(f"Unknown candidate: {mode}")
    result = deepcopy(specification)
    if mode == "recompile":
        return result
    function = result.mlProgram.functions["main"]
    if len(function.block_specializations) != 1:
        raise ValueError("Expected one decoder specialization")
    block = next(iter(function.block_specializations.values()))
    operations = list(block.operations)
    types = {value.name: deepcopy(value.type) for value in function.inputs}
    producers, consumers = {}, {}
    for index, operation in enumerate(operations):
        if operation.blocks:
            raise ValueError("Nested decoder blocks are not supported by this probe")
        for value in operation.outputs:
            types[value.name] = deepcopy(value.type)
            producers[value.name] = index
        for argument in operation.inputs.values():
            for binding in argument.arguments:
                if binding.WhichOneof("binding") != "name":
                    raise ValueError("Expected named decoder bindings")
                consumers.setdefault(binding.name, []).append(index)
    # These templates preserve the source graph's protobuf representation.
    cast_template = next(op for op in operations if op.type == "cast")
    string_template = next(op for op in operations if op.type == "const"
        and op.outputs[0].type.tensorType.dataType == mil.STRING
        and list(op.attributes["val"].immediateValue.tensor.strings.values) == ["fp16"])
    scalar_template = next(op for op in operations if op.type == "const"
        and op.outputs[0].type.tensorType.dataType == mil.FLOAT32
        and op.outputs[0].type.tensorType.rank == 0)

    def dtype(name, value):
        op = renamed(string_template, name)
        op.attributes["val"].immediateValue.tensor.strings.values[:] = [value]
        return op

    def scalar(name, value):
        op = renamed(scalar_template, name)
        op.attributes["val"].immediateValue.tensor.floats.values[:] = [float(value)]
        return op

    def cast(name, input_name, value_type, dtype_name):
        op = renamed(cast_template, name)
        op.inputs["x"].arguments[0].name = input_name
        op.inputs["dtype"].arguments[0].name = dtype_name
        op.outputs[0].type.CopyFrom(value_type)
        return op

    def immediate_scalar(name):
        op = operations[producers[name]]
        if op.type != "const":
            raise ValueError("Expected an immediate normalization constant")
        value = op.attributes["val"]
        if value.WhichOneof("value") != "immediateValue":
            raise ValueError("Expected an immediate normalization constant")
        if value.type.tensorType.dataType == mil.FLOAT16:
            values = np.frombuffer(value.immediateValue.tensor.bytes.values, dtype="<f2")
        elif value.type.tensorType.dataType == mil.FLOAT32:
            values = value.immediateValue.tensor.floats.values
        else:
            raise ValueError("Expected a floating normalization constant")
        if len(values) != 1:
            raise ValueError("Expected a scalar normalization constant")
        return float(values[0])

    new_operations = [dtype("cpu_compat_dtype_fp32", "fp32"),
                      dtype("cpu_compat_dtype_fp16", "fp16")]
    if mode == "output32":
        last = operations[-1]
        if (len(last.outputs) != 1 or last.outputs[0].name != "logits"
                or last.outputs[0].type.tensorType.dataType != mil.FLOAT16
                or list(block.outputs) != ["logits"]
                or len(result.description.output) != 1
                or result.description.output[0].name != "logits"):
            raise ValueError("Expected the FP16 logits graph output")
        output_type = deepcopy(last.outputs[0].type)
        output_type.tensorType.dataType = mil.FLOAT32
        new_operations.extend(operations[:-1])
        new_operations.append(renamed(last, "cpu_compat_logits_fp16"))
        new_operations.append(cast("logits", "cpu_compat_logits_fp16",
                                   output_type, "cpu_compat_dtype_fp32"))
        result.description.output[0].type.multiArrayType.dataType = ArrayFeatureType.FLOAT32
    else:
        before, after, edited = {}, {}, {}
        count = 0
        for rsqrt_index, rsqrt in enumerate(operations):
            if rsqrt.type != "rsqrt":
                continue
            add_index = producers[reference(rsqrt, "x")]
            add = operations[add_index]
            mean_index = producers[reference(add, "x")]
            mean = operations[mean_index]
            power_index = producers[reference(mean, "x")]
            power = operations[power_index]
            if (add.type, mean.type, power.type) != ("add", "reduce_mean", "pow"):
                raise ValueError("Unrecognized RMS normalization chain")
            x = reference(power, "x")
            inverse = rsqrt.outputs[0].name
            matches = [i for i in consumers[inverse] if operations[i].type == "mul"
                and {reference(operations[i], key) for key in ("x", "y")} == {x, inverse}]
            if len(matches) != 1:
                raise ValueError("Expected one unweighted RMS normalization product")
            norm_index = matches[0]
            norm = operations[norm_index]
            old_output = norm.outputs[0].name
            prefix = f"cpu_norm_{count}"
            input32, power32, epsilon32 = (prefix + suffix for suffix in ("_input", "_power", "_epsilon"))
            value_type = deepcopy(types[x])
            value_type.tensorType.dataType = mil.FLOAT32
            before[power_index] = [
                scalar(power32, immediate_scalar(reference(power, "y"))),
                scalar(epsilon32, immediate_scalar(reference(add, "y"))),
                cast(input32, x, value_type, "cpu_compat_dtype_fp32")]
            for index in (power_index, mean_index, add_index, rsqrt_index, norm_index):
                if index in edited or operations[index].outputs[0].type.tensorType.dataType != mil.FLOAT16:
                    raise ValueError("Overlapping or non-FP16 normalization chain")
                edited[index] = deepcopy(operations[index])
                edited[index].outputs[0].type.tensorType.dataType = mil.FLOAT32
            edited[power_index].inputs["x"].arguments[0].name = input32
            edited[power_index].inputs["y"].arguments[0].name = power32
            edited[add_index].inputs["y"].arguments[0].name = epsilon32
            for key in ("x", "y"):
                if reference(edited[norm_index], key) == x:
                    edited[norm_index].inputs[key].arguments[0].name = input32
            edited[norm_index] = renamed(edited[norm_index], prefix + "_normalized")
            after[norm_index] = [cast(old_output, prefix + "_normalized",
                                      types[old_output], "cpu_compat_dtype_fp16")]
            count += 1
        if count != expected_norms:
            raise ValueError(f"Expected {expected_norms} normalization chains, found {count}")
        for index, operation in enumerate(operations):
            new_operations.extend(before.get(index, []))
            new_operations.append(edited.get(index, operation))
            new_operations.extend(after.get(index, []))
    del block.operations[:]
    block.operations.extend(new_operations)
    return result


def download_source(package, compiled_weights):
    """Fetch only the small pinned specification; reuse verified cached weights."""
    if sha256(compiled_weights) != WEIGHT_SHA256:
        raise ValueError("The compiled decoder weights do not match the pinned source")
    with urlopen(f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}?recursive=true", timeout=90) as response:
        tree = json.load(response)
    allowed = {"text_decoder.mlpackage/Manifest.json",
               "text_decoder.mlpackage/Data/com.apple.CoreML/model.mlmodel"}
    entries = {entry["path"]: entry for entry in tree if entry["type"] == "file"}
    if not allowed <= entries.keys():
        raise ValueError("The pinned source package is incomplete")
    weight_entry = entries["text_decoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin"]
    if weight_entry["lfs"]["oid"] != WEIGHT_SHA256:
        raise ValueError("The source and compiled decoder weights differ")
    for name in sorted(allowed):
        entry = entries[name]
        # Paths come from this allowlist, never from unchecked remote metadata.
        target = package / Path(name).relative_to("text_decoder.mlpackage")
        target.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}", timeout=90) as response:
            data = response.read(2 * 1024 * 1024)
        if len(data) != entry["size"]:
            raise ValueError(f"Invalid source file size: {name}")
        lfs = entry.get("lfs")
        digest = (hashlib.sha256(data).hexdigest() if lfs else
                  hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest())
        if digest != (lfs["oid"] if lfs else entry["oid"]):
            raise ValueError(f"Invalid source file hash: {name}")
        target.write_bytes(data)
    weight_target = package / "Data/com.apple.CoreML/weights/weight.bin"
    weight_target.parent.mkdir(parents=True, exist_ok=True)
    os.link(compiled_weights, weight_target)


def link_copy(source, target):
    os.link(source, target)
    return target


def prepare_candidates(cache, destination):
    if destination.exists():
        raise ValueError("Use a new output directory; existing candidates are never overwritten")
    base = cache / "int8"
    weights = base / "text_decoder.mlmodelc/weights/weight.bin"
    package = destination / "source/text_decoder.mlpackage"
    package.mkdir(parents=True)
    download_source(package, weights)
    specification = ct.utils.load_spec(str(package / "Data/com.apple.CoreML/model.mlmodel"))
    for mode in MODES:
        candidate_root = destination / mode / "int8"
        candidate_root.mkdir(parents=True)
        modified = candidate_root / "text_decoder.mlpackage"
        shutil.copytree(package, modified, copy_function=link_copy)
        model = modified / "Data/com.apple.CoreML/model.mlmodel"
        # The source package is immutable even though large weights are shared.
        model.unlink()
        ct.utils.save_spec(transform(specification, mode), str(model))
        compiled = Path(ct.models.utils.compile_model(str(modified)))
        final = candidate_root / "text_decoder.mlmodelc"
        try:
            shutil.copytree(compiled, final, copy_function=link_copy)
        finally:
            shutil.rmtree(compiled)
        new_weights = final / "weights/weight.bin"
        if sha256(new_weights) != WEIGHT_SHA256:
            raise ValueError(f"Candidate {mode} unexpectedly changed decoder weights")
        new_weights.unlink()
        os.link(weights, new_weights)
        for item in base.iterdir():
            if item.name == "text_decoder.mlmodelc":
                continue
            target = candidate_root / item.name
            if item.is_dir():
                shutil.copytree(item, target, copy_function=link_copy)
            else:
                link_copy(item, target)
        print(json.dumps({"candidate": mode, "cache_root": str(destination / mode),
                          "weights_sha256": WEIGHT_SHA256}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    arguments = parser.parse_args()
    prepare_candidates(arguments.cache_root.resolve(), arguments.output_root.resolve())
