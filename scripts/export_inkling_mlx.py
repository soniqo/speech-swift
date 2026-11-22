#!/usr/bin/env python3
"""Export Inkling 2 Mini's mixed NVFP4/MXFP8 checkpoint to affine MLX weights.

Dependencies: mlx, numpy, huggingface_hub (only for downloading).
The output uses the MLX-VLM Inkling 2 parameter layout and loads in the
package's native Inkling2Chat backend.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import math
import re
import shutil
import struct
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

SOURCE_MODEL = "thinkingmachines/Inkling-2-Mini"
SOURCE_REVISION = "ed40e7da625f594fc9c4990f5f66be24873c22d7"
FORMAT = "speech-swift-inkling2-mlx-v1"
EXPECTED_SOURCE_TENSORS = 1244
ASSETS = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "chat_template.jinja", "generation_config.json", "processor_config.json",
)
RAW_DTYPES = {"F32": "<f4", "F16": "<f2", "BF16": "<u2", "U8": "u1", "F8_E4M3": "u1"}
FP4_VALUES = np.array([0, .5, 1, 1.5, 2, 3, 4, 6,
                       -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Use a complete local source snapshot.")
    parser.add_argument("--stream-source", action="store_true",
                        help="Read pinned remote tensor ranges without caching the 19.5 GB source weights.")
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--output", type=Path, required=True, help="New or empty output directory.")
    parser.add_argument("--bits", type=int, choices=(4, 5, 8), default=4,
                        help="Expert precision; attention and vocabulary matrices use 8 bits.")
    parser.add_argument("--group-size", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--max-shard-size-mb", type=int, default=512)
    parser.add_argument("--include-mtp", action="store_true", help="Copy the upstream BF16 draft unchanged.")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
        parser.error("--source-revision must be an immutable 40-character commit")
    if args.max_shard_size_mb < 16:
        parser.error("--max-shard-size-mb must be at least 16")
    if args.source_dir and args.stream_source:
        parser.error("--source-dir and --stream-source are mutually exclusive")
    return args


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fp8_e4m3(raw: np.ndarray) -> np.ndarray:
    """Decode E4M3FN, including subnormals, signed zero, and its two NaNs."""
    raw = np.asarray(raw, dtype=np.uint8)
    exponent = ((raw >> 3) & 15).astype(np.int32)
    mantissa = (raw & 7).astype(np.float32)
    magnitude = np.where(exponent == 0, mantissa * (2. ** -9),
                         np.ldexp(1 + mantissa / 8, exponent - 7))
    result = np.copysign(magnitude, np.where(raw & 128, -1., 1.)).astype(np.float32)
    return np.where((raw & 127) == 127, np.float32(np.nan), result)


def dequantize_expert(
    raw: np.ndarray, scales: np.ndarray, transform: str,
    outer_scale: float | np.ndarray = 1.,
) -> np.ndarray:
    """Decode one expert to FP32; NVFP4 nibbles are low-first along the input axis."""
    if transform == "nvfp4_nearest":
        raw = np.asarray(raw, dtype=np.uint8)
        values = FP4_VALUES[np.stack((raw & 15, raw >> 4), axis=-1)].reshape(
            *raw.shape[:-1], raw.shape[-1] * 2)
        decoded_scales = fp8_e4m3(scales)
        group = 16
    elif transform == "mxfp8_nearest":
        values = fp8_e4m3(raw)
        raw_scales = np.asarray(scales, dtype=np.uint8)
        if np.any(raw_scales == 255):
            raise ValueError("Invalid UE8M0 scale (NaN)")
        decoded_scales = np.ldexp(np.ones(raw_scales.shape, dtype=np.float32),
                                  raw_scales.astype(np.int32) - 127)
        group = 32
    else:
        raise ValueError(f"Unsupported expert transform: {transform}")
    if decoded_scales.shape != (*values.shape[:-1], values.shape[-1] // group):
        raise ValueError(f"Expert scales do not match weight shape {values.shape}")
    result = (values.reshape(*values.shape[:-1], -1, group)
              * decoded_scales[..., None]).reshape(values.shape)
    result *= outer_scale
    if not np.isfinite(result).all():
        raise ValueError("Non-finite expert weights or scales")
    return result


class TensorSource:
    """Memory-map source tensors without loading a model or converting its FP8 globally."""

    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        index = read_json(directory / "model.safetensors.index.json")
        self.weight_map = index["weight_map"]
        if not isinstance(self.weight_map, dict) or not self.weight_map:
            raise ValueError("Source index must contain a nonempty weight_map")
        self.headers: dict[str, dict] = {}
        self.starts: dict[str, int] = {}
        for shard in sorted(set(self.weight_map.values())):
            # A Hub shard is a filename, never a path outside its snapshot.
            if not isinstance(shard, str) or Path(shard).name != shard or "\\" in shard:
                raise ValueError(f"Invalid source shard filename: {shard}")
            path = directory / shard
            with path.open("rb") as handle:
                prefix = handle.read(8)
                if len(prefix) != 8:
                    raise ValueError(f"Truncated safetensors header: {shard}")
                size = struct.unpack("<Q", prefix)[0]
                if not 0 < size <= 16 * 1024 * 1024:
                    raise ValueError(f"Invalid safetensors header length: {shard}")
                header = json.loads(handle.read(size))
            start = size + 8
            file_size = path.stat().st_size
            for name, tensor in header.items():
                if name == "__metadata__":
                    continue
                dtype = tensor["dtype"]
                shape = tensor["shape"]
                begin, end = tensor["data_offsets"]
                if dtype not in RAW_DTYPES or any(type(n) is not int or n < 0 for n in shape):
                    raise ValueError(f"Unsupported source dtype or shape: {name}")
                expected = math.prod(shape) * np.dtype(RAW_DTYPES[dtype]).itemsize
                if begin < 0 or end - begin != expected or start + end > file_size:
                    raise ValueError(f"Invalid source tensor offsets: {name}")
                if self.weight_map.get(name) != shard:
                    raise ValueError(f"Source index does not match shard: {name}")
            self.headers[shard], self.starts[shard] = header, start
        for name, shard in self.weight_map.items():
            if name not in self.headers[shard]:
                raise ValueError(f"Indexed tensor is missing: {name}")

    def info(self, name: str) -> dict:
        return self.headers[self.weight_map[name]][name]

    def raw(self, name: str) -> np.ndarray:
        shard = self.weight_map[name]
        tensor = self.info(name)
        return np.memmap(self.directory / shard, mode="r", dtype=RAW_DTYPES[tensor["dtype"]],
                         offset=self.starts[shard] + tensor["data_offsets"][0], shape=tuple(tensor["shape"]))


class RemoteTensorSource(TensorSource):
    """Read one pinned tensor at a time; retain no full source shard on disk."""

    def __init__(self, directory: Path, revision: str):
        self.directory = directory
        self.revision = revision
        self.weight_map = read_json(directory / "model.safetensors.index.json")["weight_map"]
        self.headers, self.starts, self.tensor_hashes = {}, {}, {}
        for shard in sorted(set(self.weight_map.values())):
            if Path(shard).name != shard or "\\" in shard:
                raise ValueError(f"Invalid source shard filename: {shard}")
            size = struct.unpack("<Q", self.fetch(shard, 0, 8))[0]
            if not 0 < size <= 16 * 1024 * 1024:
                raise ValueError("Invalid remote safetensors header size")
            self.headers[shard] = json.loads(self.fetch(shard, 8, size))
            self.starts[shard] = size + 8
        for name, shard in self.weight_map.items():
            info = self.headers[shard][name]
            if info["dtype"] not in RAW_DTYPES:
                raise ValueError(f"Unsupported remote dtype: {name}")
            expected = math.prod(info["shape"]) * np.dtype(RAW_DTYPES[info["dtype"]]).itemsize
            begin, end = info["data_offsets"]
            if begin < 0 or end - begin != expected:
                raise ValueError(f"Invalid remote tensor offsets: {name}")

    def fetch(self, shard: str, offset: int, length: int) -> bytes:
        # Parallelize bounded network ranges, not model computation. This avoids
        # one slow CDN connection holding up a nearly-gigabyte vocabulary matrix.
        chunk = 16 * 1024 * 1024
        if length > chunk:
            spans = [(start, min(chunk, length - start)) for start in range(0, length, chunk)]
            buffers = [None] * len(spans)
            complete = 0
            with ThreadPoolExecutor(max_workers=4) as pool:
                pending = {pool.submit(self.fetch, shard, offset + start, size): i
                           for i, (start, size) in enumerate(spans)}
                for future in as_completed(pending):
                    i = pending[future]
                    buffers[i] = future.result()
                    complete += len(buffers[i])
                    if len(spans) > 8 and (complete // chunk) % 8 == 0:
                        print(f"  fetched {complete / 1024**2:.0f}/{length / 1024**2:.0f} MiB", flush=True)
            return b"".join(buffers)
        url = f"https://huggingface.co/{SOURCE_MODEL}/resolve/{self.revision}/{shard}"
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{offset + length - 1}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206 or response.headers.get("Content-Range", "").split("/")[0] != f"bytes {offset}-{offset + length - 1}":
                raise ValueError("Source server did not honor the exact byte-range request")
            data = response.read(length + 1)
            if len(data) != length:
                raise ValueError("Truncated or oversized source tensor response")
        return data

    def raw(self, name: str) -> np.ndarray:
        shard = self.weight_map[name]
        info = self.info(name)
        begin, end = info["data_offsets"]
        if end - begin > 16 * 1024 * 1024:
            print(f"Reading {name} ({(end - begin) / 1024**2:.0f} MiB)", flush=True)
        data = self.fetch(shard, self.starts[shard] + begin, end - begin)
        self.tensor_hashes[name] = hashlib.sha256(data).hexdigest()
        return np.frombuffer(data, dtype=RAW_DTYPES[info["dtype"]]).reshape(info["shape"])


def validate_config(config: dict) -> None:
    text = config.get("text_config", {})
    expected = {"model_type": "inkling2_loop_text", "hidden_size": 2304,
                "num_hidden_layers": 24, "n_routed_experts": 128,
                "intermediate_size": 1152, "num_experts_per_tok": 8}
    if config.get("model_type") != "inkling2_loop" or any(text.get(k) != v for k, v in expected.items()):
        raise ValueError("Expected the Inkling 2 Mini looped 26B checkpoint")
    loop = text.get("loop_config", {})
    if loop != {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64, "is_varloop": False}:
        raise ValueError("Unsupported Inkling 2 Mini loop_config")
    if config.get("quantization_config", {}).get("quant_method") != "modelopt":
        raise ValueError("Expected the upstream mixed NVFP4/MXFP8 checkpoint")


def weight_transforms(metadata: dict, names: set[str]) -> dict[str, str]:
    if metadata.get("quant_method") != "inkling2":
        raise ValueError("Expected Inkling 2 hf_quant_config.json")
    result = {}
    for group in metadata["model"]["parameter_groups"]:
        transform = group["weight_transform"]
        if transform not in ("cast", "nvfp4_nearest", "mxfp8_nearest"):
            raise ValueError(f"Unsupported source weight transform: {transform}")
        for name in group["tensors"]:
            if name in result or name not in names:
                raise ValueError(f"Duplicate or missing quantization tensor: {name}")
            result[name] = transform
    # Scaling auxiliaries have no parameter-group entry. Every other tensor must have one.
    auxiliaries = {n + suffix for n, t in result.items() if t != "cast"
                   for suffix in ("_scale", "_amax", "_outer_scale_denominator")}
    if names - result.keys() - auxiliaries:
        raise ValueError(f"Unclassified source tensors: {sorted(names - result.keys() - auxiliaries)[:3]}")
    return result


def validate_source_shapes(source: TensorSource) -> None:
    if len(source.weight_map) != EXPECTED_SOURCE_TENSORS:
        raise ValueError(f"Expected {EXPECTED_SOURCE_TENSORS} source tensors; checkpoint is incomplete or unsupported")
    for name in ("model.llm.embed.weight", "model.llm.unembed.weight"):
        if source.info(name)["shape"] != [201024, 2304]:
            raise ValueError(f"Unexpected vocabulary matrix shape: {name}")
    for layer in range(24):
        for projection, rows, columns in (("w13", 2304, 2304), ("w2", 2304, 1152)):
            name = f"model.llm.layers.{layer}.mlp.experts.{projection}_weight"
            if source.info(name)["shape"] != [128, rows, columns // (2 if layer < 20 else 1)]:
                raise ValueError(f"Unexpected expert matrix shape: {name}")
            if source.info(name + "_scale")["shape"] != [128, rows, columns // (16 if layer < 20 else 32)]:
                raise ValueError(f"Unexpected expert scale shape: {name}")


def mapped_name(name: str) -> tuple[str, bool]:
    """Return the MLX-VLM parameter name and whether its Conv1d axes need reordering."""
    llm, layers = "model.llm.", "model.llm.layers."
    if name.startswith(layers):
        index, sub = name[len(layers):].split(".", 1)
        prefix = f"language_model.model.layers.{index}."
        if sub.startswith("attn."):
            sub = sub[5:]
            exact = {"q_l2_clip.scale": "q_scale", "rel_l2_clip.scale": "rel_scale",
                     "rel_logits_proj.proj": "rel_proj"}
            if sub in exact:
                return prefix + "self_attn." + exact[sub], False
            module, _, leaf = sub.partition(".")
            module = {"wq_du": "q_proj", "wk_dv": "k_proj", "wv_dv": "v_proj",
                      "wr_du": "r_proj", "wo_ud": "o_proj", "gate_attn": "gate_proj",
                      "attn_sink_proj": "sink_proj"}.get(module, module)
            if module in ("k_sconv", "v_sconv"):
                if leaf == "weight":
                    return prefix + f"self_attn.{module}.conv.weight", True
                leaf = {"static_w": "static_weight", "weight_proj": "weight_proj.weight"}.get(leaf, leaf)
            return prefix + f"self_attn.{module}.{leaf}", False
        if sub in ("attn_sconv.weight", "mlp_sconv.weight"):
            return prefix + sub.replace(".weight", ".conv.weight"), True
        if sub in ("mlp.gate.weight", "mlp.gate.bias"):
            return prefix + ("mlp.gate_weight" if sub.endswith("weight") else "mlp.e_score_correction_bias"), False
        module, _, leaf = sub.partition(".")
        module = {"attn_norm": "input_layernorm", "mlp_norm": "post_attention_layernorm",
                  "q_input_norm": "q_input_layernorm", "k_input_norm": "k_input_layernorm",
                  "v_input_norm": "v_input_layernorm"}.get(module, module)
        return prefix + f"{module}.{leaf}", False
    if name.startswith(llm + "attn_res."):
        sub = name[len(llm + "attn_res."):]
        if sub == "output_query":
            return "language_model.model.final_input.query", False
        kind, index = sub.split(".")
        if kind not in ("attn_queries", "mlp_queries"):
            raise ValueError(f"Unknown attention residual tensor: {name}")
        layer = int(index) + (kind == "attn_queries")
        slot = "attention_input" if kind == "attn_queries" else "mlp_input"
        return f"language_model.model.layers.{layer}.{slot}.query", False
    if name.startswith(llm):
        sub = name[len(llm):]
        if sub == "unembed.weight":
            return "language_model.lm_head.weight", False
        return "language_model.model." + ("embed_tokens.weight" if sub == "embed.weight" else sub), False
    if name.startswith("model.visual."):
        sub = name[len("model.visual."):].removeprefix("backbone.")
        for before, after in (("ln_pre.", "pre_layernorm."), (".ln_1.", ".layernorm_before."),
                              (".ln_2.", ".layernorm_after."), (".ls_1.gamma", ".lambda_1"),
                              (".ls_2.gamma", ".lambda_2"), (".mlp.c_fc.", ".mlp.fc1."),
                              (".mlp.c_proj.", ".mlp.fc2.")):
            sub = sub.replace(before, after)
        return "vision_tower." + sub, False
    raise ValueError(f"Unknown Inkling parameter: {name}")


def quantize_dense(name: str, shape: tuple[int, ...]) -> bool:
    # Retain BF16 vision, routers, relative-position projections, and loop controllers.
    if not name.startswith("language_model.") or len(shape) != 2 or not name.endswith(".weight"):
        return False
    if name in ("language_model.model.embed_tokens.weight", "language_model.lm_head.weight"):
        return True
    return ".self_attn." in name and name.rsplit(".", 2)[-2] in {
        "q_proj", "k_proj", "v_proj", "r_proj", "o_proj", "gate_proj", "sink_proj"}


def estimated_weight_bytes(source: TensorSource, transforms: dict[str, str], bits: int, group_size: int) -> int:
    total = 0
    for name, transform in transforms.items():
        info = source.info(name)
        elements = math.prod(info["shape"])
        if transform != "cast":
            logical = elements * (2 if transform == "nvfp4_nearest" else 1)
            total += logical * bits // 8 + logical // group_size * 4
        else:
            target, _ = mapped_name(name)
            itemsize = np.dtype(RAW_DTYPES[info["dtype"]]).itemsize
            if quantize_dense(target, tuple(info["shape"])):
                total += elements + elements // group_size * itemsize * 2
            else:
                total += elements * itemsize
    return total


class ShardWriter:
    def __init__(self, output: Path, max_bytes: int):
        self.output, self.max_bytes = output, max_bytes
        self.pending, self.weight_map = {}, {}
        self.pending_bytes = self.total_bytes = 0
        self.shards: list[Path] = []

    def add(self, name: str, value) -> None:
        import mlx.core as mx
        if name in self.weight_map or name in self.pending:
            raise ValueError(f"Duplicate output tensor: {name}")
        if self.pending and self.pending_bytes + value.nbytes > self.max_bytes:
            self.flush()
        mx.eval(value)
        self.pending[name] = value
        self.pending_bytes += value.nbytes

    def flush(self) -> None:
        import mlx.core as mx
        if not self.pending:
            return
        path = self.output / f"part-{len(self.shards) + 1:05d}.safetensors"
        mx.save_safetensors(str(path), self.pending, metadata={"format": "mlx"})
        self.weight_map.update({name: path.name for name in self.pending})
        self.total_bytes += self.pending_bytes
        self.shards.append(path)
        self.pending.clear()
        self.pending_bytes = 0
        mx.clear_cache()

    def finish(self) -> dict:
        self.flush()
        renamed = {}
        for index, path in enumerate(self.shards, 1):
            filename = f"model-{index:05d}-of-{len(self.shards):05d}.safetensors"
            path.rename(self.output / filename)
            renamed[path.name] = filename
        result = {"metadata": {"total_size": self.total_bytes},
                  "weight_map": {k: renamed[v] for k, v in sorted(self.weight_map.items())}}
        write_json(self.output / "model.safetensors.index.json", result)
        return result


def cast_tensor(source: TensorSource, name: str):
    import mlx.core as mx
    raw = np.array(source.raw(name))
    dtype = source.info(name)["dtype"]
    if dtype == "BF16":
        if np.any((raw & 0x7F80) == 0x7F80):
            raise ValueError(f"Non-finite BF16 parameter: {name}")
        return mx.array(raw).view(mx.bfloat16)
    if dtype == "F8_E4M3":
        raise ValueError(f"FP8 tensor outside an expert parameter group: {name}")
    if not np.isfinite(raw).all():
        raise ValueError(f"Non-finite parameter: {name}")
    return mx.array(raw)


def export_experts(source: TensorSource, name: str, transform: str, bits: int, group_size: int):
    import mlx.core as mx
    raw, scales = source.raw(name), source.raw(name + "_scale")
    fused = name.endswith("w13_weight")
    shape = source.info(name)["shape"]
    if len(shape) != 3 or shape[0] != 128:
        raise ValueError(f"Expected 128 expert matrices: {name}")
    if transform == "nvfp4_nearest":
        if source.info(name)["dtype"] != "U8" or source.info(name + "_scale")["dtype"] != "F8_E4M3":
            raise ValueError(f"NVFP4 source dtypes do not match: {name}")
        amax = source.raw(name + "_amax")
        denominator = source.raw(name + "_outer_scale_denominator")
        if amax.shape != (128,) or denominator.size != 1 or not np.isfinite(denominator).all() or denominator.item() <= 0:
            raise ValueError(f"Invalid NVFP4 outer scales: {name}")
        outer = amax / denominator.item()
    else:
        if source.info(name)["dtype"] != "F8_E4M3" or source.info(name + "_scale")["dtype"] != "U8":
            raise ValueError(f"MXFP8 source dtypes do not match: {name}")
        outer = np.ones((128,), dtype=np.float32)
    projections = ("gate_proj", "up_proj") if fused else ("down_proj",)
    outputs = {p: [[], [], []] for p in projections}
    for expert in range(128):
        dense = dequantize_expert(raw[expert], scales[expert], transform, outer[expert])
        parts = (dense[0::2], dense[1::2]) if fused else (dense,)
        for projection, part in zip(projections, parts):
            if part.shape[-1] % group_size:
                raise ValueError(f"Expert input dimension is not divisible by {group_size}: {name}")
            quantized = mx.quantize(mx.array(np.ascontiguousarray(part)).astype(mx.bfloat16),
                                    group_size=group_size, bits=bits)
            mx.eval(*quantized)
            for values, value in zip(outputs[projection], quantized):
                values.append(value)
    layer = name.split(".layers.")[1].split(".")[0]
    base = f"language_model.model.layers.{layer}.mlp.switch_mlp."
    for projection, tensors in outputs.items():
        for suffix, values in zip(("weight", "scales", "biases"), tensors):
            yield base + projection + "." + suffix, mx.stack(values)


def resolve_source(args: argparse.Namespace) -> Path:
    if args.source_dir:
        return args.source_dir.expanduser().resolve()
    from huggingface_hub import snapshot_download
    patterns = ["config.json", "hf_quant_config.json", "model.safetensors.index.json",
                "README.md", "LICENSE*", "NOTICE*", "tiktoken/*", *ASSETS]
    if not args.stream_source:
        patterns.append("model-*.safetensors")
    if args.include_mtp:
        patterns.append("mtp.safetensors")
    return Path(snapshot_download(SOURCE_MODEL, revision=args.source_revision, allow_patterns=patterns))


def export(
    source: Path, output: Path, *, bits: int = 4, group_size: int = 64,
    source_revision: str = SOURCE_REVISION, max_shard_bytes: int = 512 * 1024 * 1024,
    include_mtp: bool = False, stream_source: bool = False,
) -> dict:
    if bits not in (4, 5, 8) or group_size not in (32, 64, 128) or max_shard_bytes <= 0:
        raise ValueError("Unsupported export precision, group size, or shard size")
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("Source revision must be an immutable commit")
    source, output = source.expanduser().resolve(), output.expanduser().resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Source and output directories must not overlap")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {output}")
    required = ("config.json", "hf_quant_config.json", "model.safetensors.index.json", "README.md", *ASSETS)
    if include_mtp:
        required += ("mtp.safetensors",)
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete source snapshot; missing: {', '.join(missing)}")
    original = read_json(source / "config.json")
    validate_config(original)
    tensors = RemoteTensorSource(source, source_revision) if stream_source else TensorSource(source)
    validate_source_shapes(tensors)
    transforms = weight_transforms(read_json(source / "hf_quant_config.json"), set(tensors.weight_map))
    for layer in range(24):
        for projection in ("w13", "w2"):
            name = f"model.llm.layers.{layer}.mlp.experts.{projection}_weight"
            expected = "nvfp4_nearest" if layer < 20 else "mxfp8_nearest"
            if transforms.get(name) != expected:
                raise ValueError(f"Missing or unexpected expert transform: {name}")
    # The staging directory becomes the destination by rename, so no second
    # output copy is needed. Retain a 2 GiB reserve for metadata and system use.
    required_bytes = estimated_weight_bytes(tensors, transforms, bits, group_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    if include_mtp:
        required_bytes += (source / "mtp.safetensors").stat().st_size
    if shutil.disk_usage(output.parent).free < required_bytes + 2 * 1024**3:
        raise ValueError("Insufficient free disk space for this export and a 2 GiB safety reserve")
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    mx.set_cache_limit(128 * 1024 * 1024)
    config = copy.deepcopy(original)
    config.pop("quantization_config", None)
    if not include_mtp:
        config.pop("mtp_config", None)
    quantization = {"bits": bits, "group_size": group_size, "mode": "affine"}
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temp:
        stage = Path(temp)
        writer = ShardWriter(stage, max_shard_bytes)
        cross_pass = ("model.llm.middle_cycle_cross_pass_queries", "model.llm.middle_cycle_cross_pass_kv_queries")
        for name, transform in sorted(transforms.items()):
            if name in cross_pass:
                continue
            if transform != "cast":
                print(f"Converting {name} ({transform} -> INT{bits})", flush=True)
                for target, value in export_experts(tensors, name, transform, bits, group_size):
                    writer.add(target, value)
                mx.clear_cache()
                continue
            target, conv = mapped_name(name)
            value = cast_tensor(tensors, name)
            if conv and value.shape[-1] != 1:
                value = value.transpose(0, 2, 1)
            if quantize_dense(target, value.shape):
                prefix = target.removesuffix(".weight")
                if value.shape[-1] % group_size:
                    raise ValueError(f"Matrix input dimension is not divisible by {group_size}: {name}")
                packed = mx.quantize(value, group_size=group_size, bits=8)
                for suffix, v in zip(("weight", "scales", "biases"), packed):
                    writer.add(prefix + "." + suffix, v)
                quantization[prefix] = {"bits": 8, "group_size": group_size, "mode": "affine"}
            else:
                writer.add(target, value)
        width = original["text_config"]["hidden_size"]
        am, kv = (cast_tensor(tensors, name).reshape(-1, 2, width) for name in cross_pass)
        if am.shape != kv.shape:
            raise ValueError("Cross-pass query shapes do not match")
        writer.add("language_model.model.message_queries", mx.concatenate([am[:, :1], kv, am[:, 1:]], axis=1))
        index = writer.finish()
        config["quantization"] = quantization
        config["quantization_config"] = quantization
        write_json(stage / "config.json", config)
        shutil.copy2(source / "config.json", stage / "source_config.json")
        shutil.copy2(source / "hf_quant_config.json", stage / "source_quant_config.json")
        for name in (*ASSETS, "README.md", "LICENSE", "NOTICE", "mtp.safetensors" if include_mtp else ""):
            if name and (source / name).is_file():
                shutil.copy2(source / name, stage / ("source_README.md" if name == "README.md" else name))
        if (source / "tiktoken").is_dir():
            shutil.copytree(source / "tiktoken", stage / "tiktoken")
        if not (stage / "LICENSE").exists():
            shutil.copy2(Path(__file__).resolve().parents[1] / "ThirdPartyNotices/Inkling2Mini.txt", stage / "LICENSE")
        (stage / "README.md").write_text(
            f"---\nlicense: apache-2.0\nbase_model: {SOURCE_MODEL}\nlibrary_name: mlx\n"
            f"pipeline_tag: image-text-to-text\n---\n\n# Inkling 2 Mini (MLX INT{bits})\n\n"
            f"Converted from [{SOURCE_MODEL}](https://huggingface.co/{SOURCE_MODEL}/tree/{source_revision}).\n\n"
            f"Expert weights use affine INT{bits}, group {group_size}; attention and vocabulary matrices use INT8. "
            "Other parameters retain source precision. The looped architecture, vision encoder and tokenizer "
            "are preserved. The parameter layout follows MLX-VLM Inkling 2.\n\n"
            "Load with Inkling2Chat.fromDirectory in speech-swift or the speech inkling CLI command. "
            "Export does not establish inference quality or performance. Source configuration, conversion "
            "settings, and file hashes are included. See LICENSE and source_README.md.\n",
            encoding="utf-8")
        metadata = {"format": FORMAT, "source_model": SOURCE_MODEL, "source_revision": source_revision,
                    "mlx_version": mx.__version__, "numpy_version": np.__version__,
                    "local_source_revision_verified": False,
                    "expert_bits": bits, "dense_bits": 8, "group_size": group_size,
                    "include_mtp": include_mtp, "stream_source": stream_source, "tensor_count": len(index["weight_map"]),
                    "source_tensors": len(tensors.weight_map), "weight_bytes": index["metadata"]["total_size"],
                    "files": {str(p.relative_to(stage)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                              for p in sorted(stage.rglob("*")) if p.is_file()}}
        # A local directory's commit is a caller-supplied label, not proof. Preserve
        # its hashes so exports remain auditable even outside the Hub cache.
        metadata["source_files"] = {name: {"sha256": sha256(source / name)}
                                    for name in (*required, *([] if stream_source else tensors.headers))}
        if stream_source:
            metadata["source_tensor_sha256"] = tensors.tensor_hashes
        write_json(stage / "export_config.json", metadata)
        if output.exists():
            output.rmdir()
        stage.rename(output)
    return metadata


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {args.output}")
    source = resolve_source(args)
    result = export(source, args.output, bits=args.bits, group_size=args.group_size,
                    source_revision=args.source_revision, max_shard_bytes=args.max_shard_size_mb * 1024 * 1024,
                    include_mtp=args.include_mtp, stream_source=args.stream_source)
    print(json.dumps({"output": str(args.output), "tensors": result["tensor_count"],
                      "weight_bytes": result["weight_bytes"]}))


if __name__ == "__main__":
    main()
