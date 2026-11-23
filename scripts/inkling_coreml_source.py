"""Read the pinned source checkpoint one Core ML partition at a time."""
from __future__ import annotations

from pathlib import Path
import math

import numpy as np

from export_inkling_mlx import (TensorSource, RemoteTensorSource, SOURCE_REVISION,
    read_json, validate_config, validate_source_shapes, weight_transforms,
    mapped_name, dequantize_expert, estimated_weight_bytes)


def round_bfloat16(value):
    value = np.ascontiguousarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("Non-finite source expert")
    bits = value.view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    result = rounded.view(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Source expert exceeds BF16 range")
    return result


class Checkpoint:
    def __init__(self, directory: Path, *, stream=False, revision=SOURCE_REVISION):
        self.directory = directory
        self.config = read_json(directory / "config.json")
        validate_config(self.config)
        self.tensors = RemoteTensorSource(directory, revision) if stream else TensorSource(directory)
        validate_source_shapes(self.tensors)
        self.transforms = weight_transforms(read_json(directory / "hf_quant_config.json"),
                                           set(self.tensors.weight_map))
        self.names = {mapped_name(name)[0]: name for name, transform in self.transforms.items()
                      if transform == "cast"}
        self.revision = revision

    def estimate_bytes(self, group_size=64):
        # The affine estimate includes more scale storage than symmetric INT8.
        return estimated_weight_bytes(self.tensors, self.transforms, 8, group_size)

    def read(self, name):
        if name == "language_model.model.message_queries":
            if not hasattr(self, "_message_queries"):
                text = self.config["text_config"]
                width = text["hidden_size"]
                loop = text["loop_config"]
                expected = (loop["core_end"] - loop["core_start"], 2, width)
                prefix = "language_model.model.middle_cycle_cross_pass"
                am = self.read(prefix + "_queries").reshape(-1, 2, width)
                kv = self.read(prefix + "_kv_queries").reshape(-1, 2, width)
                if am.shape != expected or kv.shape != expected:
                    raise ValueError("Cross-pass query shapes do not match the core layers")
                self._message_queries = np.concatenate([am[:, :1], kv, am[:, 1:]], axis=1)
            return self._message_queries
        original = self.names[name]
        raw = np.asarray(self.tensors.raw(original))
        dtype = self.tensors.info(original)["dtype"]
        if dtype == "BF16":
            result = (raw.astype(np.uint32) << 16).view(np.float32)
        elif dtype in ["F32", "F16"]:
            result = np.array(raw, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported ordinary parameter dtype: {dtype}")
        if not np.isfinite(result).all():
            raise ValueError(f"Non-finite source parameter: {original}")
        if mapped_name(original)[1]:
            result = result.transpose(0, 2, 1).copy()
        return result

    def ordinary(self, prefix):
        return {name: self.read(name) for name in sorted(self.names) if name.startswith(prefix)}

    def experts(self, layer):
        result = {}
        for original, transform in self.transforms.items():
            if not original.startswith(f"model.llm.layers.{layer}.") or transform == "cast":
                continue
            raw = self.tensors.raw(original)
            scales = self.tensors.raw(original + "_scale")
            if transform == "nvfp4_nearest":
                maximum = self.tensors.raw(original + "_amax")
                denominator = self.tensors.raw(original + "_outer_scale_denominator")
                if denominator.size != 1 or not np.isfinite(denominator).all() or denominator.item() <= 0:
                    raise ValueError("Invalid NVFP4 outer-scale denominator")
                outer = maximum / denominator.item()
            else:
                outer = np.ones(raw.shape[0], dtype=np.float32)
            fused = original.endswith("w13_weight")
            projections = ["gate_proj", "up_proj"] if fused else ["down_proj"]
            rows = raw.shape[1] // (2 if fused else 1)
            columns = raw.shape[2] * (2 if transform == "nvfp4_nearest" else 1)
            arrays = {name: np.empty((raw.shape[0], rows, columns), dtype=np.float32)
                      for name in projections}
            for expert in range(raw.shape[0]):
                dense = dequantize_expert(raw[expert], scales[expert], transform, outer[expert])
                parts = [dense[::2], dense[1::2]] if fused else [dense]
                for name, value in zip(projections, parts):
                    arrays[name][expert] = round_bfloat16(value)
            prefix = f"language_model.model.layers.{layer}.mlp.switch_mlp."
            result.update({prefix + name + ".weight": value for name, value in arrays.items()})
        if len(result) != 3:
            raise ValueError(f"Expected all three expert projections for layer {layer}")
        return result

    def embedding(self):
        return {name: self.read(name) for name in ["language_model.model.embed_tokens.weight",
                                                 "language_model.model.embed_norm.weight"]}

    def layer(self, index):
        weights = self.ordinary(f"language_model.model.layers.{index}.")
        loop = self.config["text_config"]["loop_config"]
        if loop["core_start"] <= index < loop["core_end"]:
            name = "language_model.model.message_queries"
            weights[name] = self.read(name)
        weights.update(self.experts(index))
        return weights

    def head(self):
        return {name: self.read(name) for name in ["language_model.lm_head.weight",
            "language_model.model.final_input.query", "language_model.model.norm.weight"]}

    def vision(self):
        return self.ordinary("vision_tower.")
