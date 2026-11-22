#!/usr/bin/env python3
"""Generate a small looped Inkling parity fixture from the pinned MLX-VLM reference.

Local-only dependencies: mlx-vlm==0.8.0, mlx, numpy, pillow.
Only Python reference source is downloaded; no model weights or hosted inference.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
import types
import urllib.request

REFERENCE_REVISION = "0d16abeb7c7167a354b36855d8088239ba675b5c"


def reference_modules(directory: Path):
    root = Path(importlib.metadata.distribution("mlx-vlm").locate_file("mlx_vlm"))
    # Import model code directly, without the optional server/audio/media extras
    # imported by the distribution's top-level initializer.
    for name in ("mlx_vlm", "mlx_vlm.models", "mlx_vlm.models.inkling", "mlx_vlm.models.inkling2", "mlx_vlm.speculative"):
        module = types.ModuleType(name)
        module.__path__ = [str(root.joinpath(*name.split(".")[1:]))]
        sys.modules[name] = module
    for filename in ("config.py", "language.py", "kernels.py", "vision.py"):
        url = (f"https://raw.githubusercontent.com/Neywa-Labs/mlx-vlm/{REFERENCE_REVISION}/"
               f"mlx_vlm/models/inkling2/{filename}")
        with urllib.request.urlopen(url, timeout=30) as response:
            (directory / filename).write_bytes(response.read())
    sys.modules["mlx_vlm.models.inkling2"].__path__ = [str(directory)]
    from mlx_vlm.models.inkling2.config import TextConfig
    from mlx_vlm.models.inkling2.language import LanguageModel
    return TextConfig, LanguageModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
    import numpy as np
    mx.set_default_device(mx.cpu)
    with tempfile.TemporaryDirectory() as temp:
        TextConfig, LanguageModel = reference_modules(Path(temp))
        config = TextConfig(model_type="inkling2_loop_text", hidden_size=64, num_hidden_layers=8,
            vocab_size=64, unpadded_vocab_size=61, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            swa_num_attention_heads=4, swa_num_key_value_heads=2, swa_head_dim=16, sliding_window_size=4,
            d_rel=4, rel_extent=8, n_layers_per_attn_res_block=2, intermediate_size=64, n_routed_experts=4,
            num_experts_per_tok=2, local_layer_ids=[0, 2, 4, 6], log_scaling_n_floor=3,
            qk_l2_clip_threshold=.75, rel_l2_clip_threshold=.5,
            loop_config={"core_start": 2, "core_end": 6, "num_passes": 3, "gate_rank": 4, "is_varloop": False})
        model = LanguageModel(config)
        rng = np.random.default_rng(872)
        parameters = []
        for name, value in sorted(tree_flatten(model.parameters())):
            array = rng.normal(0, .05, value.shape).astype(np.float32)
            if name.endswith("norm.weight") or name.endswith("q_scale") or name.endswith("rel_scale"):
                array += 1
            parameters.append((name, mx.array(array)))
        model.update(tree_unflatten(parameters))
        overrides = {}
        def predicate(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            if ".switch_mlp." in path:
                return {"bits": 5, "group_size": 32}
            if path in ("model.embed_tokens", "lm_head") or (
                ".self_attn." in path and path.rsplit(".", 1)[-1] in
                {"q_proj", "k_proj", "v_proj", "r_proj", "o_proj", "gate_proj", "sink_proj"}):
                overrides["language_model." + path] = {"bits": 8, "group_size": 32, "mode": "affine"}
                return {"bits": 8, "group_size": 32}
            return False
        nn.quantize(model, group_size=32, bits=5, class_predicate=predicate)
        weights = {"language_model." + k: v for k, v in tree_flatten(model.parameters())}
        mx.eval(weights)
        args.output.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(args.output / "model.safetensors"), weights)
        quant = {"bits": 5, "group_size": 32, "mode": "affine", **overrides}
        outer = {"model_type": "inkling2_loop", "eos_token_id": 60,
                 "text_config": dataclasses.asdict(config), "quantization": quant}
        (args.output / "config.json").write_text(json.dumps(outer, indent=2) + "\n")
        (args.output / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {k: "model.safetensors" for k in weights}}, indent=2) + "\n")
        tokens = [1, 2, 3, 4, 5, 6, 7]
        cache = model.make_cache()
        full = model(mx.array([tokens]), cache=cache).logits
        mx.eval(full)
        continued = model(mx.array([[8]]), cache=cache).logits
        mx.eval(continued)
        cache = model.make_cache()
        for token in tokens:
            sequential = model(mx.array([[token]]), cache=cache).logits
            mx.eval(sequential)
        result = {"reference_revision": REFERENCE_REVISION, "tokens": tokens,
                  "full_logits": np.array(full).reshape(-1).tolist(),
                  "continued_logits": np.array(continued).reshape(-1).tolist(),
                  "sequential_logits": np.array(sequential).reshape(-1).tolist()}
        # At least 64 selected routes activates grouped expert prefill kernels.
        grouped_tokens = [1 + i % 59 for i in range(40)]
        grouped = model(mx.array([grouped_tokens]), cache=model.make_cache()).logits
        mx.eval(grouped)
        result["grouped_tokens"] = grouped_tokens
        result["grouped_logits"] = np.array(grouped).reshape(-1).tolist()
        (args.output / "reference.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {len(weights)} tensors and full/cached reference logits to {args.output}")


if __name__ == "__main__":
    main()
