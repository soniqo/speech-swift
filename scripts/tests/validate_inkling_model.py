#!/usr/bin/env python3
"""Record full-export text/vision reference outputs for local Swift verification.

Run this separately from Swift model tests and other model-loading jobs.
Requires the dependencies of generate_inkling_reference.py plus tokenizers.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import tempfile

from generate_inkling_reference import REFERENCE_REVISION, reference_modules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from tokenizers import Tokenizer
    mx.set_cache_limit(128 * 1024 * 1024)
    config = json.loads((args.model_dir / "config.json").read_text())
    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())
    weights = {}
    for shard in sorted(set(index["weight_map"].values())):
        weights.update(mx.load(str(args.model_dir / shard)))
    with tempfile.TemporaryDirectory() as temporary:
        TextConfig, LanguageModel = reference_modules(Path(temporary))
        from mlx_vlm.models.inkling2.config import VisionConfig
        from mlx_vlm.models.inkling2.vision import VisionModel
        model = LanguageModel(TextConfig.from_dict(config["text_config"]))
        quant = config["quantization"]
        def predicate(path, module):
            prefix = "language_model." + path
            if prefix + ".scales" not in weights:
                return False
            return quant.get(prefix, {k: quant[k] for k in ["bits", "group_size", "mode"]})
        nn.quantize(model, group_size=quant["group_size"], bits=quant["bits"], class_predicate=predicate)
        model.load_weights([(k.removeprefix("language_model."), v) for k, v in weights.items()
                            if k.startswith("language_model.")], strict=True)
        tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
        texts = ["Hello world!", "one\ttwo\nthree", "你好，世界！", "Grüße 👋🏽", "e\u0301",
                 "def foo(x):\n    return x + 1", "can't I'll", "  a   b \n", "a/b\\c"]
        tokenization = [{"text": text, "tokens": tokenizer.encode(text, add_special_tokens=False).ids} for text in texts]
        prompt = ("<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
                  "<|message_user|><|content_text|>Reply with only the result of 2 + 2.<|end_message|>"
                  "<|message_model|><|content_text|>")
        tokens = tokenizer.encode(prompt, add_special_tokens=False).ids
        logits = model(mx.array([tokens]), cache=model.make_cache()).logits[0, -1].astype(mx.float32)
        mx.eval(logits)
        scores = np.array(logits)
        top = np.argsort(-scores)[:20].tolist()
        result = {"reference_revision": REFERENCE_REVISION, "mlx_version": importlib.metadata.version("mlx"), "tokens": tokens,
                  "text_top_ids": top, "text_top_logits": scores[top].tolist(), "tokenization": tokenization}
        print("Text reference argmax:", top[0], tokenizer.decode([top[0]]), flush=True)
        vision = VisionModel(VisionConfig.from_dict(config["vision_config"]))
        vision.load_weights([(k.removeprefix("vision_tower."), v) for k, v in weights.items()
                             if k.startswith("vision_tower.")], strict=True)
        # Rectangular input exercises learned-grid interpolation and patch order.
        rgb = np.arange(80 * 40 * 3, dtype=np.int32).astype(np.uint8).reshape(40, 80, 3)
        pixels = (rgb.astype(np.float32) / 127.5 - 1).reshape(1, 40, 2, 40, 3).transpose(0, 2, 1, 3, 4).reshape(2, 40, 40, 3)
        image = vision(mx.array(pixels), image_grid_thw=mx.array([[1, 1, 2]]))
        mx.eval(image)
        result["vision_embeddings"] = np.array(image.astype(mx.float32)).reshape(-1).tolist()
        result["vision_shape"] = list(image.shape)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("Saved text, tokenizer, and vision reference:", args.output)


if __name__ == "__main__":
    main()
