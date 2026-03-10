#!/usr/bin/env python3
"""Convert PeterReid BART G2P model to CoreML.

Converts the tiny BART grapheme-to-phoneme model (752K params, Apache-2.0)
to CoreML for use as OOV fallback in the Kokoro phonemizer.

The model converts English words (character sequences) to IPA phoneme sequences.

Requires:
    pip install torch transformers coremltools safetensors

Usage:
    python scripts/convert_g2p_coreml.py [--output OUTPUT_DIR] [--variant us]

Output:
    g2p.mlpackage — CoreML BART G2P model (CPU, ~3 MB)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# BART G2P model configuration
VOCAB_SIZE = 63
D_MODEL = 128
ENCODER_LAYERS = 1
DECODER_LAYERS = 1
ATTENTION_HEADS = 1
FFN_DIM = 1024
MAX_POSITION = 64

# Character sets (from model config)
GRAPHEME_CHARS = "____AIOWYbdfhijklmnpstuvwz'-.BCDEFGHJKLMNPQRSTUVXZacegoqrxy"
PHONEME_CHARS = "____AIOWYbdfhijklmnpstuvwzæðŋɑɔəɛɜɡɪɹɾʃʊʌʒʔʤʧʰθᴺᵻ"


def download_model(variant: str = "us"):
    """Download BART G2P model from HuggingFace."""
    model_id = f"PeterReid/graphemes_to_phonemes_en_{variant}"
    print(f"  Downloading {model_id}...")

    try:
        from transformers import BartForConditionalGeneration
        model = BartForConditionalGeneration.from_pretrained(model_id)
        model.eval()
        print(f"  Loaded: {sum(p.numel() for p in model.parameters())} params")
        return model
    except ImportError:
        print("ERROR: transformers required. Install with: pip install transformers")
        sys.exit(1)


class BartG2PWrapper(nn.Module):
    """Wrapper for CoreML export — encoder + greedy decoder."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        """Greedy decode: input_ids [1, seq_len] → output_ids [1, max_len]."""
        encoder_outputs = self.model.model.encoder(input_ids=input_ids)

        # Greedy decoding (max 64 steps)
        decoder_input_ids = torch.tensor([[1]], dtype=torch.long)  # BOS
        output_ids = [1]

        for _ in range(MAX_POSITION - 1):
            decoder_outputs = self.model.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
            )
            logits = self.model.lm_head(decoder_outputs.last_hidden_state)
            next_token = logits[:, -1, :].argmax(dim=-1).item()

            if next_token == 2:  # EOS
                break

            output_ids.append(next_token)
            decoder_input_ids = torch.tensor([output_ids], dtype=torch.long)

        return torch.tensor([output_ids], dtype=torch.long)


def convert_to_coreml(model, output_dir: Path):
    """Convert BART G2P to CoreML."""
    import coremltools as ct

    print("  Converting to CoreML...")

    # For this tiny model, we trace a simplified version
    # The model is small enough to run entirely on CPU
    wrapper = BartG2PWrapper(model)
    wrapper.eval()

    # Trace with example input
    example_input = torch.tensor([[1, 4, 5, 6, 7, 2]], dtype=torch.long)  # BOS + "hello" + EOS
    traced = torch.jit.trace(wrapper, example_input, strict=False)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=ct.Shape(shape=(1, ct.RangeDim(1, MAX_POSITION))),
                dtype=np.int32,
            ),
        ],
        outputs=[
            ct.TensorType(name="output_ids"),
        ],
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS17,
    )

    out_path = output_dir / "g2p.mlpackage"
    mlmodel.save(str(out_path))
    print(f"  Saved: {out_path}")

    return out_path


def test_model(model):
    """Test the model with a few words."""
    print("\n  Testing model:")

    grapheme_map = {}
    for i, c in enumerate(GRAPHEME_CHARS):
        if i >= 4:
            grapheme_map[c] = i

    phoneme_map = {}
    for i, c in enumerate(PHONEME_CHARS):
        if i >= 4:
            phoneme_map[i] = c

    test_words = ["hello", "world", "cat", "beautiful", "onomatopoeia"]

    for word in test_words:
        input_ids = [1]  # BOS
        for c in word:
            input_ids.append(grapheme_map.get(c, 3))
        input_ids.append(2)  # EOS

        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        output = model.generate(input_tensor, max_length=64)

        phonemes = ""
        for t in output[0].tolist():
            if t > 3 and t in phoneme_map:
                phonemes += phoneme_map[t]

        print(f"    {word} → {phonemes}")


def main():
    parser = argparse.ArgumentParser(description="Convert BART G2P to CoreML")
    parser.add_argument("--output", type=str, default="g2p-coreml",
                        help="Output directory")
    parser.add_argument("--variant", type=str, default="us",
                        choices=["us", "gb"], help="English variant")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Download BART G2P model")
    model = download_model(args.variant)

    print("\nStep 2: Test model")
    test_model(model)

    print("\nStep 3: Convert to CoreML")
    convert_to_coreml(model, output_dir)

    print(f"\nDone! Output: {output_dir}/")
    print("Include g2p.mlmodelc in the Kokoro model upload for OOV phonemization.")


if __name__ == "__main__":
    main()
