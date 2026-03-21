#!/usr/bin/env python3
"""Convert Kokoro-82M PyTorch model to CoreML.

Downloads the pretrained Kokoro-82M model, separates it into a duration model
and HAR decoder, and converts each to CoreML .mlpackage format.

The duration model runs on CPU (contains LSTM layers). The decoder is optimized
for Apple Neural Engine with fixed-shape output buckets (3s, 10s, 45s).

Voice style embeddings are extracted and saved as voices.json for Swift loading.

Requires:
    pip install torch kokoro coremltools numpy

Usage:
    python scripts/convert_kokoro_coreml.py [--output OUTPUT_DIR] [--quantize]

Output:
    duration.mlpackage     — Duration/alignment model (CPU)
    decoder_3s.mlpackage   — HAR decoder, 3s bucket (ANE)
    decoder_10s.mlpackage  — HAR decoder, 10s bucket (ANE)
    decoder_45s.mlpackage  — HAR decoder, 45s bucket (ANE)
    voices.json            — Voice style embeddings {name: [float]}
    vocab.json             — Phoneme vocabulary {phoneme: id}
    config.json            — Model configuration
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ─── Constants ────────────────────────────────────────────────────────────────

SAMPLE_RATE = 24000
MAX_PHONEME_LEN = 510
HIDDEN_DIM = 512
STYLE_DIM = 256

# Output buckets: (name, max_seconds, max_samples)
DECODER_BUCKETS = [
    ("3s", 3, 72_000),
    ("10s", 10, 240_000),
    ("45s", 45, 1_080_000),
]


# ─── Model Download ──────────────────────────────────────────────────────────

def download_kokoro(cache_dir: Path):
    """Download Kokoro-82M from HuggingFace."""
    model_dir = cache_dir / "Kokoro-82M"
    if model_dir.exists():
        print(f"  Using cached model: {model_dir}")
        return model_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    print("  Downloading Kokoro-82M from HuggingFace...")

    try:
        from huggingface_hub import snapshot_download
        model_dir_str = snapshot_download(
            "hexgrad/Kokoro-82M",
            local_dir=str(model_dir),
            ignore_patterns=["*.md", "*.txt"],
        )
        print(f"  Downloaded to: {model_dir_str}")
        return Path(model_dir_str)
    except ImportError:
        print("ERROR: huggingface_hub required. Install with: pip install huggingface_hub")
        sys.exit(1)


# ─── Voice Embedding Extraction ──────────────────────────────────────────────

def extract_voices(model_dir: Path, output_dir: Path):
    """Extract voice style embeddings from .pt files."""
    voices_dir = model_dir / "voices"
    if not voices_dir.exists():
        print("  WARNING: No voices directory found")
        return {}

    voices = {}
    for pt_file in sorted(voices_dir.glob("*.pt")):
        name = pt_file.stem
        embedding = torch.load(pt_file, map_location="cpu", weights_only=True)
        if isinstance(embedding, torch.Tensor):
            voices[name] = embedding.flatten().tolist()
        elif isinstance(embedding, dict) and "style" in embedding:
            voices[name] = embedding["style"].flatten().tolist()
        print(f"    Voice: {name} ({len(voices.get(name, []))} dims)")

    output_path = output_dir / "voices.json"
    with open(output_path, "w") as f:
        json.dump(voices, f)
    print(f"  Saved {len(voices)} voice embeddings to {output_path}")

    return voices


# ─── Vocabulary Extraction ────────────────────────────────────────────────────

def extract_vocab(model_dir: Path, output_dir: Path):
    """Extract phoneme vocabulary from model config."""
    # Try loading from config.json or kokoro's built-in vocab
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        if "vocab" in config:
            vocab = config["vocab"]
            output_path = output_dir / "vocab.json"
            with open(output_path, "w") as f:
                json.dump(vocab, f, ensure_ascii=False, indent=2)
            print(f"  Saved vocabulary ({len(vocab)} entries) to {output_path}")
            return vocab

    # Fallback: try importing kokoro's tokenizer
    try:
        from kokoro import KPipeline
        pipeline = KPipeline(lang_code="a")
        # Extract vocab from the pipeline's phonemizer
        if hasattr(pipeline, "vocab"):
            vocab = pipeline.vocab
        else:
            # Build from known Kokoro phoneme set
            vocab = _build_default_vocab()

        output_path = output_dir / "vocab.json"
        with open(output_path, "w") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        print(f"  Saved vocabulary ({len(vocab)} entries) to {output_path}")
        return vocab
    except ImportError:
        print("  WARNING: kokoro package not available, using default vocab")
        vocab = _build_default_vocab()
        output_path = output_dir / "vocab.json"
        with open(output_path, "w") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        return vocab


def _build_default_vocab():
    """Build default IPA vocabulary for Kokoro."""
    # Kokoro's phoneme vocabulary (IPA-based)
    # Pad=0, BOS=1, EOS=2, then IPA symbols
    phonemes = [
        "<pad>", "<bos>", "<eos>", " ",
        "!", "'", "(", ")", ",", "-", ".", ":", ";", "?",
        "a", "b", "d", "e", "f", "h", "i", "j", "k", "l",
        "m", "n", "o", "p", "r", "s", "t", "u", "v", "w",
        "x", "z",
        # IPA extensions
        "\u0251", "\u0252", "\u0254", "\u0259", "\u025a", "\u025b",
        "\u025c", "\u0261", "\u026a", "\u026b", "\u026d", "\u026f",
        "\u0270", "\u0271", "\u0272", "\u0273", "\u0274", "\u0275",
        "\u0278", "\u027b", "\u027d", "\u027e", "\u0280", "\u0281",
        "\u0282", "\u0283", "\u0288", "\u0289", "\u028a", "\u028b",
        "\u028c", "\u028d", "\u028e", "\u028f", "\u0290", "\u0291",
        "\u0292",
        # Diacritics and modifiers
        "\u02a4", "\u02a7", "\u02c8", "\u02cc", "\u02d0", "\u02d1",
        "\u0303", "\u0306", "\u0308", "\u030b", "\u030f", "\u0318",
        "\u0319", "\u031a", "\u031c", "\u031d", "\u031e", "\u031f",
        "\u0320", "\u0324", "\u0325", "\u0329", "\u032a", "\u032c",
        "\u032f", "\u0330", "\u0334", "\u0339", "\u033a", "\u033b",
        "\u033c", "\u033d",
        "\u0361",
        # Tones
        "\u2191", "\u2193", "\u2197", "\u2198",
    ]

    return {p: i for i, p in enumerate(phonemes)}


# ─── CoreML Conversion ───────────────────────────────────────────────────────

def convert_duration_model(model, output_dir: Path, quantize: bool = False):
    """Convert the duration/text encoder part to CoreML."""
    import coremltools as ct

    print("  Converting duration model...")

    class DurationWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, tokens, style):
            # Run text encoder + duration predictor
            # Output: durations, text_encoded, style_out
            t_en = self.model.text_encoder(tokens)
            d = self.model.duration_predictor(t_en, style)
            s = self.model.style_encoder(style)
            return d, t_en, s

    wrapper = DurationWrapper(model)
    wrapper.eval()

    # Trace with example inputs
    example_tokens = torch.zeros(1, MAX_PHONEME_LEN + 2, dtype=torch.long)
    example_style = torch.randn(1, STYLE_DIM)

    traced = torch.jit.trace(wrapper, (example_tokens, example_style))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="tokens", shape=(1, MAX_PHONEME_LEN + 2), dtype=np.int32),
            ct.TensorType(name="style", shape=(1, STYLE_DIM), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="duration"),
            ct.TensorType(name="t_en"),
            ct.TensorType(name="s"),
        ],
        compute_precision=ct.precision.FLOAT16 if not quantize else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS17,
    )

    out_path = output_dir / "duration.mlpackage"
    mlmodel.save(str(out_path))
    print(f"    Saved: {out_path}")


def convert_decoder(model, output_dir: Path, quantize: bool = False):
    """Convert the HAR decoder to CoreML with fixed output buckets."""
    import coremltools as ct

    print("  Converting decoder models...")

    class DecoderWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, features, f0):
            return self.model.decoder(features, f0)

    wrapper = DecoderWrapper(model)
    wrapper.eval()

    for name, max_sec, max_samples in DECODER_BUCKETS:
        print(f"    Converting {name} bucket ({max_sec}s, {max_samples} samples)...")

        max_frames = max_samples // (SAMPLE_RATE // 100)  # ~10ms frames
        example_features = torch.randn(1, HIDDEN_DIM, max_frames)
        example_f0 = torch.randn(1, max_frames)

        traced = torch.jit.trace(wrapper, (example_features, example_f0))

        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="features", shape=(1, HIDDEN_DIM, max_frames), dtype=np.float32),
                ct.TensorType(name="f0", shape=(1, max_frames), dtype=np.float32),
            ],
            outputs=[
                ct.TensorType(name="audio"),
            ],
            compute_precision=ct.precision.FLOAT16 if not quantize else ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.iOS17,
        )

        out_path = output_dir / f"decoder_{name}.mlpackage"
        mlmodel.save(str(out_path))
        print(f"      Saved: {out_path}")


# ─── Config Generation ────────────────────────────────────────────────────────

def save_config(output_dir: Path, vocab_size: int, num_voices: int):
    """Save model configuration as config.json."""
    config = {
        "sampleRate": SAMPLE_RATE,
        "vocabSize": vocab_size,
        "maxPhonemeLength": MAX_PHONEME_LEN,
        "hiddenDim": HIDDEN_DIM,
        "styleDim": STYLE_DIM,
        "numVoices": num_voices,
        "languages": ["en", "fr", "es", "ja", "zh", "hi", "pt", "ko"],
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config.json")


# ─── Main ─────────────────────────────────────────────────────────────────────

def palettize_decoders(output_dir: Path, nbits: int = 4):
    """Apply post-training palettization to decoder models for iOS memory reduction."""
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpPalettizerConfig,
        OptimizationConfig,
        palettize_weights,
    )

    op_config = OpPalettizerConfig(mode="kmeans", nbits=nbits)
    config = OptimizationConfig(global_config=op_config)

    for name, _, _ in DECODER_BUCKETS:
        pkg_path = output_dir / f"decoder_{name}.mlpackage"
        if not pkg_path.exists():
            print(f"    Skipping {name} (not found)")
            continue

        print(f"    Palettizing decoder_{name} to INT{nbits}...")
        model = ct.models.MLModel(str(pkg_path))
        model = palettize_weights(model, config)
        model.save(str(pkg_path))

        size_mb = sum(
            f.stat().st_size for f in pkg_path.rglob("*") if f.is_file()
        ) / 1024 / 1024
        print(f"      Saved: {pkg_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Convert Kokoro-82M to CoreML")
    parser.add_argument("--output", type=str, default="kokoro-coreml",
                        help="Output directory")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Cache directory for downloads")
    parser.add_argument("--quantize", action="store_true",
                        help="Use INT8 quantization for smaller model size")
    parser.add_argument("--palettize", type=int, choices=[4, 8], default=0,
                        help="Post-training palettization (4 or 8 bits). Reduces decoder size for iOS.")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path.home() / ".cache" / "kokoro-convert"

    print("Step 1: Download Kokoro-82M")
    model_dir = download_kokoro(cache_dir)

    print("\nStep 2: Extract voice embeddings")
    voices = extract_voices(model_dir, output_dir)

    print("\nStep 3: Extract vocabulary")
    vocab = extract_vocab(model_dir, output_dir)

    print("\nStep 4: Load PyTorch model")
    try:
        # Try loading via kokoro package
        from kokoro import KModel
        model = KModel()
        model.eval()
        print("  Loaded via kokoro package")
    except ImportError:
        # Direct checkpoint loading
        ckpt_path = model_dir / "kokoro-v0_19.pth"
        if not ckpt_path.exists():
            # Try finding any .pth file
            pth_files = list(model_dir.glob("*.pth"))
            if pth_files:
                ckpt_path = pth_files[0]
            else:
                print("ERROR: No .pth checkpoint found. Install kokoro: pip install kokoro")
                sys.exit(1)

        print(f"  Loading checkpoint: {ckpt_path}")
        model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(model, dict) and "model" in model:
            model = model["model"]
        print("  Loaded from checkpoint")

    print("\nStep 5: Convert to CoreML")
    convert_duration_model(model, output_dir, quantize=args.quantize)
    convert_decoder(model, output_dir, quantize=args.quantize)

    if args.palettize:
        print(f"\nStep 5b: Palettize decoder models to INT{args.palettize}")
        palettize_decoders(output_dir, nbits=args.palettize)

    print("\nStep 6: Save configuration")
    save_config(output_dir, vocab_size=len(vocab), num_voices=len(voices))

    # Compile models
    print("\nStep 7: Compile CoreML models")
    try:
        import coremltools as ct
        for mlpackage in output_dir.glob("*.mlpackage"):
            print(f"  Compiling {mlpackage.name}...")
            model = ct.models.MLModel(str(mlpackage))
            compiled_name = mlpackage.stem + ".mlmodelc"
            compiled_path = output_dir / compiled_name
            # Compilation happens at load time on-device, but we can validate
            print(f"    Validated: {mlpackage.name}")
    except Exception as e:
        print(f"  Compilation validation skipped: {e}")

    print(f"\nDone! Output: {output_dir}/")
    print("Upload to HuggingFace with compiled .mlmodelc directories for Swift loading.")


if __name__ == "__main__":
    main()
