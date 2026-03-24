#!/usr/bin/env python3
"""Convert Open-Unmix (UMX-HQ) PyTorch weights to MLX safetensors.

Downloads pretrained UMX-HQ models (4 targets: vocals, drums, bass, other)
from Zenodo and converts each to a safetensors file for Swift/MLX loading.

Usage:
    python scripts/convert_open_unmix.py [--output-dir OUTPUT] [--model umxhq|umxl]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file


# UMX-HQ: Zenodo record 3370489
UMXHQ_URLS = {
    "vocals": "https://zenodo.org/records/3370489/files/vocals-b62c91ce.pth",
    "drums": "https://zenodo.org/records/3370489/files/drums-9619578f.pth",
    "bass": "https://zenodo.org/records/3370489/files/bass-8d85a5bd.pth",
    "other": "https://zenodo.org/records/3370489/files/other-b52fbbf7.pth",
}

# UMX-L: Zenodo record 5069601
UMXL_URLS = {
    "vocals": "https://zenodo.org/records/5069601/files/vocals-bccbd9aa.pth",
    "drums": "https://zenodo.org/records/5069601/files/drums-69e0ebd4.pth",
    "bass": "https://zenodo.org/records/5069601/files/bass-2ca1ce51.pth",
    "other": "https://zenodo.org/records/5069601/files/other-c8c5b3e6.pth",
}

SKIP_KEYS = {"stft.window", "transform.0.window", "sample_rate",
             "bn1.num_batches_tracked", "bn2.num_batches_tracked",
             "bn3.num_batches_tracked"}

# Key mapping: PyTorch → MLX module path
def map_key(key: str) -> str:
    """Map PyTorch state_dict key to MLX module path."""
    # LSTM keys: lstm.weight_ih_l0 → lstm.layers.0.forward.weight_ih
    # LSTM reverse: lstm.weight_ih_l0_reverse → lstm.layers.0.backward.weight_ih
    if key.startswith("lstm."):
        parts = key.replace("lstm.", "").split("_")
        # e.g., weight_ih_l0, weight_hh_l2_reverse
        if "reverse" in key:
            layer_idx = key.split("_l")[1].split("_reverse")[0]
            param_name = "_".join(key.replace("lstm.", "").split("_l")[0].split("_"))
            return f"lstm.layers.{layer_idx}.backward.{param_name}"
        else:
            layer_idx = key.split("_l")[1].split("_")[0]
            param_name = key.replace("lstm.", "").split(f"_l{layer_idx}")[0]
            return f"lstm.layers.{layer_idx}.forward.{param_name}"

    # BatchNorm running stats
    if "running_mean" in key or "running_var" in key:
        return key

    return key


def convert_target(pth_path: str, output_path: str):
    """Convert a single target's .pth to .safetensors."""
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=False)

    # Handle nested state_dict
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    tensors = {}
    for key, value in state_dict.items():
        if key in SKIP_KEYS:
            continue
        if not isinstance(value, torch.Tensor):
            continue

        mlx_key = map_key(key)
        arr = value.detach().cpu().numpy().astype(np.float32)
        tensors[mlx_key] = arr
        # print(f"  {key} → {mlx_key}: {arr.shape}")

    save_file(tensors, output_path)
    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"  Saved {output_path} ({len(tensors)} tensors, {size_mb:.1f} MB)")


def download_file(url: str, dest: Path):
    """Download a file if not cached."""
    if dest.exists():
        return
    print(f"  Downloading {dest.name}...")
    import urllib.request
    urllib.request.urlretrieve(url, str(dest))


def main():
    parser = argparse.ArgumentParser(description="Convert Open-Unmix to safetensors")
    parser.add_argument("--output-dir", type=str, default="open-unmix-hq",
                        help="Output directory")
    parser.add_argument("--model", choices=["umxhq", "umxl"], default="umxhq",
                        help="Model variant")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Cache dir for downloads")
    args = parser.parse_args()

    urls = UMXHQ_URLS if args.model == "umxhq" else UMXL_URLS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path.home() / ".cache" / "open-unmix"
    cache_dir.mkdir(parents=True, exist_ok=True)

    targets = ["vocals", "drums", "bass", "other"]

    print(f"Converting {args.model} to safetensors...")
    for target in targets:
        print(f"\n{target}:")
        pth_path = cache_dir / f"{target}.pth"
        download_file(urls[target], pth_path)
        output_path = output_dir / f"{target}.safetensors"
        convert_target(str(pth_path), str(output_path))

    # Save config
    config = {
        "model": args.model,
        "hidden_size": 512 if args.model == "umxhq" else 1024,
        "nb_bins": 2049,
        "max_bin": 1487,
        "nb_channels": 2,
        "sample_rate": 44100,
        "n_fft": 4096,
        "n_hop": 1024,
        "targets": targets,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved config.json")

    print(f"\nDone! Output: {output_dir}/")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size:.1f} MB")


if __name__ == "__main__":
    main()
