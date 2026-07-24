#!/usr/bin/env python3
"""Export MOSS-Transcribe-Diarize weights to a native MLX bundle.

The audio encoder and VQ adaptor remain FP16. The tied Qwen3 decoder is
quantized with MLX affine group quantization so INT5 and INT8 differ only in
decoder weight precision. Runtime assets are copied from the exact pinned
source revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
from huggingface_hub import snapshot_download


DEFAULT_SOURCE_MODEL = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
DEFAULT_SOURCE_REVISION = "e6d68cdfcddbdad1a7e8454f0cb859cad76e2502"
DEFAULT_REPOSITORIES = {
    5: "aufklarer/MOSS-Transcribe-Diarize-0.9B-MLX-5bit",
    8: "aufklarer/MOSS-Transcribe-Diarize-0.9B-MLX-INT8",
}
WEIGHT_PATTERN = "model-*.safetensors"
RUNTIME_ASSETS = [
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processing_moss_transcribe_diarize.py",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]
GENERATED_FILES = [
    ".gitattributes",
    "LICENSE",
    "README.md",
    "audio_encoder.safetensors",
    "config.json",
    "decoder.safetensors",
    "export_config.json",
    "source_config.json",
    "validation.json",
    *RUNTIME_ASSETS,
]
EXPECTED_AUDIO_TENSORS = 373
EXPECTED_DECODER_SOURCE_TENSORS = 310
EXPECTED_DECODER_MATRICES = 197
VALIDATION_TENSOR = "layers.0.self_attn.q_proj.weight"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the pinned MOSS 0.9B checkpoint to an MLX bundle with "
            "an FP16 audio frontend and affine INT5 or INT8 decoder."
        )
    )
    parser.add_argument(
        "--bits",
        type=int,
        choices=(5, 8),
        required=True,
        help="Decoder weight precision.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination directory for the complete MLX bundle.",
    )
    parser.add_argument(
        "--source-model",
        default=DEFAULT_SOURCE_MODEL,
        help="Hugging Face source repository.",
    )
    parser.add_argument(
        "--source-revision",
        default=DEFAULT_SOURCE_REVISION,
        help="Immutable source commit.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Use an already downloaded source snapshot instead of downloading.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional huggingface_hub cache directory.",
    )
    parser.add_argument(
        "--repo-id",
        help="Repository ID recorded in metadata and the generated model card.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        choices=(32, 64, 128),
        help="MLX affine quantization group size (default: 64).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace known generated files in an existing output directory.",
    )
    return parser.parse_args()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_output(directory: Path, overwrite: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    existing = [
        directory / name
        for name in GENERATED_FILES
        if (directory / name).exists()
    ]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing[:5])
        raise RuntimeError(
            f"{directory} already contains generated files ({names}); "
            "pass --overwrite to replace this bundle"
        )
    if overwrite:
        for path in existing:
            if path.is_file() or path.is_symlink():
                path.unlink()


def resolve_source(args: argparse.Namespace) -> Path:
    if args.source_dir:
        source = args.source_dir.expanduser().resolve()
    else:
        allow_patterns = [
            WEIGHT_PATTERN,
            "config.json",
            "model.safetensors.index.json",
            *RUNTIME_ASSETS,
        ]
        source = Path(
            snapshot_download(
                repo_id=args.source_model,
                revision=args.source_revision,
                cache_dir=(
                    str(args.cache_dir.expanduser())
                    if args.cache_dir
                    else None
                ),
                allow_patterns=allow_patterns,
            )
        )
    required = [
        source / "config.json",
        source / "model.safetensors.index.json",
        *[source / name for name in RUNTIME_ASSETS],
    ]
    missing = [path.name for path in required if not path.is_file()]
    shards = sorted(source.glob(WEIGHT_PATTERN))
    if not shards:
        missing.append(WEIGHT_PATTERN)
    if missing:
        raise RuntimeError(
            f"source snapshot is incomplete; missing: {', '.join(missing)}"
        )
    return source


def validate_source_config(config: dict[str, Any]) -> None:
    audio = config["audio_config"]
    text = config["text_config"]
    expected = {
        "audio d_model": (audio["d_model"], 1024),
        "audio layers": (audio["encoder_layers"], 24),
        "audio heads": (audio["encoder_attention_heads"], 16),
        "audio mel bins": (audio["num_mel_bins"], 80),
        "audio positions": (audio["max_source_positions"], 1500),
        "merge size": (config["audio_merge_size"], 4),
        "audio token": (config["audio_token_id"], 151671),
        "decoder hidden": (text["hidden_size"], 1024),
        "decoder layers": (text["num_hidden_layers"], 28),
        "decoder heads": (text["num_attention_heads"], 16),
        "decoder KV heads": (text["num_key_value_heads"], 8),
        "decoder head dim": (text["head_dim"], 128),
        "decoder vocabulary": (text["vocab_size"], 151936),
        "decoder context": (text["max_position_embeddings"], 131072),
    }
    mismatches = [
        f"{name}={actual} (expected {wanted})"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise RuntimeError(
            "source architecture does not match the Swift runtime: "
            + "; ".join(mismatches)
        )


def load_source_weights(source: Path) -> dict[str, mx.array]:
    weights: dict[str, mx.array] = {}
    for shard in sorted(source.glob(WEIGHT_PATTERN)):
        loaded = mx.load(str(shard))
        if not isinstance(loaded, dict):
            raise RuntimeError(f"{shard} did not contain named tensors")
        duplicate = set(weights).intersection(loaded)
        if duplicate:
            raise RuntimeError(
                f"duplicate tensor across source shards: {sorted(duplicate)[0]}"
            )
        weights.update(loaded)
    return weights


def export_audio(
    source_weights: dict[str, mx.array],
    destination: Path,
    metadata: dict[str, str],
) -> tuple[int, int]:
    output: dict[str, mx.array] = {}
    parameter_count = 0
    prefixes = ("model.whisper_encoder.", "model.vq_adaptor.")
    for source_name, value in source_weights.items():
        if not source_name.startswith(prefixes):
            continue
        name = source_name.removeprefix("model.")
        array = value.astype(mx.float16)
        if name in {
            "whisper_encoder.conv1.weight",
            "whisper_encoder.conv2.weight",
        }:
            # PyTorch Conv1d [out, in, kernel] -> MLX [out, kernel, in].
            array = array.transpose(0, 2, 1)
        mx.eval(array)
        output[name] = array
        parameter_count += value.size
    if len(output) != EXPECTED_AUDIO_TENSORS:
        raise RuntimeError(
            f"expected {EXPECTED_AUDIO_TENSORS} audio tensors, got {len(output)}"
        )
    mx.save_safetensors(str(destination), output, metadata=metadata)
    return len(output), parameter_count


def export_decoder(
    source_weights: dict[str, mx.array],
    destination: Path,
    bits: int,
    group_size: int,
    metadata: dict[str, str],
) -> tuple[int, int, dict[str, float]]:
    decoder_source = {
        name.removeprefix("model.language_model."): value
        for name, value in source_weights.items()
        if name.startswith("model.language_model.")
    }
    if len(decoder_source) != EXPECTED_DECODER_SOURCE_TENSORS:
        raise RuntimeError(
            "expected "
            f"{EXPECTED_DECODER_SOURCE_TENSORS} decoder source tensors, "
            f"got {len(decoder_source)}"
        )

    output: dict[str, mx.array] = {}
    matrix_count = 0
    parameter_count = 0
    validation: dict[str, float] | None = None
    for name, value in decoder_source.items():
        parameter_count += value.size
        converted = value.astype(mx.float16)
        if name.endswith(".weight") and converted.ndim >= 2:
            if converted.shape[-1] % group_size != 0:
                raise RuntimeError(
                    f"{name} input dimension {converted.shape[-1]} is not "
                    f"divisible by group size {group_size}"
                )
            quantized, scales, biases = mx.quantize(
                converted,
                group_size=group_size,
                bits=bits,
                mode="affine",
            )
            mx.eval(quantized, scales, biases)
            base = name.removesuffix(".weight")
            output[name] = quantized
            output[f"{base}.scales"] = scales
            output[f"{base}.biases"] = biases
            matrix_count += 1
            if name == VALIDATION_TENSOR:
                restored = mx.dequantize(
                    quantized,
                    scales,
                    biases,
                    group_size=group_size,
                    bits=bits,
                    mode="affine",
                    dtype=mx.float16,
                )
                source_flat = converted.flatten().astype(mx.float32)
                restored_flat = restored.flatten().astype(mx.float32)
                cosine = mx.sum(source_flat * restored_flat) / (
                    mx.sqrt(mx.sum(source_flat * source_flat))
                    * mx.sqrt(mx.sum(restored_flat * restored_flat))
                )
                maximum_error = mx.max(
                    mx.abs(source_flat - restored_flat)
                )
                mx.eval(cosine, maximum_error)
                validation = {
                    "cosine": float(cosine.item()),
                    "max_abs": float(maximum_error.item()),
                }
        else:
            mx.eval(converted)
            output[name] = converted

    if matrix_count != EXPECTED_DECODER_MATRICES:
        raise RuntimeError(
            f"expected {EXPECTED_DECODER_MATRICES} decoder matrices, "
            f"got {matrix_count}"
        )
    if validation is None:
        raise RuntimeError(f"validation tensor {VALIDATION_TENSOR} is absent")
    mx.save_safetensors(str(destination), output, metadata=metadata)
    return len(output), parameter_count, validation


def make_runtime_config(
    source: dict[str, Any],
    *,
    bits: int,
    group_size: int,
    repo_id: str,
    source_model: str,
    source_revision: str,
) -> dict[str, Any]:
    audio = source["audio_config"]
    text = source["text_config"]
    return {
        "audio_config": {
            "hidden_size": audio["d_model"],
            "intermediate_size": audio["encoder_ffn_dim"],
            "layer_norm_eps": audio.get("layer_norm_eps", 1e-5),
            "max_source_positions": audio["max_source_positions"],
            "merge_size": source["audio_merge_size"],
            "num_heads": audio["encoder_attention_heads"],
            "num_layers": audio["encoder_layers"],
            "num_mel_bins": audio["num_mel_bins"],
        },
        "audio_token_id": source["audio_token_id"],
        "backend": "mlx",
        "decoder_config": {
            "head_dim": text["head_dim"],
            "hidden_size": text["hidden_size"],
            "intermediate_size": text["intermediate_size"],
            "num_heads": text["num_attention_heads"],
            "num_kv_heads": text["num_key_value_heads"],
            "num_layers": text["num_hidden_layers"],
            "rms_norm_eps": text["rms_norm_eps"],
            "rope_theta": text["rope_theta"],
            "vocab_size": text["vocab_size"],
        },
        "files": {
            "audio_encoder": "audio_encoder.safetensors",
            "decoder": "decoder.safetensors",
        },
        "max_context_tokens": text["max_position_embeddings"],
        "model_type": "moss-transcribe-diarize-mlx",
        "precision": f"fp16-audio-int{bits}-decoder",
        "quantization_config": {
            "bits": bits,
            "group_size": group_size,
            "mode": "affine",
        },
        "repo_id": repo_id,
        "source_model": source_model,
        "source_revision": source_revision,
    }


def make_model_card(
    *,
    bits: int,
    group_size: int,
    repo_id: str,
    source_model: str,
    source_revision: str,
    audio_parameters: int,
    decoder_parameters: int,
) -> str:
    total_parameters = audio_parameters + decoder_parameters
    return f"""---
license: apache-2.0
library_name: mlx
base_model: {source_model}
pipeline_tag: automatic-speech-recognition
tags:
- mlx
- speech-swift
- asr
- speaker-diarization
- long-form-audio
- multilingual
---

# MOSS-Transcribe-Diarize 0.9B MLX INT{bits}

Native MLX weights for offline, speaker-attributed transcription on Apple
Silicon. The Whisper encoder and VQ adaptor remain FP16; the tied Qwen3
decoder uses {bits}-bit affine group-{group_size} weights.

| Property | Value |
|---|---|
| Source | `{source_model}` |
| Source revision | `{source_revision}` |
| Parameters | {total_parameters:,} |
| Audio/VQ precision | FP16 |
| Decoder precision | INT{bits}, affine group {group_size} |
| Context | 131,072 tokens |
| License | Apache-2.0 |

The model is offline, not streaming. Audio is encoded in 30-second chunks,
then all audio embeddings are concatenated into one globally contextualized
prompt. Transcript tokens are generated autoregressively after the complete
recording is available.

## speech-swift

```bash
speech transcribe meeting.wav \\
  --engine moss \\
  --backend mlx \\
  --model {repo_id} \\
  --kv-cache fp16
```

`--kv-cache int8` reduces long-context memory separately from decoder weight
quantization. Use FP16 cache when comparing INT5 and INT8 model quality.
INT4 KV cache is not exposed because it failed the structured-output quality
gate.

## Export provenance

The bundle is produced by `scripts/export_moss_mlx.py` in speech-swift from
the immutable source revision above. `export_config.json` records tensor
counts, checksums, quantization verification, and the runtime contract.

No training or fine-tuning was performed. Benchmark results are intentionally
not claimed by the exporter; release WER/CER and DER/JER measurements belong
in `validation.json` after running the ASR and diarization benchmarks.

## Limitations

The upstream project reports support for 50+ languages and recordings up to
about 90 minutes, but does not publish an exhaustive language list. Actual
duration is constrained by unified memory, KV-cache precision, transcript
length, and the 131,072-token combined context. Speaker IDs are anonymous per
recording. Timestamps may overlap or be malformed, and quantization can affect
both word accuracy and speaker attribution.

## License

Apache License 2.0, inherited from the upstream model.
"""


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    prepare_output(output, args.overwrite)
    source = resolve_source(args)
    repo_id = args.repo_id or DEFAULT_REPOSITORIES[args.bits]

    source_config = json.loads(
        (source / "config.json").read_text(encoding="utf-8")
    )
    validate_source_config(source_config)
    source_weights = load_source_weights(source)

    print("Exporting FP16 Whisper encoder and VQ adaptor...", flush=True)
    audio_count, audio_parameters = export_audio(
        source_weights,
        output / "audio_encoder.safetensors",
        {
            "format": "mlx",
            "precision": "float16",
        },
    )
    print(
        f"Quantizing Qwen3 decoder to affine INT{args.bits} "
        f"group {args.group_size}...",
        flush=True,
    )
    decoder_count, decoder_parameters, verification = export_decoder(
        source_weights,
        output / "decoder.safetensors",
        args.bits,
        args.group_size,
        {
            "format": "mlx",
            "group_size": str(args.group_size),
            "precision": f"int{args.bits}",
            "quantization": "affine",
        },
    )

    for name in RUNTIME_ASSETS:
        shutil.copy2(source / name, output / name)
    shutil.copy2(source / "config.json", output / "source_config.json")
    package_license = Path(__file__).resolve().parent.parent / "LICENSE"
    if not package_license.is_file():
        raise RuntimeError(f"Apache license file is missing: {package_license}")
    shutil.copy2(package_license, output / "LICENSE")
    (output / ".gitattributes").write_text(
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )

    runtime_config = make_runtime_config(
        source_config,
        bits=args.bits,
        group_size=args.group_size,
        repo_id=repo_id,
        source_model=args.source_model,
        source_revision=args.source_revision,
    )
    json_dump(output / "config.json", runtime_config)
    artifacts = {}
    for name in ("audio_encoder.safetensors", "decoder.safetensors"):
        path = output / name
        artifacts[name] = {
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
    export_config = {
        **runtime_config,
        "artifacts": artifacts,
        "host_contract": {
            "audio_chunk_samples": 480_000,
            "audio_tokens_per_second": 12.5,
            "generation": (
                "Greedy autoregressive decode; parse "
                "[start][Sxx]text[end]"
            ),
            "processor": "Use copied assets from the pinned source revision",
            "sample_rate": 16_000,
        },
        "parameter_counts": {
            "audio_and_vq": audio_parameters,
            "decoder": decoder_parameters,
            "total": audio_parameters + decoder_parameters,
        },
        "runtime_assets": RUNTIME_ASSETS,
        "schema_version": 1,
        "tensor_counts": {
            "audio": audio_count,
            "decoder_exported": decoder_count,
            "decoder_matrices_quantized": EXPECTED_DECODER_MATRICES,
        },
        "verification": {
            **verification,
            "status": "passed",
            "tensor": VALIDATION_TENSOR,
        },
    }
    json_dump(output / "export_config.json", export_config)
    json_dump(
        output / "validation.json",
        {
            "status": "not_run",
            "required_comparisons": {
                "asr": ["WER", "CER", "RTF", "peak RSS"],
                "diarization": [
                    "DER",
                    "JER",
                    "speaker-count accuracy",
                    "RTF",
                    "peak RSS",
                ],
            },
            "quality_policy": (
                "Compare INT5 and INT8 with the same FP16 audio weights, "
                "greedy decoding, prompts, datasets, and FP16 KV cache."
            ),
            "kv_cache_policy": (
                "FP16 is the quality baseline; INT8 requires separate "
                "structured-output validation; INT4 is unsupported."
            ),
        },
    )
    (output / "README.md").write_text(
        make_model_card(
            bits=args.bits,
            group_size=args.group_size,
            repo_id=repo_id,
            source_model=args.source_model,
            source_revision=args.source_revision,
            audio_parameters=audio_parameters,
            decoder_parameters=decoder_parameters,
        ),
        encoding="utf-8",
    )

    print(f"Exported {repo_id} to {output}")
    print(
        "Decoder verification: "
        f"cosine={verification['cosine']:.8f}, "
        f"max_abs={verification['max_abs']:.8f}"
    )
    for name, artifact in artifacts.items():
        size_mb = artifact["size_bytes"] / (1024 * 1024)
        print(f"{name}: {size_mb:.1f} MiB {artifact['sha256']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
