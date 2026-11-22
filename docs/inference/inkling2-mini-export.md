# Export Inkling 2 Mini to MLX

`scripts/export_inkling_mlx.py` converts the pinned upstream Inkling 2 Mini
checkpoint to sharded affine MLX safetensors for `Inkling2Chat` and MLX-VLM.
See [native Swift and CLI usage](inkling2-mini.md).

Use Python 3.10 or later on Apple Silicon, with a separate environment:

```bash
python3 -m venv /tmp/inkling-export
source /tmp/inkling-export/bin/activate
python -m pip install 'mlx>=0.29' numpy huggingface_hub

python scripts/export_inkling_mlx.py \
  --stream-source --bits 4 \
  --output /path/to/Inkling-2-Mini-MLX-4bit
```

The default source revision is immutable. `--stream-source` downloads metadata
and tokenizer assets through `huggingface_hub`, then reads exact remote tensor
ranges without retaining the complete 19.5 GB source shard. At most four bounded
network requests run at once; quantization remains sequential. Without that
flag, the complete original checkpoint is cached before conversion. The script
does not upload or publish files.

Use an existing complete snapshot to avoid downloading again:

```bash
python scripts/export_inkling_mlx.py \
  --source-dir /path/to/Inkling-2-Mini-snapshot \
  --bits 5 --group-size 64 --max-shard-size-mb 512 \
  --output /path/to/Inkling-2-Mini-MLX-5bit
```

To create INT5 directly from the same pinned remote source without caching its
full shard:

```bash
python scripts/export_inkling_mlx.py \
  --stream-source --bits 5 --group-size 64 \
  --output /path/to/Inkling-2-Mini-MLX-5bit
```

INT4 and INT5 exports decode the original source expert tensors independently.
Changing precision does not require or reuse a lower-bit export. The Swift
runtime reads the expert precision from each bundle's configuration; the CLI
selects a bundle with `--model-dir`.

`--bits` accepts 4, 5, or 8 for experts; attention and vocabulary matrices
always use INT8. Vision and sensitive routing, normalization, relative-position,
and loop parameters retain their source precision. `--group-size` accepts
32, 64, or 128. `--include-mtp` copies the upstream BF16 `mtp.safetensors`
draft unchanged; without it, the output omits both the draft and `mtp_config`.

The exporter memory-maps the source, dequantizes one expert at a time, and
writes output shards incrementally. It runs MLX conversion on the CPU and
limits the MLX allocation cache to 128 MiB. The shard target defaults to 512
MiB; an indivisible tensor may exceed the target. This is not a hard process
memory cap: vocabulary matrices and a complete output expert tensor must
still fit in memory. Run one export or model-loading job at a time. Disk preflight
accounts for packed weights, affine scales/biases, an optional draft, and a 2 GiB reserve.

Outputs must be new or empty directories outside the source snapshot. Files
are assembled in a temporary sibling directory, then renamed into place only
after conversion, hashing, and asset copying complete. Failures leave the
destination untouched. The bundle includes:

- Sharded safetensors and `model.safetensors.index.json`.
- `config.json` with per-module affine quantization settings and original loop settings.
- Original tokenizer, chat template, processor, generation settings, and available Tiktoken assets.
- `source_config.json`, `source_quant_config.json`, `source_README.md`, and Apache 2.0 `LICENSE`.
- `export_config.json` with the source revision label, source/output hashes, precision, and tensor count.

For a local `--source-dir`, the supplied revision is a provenance label; the
exporter cannot attest that arbitrary local files came from that commit.
`local_source_revision_verified` is therefore false. File hashes allow
independent comparison. Streaming records per-tensor source hashes instead of
a locally read full-shard hash. The output follows the MLX-VLM Inkling 2 parameter
layout, and does not contain source activation quantizers. Weight export
alone does not verify text/image generation quality.

CPU-only regression checks require NumPy and do not download weights:

```bash
python -m unittest discover -s scripts/tests -p 'test_export_inkling_mlx.py' -v
```

On Apple Silicon, run the explicit MLX checks separately:

```bash
python scripts/tests/e2e_inkling_export.py
```

These download about 8 MB of byte-range samples from the pinned checkpoint,
compare NVFP4/MXFP8 decoding with native MLX decoding, check INT4/INT5/INT8
round trips and quantized matrix products, verify interleaved gate/up export,
and reload shard files through their index. They do not run full model
inference or validate the quality of a complete converted checkpoint.
