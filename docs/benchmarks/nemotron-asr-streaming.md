# Nemotron-3.5 ASR Streaming — benchmarks

All numbers from M5 Pro / 48 GB. 50 samples per language from FLEURS test split (chunk 320 ms). Methodology matches NVIDIA's published table: `EnglishTextNormalizer` for English, `BasicTextNormalizer` for European/Arabic, `BasicTextNormalizer(split_letters=True)` for CJK + Indic (char-level scoring).

## WER % vs upstream

| lang | NVIDIA dev | fp32 NeMo | CoreML INT8 | MLX bf16 | MLX int8 | MLX int4 |
|------|-:|-:|-:|-:|-:|-:|
| en-US |  8.27 |  9.33 |  9.59 | 10.36 | 10.28 | 15.98 |
| de-DE |  8.83 | 10.22 | 10.41 | 10.87 | 10.59 | 14.96 |
| fr-FR |  9.79 | 11.13 | 12.18 | 11.62 | 11.20 | 15.85 |
| ar    | 12.55 | 13.27 | 13.37 | 13.76 | 13.95 | 20.88 |
| hi-IN |  7.41 |  5.26 |  4.42 |  5.36 |  5.68 |  8.13 |
| ja-JP | 12.22 | 16.97 | 17.66 | 17.33 | 17.98 | 19.56 |

## CER %

| lang | fp32 | CoreML INT8 | MLX bf16 | MLX int8 | MLX int4 |
|------|-:|-:|-:|-:|-:|
| en-US |  3.96 |  4.26 |  4.41 |  4.32 |  8.69 |
| de-DE |  5.32 |  5.37 |  5.10 |  5.06 |  6.43 |
| fr-FR |  4.56 |  4.84 |  4.83 |  4.79 |  5.98 |
| ar    |  3.72 |  3.80 |  3.85 |  3.87 |  5.61 |
| hi-IN |  4.37 |  3.61 |  4.31 |  4.47 |  6.52 |
| ja-JP | 11.27 | 12.09 | 11.50 | 11.97 | 12.89 |

## Streaming throughput + memory (60 s long-form en_us, chunk 320 ms)

| variant | on-disk | RSS post-load | RSS peak | offline RTF | streaming RTF | p50 / p99 chunk ms |
|---------|--------:|--------------:|---------:|-:|-:|-:|
| CoreML INT8 |  612 MB |  1046 MB |  1238 MB | 0.059 | 0.068 | 18.6 / 23.4 |
| MLX bf16    | 1217 MB |   192 MB |  1474 MB | 0.014 | 0.062 | 18.4 / 23.5 |
| MLX int8    |  732 MB |   262 MB |   997 MB | 0.013 | 0.044 | 13.5 / 16.8 |
| MLX int4    |  473 MB |   270 MB |   747 MB | 0.012 | 0.041 | 12.8 / 15.6 |

MLX int4 has the lowest absolute streaming peak (747 MB). CoreML INT8 has the smallest in-streaming Δ (+192 MB above post-load) because the encoder mmap is fully resident after first chunk.

## Round-trip equivalence Δ-CER vs fp32 source

Per-variant Δ-CER averaged over 6 languages, same audio through fp32 NeMo + variant pipelines:

| variant | avg Δ-CER vs fp32 |
|---------|------------------:|
| CoreML INT8 | +0.13 pp |
| MLX bf16    | +0.20 pp |
| MLX int8    | +0.32 pp |
| MLX int4    | +2.32 pp |

CoreML INT8, MLX bf16, and MLX int8 are essentially lossless. MLX int4 trades +2.3 pp avg CER for the smallest disk + streaming footprint.

## Bench setup notes

- All MLX runs used `mx.set_cache_limit` defaults — sweep showed cache_limit changes don't affect peak RSS (peak Δ ≈ weight size for any variant, structural cost of forward through 24-layer Conformer).
- Per-N-layer `mx.eval` sweep showed N=8 gives a small free RTF speedup (-5–8 %) on bf16 with no peak / accuracy effect. N=1 is harmful.
- The Swift bench uses `autoreleasepool` per utterance + per-language model reload to avoid CoreML IOSurface exhaustion (~250+ predicts in one MLModel lifetime triggers segfault).
- All Swift WER numbers match the Python pipeline byte-for-byte on every sample (49/50 on Hindi; the 1 divergence is a real model-level non-determinism in CoreML ANE scheduling, not a wrapper bug).

## Reproducing

```bash
# Python (50 samples per lang)
cd speech-models/models/nemotron-asr-streaming-multilingual/export
poetry run python bench_wer_torch_fp32.py --langs en_us,de_de,fr_fr,ar_eg,hi_in,ja_jp --limit 50
poetry run python bench_wer_coreml.py --bundle /tmp/Nemotron-3.5-CoreML-320ms --langs en_us,de_de,fr_fr,ar_eg,hi_in,ja_jp --limit 50
poetry run python bench_wer_mlx.py --root /tmp/Nemotron-3.5-MLX --variants bf16,int8,int4 --limit 50

# Swift
cd speech-swift-asr-bench  # or your worktree
NEMOTRON_35_LOCAL_BUNDLE=/tmp/Nemotron-3.5-CoreML-320ms \
  swift test --filter E2ENemotronMultilingualBench
```

Results land in `/tmp/nem35-logs/wer_*.json` (Python) and `/tmp/nem35-logs/wer_swift.json` (Swift).
