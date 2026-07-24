# Nemotron-3.5 ASR Streaming — benchmarks

All numbers are from an M5 Pro / 48 GB. Quality uses 50 samples per
language from the FLEURS test split. Methodology matches NVIDIA's published
table: `EnglishTextNormalizer` for English, `BasicTextNormalizer` for
European/Arabic, and `BasicTextNormalizer(split_letters=True)` for CJK +
Indic (character-level scoring).

## Whole-utterance conversion diagnostic

This historical table checks exported weights without advancing the release
streaming caches. Use the next section for the published INT5/INT8 runtime
quality.

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

## Published MLX 320 ms streaming quality

These results use the exact cache-aware runtime shipped with the release
bundles.

| lang | INT5 WER | INT5 CER | INT8 WER | INT8 CER |
|------|---------:|---------:|---------:|---------:|
| en-US | 8.64 | 3.77 | 8.98 | 3.96 |
| de-DE | 11.15 | 6.25 | 10.59 | 5.73 |
| fr-FR | 13.10 | 5.18 | 11.83 | 4.66 |
| ar | 13.66 | 3.94 | 13.37 | 3.77 |
| hi-IN | 4.77 | 3.85 | 4.28 | 3.50 |
| ja-JP | 17.86 | 12.11 | 17.01 | 11.42 |
| **Mean** | **11.53** | **5.85** | **11.01** | **5.51** |

INT5 costs 0.52 percentage points mean WER and 0.34 points mean CER
versus INT8.

## Streaming throughput + memory (63.7 s long-form en-US, 320 ms chunks)

| variant | on-disk | RSS post-load | RSS peak | streaming RTF | p50 / p95 / p99 chunk ms |
|---------|--------:|--------------:|---------:|--------------:|--------------------------:|
| CoreML INT8 | 612 MB | 1046 MB | 1238 MB | 0.068 | 18.6 / — / 23.4 |
| MLX bf16 | 1217 MB | 192 MB | 1474 MB | 0.062 | 18.4 / — / 23.5 |
| **MLX INT5** | **538.6 MB** | **196 MB** | **800 MB** | **0.0467** | **13.8 / 15.9 / 16.8** |
| MLX INT8 | 732.6 MB | 196 MB | 992 MB | 0.0485 | 14.3 / 16.2 / 18.9 |
| MLX int4 | 473 MB | 270 MB | 747 MB | 0.041 | 12.8 / — / 15.6 |

INT5 saves about 193 MB on disk and 192 MB of peak RSS versus INT8 while
remaining slightly faster in this run. INT4 has the lowest absolute peak, but
the native Swift runtime deliberately supports only the quality-gated INT5
and INT8 releases.

## Native Swift regression gate

The release `asr-bench` executable was also run three times with a fresh
isolated process per variant. The fixture is one 20-second English utterance
(11 reference words); model warmup is excluded from RTF. Values below are
three-run medians on the same M5 Pro / 48 GB host.

| variant | WER / CER | RTF | throughput | peak RSS | RSS delta | peak physical footprint |
|---------|----------:|----:|-----------:|---------:|----------:|------------------------:|
| **MLX INT5** | **0 / 0** | **0.0360** | **27.8×** | **611 MB** | **587 MB** | **676 MB** |
| MLX INT8 | 0 / 0 | 0.0375 | 26.7× | 805 MB | 781 MB | 870 MB |

The maximum observed RTF was 0.0362 for INT5 and 0.0379 for INT8. INT5 saved
194 MB of peak process RSS in the native runtime. This short fixture is a
repeatable performance and exact-output smoke gate, not a replacement for the
six-language quality benchmark above. Its workload also differs from the
63.7-second Python benchmark, so absolute Python and Swift RTF values should
not be compared directly.

## Round-trip equivalence Δ-CER vs fp32 source

Per-variant Δ-CER averaged over 6 languages, same audio through fp32 NeMo + variant pipelines:

| variant | avg Δ-CER vs fp32 |
|---------|------------------:|
| CoreML INT8 | +0.13 pp |
| MLX bf16    | +0.20 pp |
| MLX int8    | +0.32 pp |
| MLX int4    | +2.32 pp |

CoreML INT8, MLX bf16, and MLX int8 are essentially lossless in this
whole-utterance diagnostic. MLX int4 trades +2.3 pp average CER for the
smallest disk + streaming footprint.

## Bench setup notes

- INT5 and INT8 release memory figures are fresh-process runs over 199
  chunks; RSS is process memory, not only active MLX allocations.
- All MLX runs used `mx.set_cache_limit` defaults — sweep showed cache-limit
  changes do not affect peak RSS (peak delta is approximately the weight
  size for any variant, a structural cost of forwarding through the
  24-layer Conformer).
- Per-N-layer `mx.eval` sweep showed N=8 gives a small free RTF speedup (-5–8 %) on bf16 with no peak / accuracy effect. N=1 is harmful.
- The Swift bench uses `autoreleasepool` per utterance + per-language model reload to avoid CoreML IOSurface exhaustion (~250+ predicts in one MLModel lifetime triggers segfault).
- All Swift WER numbers match the Python pipeline byte-for-byte on every sample (49/50 on Hindi; the 1 divergence is a real model-level non-determinism in CoreML ANE scheduling, not a wrapper bug).

## Reproducing

```bash
# Python (50 samples per lang)
cd speech-models/models/nemotron-asr-streaming-multilingual/export
poetry run python bench_wer_torch_fp32.py --langs en_us,de_de,fr_fr,ar_eg,hi_in,ja_jp --limit 50
poetry run python bench_wer_coreml.py --bundle /tmp/Nemotron-3.5-CoreML-320ms --langs en_us,de_de,fr_fr,ar_eg,hi_in,ja_jp --limit 50
python bench_wer_mlx.py \
  --root /tmp/Nemotron-3.5-MLX \
  --variants int5,int8 \
  --langs en_us,de_de,fr_fr,ar_eg,hi_in,ja_jp \
  --streaming \
  --limit 50
python bench_stream_all.py \
  --root /tmp/Nemotron-3.5-MLX \
  --runs mlx_int5 \
  --seconds 60
python bench_stream_all.py \
  --root /tmp/Nemotron-3.5-MLX \
  --runs mlx_int8 \
  --seconds 60

# Swift
cd speech-swift
NEMOTRON_35_LOCAL_BUNDLE=/tmp/Nemotron-3.5-CoreML-320ms \
  swift test --filter E2ENemotronMultilingualBench
NEMOTRON_MLX_LOCAL_BUNDLE=/tmp/Nemotron-3.5-MLX/int5 \
NEMOTRON_MLX_INT8_LOCAL_BUNDLE=/tmp/Nemotron-3.5-MLX/int8 \
  swift test --filter E2ENemotronMLXTests

# Native isolated WER / RTF / RSS gate. The TSV contains:
# /absolute/path/to/audio.wav<TAB>reference transcript
make build
NEMOTRON_MLX_LOCAL_BUNDLE=/tmp/Nemotron-3.5-MLX/int5 \
NEMOTRON_MLX_INT8_LOCAL_BUNDLE=/tmp/Nemotron-3.5-MLX/int8 \
  .build/release/asr-bench \
    --dataset /path/to/benchmark.tsv \
    --engines nemotron-mlx-int5 nemotron-mlx-int8 \
    --language en-US \
    --isolated \
    --output /tmp/nemotron-swift-bench.json
```

Results land in `/tmp/nem35-logs/wer_*.json` (Python) and `/tmp/nem35-logs/wer_swift.json` (Swift).
