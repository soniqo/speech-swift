---
name: benchmark
description: Run performance + quality benchmarks. ASR reports WER, RTF, process memory, and throughput across engines/variants. Arguments include asr, tts, vad, diarize, asr-quick.
disable-model-invocation: false
argument-hint: [asr|asr-quick|tts|vad|diarize] [extra args]
allowed-tools: Bash
---

# Benchmark

Run benchmarks using the release build. Build first with `/build`.

## Usage

- `/benchmark asr` — full WER + RTF + peakRSS + throughput on a labeled
  dataset (LibriSpeech-style dir or `.tsv` manifest). Extra args pass
  through to `asr-bench`. **Requires** `BENCH_DATASET` env var or
  `--dataset <path>` in the trailing args.
- `/benchmark asr-quick` — single-file RTF smoke test (no WER, no dataset
  required). Equivalent to the old `asr` behavior.
- `/benchmark tts` — synthesize test text, report RTF
- `/benchmark vad` — VAD on VoxConverse (all engines)
- `/benchmark diarize` — DER on VoxConverse (requires downloaded test set)

### Examples

Compare our MLX quantizations on LibriSpeech test-clean (WER + RTF + peakRSS,
each engine isolated in its own process so RSS reflects per-engine cost):

```bash
BENCH_DATASET=$HOME/datasets/LibriSpeech/test-clean /benchmark asr \
  --engines qwen3-mlx-0.6b-4bit qwen3-mlx-0.6b-8bit \
  --isolated --limit 50
```

Default engine set (qwen3-coreml + parakeet + whisperkit) on a TSV manifest:

```bash
/benchmark asr --dataset bench.tsv --limit 100 --output /tmp/run.json
```

```bash
module="$1"
shift || true
cli=".build/release/speech"
bench=".build/release/asr-bench"

case "$module" in
  asr)
    if [ ! -x "$bench" ]; then
      echo "asr-bench binary missing — run /build first (release)." >&2
      exit 1
    fi
    # Honor BENCH_DATASET if --dataset isn't already in the trailing args.
    has_dataset=0
    for a in "$@"; do
      if [ "$a" = "--dataset" ]; then has_dataset=1; break; fi
    done
    if [ "$has_dataset" = "0" ] && [ -n "$BENCH_DATASET" ]; then
      set -- --dataset "$BENCH_DATASET" "$@"
    fi
    "$bench" "$@" 2>&1
    ;;
  asr-quick)
    $cli transcribe Tests/Qwen3ASRTests/Resources/test_audio.wav 2>&1
    ;;
  tts)
    $cli speak "The quick brown fox jumps over the lazy dog." --output /tmp/bench_tts.wav 2>&1
    ;;
  vad)
    python3 scripts/benchmark_vad.py --compare --num-files 5 2>&1
    ;;
  diarize)
    python3 scripts/benchmark_diarization.py --num-files 5 2>&1
    ;;
  *)
    echo "Usage: /benchmark [asr|asr-quick|tts|vad|diarize] [args...]"
    echo "  asr       — full WER + RTF + peakRSS via asr-bench (needs dataset)"
    echo "  asr-quick — single-file RTF smoke test (no dataset)"
    ;;
esac
```

## What `/benchmark asr` reports

Per engine, in the printed table and the JSON output:

| Metric | Source |
|--------|--------|
| `WER%` | substitutions + insertions + deletions over normalized reference words (`AsrBenchmark/WER.swift`) |
| `RTF` | mean transcribe-elapsed / audio-duration per utterance |
| `xRT` | throughput = 1 / RTF |
| `peakRSS` | high-water resident-set size via `mach_task_basic_info` (historical compatibility metric) |
| `RSSΔ` | RSS gained from pre-load to peak |
| `Phys` | high-water physical footprint via `TASK_VM_INFO`; use this for unified-memory sizing |
| `PhysΔ` | physical footprint gained from pre-load to peak (engine cost vs. baseline) |
| `loadSec` | model load + warmup wall time |

Use `--isolated` to run each engine in a child process. RSS and physical-
footprint high-water marks then reflect one engine instead of the cumulative
state of a sequential multi-engine run.

## Available engines

`qwen3-coreml`, `qwen3-mlx-{0.6b,1.7b}-{4bit,8bit}`, `parakeet`,
`nemotron`, `nemotron-mlx-{int5,int8}`, `omnilingual`,
`omnilingual-mlx-{300m,1b,3b,7b}-4bit`,
`whisperkit-{large-v3-turbo,large-v3,distil-large-v3}`.

## Performance targets (M2 Max)

| Module | Metric | Target |
|--------|--------|--------|
| ASR (Qwen3 MLX) | RTF | ~0.06 |
| ASR (Parakeet) | RTF | ~0.025 |
| TTS | RTF | ~0.7 |
| VAD (Silero) | RTF | >20x real-time |
| Diarization | DER | <10% (VoxConverse) |
