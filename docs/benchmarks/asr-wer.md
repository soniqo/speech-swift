# ASR Word Error Rate (WER) Benchmark

## Datasets

- **LibriSpeech test-clean** — 2620 utterances, ~5.4 hours, English read speech (standard ASR benchmark)
- **FLEURS** — multilingual (10 languages), ~400-900 utterances per language, freely downloadable

## Cross-engine isolated benchmark — M5 Pro, n=200

The `asr-bench` tool runs each engine in a separate child process (`--isolated`) so the peak RSS column is the **real per-engine** allocation rather than a cumulative high-water mark across a sequential run. WER computed via a Whisper-style normalizer + Levenshtein on whitespace tokens.

**Machine**: Apple M5 Pro, 48 GB, macOS 25.5, release build with compiled metallib. LibriSpeech test-clean, first 200 utterances (~30 min audio).

| Engine | Backend | Quant | WER% | RTF | xRT | Peak RSS | Cold load |
|---|---|---|---|---|---|---|---|
| Qwen3-ASR 1.7B | MLX (GPU) | 8-bit | **1.52** | 0.033 | 30.5x | 2 706 MB | 2.5s |
| WhisperKit Large-v3 Turbo | CoreML (ANE) | FP16 | 1.71 | 0.084 | 11.9x | 428 MB | 5.9s |
| Qwen3-ASR 0.6B | MLX (GPU) | 8-bit | 1.82 | 0.015 | 66.0x | 1 306 MB | 2.2s |
| Qwen3-ASR 0.6B | MLX (GPU) | 4-bit | 2.20 | 0.012 | 85.6x | 1 022 MB | 2.1s |
| Parakeet TDT 0.6B v3 | CoreML (ANE) | INT8 | 2.37 | 0.009 | **117.4x** | 897 MB | 3.1s |
| Qwen3-ASR 0.6B | CoreML (ANE) | INT8 | 3.02 | 0.098 | 10.2x | 1 379 MB | 7.3s |
| Omnilingual CTC 300M | MLX (GPU) | 4-bit | 4.26 | 0.005 | **222.1x** | **384 MB** | 1.6s |
| Omnilingual CTC 300M | CoreML (ANE) | INT8 | 5.67 | 0.128 | 7.8x | 543 MB | 2.6s |
| Nemotron-Speech-Streaming | CoreML (ANE) | INT8 | 2.82 | 0.058 | 17.1x | 961 MB | 7.3s |

**Headline picks:**
- **Best WER**: Qwen3-ASR MLX 1.7B 8-bit at 1.52% — beats WhisperKit Large-v3 Turbo (1.71%) and runs 2.6x faster, at a 6x memory cost.
- **Best WER under WhisperKit memory**: WhisperKit Large-v3 Turbo itself (1.71%, 428 MB) and Qwen3-ASR MLX 0.6B 4-bit (2.20%, 1 022 MB) for the next size class.
- **Fastest English**: Parakeet TDT v3 INT8 — 117x real-time at 897 MB, English-only (25 European languages).
- **Multilingual throughput leader**: Omnilingual MLX 300M 4-bit — 222x real-time, 384 MB peak, 1 672 languages, 4.26% WER on English test-clean.
- **Streaming**: Nemotron at 2.82% on whole-utterance batch (post-PR #304 vocab fix that strips `<en-US>`/`<ar-AR>` language tags from decoded output) — competitive with offline engines while retaining 160 ms streaming chunks with 1-chunk right context.

**Reading the memory column**: Sequential-run peak RSS is much higher (4–10 GB) because MLX's Metal cache and CoreML's compiled plans don't release between engines. The `--isolated` mode spawns one child per engine, so each row is the actual per-engine cost.

### Qwen3-ASR CoreML encoder rebuild

The Qwen3-ASR-CoreML row in the table above is the **rebuilt** encoder (chunked block-attention export, [`aufklarer/Qwen3-ASR-CoreML`](https://huggingface.co/aufklarer/Qwen3-ASR-CoreML)). The previous export ran unmasked global self-attention over the zero-padded mel input under `EnumeratedShapes`; padding-derived audio tokens contaminated the real ones via attention, causing the text decoder to emit `<|im_end|>` right after the first sentence-final period — **24.88% WER** on the same n=200 fixture. The rebuilt encoder mirrors upstream's 100-frame chunks + 800-frame attention windows (in-graph block-attention bias from a new `mel_length` input; outputs `(audio_embeddings, output_length)`); encoder time also drops from 113 ms to 24 ms per call.

### WhisperASR native runtime comparison

The `whisper-asr-turbo` engine loads the published `aufklarer/Whisper-Large-v3-Turbo-CoreML` bundle through speech-swift's native `WhisperASRModel` CoreML runtime. It no longer calls WhisperKit for inference. The direct WhisperKit baseline remains in the benchmark harness as an external comparison via `WhisperKitConfig(model: "openai_whisper-large-v3-v20240930_turbo")`.

Current native runtime scope: fixed 30 s CoreML mel input, CoreML encoder, CoreML decoder prefill, explicit KV-cache updates, greedy no-timestamp decoding, byte-level tokenizer decode, and optional language detection. Timestamp decoding, word timestamps, VAD/chunk seeking, and fallback heuristics are not yet parity with WhisperKit.

Quick LibriSpeech test-clean slice on Apple M5 Pro, 48 GB, macOS 26.5.1, debug build, 2026-07-01:

```bash
.build/debug/asr-bench \
  --dataset ~/Library/Caches/qwen3-speech/datasets/LibriSpeech/test-clean \
  --engines whisper-asr-turbo whisperkit-large-v3-turbo \
  --language en \
  --limit 20
```

| Engine | WER% | CER% | Mean RTF | Median RTF | Overall xRT | Load | Peak RSS | RSS Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WhisperASR native | 1.70 | 0.82 | 0.086 | 0.083 | 13.9x | 27.8s | 2794 MB | +2751 MB |
| Direct WhisperKit | 0.73 | 0.27 | 0.091 | 0.091 | 13.6x | 101.3s | 2919 MB | +125 MB |

The native runtime is throughput-competitive on this slice. The remaining WER gap is concentrated in a few decoding-policy/normalization-sensitive utterances (`10` vs `ten`, `ardor` vs `ardour`, `st` vs `saint`, plus one first-word merge). Re-run the full isolated release benchmark before using these quick-slice numbers as release data.

## Comparison with published models

| Model | Params | Size | Precision | WER% (test-clean) | Source |
|-------|--------|------|-----------|-------------------|--------|
| **Qwen3-ASR 1.7B 8-bit** | **1.7B** | **2.3 GB** | **8-bit** | **2.35** | **This benchmark** |
| Whisper Large v3 Turbo | 809M | 1.6 GB | FP16 | 2.5 | OpenAI (2024) |
| **Qwen3-ASR 1.7B 4-bit** | **1.7B** | **1.2 GB** | **4-bit** | **2.57** | **This benchmark** |
| Whisper Large v3 | 1.5B | 3.1 GB | FP16 | 2.7 | OpenAI (2023) |
| **Parakeet TDT 0.6B INT8** | **600M** | **634 MB** | **INT8** | **2.74** | **This benchmark** |
| **Qwen3-ASR 0.6B 8-bit** | **600M** | **960 MB** | **8-bit** | **2.80** | **This benchmark** |
| Whisper Medium | 769M | 1.5 GB | FP16 | 3.0 | OpenAI (2022) |
| **Qwen3-ASR 0.6B 4-bit** | **600M** | **675 MB** | **4-bit** | **3.34** | **This benchmark** |
| Whisper Small | 244M | 483 MB | FP16 | 3.4 | OpenAI (2022) |
| FireRedASR2-AED | 1B | ~2 GB | FP16 | 4.57 | Xiaohongshu (2025) |
| Whisper Base | 74M | 142 MB | FP16 | 5.0 | OpenAI (2022) |

Whisper numbers from original papers (FP16 inference).

## Multilingual results (FLEURS)

CER used for CJK languages (no word boundaries). Parakeet is English-only (25 European languages).

| Language | Metric | Qwen3 4-bit | Qwen3 8-bit | Parakeet INT8 |
|----------|--------|-------------|-------------|---------------|
| Spanish | WER | 6.44 | 5.06 | 5.18 |
| English | WER | 6.57 | 5.64 | 9.30 |
| Chinese | CER | 8.41 | 7.71 | — |
| German | WER | 9.45 | 6.81 | 12.33 |
| French | WER | 11.42 | 8.50 | 13.02 |
| Japanese | CER | 16.11 | 8.64 | — |
| Russian | WER | 16.35 | 10.52 | 11.49 |
| Korean | WER | 19.95 | 6.89 | — |
| Hindi | WER | 25.93 | 18.57 | — |
| Arabic | WER | 33.47 | 20.31 | — |

**Qwen3-ASR 8-bit** consistently outperforms 4-bit across all languages. Largest gains on Korean (19.95% → 6.89%, 65% reduction) and Japanese (16.11% → 8.64%, 46% reduction).

**Qwen3 vs Parakeet**: Qwen3 8-bit is better on all languages except Spanish (5.06% vs 5.18%). Qwen3 supports 52 languages; Parakeet supports ~25 European languages (no CJK).

## Compression delta

How much accuracy do we lose by quantizing to lower bit widths? This establishes the baseline quality cost of our current quantization before trying more advanced techniques like mixed-bit allocation or outlier decomposition.

| Variant | WER% | Substitutions | Insertions | Deletions | Total errors | Size |
|---------|------|---------------|------------|-----------|-------------|------|
| Qwen3 0.6B 8-bit | 2.80 | 1111 | 92 | 268 | 1471 | 960 MB |
| Qwen3 0.6B 4-bit | 3.34 | 1323 | 123 | 308 | 1754 | 675 MB |
| Delta | +0.54 | +212 | +31 | +40 | +283 | -30% |
| Parakeet TDT INT8 | 2.74 | 990 | 125 | 308 | 1423 | 634 MB |

**Qwen3-ASR**: 4-bit adds 0.54% WER (19% more errors) for 30% size reduction.

## Long-Form Stability (Sustained Neural Engine Load)

Tested whether WER or latency degrade under sustained transcription sessions (simulating meeting transcription). 200 LibriSpeech test-clean utterances processed sequentially (~30 min of audio) on M2 Max.

| Metric | First 25% | Last 25% | Overall |
|--------|-----------|----------|---------|
| WER% | 1.30 | 1.23 | 2.43 |
| RTF | 0.672 | 0.400 | 0.539 |

**Key findings:**
- No WER degradation — last quarter is actually slightly better (1.23% vs 1.30%), within noise
- RTF **improves** over the session (0.67 → 0.40) as CoreML warms up its execution plan cache
- No thermal throttling detected on M2 Max after 42 minutes of continuous Neural Engine inference
- Parakeet processes each chunk independently (no cross-chunk state), so quality cannot accumulate errors

RTF includes per-invocation model loading overhead (~3s). Pure inference RTF is ~0.023 (43x real-time).

## Reproduction

The cross-engine table at the top is produced by the `asr-bench` tool that ships in this repo (`Sources/AsrBenchmark/`, executable `asr-bench`). It supports any LibriSpeech-style directory layout (`<speaker>/<chapter>/{*.flac,*.trans.txt}`) or a flat TSV manifest (`<audio_path>\t<reference_text>`).

```bash
# Build (release + metallib)
make build

# Download LibriSpeech test-clean (~350 MB compressed)
mkdir -p ~/Library/Caches/qwen3-speech/datasets
cd ~/Library/Caches/qwen3-speech/datasets
curl -L https://www.openslr.org/resources/12/test-clean.tar.gz | tar -xz

# Cross-engine bench, isolated per-engine peak RSS
.build/release/asr-bench \
  --dataset ~/Library/Caches/qwen3-speech/datasets/LibriSpeech/test-clean \
  --limit 200 \
  --engines qwen3-mlx-1.7b-8bit qwen3-mlx-0.6b-8bit qwen3-mlx-0.6b-4bit \
            qwen3-coreml parakeet omnilingual omnilingual-mlx-300m-4bit \
            nemotron whisperkit-large-v3-turbo \
  --isolated \
  --output report.json
```

Available engine IDs (see `Sources/AsrBenchmark/Engine.swift::EngineID`):

- `qwen3-coreml` — Qwen3-ASR 0.6B CoreML INT8 (full pipeline)
- `qwen3-mlx-{0.6b,1.7b}-{4,8}bit` — Qwen3-ASR MLX variants
- `parakeet` — Parakeet TDT 0.6B v3 CoreML INT8
- `nemotron` — Nemotron Streaming ASR
- `omnilingual` — Omnilingual CTC 300M CoreML INT8
- `omnilingual-mlx-{300m,1b,3b,7b}-4bit` — Omnilingual CTC MLX variants
- `whisper-asr-turbo` — speech-swift native WhisperASR CoreML runtime over `aufklarer/Whisper-Large-v3-Turbo-CoreML`
- `whisperkit-large-v3-turbo` / `whisperkit-large-v3` / `whisperkit-distil-large-v3` — Argmax WhisperKit

Without `--isolated`, peak RSS reflects the sequential high-water mark across the whole run (MLX/CoreML caches don't release between engines). With `--isolated`, each engine runs in a child process and its peak RSS is its own.

### FLEURS (multilingual, separate runner)

```bash
python scripts/benchmark_asr.py --dataset fleurs --language en_us --batch
python scripts/benchmark_asr.py --dataset fleurs --language cmn_hans_cn --batch
python scripts/benchmark_asr.py --dataset fleurs --language de_de --batch
```

(FLEURS reproduction still lives in the legacy Python script; the FLEURS rows in this doc are historical.)
