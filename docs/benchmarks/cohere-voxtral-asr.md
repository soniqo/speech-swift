# Cohere and Voxtral MLX acceptance benchmark

The Cohere Transcribe 2B and Voxtral Mini 3B FP16, INT5, and INT8 variants
passed the cross-precision acceptance gate on 2026-07-22. The benchmark
reports aggregate/mean WER, CER, mean/median RTF, overall and per-utterance
×RT throughput, load time, peak RSS, and macOS physical footprint. RSS is
retained for historical comparisons; physical footprint is the deployment-
relevant unified-memory metric because MLX may memory-map weight files.

## Validated English result

The run used 80 English FLEURS read-speech utterances totaling 759.56 seconds
and 1,813 reference words. It ran on a MacBook Pro (Mac17,9), Apple M5 Pro,
48 GB memory, and macOS 26.5.2. Each variant ran in its own process from a
release build with the release MLX metallib precompiled. All 80 utterances
completed for every variant.

### Cohere Transcribe 2B

| Variant | Bundle | WER | Δ WER | Mean RTF | Overall ×RT | Peak RSS | Physical footprint |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 3.85 GiB | 6.178% | — | 0.0444 | 23.59× | 3,964 MiB | 6,724 MiB |
| INT5 | 1.62 GiB | 6.288% | +0.110 pp | 0.0150 | 69.64× | 1,776 MiB | 2,582 MiB |
| INT8 | 2.25 GiB | 6.178% | +0.000 pp | 0.0159 | 65.74× | 2,428 MiB | 3,292 MiB |

### Voxtral Mini 3B 2507

| Variant | Bundle | WER | Δ WER | Mean RTF | Overall ×RT | Peak RSS | Physical footprint |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 8.71 GiB | 4.633% | — | 0.1305 | 8.05× | 9,091 MiB | 10,568 MiB |
| INT5 | 3.77 GiB | 4.744% | +0.110 pp | 0.0739 | 14.47× | 4,035 MiB | 6,012 MiB |
| INT8 | 5.18 GiB | 4.578% | -0.055 pp | 0.0906 | 11.79× | 5,487 MiB | 7,233 MiB |

The FP16 rows use verified native-F16 weights. Superseded measurements are not
included in these tables.

The current runtime projects only the final prompt hidden state through the
131,072-token language-model head. Projecting every prompt and audio state is
unnecessary for greedy decoding. Against the same-dtype pre-optimization runs,
this cut clean-run mean RTF by 25.8% for INT5 and 15.1% for INT8 without
changing their transcript or WER results. There is no comparable native-FP16
pre-change run, so no FP16 speedup percentage is claimed.

INT8 preserved Cohere FP16 aggregate WER and scored 0.055 point below true
Voxtral FP16 on this set. Standard INT5 reduced Voxtral physical footprint by
43.1% and remained within 0.111 absolute WER point of FP16. Cohere INT5
remains the practical low-memory/default candidate; Voxtral INT5 is the
stronger English-FLEURS quality candidate at a larger memory and latency cost.

This result validates the Swift runtimes and precision variants on English
read speech. FLEURS is not the Artificial Analysis Conversational benchmark,
so these numbers do not establish parity with ElevenLabs Scribe on meetings,
telephone audio, noise, accents, code-switching, or speaker-heavy dialogue.

## Build

Use a release build and compile the MLX metallib before measuring inference:

```bash
swift build -c release --disable-sandbox
./scripts/build_mlx_metallib.sh release
```

## Run

Point each benchmark engine at the matching local model bundle. Run engines in
separate processes so process-memory high-water marks do not leak between
variants:

```bash
COHERE_MLX_FP16_MODEL_PATH=/models/cohere/fp16 \
COHERE_MLX_INT5_MODEL_PATH=/models/cohere/int5 \
COHERE_MLX_INT8_MODEL_PATH=/models/cohere/int8 \
.build/release/asr-bench \
  --dataset /data/english.tsv \
  --engines cohere-transcribe-mlx-fp16 \
            cohere-transcribe-mlx-int5 \
            cohere-transcribe-mlx-int8 \
  --language en --isolated --output /tmp/cohere-asr.json

VOXTRAL_MLX_FP16_MODEL_PATH=/models/voxtral/fp16 \
VOXTRAL_MLX_INT5_MODEL_PATH=/models/voxtral/int5 \
VOXTRAL_MLX_INT8_MODEL_PATH=/models/voxtral/int8 \
.build/release/asr-bench \
  --dataset /data/english.tsv \
  --engines voxtral-mini-mlx-fp16 \
            voxtral-mini-mlx-int5 \
            voxtral-mini-mlx-int8 \
  --language en --isolated --output /tmp/voxtral-asr.json
```

The validated compatibility pass uses English FLEURS. A multilingual pass
must also cover every shared supported language before making multilingual
quality claims.

## Validation scope

The published variants completed the same 80 utterances and were compared
within each family on WER, RTF, throughput, RSS, and physical footprint. These
remain compatibility measurements rather than Conversational-board claims.
