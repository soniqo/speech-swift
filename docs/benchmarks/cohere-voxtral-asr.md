# Cohere and Voxtral MLX acceptance benchmark

The Cohere Transcribe 2B and Voxtral Mini 3B FP16, INT5, and INT8 exports
passed the cross-precision acceptance gate on 2026-07-22. The benchmark
reports aggregate/mean WER, CER, mean/median RTF, overall and per-utterance
×RT throughput, load time, peak RSS, and macOS physical footprint. RSS is
retained for historical comparisons; physical footprint is the deployment-
relevant unified-memory metric because MLX may memory-map weight files.

## Validated English result

The run used all 80 rows from the `transcription/english` slice of
`speech-bench-data` at revision
`99dd7ea4b057dbc1bea18694d639e841e2e569ae`: English FLEURS read speech
totaling 759.56 seconds and 1,813 reference words. It ran on a MacBook Pro
(Mac17,9), Apple M5 Pro, 48 GB memory, and macOS 26.5.2. Each variant ran in
its own process from a release build with the release MLX metallib
precompiled. All 80 utterances completed for every variant.

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
| INT5 + audio FFN (candidate) | 3.26 GiB | 4.523% | -0.110 pp | 0.0757 | 14.19× | 3,517 MiB | 5,671 MiB |

The FP16 rows are verified native-F16 exports. Earlier Cohere and Voxtral
artifacts bearing the FP16 name retained upstream BF16 tensors because of an
`mlx-audio` non-quantized conversion defect; those numbers were discarded.
Both exporters now rewrite and inspect safetensor headers, so a mismatched FP16
artifact fails before acceptance.

The current runtime projects only the final prompt hidden state through the
131,072-token language-model head. Projecting every prompt and audio state is
unnecessary for greedy decoding. Against the same-dtype pre-optimization runs,
this cut clean-run mean RTF by 25.8% for INT5 and 15.1% for INT8 without
changing their transcript or WER results. There is no comparable native-FP16
pre-change run, so no FP16 speedup percentage is claimed.

The candidate row is a separate local export, not a published
`VoxtralVariant`. It additionally quantizes the audio tower's 64 feed-forward
linear layers while retaining audio attention, convolutions, positions, and
norms in floating point. Relative to standard INT5 it reduced the bundle by
0.51 GiB (13.6%), peak RSS by 12.9%, and physical footprint by 5.7%. The final
kernel-high-water pass measured 0.0839 mean RTF; another heat-soaked run
measured 0.0921. Both remained faster than the original 0.0995 INT5 baseline,
while the clean candidate run was 0.0757 versus standard INT5's 0.0739.

INT8 preserved Cohere FP16 aggregate WER and scored 0.055 point below true
Voxtral FP16 on this set. Standard INT5 reduced Voxtral physical footprint by
43.1% and remained within 0.111 absolute WER point of FP16. The selective
Voxtral candidate reduced memory further with no observed English-FLEURS
quality regression. Cohere INT5 remains the practical low-memory/default
candidate; Voxtral INT5 is the stronger English-FLEURS quality candidate at a
larger memory and latency cost.

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

Point each benchmark engine at the matching local export. Run engines in
separate processes so process-memory high-water marks do not leak between
variants:

```bash
COHERE_MLX_FP16_MODEL_PATH=/exports/cohere/fp16 \
COHERE_MLX_INT5_MODEL_PATH=/exports/cohere/int5 \
COHERE_MLX_INT8_MODEL_PATH=/exports/cohere/int8 \
.build/release/asr-bench \
  --dataset /data/english.tsv \
  --engines cohere-transcribe-mlx-fp16 \
            cohere-transcribe-mlx-int5 \
            cohere-transcribe-mlx-int8 \
  --language en --isolated --output /tmp/cohere-asr.json

VOXTRAL_MLX_FP16_MODEL_PATH=/exports/voxtral/fp16 \
VOXTRAL_MLX_INT5_MODEL_PATH=/exports/voxtral/int5 \
VOXTRAL_MLX_INT8_MODEL_PATH=/exports/voxtral/int8 \
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

## Compatibility gate

The `speech-models` validator applies these default requirements to INT5 and
INT8 relative to the same family's FP16 result:

- aggregate WER degradation no greater than 1.0 percentage point;
- mean RTF no greater than 1.0 (faster than real time);
- overall ×RT throughput at least 75% of FP16;
- peak RSS and model-load RSS delta no greater than 105% of FP16;
- when present, peak physical footprint and its delta no greater than 105% of
  FP16;
- identical successful-utterance counts across all three variants.

```bash
python benchmarks/validate_mlx_asr_variants.py /tmp/cohere-asr.json --family cohere
python benchmarks/validate_mlx_asr_variants.py /tmp/voxtral-asr.json --family voxtral
```

These remain compatibility measurements rather than Conversational-board
claims. Record the machine, OS, dataset revision, utterance count, audio
duration, and generated JSON with every accepted result.
