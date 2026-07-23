# MOSS CoreML Native Runtime Benchmark

## Recommendation

Use `moss-coreml-int8`. Across two complete order-reversed runs it preserved
the same aggregate English WER as FP16, reduced pooled RTF by about 11%, and
used about 36% less sampled peak RSS.

## Setup

- Machine: Apple M5 Pro, 48 GB
- OS: macOS 26.5.2
- Build: release
- Dataset: 80 English FLEURS clips from the speech-bench-data Hugging Face
  fixture
- Audio: 759.56 seconds total
- Reference: 1,813 words
- Execution: one fresh child process per variant with `--isolated`
- Warm-up: first dataset utterance before measurement
- Repeats: two full runs with reversed model order

The initial FP16 download time is excluded from all RTF and throughput values.
The second run measured cached load plus warm-up at 2.6 seconds for FP16 and
2.8 seconds for INT8.

## Pooled result

Pooled RTF is total wall time divided by total audio across both runs.

| Variant | Aggregate WER | CER | Pooled RTF | Pooled ×RT | Peak RSS range |
|---|---:|---:|---:|---:|---:|
| CoreML INT8 | 8.16% | 5.90% | **0.070** | **14.3×** | **2,382–2,399 MB** |
| CoreML FP16 | 8.16% | 5.90% | 0.079 | 12.6× | 3,689–3,735 MB |

INT8 has an 11.4% lower pooled RTF. Comparing the observed peak ranges, it
uses approximately 35–36% less sampled RSS.

## Per-run measurements

| Order | Variant | Mean RTF | Median RTF | Overall ×RT | Peak RSS |
|---|---|---:|---:|---:|---:|
| Run 1, first | INT8 | 0.064 | 0.060 | 16.4× | 2,399 MB |
| Run 1, second | FP16 | 0.088 | 0.079 | 11.8× | 3,735 MB |
| Run 2, first | FP16 | 0.076 | 0.074 | 13.6× | 3,689 MB |
| Run 2, second | INT8 | 0.080 | 0.070 | 12.7× | 2,382 MB |

The order reversal exposes normal machine/thermal variance. Pooling both runs
preserves the approximately 10% INT8 speed advantage seen during exporter
validation, while the memory difference is stable.

## Quality validation

Both variants produced:

- 148 total word errors;
- 106 substitutions;
- 28 insertions;
- 14 deletions;
- 504 character errors.

The E2E regressions additionally pin the same complete reference round trip
for both published CoreML bundles:

```text
[5.00][S01] Can you guarantee that the replacement part will be shipped tomorrow?[8.40]
```

The Swift Whisper frontend matches pinned Hugging Face values within `5e-4`,
and the complete prompt token IDs match the exporter reference exactly.

## Reproduction

Convert the HF `metadata.jsonl` fixture to the TSV form accepted by
`asr-bench` (`<absolute-audio-path><tab><reference>`), then run:

```bash
make build

.build/release/asr-bench \
  --dataset /path/to/english.tsv \
  --engines moss-coreml-int8 moss-coreml-fp16 \
  --isolated \
  --output moss-native-coreml.json

# Reverse-order replication
.build/release/asr-bench \
  --dataset /path/to/english.tsv \
  --engines moss-coreml-fp16 moss-coreml-int8 \
  --isolated \
  --output moss-native-coreml-reverse.json
```

See [MOSS inference](../inference/moss-transcribe-diarize.md) for the CLI and
Swift APIs.
