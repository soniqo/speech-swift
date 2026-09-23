# Speaker Diarization Benchmarks

## Hugging Face multilingual benchmark (23 September 2026)

The shipped Nemotron 3 INT8 MLX and Core ML runtimes were evaluated on all 132
recordings (12.82 hours) in [`soniqo/speech-bench-data`](https://huggingface.co/datasets/soniqo/speech-bench-data), revision
`9c91223705488a5f66bfe747a42ca0699124b950`. The model export is from
[`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization), revision
`a435e9867d79e789e90053f9b6d6834053af564a`. The
[`speech-bench` diarization runner](https://github.com/soniqo/speech-bench)
invoked `speech diarize --engine nemotron3` with automatic speaker count and
the package's default postprocessing. `pyannote.metrics` scored each recording
inside its UEM, including overlap, with a 0.25-second collar. DER is pooled by
reference speaker time; no reference speaker count was passed to the model.

| Language | Files | Nemotron 3 MLX | Nemotron 3 Core ML | Sortformer streaming | MOSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Arabic | 12 | 16.54% | 18.16% | 22.32% | 14.57% |
| German | 12 | 7.32% | 7.21% | 8.38% | 16.95% |
| English | 48 | 10.31% | 10.24% | 13.81% | 16.35% |
| Spanish | 12 | 12.56% | 13.19% | 16.33% | 22.51% |
| Mandarin | 12 | 10.00% | 9.68% | 9.95% | 10.12% |
| Azerbaijani | 12 | 2.53% | 2.55% | 5.25% | 11.30% |
| Swiss German | 12 | 15.72% | 15.69% | 21.59% | 26.59% |
| Russian | 12 | 4.47% | 4.44% | 5.35% | 15.14% |
| **All** | **132** | **10.98%** | **11.20%** | — | — |

Nemotron 3 had no failed recordings. Exact speaker-count agreement was 85.6%
for MLX and 84.8% for Core ML. The earlier Sortformer and MOSS columns come
from the [same-corpus reference benchmark](https://github.com/soniqo/speech-bench/blob/main/reports/2026-08-16-diarization.md),
which used direct source-model inference rather than the conversions in this
package. Azerbaijani, Swiss German, and Russian are synthetic turn-taking
rails; the other five languages are real conversation. The Mandarin difference
between Nemotron 3 and Sortformer is small, and MOSS is stronger on Arabic.

As a postprocessing check, setting both minimum speech and minimum silence to
zero (the source model's default) changed MLX pooled DER from 10.98% to 10.88%
and count agreement from 85.6% to 84.8%. Results varied by language, so this
check does not change the package default.

## Local VoxConverse comparison (23 September 2026)

`diarization-bench` scored five VoxConverse recordings (2,346.6 seconds total) on an Apple Silicon Mac with a 0.25-second collar and 0.01-second scoring resolution. DER includes missed speech, false alarms, and speaker confusion. Throughput excludes model loading.

| Engine | DER | Throughput | Load | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Nemotron 3 MLX INT8 | 17.26% | 333.4× real time | 7.3 s | 498 MB |
| Nemotron 3 Core ML INT8 | 17.55% | 51.0× real time | 15.9 s | 782 MB |
| Sortformer default Core ML | 20.77% | 113.1× real time | 36.5 s | 305 MB |

MOSS Core ML cannot process these whole recordings through the current benchmark adapter: all five exceed its 1,024-token decoder prompt limit. A matched excerpt run used the first 25 seconds of each recording (125 seconds total), with the RTTM references clipped at the same boundary:

| Engine | DER | Throughput | Load | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Nemotron 3 MLX INT8 | 1.25% | 307.1× real time | 0.3 s | 184 MB |
| Nemotron 3 Core ML INT8 | 1.25% | 52.4× real time | 14.4 s | 425 MB |
| Sortformer default Core ML | 0.22% | 109.4× real time | 40.3 s | 132 MB |
| MOSS Core ML INT8 | 2.73% | 3.5× real time | 56.6 s | 1,737 MB |

The short excerpts contain one to three reference speakers each; their DER cannot be compared directly with the full-recording DER. Speaker-count accuracy and per-count errors are available in the benchmark JSON output. These local measurements are a small regression check, not a corpus-wide quality estimate.

## Published Streaming Sortformer baseline

The table below is NVIDIA's published raw-diarization baseline for
`nvidia/diar_streaming_sortformer_4spk-v2`. It is an external reference, not a
measurement of the CoreML conversion in this package and not a downstream
speaker-naming or attribution result.

| Dataset | Condition | DER |
|---|---|---:|
| CALLHOME Part 2 | 2 reference speakers | 6.57% |
| CALLHOME Part 2 | 3 reference speakers | 10.05% |
| CALLHOME Part 2 | 4 reference speakers | 12.44% |

All three conditions use 1.04-second input-buffer latency, include overlapping
speech, use a 0.25-second scoring collar, and apply post-processing tuned on
the disjoint CALLHOME Part 1 split. The respective Part 2 subsets contain 148,
74, and 20 sessions. See the
[official model card](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2#performance)
for the source evaluation and additional latency configurations.

CALLHOME is an LDC-licensed corpus and is not downloaded or redistributed by
this package. A local CoreML result must therefore be reported separately and
must identify its exact manifest, model revision, collar, overlap policy, and
post-processing configuration.

## Local speaker-count breakdown

`diarization-bench` automatically counts unique speakers in each reference
RTTM and reports pooled DER separately for every count present in the manifest.
This prevents a corpus-wide average from hiding a regression that appears only
when a fourth speaker is present.

```bash
swift run -c release diarization-bench \
  --manifest callhome-part2.tsv \
  --engines sortformer-session \
  --collar 0.25 \
  --output callhome-part2.json
```

The console report appends a `DER by reference speaker count` table. The JSON
result for each engine includes `byReferenceSpeakerCount`, with pooled DER,
miss, false alarm, speaker error, scored speech duration, file count, and
speaker-count accuracy for each condition. Failed files are excluded from the
corresponding aggregate and remain listed in the engine's `failures` field.

These are raw anonymous-speaker diarization metrics. Any application-level
speaker enrollment, naming, or identity propagation must be evaluated in a
separate report rather than presented as this baseline.
