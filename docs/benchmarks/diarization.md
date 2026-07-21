# Speaker Diarization Benchmarks

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
