# ReDimNet2-B6 Speaker Identity

## Purpose

`ReDimNet2SpeakerModel` extracts a stable voice representation from a clean
speaker-only sample. It is intended for recording-local identity continuity and
persistent named voice profiles across recordings.

It is not a diarization model. Community-1 continues to use its own masked
WeSpeaker embeddings, PLDA transform, and VBx clustering. Those centroids live
in a different vector space and must not be compared with ReDimNet2 profiles.

## Model

The source model is the official
[PalabraAI/ReDimNet2](https://github.com/PalabraAI/redimnet2) B6 large-margin
checkpoint trained on VoxBlink2 and VoxCeleb2.

| Property | Value |
|---|---:|
| Parameters | 12.3 million |
| Input | 96,000 mono Float32 samples |
| Sample rate | 16 kHz |
| Window | 6 seconds |
| Output | 192 floats, L2-normalized |
| Runtime | Compiled Core ML |
| Compiled size | approximately 29 MiB |
| Export revision | `frontend-fp32-v1` |
| Compute precision | FP32 frontend/head, FP16 backbone |

The graph includes waveform normalization, pre-emphasis, mel feature
extraction, ReDimNet2-B6, attentive statistics pooling, the embedding head, and
final L2 normalization.

The frontend, statistics pooling, projection and final normalization retain
FP32 types. Only the learned backbone uses FP16. This avoids preprocessing
overflow for sparse or quiet inputs while preserving the fixed input/output
contract and the original checkpoint.

## Fixed Window

A fixed six-second waveform avoids the CPU fallback seen with flexible Core ML
shapes. Runtime preparation follows the benchmarked policy:

- reject less than two seconds of clean speech;
- repeat two-to-six-second speech until the window is full;
- keep an exact six-second input unchanged;
- center-crop longer input.

An explicit `embedShortUtterance` path accepts 0.6-to-2 seconds and repeats it
to the same fixed window. Repetition satisfies the graph shape but adds no voice
evidence. These embeddings are lower-confidence retrieval probes only: match
them against an identity established from at least two seconds, and do not use
them for enrollment, new-cluster creation, or centroid updates. Calibrate a
stricter threshold and ambiguity margin for each target domain.

Inference failures throw an error. The model never substitutes a zero embedding.

## Validation

The conversion checks five deterministic waveforms under four Core ML compute
configurations against the checksum-pinned original FP32 model. Every output
must be finite, normalized, and reach cosine agreement >= 0.999. These are
numerical checks, not diarization or speaker-recognition accuracy measurements.
Matching thresholds still need microphone, language and duration calibration.

## Cache and Offline Loading

`cacheDir` is the repository cache root. The default model loads from
`revisions/frontend-fp32-v1/` under that root, preserving earlier cached exports.
`modelCacheDirectory(in:)` resolves this directory; `isCached(at:)` checks the
expected export metadata and required-file presence for routing. The loader
then verifies the pinned compiled-file SHA-256 values before inference.
`offlineMode: true` requires this corrected generation; a retired export is not
an offline hit. Explicit custom repositories retain their direct cache paths.

The pinned compiled bytes are published at
[revision `112dd8f4`](https://huggingface.co/aufklarer/ReDimNet2-B6-CoreML/tree/112dd8f4f836abdf8a420e66a5e2885cf8ec64ab).

## API

```swift
let model = try await ReDimNet2SpeakerModel.fromPretrained()
try model.prewarm()

let profile = try model.embed(audio: cleanAudio, sampleRate: sampleRate)
let shortProbe = try model.embedShortUtterance(
    audio: shortCleanAudio,
    sampleRate: sampleRate)
let score = ReDimNet2SpeakerModel.cosineSimilarity(profile, candidate)
```

The model produces features for labeling. It is not an authentication system
and does not provide anti-spoofing guarantees.
