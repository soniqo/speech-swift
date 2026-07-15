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
| Compiled size | approximately 25 MiB |

The graph includes waveform normalization, pre-emphasis, mel feature
extraction, ReDimNet2-B6, attentive statistics pooling, the embedding head, and
final L2 normalization.

## Fixed Window

A fixed six-second waveform avoids the CPU fallback seen with flexible Core ML
shapes. Runtime preparation follows the benchmarked policy:

- reject less than two seconds of clean speech;
- repeat two-to-six-second speech until the window is full;
- keep an exact six-second input unchanged;
- center-crop longer input.

Inference failures throw an error. The model never substitutes a zero embedding.

## Validation

The conversion compares the checksum-pinned official PyTorch checkpoint with
the compiled Core ML bundle. The release candidate measured 0.999990 cosine
agreement, a 0.999916 output norm, and 13.6 ms median warm inference on an Apple
M2 Max.

On the initial recurring-speaker meeting pilot, the fixed Core ML model reduced
equal error rate from 3.76% with standalone WeSpeaker to 1.88%. This is a small
five-speaker pilot, not a universal quality claim. Production thresholds need
additional accented and multilingual calibration data.

## API

```swift
let model = try await ReDimNet2SpeakerModel.fromPretrained()
try model.prewarm()

let profile = try model.embed(audio: cleanAudio, sampleRate: sampleRate)
let score = ReDimNet2SpeakerModel.cosineSimilarity(profile, candidate)
```

The model produces features for labeling. It is not an authentication system
and does not provide anti-spoofing guarantees.
