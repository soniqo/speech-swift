# CSM Inference

CSM (Conversational Speech Model, Sesame CSM-1B) is a native MLX text→audio engine
with zero-shot voice cloning. It requires a reference voice sample and the exact
transcript for that sample.

Model architecture, bundle layout, and validation notes are documented in
[`docs/models/csm.md`](../models/csm.md).

## Swift API

```swift
import CSM
import MLX
import AudioCommon

// Download + load the published int8 model (cached after first run).
let pipeline = try await CSMPipeline.fromPretrained("aufklarer/CSM-1B-MLX-8bit")

// Reference voice: any sample rate, resampled to 24 kHz internally.
let ref = try AudioFileLoader.load(
    url: URL(fileURLWithPath: "/path/to/reference.wav"), targetSampleRate: 24000)

let audio = pipeline.synthesize(
    text: "Nice to meet you — this voice is running entirely on device.",
    refAudio: MLXArray(ref),
    refText: "The words spoken in the reference recording.",
    temperature: 0.9,
    topK: 50)

let samples = audio.asArray(Float.self)          // 24 kHz mono
try WAVWriter.write(samples: samples, sampleRate: 24000,
                    to: URL(fileURLWithPath: "out.wav"))
```

Pass `offlineMode: true` to `fromPretrained` to skip the network and load a
previously cached model. Use `generateFrames(...)` if you want the raw codebook
frames instead of a decoded waveform.

## CLI

```bash
speech csm "Nice to meet you — this voice is running entirely on device." \
  --ref-audio reference.wav \
  --ref-text "The words spoken in the reference recording." \
  --output out.wav
```

The command downloads (or reuses) the model, loads the reference audio, runs the
autoregressive frame loop, Mimi-decodes to a 24 kHz waveform, and writes mono WAV.

## Flags

| Flag | Notes |
|---|---|
| `<text>` | Positional. Text to synthesize. |
| `--ref-audio <wav>` | Required. Reference audio to clone the voice from (any sample rate). |
| `--ref-text <text>` | Required. Transcript of the reference audio. |
| `-o, --output <wav>` | Output WAV path. Defaults to `output.wav`. |
| `--model <repo>` | HuggingFace model ID. Defaults to `aufklarer/CSM-1B-MLX-8bit`. |
| `--temperature <float>` | Sampling temperature; `0` = greedy. Defaults to `0.9`. |
| `--top-k <int>` | Top-k sampling. Defaults to `50`. |

## Notes

- Give an accurate transcript for the reference clip; a clean 10–15 s sample
  clones best. Mismatched reference text degrades quality.
- Output is 24 kHz mono. `temperature = 0` is deterministic (greedy); raise it for
  more expressive, higher-variance takes.
- Build in Release for real-time-class throughput — Debug MLX is several times
  slower. Call `MLX.GPU.clearCache()` between repeated generations.

## E2E Tests

`Tests/CSMTests` covers weight loading, single-frame generation, the frame loop,
and full text→audio synthesis, including an end-to-end
`fromPretrained("aufklarer/CSM-1B-MLX-8bit")` download-and-synthesize check.
