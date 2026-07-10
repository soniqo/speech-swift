# F5-TTS Inference

F5-TTS is exposed as a native MLX voice-cloning engine. It requires a reference
voice sample and the exact transcript for that sample.

Model architecture, bundle layout, and validation notes are documented in
[`docs/models/f5-tts.md`](../models/f5-tts.md).

## Swift API

```swift
import F5TTS

let bundle = URL(fileURLWithPath: "/path/to/F5TTS-v1-Base-MLX-fp16")
let model = try await F5TTSModel.fromBundle(bundle)

let audio = try await model.generate(
    text: "This is a short F five TTS voice cloning test.",
    referenceAudio: URL(fileURLWithPath: "/path/to/reference.wav"),
    referenceText: "The words spoken in the reference recording.",
    options: try F5TTSSynthesisOptions(
        steps: 32,
        cfgStrength: 2.0,
        swaySamplingCoef: -1.0,
        speed: 1.0,
        seed: 0))

print(audio.count)
print(model.sampleRate) // 24000
```

Use `F5TTSModel.fromPretrained()` to download the default published bundle, or
`fromBundle(_:)` for local exporter output.

## CLI

Use a local exported bundle:

```bash
speech speak "This is a short F five TTS voice cloning test." \
  --engine f5 \
  --f5-bundle-dir /path/to/F5TTS-v1-Base-MLX-fp16 \
  --voice-sample reference.wav \
  --f5-reference-text "The words spoken in the reference recording." \
  --output f5.wav
```

Use the published default bundle:

```bash
speech speak "This is a short F five TTS voice cloning test." \
  --engine f5 \
  --voice-sample reference.wav \
  --f5-reference-text "The words spoken in the reference recording." \
  --f5-model-id aufklarer/F5TTS-v1-Base-MLX-fp16 \
  --output f5.wav
```

The command loads the exported bundle, prepares a reference mel from
`--voice-sample`, samples a target mel with the DiT flow model, decodes it with
Vocos, and writes 24 kHz mono WAV.

## Flags

| Flag | Notes |
|---|---|
| `--engine f5` | Selects F5-TTS. |
| `--voice-sample <wav>` | Required reference audio. |
| `--f5-reference-text <text>` | Required transcript for the reference audio. |
| `--f5-model-id <repo>` | Defaults to `aufklarer/F5TTS-v1-Base-MLX-fp16`. |
| `--f5-bundle-dir <path>` | Loads a local exported bundle instead of Hugging Face. |
| `--f5-steps <n>` | Flow-matching steps; defaults to `32`. |
| `--f5-cfg-strength <value>` | Classifier-free guidance strength; defaults to `2.0`. |
| `--f5-sway <value>` | Sway sampling coefficient; defaults to `-1.0`. |
| `--f5-speed <value>` | Speaking-rate multiplier; values above `1.0` shorten generated speech. |
| `--f5-seed <n>` | Deterministic MLX flow-sampling seed; defaults to `0`. |
| `--f5-target-rms <value>` | RMS target for quiet reference normalization; defaults to `0.1`. |
| `--clean-reference` | Optionally denoises/dereverbs the reference with Sidon before cloning. |

`--stream`, `--speaker`, and `--instruct` are intentionally rejected for
`--engine f5`.

## E2E Tests

Fast unit tests avoid model downloads:

```bash
swift test --filter F5TTSTests --skip E2E --disable-sandbox
```

Full local-bundle synthesis with optional ASR roundtrip:

```bash
F5TTS_E2E_BUNDLE=/path/to/F5TTS-v1-Base-MLX-fp16 \
F5TTS_REFERENCE_WAV=/path/to/reference.wav \
F5TTS_REFERENCE_TEXT='The words spoken in the reference recording.' \
F5TTS_TARGET_TEXT='This is a short F five TTS voice cloning test.' \
F5TTS_E2E_WAV=/tmp/f5.wav \
F5TTS_ASR_E2E=1 \
  swift test --filter E2EF5TTSTests/testLocalBundleSynthesizesEnglishCloneAndOptionalASRRoundTrip --disable-sandbox
```

Parity diagnostics:

```bash
F5TTS_E2E_BUNDLE=/path/to/F5TTS-v1-Base-MLX-fp16 \
F5TTS_PARITY_X_F32=/tmp/x.f32 \
F5TTS_PARITY_COND_F32=/tmp/cond.f32 \
F5TTS_PARITY_TOKENS_I32=/tmp/tokens.i32 \
F5TTS_PARITY_SEQ=1624 \
  swift test --filter E2EF5TTSTransformerParityTests/testFixedInputVelocityDump --disable-sandbox
```

`E2EF5TTSFlowParityTests` can dump sampler steps with
`F5TTS_STEP_DUMP_DIR=/tmp/f5-steps` and compare Swift flow states against a
Python reference. `E2EF5TTSVocosParityTests` decodes a supplied 100-band mel and
can run a Qwen3-ASR roundtrip with `F5TTS_ASR_E2E=1`.
