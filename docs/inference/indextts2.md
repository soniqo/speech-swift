# IndexTTS2 Inference

IndexTTS2 is exposed as a native MLX voice-cloning engine. It requires a
reference voice sample and currently runs batch synthesis only.

Model architecture, bundle layout, and remaining port work are documented in
[`docs/models/indextts2.md`](../models/indextts2.md).

## Swift API

```swift
import IndexTTS2TTS

let bundle = URL(fileURLWithPath: "/path/to/IndexTTS2-MLX-fp16")
let model = try await IndexTTS2TTSModel.fromBundle(bundle)

print(model.manifest.parameterCount ?? "unknown")
print(model.memoryFootprint)
print(IndexTTS2TTSModel.auxiliaryModels.map(\.repository))

let tokenIds = try model.tokenizer?.encode("Hello from IndexTTS2")

let conditioning = try model.prepareReferenceConditioning(
    referenceAudio: URL(fileURLWithPath: "/path/to/reference.wav"))
print(conditioning.promptConditionShape)

let audio = try await model.generate(
    text: "Hello from IndexTTS2",
    referenceAudio: URL(fileURLWithPath: "/path/to/reference.wav"),
    synthesisOptions: try IndexTTS2SynthesisOptions(
        speakingRate: 1.35,
        maxInternalPauseDuration: 0.05))
print(audio.count)
```

For identity-first cloning, omit explicit emotion control and use
`IndexTTS2SynthesisOptions` for tempo and pause shaping. High emotion weights can
reduce speaker similarity.

## CLI

Use a local exported bundle:

```bash
speech speak "Hello from IndexTTS2" \
  --engine indextts2 \
  --voice-sample reference.wav \
  --indextts2-bundle-dir /path/to/IndexTTS2-MLX-fp16
```

Use the published default bundle:

```bash
speech speak "Hello from IndexTTS2" \
  --engine indextts2 \
  --voice-sample reference.wav \
  --indextts2-speaking-rate 1.35 \
  --indextts2-max-pause 0.05 \
  --indextts2-model-id aufklarer/IndexTTS2-MLX-fp16
```

The command downloads or loads the exported bundle, prepares native reference
conditioning, generates semantic codes with the native GPT path, decodes 80-band
mels with S2Mel, and vocodes the result with BigVGAN.

## Flags

| Flag | Notes |
|---|---|
| `--engine indextts2` | Selects IndexTTS2. |
| `--voice-sample <wav>` | Required reference audio. |
| `--indextts2-model-id <repo>` | Defaults to `aufklarer/IndexTTS2-MLX-fp16`. |
| `--indextts2-bundle-dir <path>` | Loads a local exported bundle instead of Hugging Face. |
| `--indextts2-emotion-audio <wav>` | Optional separate emotion/style reference. Defaults to the speaker reference. |
| `--indextts2-emotion <preset-or-vector>` | Optional upstream-style emotion vector control. Presets: `eager`, `happy`, `excited`, `angry`, `sad`, `afraid`, `disgusted`, `melancholic`, `surprised`, `calm`. Custom vectors use the 8-value order `happy,angry,sad,afraid,disgusted,melancholic,surprised,calm` and must sum to `<= 0.8`. Mutually exclusive with `--indextts2-emotion-audio`. |
| `--indextts2-emotion-weight <0...1>` | Scales `--indextts2-emotion`; defaults to `1.0`. Keep this modest when speaker identity matters. |
| `--indextts2-speaking-rate <0.5...1.5>` | Shortens or lengthens generated speech by changing the S2Mel frame expansion. Values above `1.0` are faster. Defaults to `1.0`. |
| `--indextts2-s2mel-steps <4...100>` | S2Mel flow steps. Defaults to the upstream `25`; `15` halves the S2Mel stage with word-identical ASR roundtrips in local testing. |
| `--indextts2-max-pause <0.05...2.0>` | Optional post-vocoder cap for long internal low-energy spans. Omit to keep raw model timing; use small values such as `0.05` only when generated speech has audible dead pauses. |

Default semantic generation uses upstream-style beam sampling with `beams=3`,
`top_k=30`, `top_p=0.8`, `temperature=0.8`, `repetition_penalty=10`, and seed
`11`.

## E2E Tests

Fast unit tests avoid model downloads:

```bash
swift test --filter IndexTTS2TTSTests
```

The expanded-bundle E2E test is opt-in because it validates a multi-GB bundle:

```bash
INDEXTTS2_E2E_BUNDLE=/path/to/IndexTTS2-MLX-fp16 \
  swift test --filter E2EIndexTTS2BundleTests --disable-sandbox

INDEXTTS2_E2E_DOWNLOAD=1 \
  swift test --filter E2EIndexTTS2BundleTests --disable-sandbox
```

The E2E verifies bundle metadata, auxiliary model records, tokenizer loading,
runtime tensor inventory, native reference conditioning, native semantic-code
generation, and bounded waveform synthesis through S2Mel and BigVGAN. The
optional ASR roundtrip path writes the generated WAV, reports RTF and
resident-memory deltas, and gates intelligibility with Qwen3-ASR:

```bash
INDEXTTS2_E2E_BUNDLE=/path/to/IndexTTS2-MLX-fp16 \
INDEXTTS2_E2E_REFERENCE=/path/to/reference.wav \
INDEXTTS2_E2E_OUTPUT=/tmp/indextts2.wav \
INDEXTTS2_E2E_TEXT='I am ready to help right now with clear energetic speech' \
INDEXTTS2_E2E_SPEAKING_RATE=1.35 \
INDEXTTS2_E2E_MAX_PAUSE=0.05 \
INDEXTTS2_E2E_ROUNDTRIP=1 \
  swift test --filter E2EIndexTTS2BundleTests/testFullSynthesisBenchmarkAndOptionalASRRoundtrip --disable-sandbox
```

Useful diagnostic overrides are `INDEXTTS2_E2E_SEED`,
`INDEXTTS2_E2E_BEAMS`, `INDEXTTS2_E2E_TEMPERATURE`,
`INDEXTTS2_E2E_TOP_K`, `INDEXTTS2_E2E_TOP_P`,
`INDEXTTS2_E2E_REPETITION_PENALTY`, `INDEXTTS2_E2E_LENGTH_PENALTY`,
`INDEXTTS2_E2E_EMOTION`, `INDEXTTS2_E2E_EMOTION_WEIGHT`,
`INDEXTTS2_E2E_SPEAKING_RATE`, `INDEXTTS2_E2E_MAX_PAUSE`,
`INDEXTTS2_E2E_SEMANTIC_CODES`, `INDEXTTS2_E2E_SEMANTIC_ONLY=1`, and
`INDEXTTS2_E2E_SEED_SWEEP=0-20`.
