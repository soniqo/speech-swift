# Voice Cloning Candidate Runtimes - Inference

These modules are loadable runtime contracts. IndexTTS2 now includes the native
reference-conditioning and synthesis path, while Higgs Audio and F5-TTS remain
bundle-loader surfaces. This page is the source of truth for runtime usage,
CLI/API controls, and E2E validation. Bundle layout, architecture coverage, and
port status live in
[`docs/models/voice-cloning-candidate-runtimes.md`](../models/voice-cloning-candidate-runtimes.md).

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

The bundle-loading shape is available for `HiggsAudioTTSModel` and `F5TTSModel`.

## CLI Surface

IndexTTS2 is exposed through `speech speak` as a native voice-cloning engine:

```bash
speech speak "Hello from IndexTTS2" \
  --engine indextts2 \
  --voice-sample reference.wav \
  --indextts2-bundle-dir /path/to/IndexTTS2-MLX-fp16
```

or, after the expanded bundle is published:

```bash
speech speak "Hello from IndexTTS2" \
  --engine indextts2 \
  --voice-sample reference.wav \
  --indextts2-speaking-rate 1.35 \
  --indextts2-max-pause 0.05 \
  --indextts2-model-id aufklarer/IndexTTS2-MLX-fp16
```

The command downloads or loads the exported bundle, prepares native IndexTTS2
reference conditioning, generates semantic codes with the native GPT path using
upstream-style beam sampling (`beams=3`, `top_k=30`, `top_p=0.8`,
`temperature=0.8`, `repetition_penalty=10`, seed `11` by default), decodes
80-band mels with the S2Mel CFM flow, and vocodes the result with BigVGAN.

| Flag | Scope | Notes |
|---|---|---|
| `--engine indextts2` | `speech speak` | Selects the IndexTTS2 exported-bundle loader. |
| `--voice-sample <wav>` | `indextts2` | Required because IndexTTS2 is a zero-shot voice-cloning model. |
| `--indextts2-model-id <repo>` | `indextts2` | Defaults to `aufklarer/IndexTTS2-MLX-fp16`. |
| `--indextts2-bundle-dir <path>` | `indextts2` | Loads a local exported bundle instead of Hugging Face. |
| `--indextts2-emotion-audio <wav>` | `indextts2` | Optional separate emotion reference; defaults to the speaker reference. |
| `--indextts2-emotion <preset-or-vector>` | `indextts2` | Optional upstream-style emotion vector control. Presets: `eager`, `happy`, `excited`, `angry`, `sad`, `afraid`, `disgusted`, `melancholic`, `surprised`, `calm`; custom vectors use the 8-value order `happy,angry,sad,afraid,disgusted,melancholic,surprised,calm` and must sum to `<= 0.8`. Mutually exclusive with `--indextts2-emotion-audio`. |
| `--indextts2-emotion-weight <0...1>` | `indextts2` | Scales `--indextts2-emotion`; defaults to `1.0`. Identity-first cloning should omit explicit emotion or keep this modest because high weights can reduce speaker similarity. |
| `--indextts2-speaking-rate <0.5...1.5>` | `indextts2` | Shortens or lengthens generated speech by changing the S2Mel frame expansion. Values above `1.0` are faster. Defaults to `1.0`. |
| `--indextts2-max-pause <0.05...2.0>` | `indextts2` | Optional post-vocoder cap for long internal low-energy spans. Omit to keep raw model timing; use small values such as `0.05` only when generated speech has audible dead pauses. |

Default model IDs are listed in the model bundle status table. The CLI defaults
to `aufklarer/IndexTTS2-MLX-fp16` for IndexTTS2 and can be pointed at a local
bundle with `--indextts2-bundle-dir`.

## Tests

Fast unit tests avoid model downloads:

```bash
swift test --filter VoiceCloneCandidateTTSTests
```

The IndexTTS2 expanded-bundle E2E test is opt-in because it validates a multi-GB
bundle:

```bash
INDEXTTS2_E2E_BUNDLE=/path/to/IndexTTS2-MLX-fp16 \
  swift test --filter E2EIndexTTS2BundleTests --disable-sandbox

INDEXTTS2_E2E_DOWNLOAD=1 \
  swift test --filter E2EIndexTTS2BundleTests --disable-sandbox
```

The E2E verifies bundle metadata, auxiliary model records, tokenizer loading,
runtime tensor inventory, native reference conditioning
(`w2v-BERT -> MaskGCT -> length regulator`, CAMPPlus style, prompt mel), native
semantic-code generation, and a bounded waveform synthesis smoke through S2Mel
and BigVGAN. The optional ASR roundtrip path writes the generated WAV, reports
RTF and resident-memory deltas, and gates intelligibility with Qwen3-ASR:

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
`INDEXTTS2_E2E_SEMANTIC_CODES` for supplying a known code sequence,
`INDEXTTS2_E2E_SEMANTIC_ONLY=1` for semantic-generation-only checks, and
`INDEXTTS2_E2E_SEED_SWEEP=0-20` for in-process seed sweeps.
