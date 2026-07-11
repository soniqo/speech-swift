# Higgs TTS 3 Inference

Higgs TTS 3 is a conversational multilingual TTS engine with zero-shot voice
cloning and inline control tokens. Model architecture, bundle layout, and
validation notes are documented in
[`docs/models/higgs-tts.md`](../models/higgs-tts.md).

## Swift API

```swift
import HiggsTTS

let model = try await HiggsTTSModel.fromPretrained(
    modelId: "aufklarer/Higgs-TTS-3-4B-MLX-bf16")

let audio = try model.generate(
    text: "<|emotion:enthusiasm|>Hello from Higgs on Apple Silicon!",
    options: try HiggsTTSSynthesisOptions(temperature: 0.8, seed: 0))

print(audio.count)          // 24 kHz mono samples
print(model.sampleRate)     // 24000
```

Voice cloning takes a reference clip and its transcript; the codec encoder
runs natively (any input sample rate, ~0.6 s for a 12 s clip):

```swift
let cloned = try model.generate(
    text: "Any text in the cloned voice.",
    referenceAudio: URL(fileURLWithPath: "reference.wav"),
    referenceText: "The words spoken in the reference recording.")
```

For repeated use, encode once and reuse the codes:

```swift
let reference = try model.encodeReference(
    audio: URL(fileURLWithPath: "reference.wav"),
    text: "The words spoken in the reference recording.")
let audio = try model.generate(text: "Another line.", references: [reference])
```

## CLI

```bash
speech speak "<|emotion:elation|>Hello from Higgs!" \
  --engine higgs \
  --voice-sample reference.wav \
  --higgs-ref-text "The words spoken in the reference recording." \
  --output higgs.wav
```

`--voice-sample` is optional — without it, Higgs synthesizes in a
model-chosen voice.

| Flag | Notes |
|---|---|
| `--engine higgs` | Selects Higgs TTS 3. |
| `--voice-sample <wav>` | Optional reference audio for cloning. |
| `--higgs-ref-text <text>` | Transcript of the reference clip (improves cloning). |
| `--higgs-model-id <repo>` | Defaults to `aufklarer/Higgs-TTS-3-4B-MLX-bf16`. |
| `--higgs-bundle-dir <path>` | Loads a local bundle instead of Hugging Face. |
| `--higgs-temperature <value>` | Defaults to `0.8` (upstream's 1.0 is more variable). |
| `--higgs-top-p <value>` / `--higgs-top-k <n>` | Optional nucleus / top-k sampling. |
| `--higgs-max-new-tokens <n>` | Frame budget; defaults to `2048` (25 frames/s). |
| `--higgs-seed <n>` | Deterministic sampling seed; defaults to `0`. |
| `--clean-reference` | Optionally denoises/dereverbs the reference with Sidon. |

`--stream`, `--speaker`, `--instruct`, and `--batch-file` are rejected for
`--engine higgs`.

## Inline control tokens

All tags use `<|category:value|>` syntax inside the target text: 21 emotions
(`<|emotion:elation|>` …), styles (`singing`, `shouting`, `whispering`), 9
sound effects paired with onomatopoeia (`<|sfx:laughter|>Haha…`), and prosody
controls (`speed_slow`, `pause`, `pitch_high`, `expressive_high`, …). See the
upstream PROMPTING.md for placement rules; the tokenizer resolves the tags as
single tokens, so they pass through the Swift API unchanged.

## Languages

The upstream card claims 102 languages, 85 of them at WER/CER < 5. Our gate
measured perfect single-sentence cloned roundtrips in English, Mandarin,
Spanish, and German (and 3% CER Japanese) through Qwen3-ASR, with
single-sentence variance at the reference's temperature 1.0 — prefer lower
temperatures for stable delivery.

## E2E Tests

All Higgs E2E tests are environment-gated and skip without their inputs:

```bash
HIGGS_E2E_BUNDLE=/path/to/Higgs-TTS-3-4B-MLX-bf16 \
HIGGS_PARITY_FIXTURE=/path/to/greedy_hello.json \
HIGGS_E2E_WAV=/tmp/higgs.wav \
  swift test --filter E2EHiggsTTSTests/testGreedyParityAgainstReferenceImplementation --disable-sandbox
```

The parity fixture (prompt token ids, greedy delayed rows, teacher-forced
logits, reference waveform) is dumped from the mlx-audio reference by the
exporter tooling. The clone test additionally takes `HIGGS_REF_CODES`,
`HIGGS_REF_TEXT`, and `HIGGS_CLONE_WAV`:

```bash
HIGGS_E2E_BUNDLE=… HIGGS_REF_CODES=ivan_ref_codes.json \
HIGGS_REF_TEXT='Transcript of the reference clip.' \
HIGGS_CLONE_WAV=/tmp/higgs-clone.wav \
  swift test --filter E2EHiggsTTSTests/testCloneFromPrecomputedReferenceCodes --disable-sandbox
```

Fully native cloning from a reference WAV (Swift encoder → LM → decoder),
with optional per-codebook agreement reporting against `HIGGS_REF_CODES` and
an ASR gate when `HIGGS_ASR_E2E=1`:

```bash
HIGGS_E2E_BUNDLE=… HIGGS_REF_WAV=reference.wav \
HIGGS_REF_TEXT='Transcript of the reference clip.' \
HIGGS_ASR_E2E=1 \
  swift test --filter E2EHiggsTTSTests/testCloneFromReferenceWav --disable-sandbox
```

The multilingual gate mirrors the F5 suite — cloned English and Mandarin
roundtrips through Qwen3-ASR with overlap/CER thresholds:

```bash
HIGGS_E2E_BUNDLE=… HIGGS_REF_WAV=reference.wav \
HIGGS_REF_TEXT='Transcript of the reference clip.' \
HIGGS_ASR_E2E=1 HIGGS_GATE_WAV_DIR=/tmp/higgs-gate \
  swift test --filter E2EHiggsTTSTests/testNativeCloneRoundtripsEnglishAndMandarin --disable-sandbox
```
