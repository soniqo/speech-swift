# Nemotron-3.5 ASR Streaming — inference

Swift bindings for the Nemotron-3.5 multilingual streaming ASR bundle (CoreML, 600 M params, 40 language-locales). Lives in `Sources/NemotronStreamingASR/`.

## Two ways to consume

```swift
import NemotronStreamingASR

// 1. Single-shot transcription (model creates a session internally)
let model = try await NemotronStreamingASRModel.fromPretrained()
let text = try model.transcribeAudio(audio, sampleRate: 16000, language: "en-US")

// 2. Streaming session that yields partial transcripts as audio is fed
for await partial in model.transcribeStream(audio: audio, sampleRate: 16000, language: "ja-JP") {
    print(partial.text, partial.isFinal)
}

// 3. Manual session for caller-controlled chunking
let session = try model.createSession(language: "de-DE")
for chunk in audioChunks {
    let partials = try session.pushAudio(chunk)
    // ...
}
let finals = try session.finalize()
```

## Language tags

`language` is a BCP-47 tag (`en-US`, `de-DE`, `fr-FR`, `ja-JP`, `hi-IN`, ...) resolved against `languages.json` via `NemotronLanguages.slot(for:)`. Resolution order:

1. Exact match (`en-US` → slot 0)
2. Underscore → hyphen normalization (`en_US` → `en-US`)
3. Base language fallback (`en-GB` → `en`)
4. `"auto"` slot (101)

The full list of 121 aliases is in the `languages` property of the loaded model.

## Loading

```swift
// From the HF cache (default repo aufklarer/Nemotron-3.5-ASR-Streaming-0.6B-CoreML-INT8)
let model = try await NemotronStreamingASRModel.fromPretrained()

// Pinned to a different HF repo
let model = try await NemotronStreamingASRModel.fromPretrained(
    modelId: "aufklarer/Nemotron-3.5-ASR-Streaming-0.6B-CoreML-INT8"
)

// From a local bundle (skips download — useful for CI and dev)
let bundleURL = URL(fileURLWithPath: "/tmp/Nemotron-3.5-CoreML-320ms")
let model = try await NemotronStreamingASRModel.fromLocal(bundleDir: bundleURL)
```

The loader expects the standard bundle layout described in `docs/models/nemotron-asr-streaming.md`. Compute units default to `.all` (encoder on ANE + GPU + CPU, decoder/joint on CPU) — measured ~40% faster RTF than `.cpuAndGPU` on M-series.

## Streaming geometry

| Param | Default |
|-------|---------|
| Chunk | 320 ms (4 output frames × 80 ms) |
| Right context | 3 frames (240 ms lookahead) |
| Left context | 56 frames (4.48 s) |
| Pre-cache | 9 mel frames |
| Sample rate | 16 kHz mono Float32 |

`transcribeStream` slices audio at the configured chunk size; `pushAudio` accepts any chunk size and buffers internally until enough samples accumulate.

## TTS round-trip and short-audio padding

`transcribeAudio` adds 100 ms of silence at both ends by default to prime the all-zero streaming cache (otherwise short TTS clips with sharp onsets can drop the first word). For natural recorded speech (FLEURS-style microphone capture), pass `padSilence: false` to skip the pad — measured ~0–5 pp WER drop on the harder languages.

```swift
// TTS-style (short clip, padding helps): default true
let ttsText = try model.transcribeAudio(ttsAudio, sampleRate: 24000, language: "en-US")

// Recorded speech (no padding shift):
let recText = try model.transcribeAudio(micAudio, sampleRate: 16000,
                                        language: "en-US", padSilence: false)
```

## Memory + throughput on M5 Pro

60 s synthetic FLEURS en_us long-form streamed in 320 ms chunks (compute units `.all`):

| metric | value |
|--------|------:|
| RSS pre-load | 56 MB |
| RSS post-load | 652 MB |
| RSS peak (mid-stream) | 1.2 GB |
| RTF (encode + decode) | 0.06 |
| p50 chunk latency | 18.6 ms |
| p99 chunk latency | 23.4 ms |

## Memory management

Conforms to `ModelMemoryManageable`:

```swift
if memoryPressure { model.unload() }
// Later
if !model.isLoaded { /* fromLocal/fromPretrained again */ }
```

Note: in long-running batch transcription (200+ utterances in one MLModel lifetime), CoreML can exhaust the IOSurface pool and segfault. Workarounds proven in `Tests/NemotronStreamingASRTests/MultilingualBenchTests.swift`:

1. Wrap each `transcribeAudio` call in `autoreleasepool` — drains the MLMultiArrays returned by `MLModel.prediction` per utterance.
2. For 1 000+ samples in one process, also `unload` + reload the model every 50–100 utterances.

## Accuracy snapshot (FLEURS test, 50 samples per lang)

Methodology: Whisper `EnglishTextNormalizer` for English; `BasicTextNormalizer` for de/fr/ar; `BasicTextNormalizer(split_letters=True)` for hi/ja/zh/ko/th (matches NVIDIA's char-level scoring for CJK/Indic).

| lang | NVIDIA dev | fp32 NeMo (ours) | CoreML INT8 (ours) |
|------|-:|-:|-:|
| en-US | 8.27 | 9.33 | 9.59 |
| de-DE | 8.83 | 10.22 | 10.41 |
| fr-FR | 9.79 | 11.13 | 12.18 |
| ar | 12.55 | 13.27 | 13.37 |
| hi-IN | 7.41 | 5.26 | 4.42 |
| ja-JP | 12.22 | 16.97 (≈ CER 11.27) | 17.66 |

Remaining gaps are scoring methodology (NVIDIA's exact normalizer + full FLEURS test split — 600 samples per lang — would close most), not model quality. Quantization is essentially lossless: CoreML INT8 is within ±1 pp of fp32 on every language.

## Tests

```bash
make test                                          # unit + E2E (downloads HF model first time)
swift test --filter E2ENemotronMultilingualBench   # 6-lang FLEURS bench, writes /tmp/nem35-logs/wer_swift.json
swift test --filter E2EHiBisect                    # Hindi divergence diagnostic (Python diff)
```

Set `NEMOTRON_35_LOCAL_BUNDLE=/tmp/Nemotron-3.5-CoreML-320ms` to skip the HF download and use a local bundle.
