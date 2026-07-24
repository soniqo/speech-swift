# Nemotron-3.5 ASR Streaming — inference

Swift bindings for the Nemotron-3.5 multilingual streaming ASR bundle
(Core ML or native MLX, 600 M parameters, 40 language-locales). Lives in
`Sources/NemotronStreamingASR/`.

## Core ML API

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

## Native MLX INT5 and INT8

`NemotronStreamingASRMLXModel` runs the quantized encoder, prompt kernel,
RNN-T predictor, and joint network directly with MLX. INT5 is the default
because it saves about 193 MB on disk and 192 MB of peak RSS versus INT8,
while the six-language streaming benchmark costs 0.52 percentage points of
mean WER. Select INT8 when the small accuracy gain matters more than memory.

```swift
import NemotronStreamingASR

// Downloads the 538.6 MB INT5 bundle by default.
let model = try await NemotronStreamingASRMLXModel.fromPretrained(
    variant: .int5
)

let session = try model.createSession(language: "en-US")
for chunk in microphoneChunks16kHz {
    for partial in try session.pushAudio(chunk) {
        print(partial.text)
    }
}
let final = try session.finalize().last?.text
```

Use `.int8` for the 732.6 MB release, or load an already downloaded bundle:

```swift
let int8 = try await NemotronStreamingASRMLXModel.fromPretrained(
    variant: .int8
)

let local = try await NemotronStreamingASRMLXModel.fromDirectory(
    URL(fileURLWithPath: "/path/to/Nemotron-MLX-5bit")
)
```

Both variants require the bundle-declared 320 ms geometry. The model may
serve multiple source-local sessions, but it serializes model creation,
`pushAudio`, and `finalize` inference because the sessions share one set of
resident MLX weights. Each session keeps independent mel, attention,
convolution, language, and RNN-T state.

The MLX API currently uses greedy decoding only. Word boosting remains a
Core ML API feature.

## Word boosting (Core ML)

Nemotron word boosting biases RNN-T decoding toward caller-provided words or phrases. It is not transcript post-processing: the decoder adjusts token scores after the joint network produces logits and before greedy token selection.

Use it for custom vocabulary that the model might otherwise miss, such as product names, speaker names, acronyms, project names, or domain terms.

This is RNN-T shallow fusion over the joint logits. NVIDIA's stricter CTC-WS context-biasing method requires CTC log probabilities from a CTC or hybrid RNN-T/CTC checkpoint; the Nemotron 3.5 streaming checkpoint is RNNT-only and has no CTC head, so that path is not available for this model.

For a small list where one strength is good enough:

```swift
let boosted = WordBoostingConfig(
    phrases: ["AcmeCloud", "customer portal", "release train"],
    boost: 0.75
)

let text = try model.transcribeAudio(
    audio,
    sampleRate: 16000,
    language: "en-US",
    wordBoosting: boosted
)
```

For mixed custom vocabulary, prefer per-phrase boosts. This lets easy terms use a lower boost while fragmented or out-of-vocabulary terms use a stronger one:

```swift
let boosted = WordBoostingConfig(phrases: [
    .init("AcmeCloud", boost: 0.75),
    .init("XQ-17", boost: 1.25),
    .init("Veltrix", boost: 1.25),
])
```

The same config works with streaming sessions:

```swift
let session = try model.createSession(
    language: "en-US",
    wordBoosting: boosted
)
```

Word boosting uses the real SentencePiece Unigram tokenizer (shipped as `tokenizer.model` alongside `vocab.json` in the `aufklarer/Nemotron-3.5-ASR-Streaming-0.6B-CoreML-INT8` bundle) to produce the same canonical token path the decoder emits. The phrase text is treated as canonical, including casing. Older bundles that ship only `vocab.json` fall back to greedy longest-match segmentation — this fallback is narrower and silently no-ops on fragmented OOV terms because the phrase token IDs diverge from what the decoder actually emits.

Check which tokenizer mode loaded at runtime:

```swift
switch model.wordBoostingTokenizerStatus.mode {
case .sentencePieceModel:
    // tokenizer.model found (default for current bundles); boost paths
    // match the decoder's segmentation and a configured boost reliably
    // changes greedy decisions when the matcher advances.
    break
case .vocabFallback:
    // Older bundle without tokenizer.model. Greedy vocab.json
    // segmentation only — boosting OOV brand/technical terms may
    // silently no-op because the phrase token IDs diverge from the
    // decoder's output IDs. Re-download or upgrade the bundle.
    break
}
```

### Batch boost suggestions

Use `wordBoostingSuggestions(for:)` to estimate a reasonable per-phrase boost from the same tokenization logic the decoder will use:

```swift
let suggestions = model.wordBoostingSuggestions(for: [
    "AcmeCloud",
    "XQ-17",
    "Veltrix",
    "release train",
])

for suggestion in suggestions {
    print(
        suggestion.phrase,
        suggestion.suggestedBoost,
        suggestion.difficulty,
        suggestion.reason
    )
}
```

This API is batch-oriented so callers can analyze a whole custom vocabulary list once, cache the results, and avoid carrying Nemotron-specific tokenization heuristics in app code.

Suggestions can be fed directly into a per-phrase config:

```swift
let boosted = WordBoostingConfig(
    phrases: suggestions.map {
        .init($0.phrase, boost: $0.suggestedBoost)
    }
)
```

### Choosing boost strength

Word boosting is a nudge, not a replacement rule. A phrase only wins when its token path is plausible enough in the acoustic context.

| boost | Use |
|------:|-----|
| 0.25-0.5 | Very gentle bias for common words or large phrase lists |
| 0.75 | Suggested starting point for easy custom vocabulary |
| 0.95 | Moderate bias for product names, people, and uncommon technical terms |
| 1.25 | Highest automatic suggestion for fragmented or unencodable phrases (4+ pieces or 3+ pieces per word). Short phrases (≤4 characters) only get this boost when they fragment — a short term the vocabulary already covers in a single piece (e.g. `AI`, `iOS`, `kHz`) is treated as easy. |
| 5.0+ | Diagnostic/risky; can force wrong words into the transcript |

For example, if the speaker says a product name but the baseline often hears nearby common words, try:

```swift
WordBoostingConfig(phrases: ["ExampleProduct"], boost: 0.75)
```

If a rare proper noun is still missed, increase gradually:

```swift
WordBoostingConfig(phrases: ["XQ-17"], boost: 1.25)
```

Some out-of-vocabulary product names are harder than others. If a word is split into many tiny BPE pieces, the first boosted token may be generic and the decoder may need a stronger nudge before the phrase-prefix state can help. For example, a made-up product name might have no single token and split into several character-like pieces, so it may need a higher boost than a name with larger pieces.

Avoid using very high boost values as a general setting. They increase the chance of forcing phonetically nearby phrases into clean audio.

Shallow fusion cannot fully prevent every false acceptance. If a boosted phrase is short, acoustically close to common words, or begins with very generic tokens, keep its boost low, keep the context list small, and prefer an app-level post-processing replacement layer for heard-phrase-to-replacement workflows.

### Good and bad uses

Good:

- Biasing real terms expected in the audio: product names, project codes, customer names, or domain-specific acronyms.
- Multi-word phrases that should stay together: `"replacement part"`, `"speech swift"`.
- Per-session context, such as names in the current meeting or terms from the active document.

Bad:

- Replacing text after recognition. Use a normal text replacement layer for that.
- Large unrelated word lists. They increase the chance of false positives.
- Very high global boost values. They can hallucinate boosted words into unrelated speech.

## Language tags

`language` is a BCP-47 tag (`en-US`, `de-DE`, `fr-FR`, `ja-JP`, `hi-IN`, ...) resolved against `languages.json` via `NemotronLanguages.slot(for:)`. Resolution order:

1. Exact match (`en-US` → slot 0)
2. Underscore → hyphen normalization (`en_US` → `en-US`)
3. Base language fallback (`en-GB` → `en`)
4. `"auto"` slot (101)

The full list of 121 aliases is in the `languages` property of the loaded model.

## Core ML loading

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

The Core ML loader expects the standard bundle layout described in
`docs/models/nemotron-asr-streaming.md`. Compute units default to `.all`
(encoder on ANE + GPU + CPU, decoder/joint on CPU) — measured about 40%
faster RTF than `.cpuAndGPU` on M-series.

## Streaming geometry

| Param | Default |
|-------|---------|
| Chunk | 320 ms (4 output frames × 80 ms) |
| Right context | 3 frames (240 ms lookahead) |
| Left context | 56 frames (4.48 s) |
| Pre-cache | 9 mel frames |
| Sample rate | 16 kHz mono Float32 |

`transcribeStream` slices audio at the configured chunk size; `pushAudio` accepts any chunk size and buffers internally until enough samples accumulate.
Published MLX bundles accept only the validated 320 ms row. Alternate chunk
geometries are research/export options, not a runtime override for those
weights.

## TTS round-trip and short-audio padding

Core ML `transcribeAudio` adds 100 ms of silence at both ends by default to
prime the all-zero streaming cache (otherwise short TTS clips with sharp
onsets can drop the first word). For natural recorded speech (FLEURS-style
microphone capture), pass `padSilence: false` to skip the pad — measured
about 0–5 pp WER drop on the harder languages. The MLX API does not add this
Core ML compatibility padding.

```swift
// TTS-style (short clip, padding helps): default true
let ttsText = try model.transcribeAudio(ttsAudio, sampleRate: 24000, language: "en-US")

// Recorded speech (no padding shift):
let recText = try model.transcribeAudio(micAudio, sampleRate: 16000,
                                        language: "en-US", padSilence: false)
```

## Memory + throughput on M5 Pro

60 s synthetic FLEURS en_us long-form streamed in 320 ms chunks (compute units `.all`):

| runtime | bundle | RSS peak | streaming RTF | p50 / p95 / p99 chunk |
|---------|-------:|---------:|--------------:|----------------------:|
| MLX INT5 | 538.6 MB | 800 MB | 0.0467 | 13.8 / 15.9 / 16.8 ms |
| MLX INT8 | 732.6 MB | 992 MB | 0.0485 | 14.3 / 16.2 / 18.9 ms |
| Core ML INT8 | 612 MB | 1.2 GB | 0.068 | 18.6 / — / 23.4 ms |

The native Swift release regression gate uses `asr-bench --isolated` and a
20-second English fixture. Across three fresh-process runs per variant, INT5
had median RTF 0.0360 and 611 MB peak RSS; INT8 had median RTF 0.0375 and
805 MB peak RSS. Both reproduced the 11-word reference exactly. See
`docs/benchmarks/nemotron-asr-streaming.md` for methodology and the
reproduction command.

## Memory management

The Core ML model conforms to `ModelMemoryManageable`:

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

The release MLX benchmark exercises the actual cache-aware 320 ms path
(50 samples per language):

| lang | MLX INT5 WER / CER | MLX INT8 WER / CER |
|------|-------------------:|-------------------:|
| en-US | 8.64 / 3.77 | 8.98 / 3.96 |
| de-DE | 11.15 / 6.25 | 10.59 / 5.73 |
| fr-FR | 13.10 / 5.18 | 11.83 / 4.66 |
| ar | 13.66 / 3.94 | 13.37 / 3.77 |
| hi-IN | 4.77 / 3.85 | 4.28 / 3.50 |
| ja-JP | 17.86 / 12.11 | 17.01 / 11.42 |
| **Mean** | **11.53 / 5.85** | **11.01 / 5.51** |

## Tests

```bash
make test                                          # unit + E2E (downloads HF model first time)
swift test --filter WordBoosting                  # word boosting unit tests + E2E A/B smoke test
swift test --filter NemotronMLXTests              # model-free MLX loader/config tests
NEMOTRON_MLX_LOCAL_BUNDLE=/path/to/int5 \
NEMOTRON_MLX_INT8_LOCAL_BUNDLE=/path/to/int8 \
  swift test --filter E2ENemotronMLXTests         # native MLX streaming E2E
NEMOTRON_MLX_LOCAL_BUNDLE=/path/to/int5 \
NEMOTRON_MLX_INT8_LOCAL_BUNDLE=/path/to/int8 \
  .build/release/asr-bench \
    --dataset /path/to/benchmark.tsv \
    --engines nemotron-mlx-int5 nemotron-mlx-int8 \
    --language en-US --isolated                   # WER + RTF + peak RSS
swift test --filter E2ENemotronMultilingualBench   # 6-lang FLEURS bench, writes /tmp/nem35-logs/wer_swift.json
swift test --filter E2EHiBisect                    # Hindi divergence diagnostic (Python diff)
```

Set `NEMOTRON_35_LOCAL_BUNDLE=/tmp/Nemotron-3.5-CoreML-320ms` to skip the HF download and use a local bundle.
