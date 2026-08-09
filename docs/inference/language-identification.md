# Language Identification

## CLI

```bash
speech language-id recording.wav
speech language-id recording.wav --engine coreml --top 3
speech language-id recording.wav --json
speech language-id recording.wav --model organization/custom-export
```

The default backend is MLX. `--model` can override the Hugging Face model ID,
and `--top` selects how many of the 107 closed-set labels are returned.
Human-readable output includes ranked language names, upstream codes,
probabilities, analyzed duration, window count, and inference time. `--json`
emits the same result as structured JSON.

## Swift API

```swift
import SpeechLanguageID

let identifier = try await SpeechLanguageIdentifier.fromPretrained(
    engine: .mlx
)
let result = try identifier.identify(
    audio: samples,
    sampleRate: sampleRate,
    topK: 5
)

if let best = result.best {
    print("\(best.label.name) [\(best.label.code)]: \(best.probability)")
}
```

Use `.coreML` to load the compiled Core ML export instead. For local artifacts,
`SpeechLanguageIdentifier.fromLocal(directory:engine:)` strictly validates the
configuration, label order, artifact name, and complete MLX parameter set.

`fromPretrained` downloads the selected export on first use and accepts
`cacheDir:`, `offlineMode:`, and a progress callback. The default public models
are:

- MLX: [`aufklarer/SpeechBrain-ECAPA-VoxLingua107-21M-MLX`](https://huggingface.co/aufklarer/SpeechBrain-ECAPA-VoxLingua107-21M-MLX)
- Core ML: [`aufklarer/SpeechBrain-ECAPA-VoxLingua107-21M-CoreML`](https://huggingface.co/aufklarer/SpeechBrain-ECAPA-VoxLingua107-21M-CoreML)

## Long recordings

The runtime analyzes up to roughly 30 seconds per model call. Longer audio is
split into non-overlapping windows; class probabilities are averaged with each
window's analyzed duration as its weight. The result exposes both
`analyzedDuration` and `windowCount`.

## Interpreting results

Probabilities only compare labels inside the model's fixed inventory. They are
not calibrated evidence that the recording contains a supported language.
Applications that may receive silence, music, unknown languages, non-native
speech, or code-switching should add VAD and an independently evaluated reject
policy before presenting a definitive label.

Do not compare confidence thresholds across backends without calibration. FP16
conversion preserves all representative top-1 outputs, but small probability
differences remain and can matter near a product rejection threshold.
