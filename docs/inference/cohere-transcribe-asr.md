# Cohere Transcribe inference

Import `CohereTranscribeASR` and load either a local export directory or a
published Hugging Face bundle:

```swift
import CohereTranscribeASR

let model = try await CohereTranscribeModel.load(
    "/path/to/Cohere-Transcribe-2B-MLX-5bit",
    offlineMode: true
)

let text = model.transcribe(
    audio: samples,
    sampleRate: sourceSampleRate,
    language: "en"
)
```

The local directory must contain `config.json`, one or more `.safetensors`
files, `tokenizer.model`, and `tokenizer_config.json`. Quantized bundles use
the standard MLX `quantization` or `quantization_config` object with `bits`,
`group_size`, and affine `mode`.

## CLI

```bash
speech transcribe recording.wav --engine cohere
speech transcribe recording.wav --engine cohere --model int8 --language de
speech transcribe recording.wav --engine cohere --model /path/to/local/export
```

The default is INT5. `--model` also accepts `int8`, `fp16`, a Hugging Face
repository ID, or a local export directory. Cohere Transcribe is currently a
non-streaming engine.

## Variants

`CohereTranscribeVariant` defines `.fp16`, `.int5`, and `.int8`. INT5 passed
the cross-precision gate and is the default candidate once the bundles are
published. INT8 is the closest supported MLX format to INT7; it prioritizes
quality over the smaller INT5 footprint.

## Decoding

Inference is deterministic greedy decoding. The prompt requests punctuation,
disables timestamps and diarization, and inserts the selected language token.
Unknown language names fall back to English. `CohereTranscribeDecodingOptions`
controls the language, generation limit, and maximum chunk duration.

Audio longer than the model's configured clip limit is processed in bounded
chunks. Each boundary is selected from the lowest-energy window in the final
five seconds of the candidate chunk, reducing mid-word cuts. The resulting
texts are joined with spaces (without an inserted space for Japanese or
Chinese), so memory is bounded by one encoder window and decoder cache rather
than total file duration.

## Validation

Unit tests cover config decoding, language aliases, preprocessing, and the
supported quantization contract. `E2ECohereTranscribeASRTests` runs a real
local bundle when `COHERE_MLX_MODEL_PATH` is set. Cross-variant quality and
performance are measured by `asr-bench --isolated`; see
[`cohere-voxtral-asr.md`](../benchmarks/cohere-voxtral-asr.md).
