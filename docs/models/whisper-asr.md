# WhisperASR CoreML

WhisperASR runs Whisper Large-v3 Turbo through speech-swift's native CoreML runtime. The default model bundle is [`aufklarer/Whisper-Large-v3-Turbo-CoreML`](https://huggingface.co/aufklarer/Whisper-Large-v3-Turbo-CoreML).

## Model

| Field | Value |
|-------|-------|
| Family | Whisper Large-v3 Turbo |
| Default repo | `aufklarer/Whisper-Large-v3-Turbo-CoreML` |
| Tokenizer source | `openai/whisper-large-v3` |
| Backend | CoreML |
| Precision | FP16 |
| Input audio | 16 kHz mono Float32 |
| Chunk size | 30 s fixed CoreML mel input |
| Languages | Whisper multilingual language set |

The model ships as a split CoreML bundle:

| CoreML model | Default compute units | Purpose |
|--------------|-----------------------|---------|
| `MelSpectrogram.mlmodelc` | CPU + GPU | 30 s waveform to Whisper log-mel features |
| `AudioEncoder.mlmodelc` | CPU + Neural Engine | Mel features to encoder embeddings |
| `TextDecoderContextPrefill.mlmodelc` | CPU + Neural Engine | Initial decoder cache for language/task tokens |
| `TextDecoder.mlmodelc` | CPU + Neural Engine | Autoregressive token generation with explicit KV-cache updates |

## Runtime Scope

The runtime is implemented in `Sources/WhisperASR` and loads the CoreML sub-models directly. It does not call WhisperKit for inference.

Current behavior:

- Greedy no-timestamp decoding.
- Optional language hint, otherwise Whisper language-token detection.
- Byte-level tokenizer decode from the model bundle.
- 30 s chunking with per-chunk autorelease pools.
- Repeated-word stop guard for obvious greedy hallucination loops.

Not yet implemented:

- Word timestamps and timestamp-token decoding.
- Temperature fallback and beam-search policies.
- WhisperKit's long-form VAD/chunk seeking heuristics.

## Swift API

```swift
import WhisperASR

let model = try await WhisperASRModel.fromPretrained()
let text = try await model.transcribeAudio(samples, sampleRate: 16000, language: "en")
```

The default model can be overridden with a full HuggingFace repo ID:

```swift
let model = try await WhisperASRModel.fromPretrained(
    modelId: "aufklarer/Whisper-Large-v3-Turbo-CoreML")
```

## CLI

```bash
speech transcribe recording.wav --engine whisper
speech transcribe recording.wav --engine whisper --model turbo --language en
```

Accepted short model aliases are `default`, `turbo`, `whisper`, `whisper-turbo`, `large-v3-turbo`, and `large-v3-v20240930_turbo`. A full HuggingFace repo ID is also accepted.

## Benchmark

Matched LibriSpeech test-clean benchmark on Apple M5 Pro, 48 GB, macOS 26.5.2, release build, 2026-07-16. Both engines ran in isolated processes over the same first 200 utterances. The table shows the WhisperKit-first run; a reverse-order replication preserved the native speed lead (0.077 versus 0.085 mean RTF and 17.6x versus 16.0x overall throughput):

| Engine | WER% | CER% | Mean RTF | Median RTF | Overall xRT | Load | Peak RSS | RSS Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WhisperASR native | 1.73 | 0.63 | **0.077** | **0.064** | **17.6x** | **1.3s** | 445 MB | +304 MB |
| Direct WhisperKit | **1.71** | **0.53** | 0.084 | 0.072 | 16.0x | 7.2s | **427 MB** | +286 MB |

Native WhisperASR has 9.4% lower mean RTF and 5.5x faster cached loading. Accuracy is one word error apart across 4,675 reference words, and peak RSS is within 18 MB.

See [ASR WER benchmark](../benchmarks/asr-wer.md#whisperasr-native-runtime-comparison) for methodology, caveats, and the wider engine table.

## Model Weights

- [`aufklarer/Whisper-Large-v3-Turbo-CoreML`](https://huggingface.co/aufklarer/Whisper-Large-v3-Turbo-CoreML) - CoreML bundle used by `WhisperASRModel.defaultModelId`.
- [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) - tokenizer fallback source.

## Thread Safety

`WhisperASRModel` owns CoreML models and per-call decoder state. Concurrent callers should create separate model instances until shared-instance concurrency is explicitly validated.
