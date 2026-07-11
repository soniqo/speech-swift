# WhisperASR Inference Pipeline

## Overview

```
Audio (16 kHz mono) -> MelSpectrogram CoreML -> AudioEncoder CoreML -> greedy TextDecoder CoreML -> text
```

WhisperASR provides a native speech-swift runtime for the published [`aufklarer/Whisper-Large-v3-Turbo-CoreML`](https://huggingface.co/aufklarer/Whisper-Large-v3-Turbo-CoreML) bundle. It is intended as the built-in Whisper baseline for `speech transcribe --engine whisper` and for direct Swift use through the `WhisperASR` SPM target.

## Pipeline Stages

### 1. Audio Loading and Resampling

Input audio is loaded as mono Float32 and resampled to 16 kHz when needed. The CoreML mel model expects a fixed 480,000-sample window, so shorter audio is zero-padded and longer audio is processed in 30 s chunks.

### 2. CoreML Mel Spectrogram

`MelSpectrogram.mlmodelc` converts the 30 s waveform window to Whisper log-mel features. The runtime uses CPU + GPU for this model because that matches the lower-memory WhisperKit-style placement observed during benchmarking.

### 3. CoreML Encoder

`AudioEncoder.mlmodelc` runs the Whisper encoder on CPU + Neural Engine and emits encoder embeddings for the decoder.

### 4. Language Token

If the caller passes `language`, the runtime uses the matching Whisper language token. Without a language hint, it runs one decoder step from the start-of-transcript token and selects the best language token.

### 5. Decoder Prefill and Greedy Decode

`TextDecoderContextPrefill.mlmodelc` creates the cache for language and task tokens. `TextDecoder.mlmodelc` then generates tokens one step at a time, with explicit `key_cache`, `value_cache`, cache masks, and cache-length updates supplied by Swift.

The current decoder uses greedy no-timestamp generation. It suppresses special tokens and stops on the end token, with a narrow repeated-word guard for obvious hallucination loops.

## CLI

```bash
# Default Whisper Large-v3 Turbo CoreML bundle
speech transcribe recording.wav --engine whisper

# Explicit alias
speech transcribe recording.wav --engine whisper --model turbo

# Pass a language hint
speech transcribe recording.wav --engine whisper --language de

# Use a full CoreML repo ID
speech transcribe recording.wav --engine whisper --model aufklarer/Whisper-Large-v3-Turbo-CoreML
```

Accepted short aliases:

- `default`
- `turbo`
- `whisper`
- `whisper-turbo`
- `large-v3-turbo`
- `large-v3-v20240930-turbo`
- `large-v3-v20240930_turbo`

Other short names are rejected so `--model medium` or `--model small` cannot accidentally load a non-CoreML layout. Pass a full HuggingFace repo ID for custom CoreML bundles.

## Swift

```swift
import WhisperASR

let model = try await WhisperASRModel.fromPretrained()
let text = try await model.transcribeAudio(audioSamples, sampleRate: 48_000, language: "en")
```

To retrieve the detected or hinted language:

```swift
let result = try await model.transcribeWithLanguageAsync(
    audio: audioSamples,
    sampleRate: 16_000,
    language: nil)
print(result.text)
print(result.language ?? "unknown")
```

## Benchmarking

The benchmark harness includes both the native runtime and direct WhisperKit baselines:

```bash
.build/debug/asr-bench \
  --dataset ~/Library/Caches/qwen3-speech/datasets/LibriSpeech/test-clean \
  --engines whisper-asr-turbo whisperkit-large-v3-turbo \
  --language en \
  --limit 100
```

The latest quick slice is recorded in [docs/benchmarks/asr-wer.md](../benchmarks/asr-wer.md#whisperasr-native-runtime-comparison). Use `--isolated` in release builds before publishing cross-engine memory numbers because CoreML and MLX caches otherwise share a process high-water mark.

## Cache and Offline Use

`fromPretrained()` downloads the CoreML bundle into the standard speech-swift HuggingFace cache. Use `cacheDir:` to override the location and `offlineMode: true` once the model files are already present:

```swift
let model = try await WhisperASRModel.fromPretrained(
    offlineMode: true)
```

The runtime also ensures the tokenizer metadata is present. If the CoreML bundle cache does not include tokenizer files, it fetches them from `openai/whisper-large-v3` unless offline mode is enabled.

## Limitations

- No word timestamps yet.
- No temperature fallback or beam-search policies yet.
- No WhisperKit VAD/chunk seeking parity yet.
- 30 s fixed mel windows define the current chunking behavior.

See [WhisperASR CoreML](../models/whisper-asr.md) for model layout and architecture notes.
