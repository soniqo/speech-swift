# Parakeet TDT ASR Inference Pipeline

## Overview

```
Audio (16kHz) → Mel (vDSP) → CoreML Encoder (Neural Engine) → TDT Greedy Decode → Text
```

Parakeet TDT is a CTC-variant ASR model running entirely on CoreML. The encoder runs on the Neural Engine, freeing the GPU for concurrent MLX workloads (e.g., TTS generation or LLM inference). This makes it well-suited for voice pipeline scenarios where ASR and TTS run in parallel.

## Pipeline Stages

### 1. Audio Preprocessing

Raw 16kHz audio is converted to 80-dim log mel features using vDSP (Accelerate framework). This is the same DSP stack used across the project — no Python dependencies or external feature extractors.

### 2. CoreML Encoder

The FastConformer encoder runs on the Apple Neural Engine via CoreML. The encoder produces per-frame logits over the SentencePiece vocabulary plus TDT duration tokens.

- **First inference**: ~3s due to CoreML model compilation and ANE scheduling. The compiled model is cached by CoreML for subsequent runs.
- **Subsequent inference**: ~0.6s for 20s of audio (RTF ~0.03).

### 3. TDT Greedy Decode

Token-and-Duration Transducer (TDT) decoding interprets the encoder output. Unlike standard CTC which emits one token per frame, TDT predicts both the token and a duration (number of frames to skip). This reduces redundant computation and improves accuracy on long-form audio.

Greedy decoding selects the highest-probability token and duration at each step. No beam search or language model rescoring is applied.

### 4. Confidence Score

The pipeline returns an overall confidence score (0.0 to 1.0) computed as the sigmoid-scaled mean of token logits. This provides a rough estimate of transcription reliability — useful for deciding whether to retry with a different model or prompt the user.

## Language Support

Parakeet TDT supports 25 European languages via its SentencePiece vocabulary. The model handles multilingual audio without explicit language selection.

## Thread Safety

The model is **not thread-safe** due to internal LSTM state in the encoder. If concurrent transcription is needed, create separate model instances. Do not share a single instance across threads or async tasks.

## Warmup

For latency-sensitive applications (voice pipelines), call a warmup inference with a short silent audio buffer during app initialization. This triggers CoreML compilation on the Neural Engine so that the first real inference runs at full speed:

```swift
let model = try await ParakeetASR.fromPretrained()
// Warmup: triggers ANE compilation (~3s, runs once)
let _ = try model.transcribe(audio: silentBuffer, sampleRate: 16000)
// Subsequent calls: ~0.6s for 20s audio
```

## Encoder Variant Selection (multi-encoder repo)

`aufklarer/Parakeet-TDT-v3-CoreML-INT8` ships multiple single-shape encoders in one repo (`encoder.mlmodelc` = 30s, `encoder_5s.mlmodelc`, `encoder_15s.mlmodelc`). Pick the variant matching your typical input length to cut per-call ANE bandwidth:

```swift
// Short voice-pipeline chunks (≤5s): ~6× less weight bandwidth per call than the 30s encoder
let model = try await ParakeetASRModel.fromPretrained(
    modelId: "aufklarer/Parakeet-TDT-v3-CoreML-INT8",
    encoderVariant: "5s")
```

`encoderVariant: nil` (default) loads `encoder.mlmodelc`, so the single-shape `-30s` and `-iOS-5s` repos work unchanged.

## Compute Units

| Platform | Default | Fallback |
|----------|---------|----------|
| iOS Simulator | `.cpuOnly` | — (only path that returns non-zero tokens; the simulator's Espresso lacks an MPSGraph backend) |
| iOS device | `.cpuAndNeuralEngine` | `.cpuAndGPU` |
| macOS | `.cpuAndNeuralEngine` | `.cpuAndGPU` |

macOS was historically pinned to `.cpuAndGPU` because the iOS17-target **EnumeratedShapes** INT8 encoder SIGSEGV'd in `bnns::GraphCompile` on Mac ANE (surfaced on M5, see issue #313). Single-shape builds aren't affected: the shipped `-30s` and `-iOS-5s` repos (iOS17 single-shape) ANE-compile cleanly on M5; the legacy multi-encoder `-INT8` repo was rebuilt with iOS18 single-shape variants since EnumeratedShapes can't be re-emitted on iOS18 either (MIL validator rejects the dynamic `tile` reps). So macOS now defaults to ANE and falls back to GPU only if loading throws.

## CLI

```bash
# Transcribe with Parakeet
.build/release/speech transcribe --engine parakeet recording.wav

# Default engine (Qwen3-ASR)
.build/release/speech transcribe recording.wav
```

## Performance (M5 Pro, macOS 26.5)

Encoder-only forward, warm best of 5, swift JIT process:

| Shape | Compute | Warm encode | Encoder RTF | Peak RSS |
|-------|---------|------|-----|----------|
| 3000 (30s) | ANE | 74 ms | 0.0025 | 820 MB |
| 1500 (15s) | ANE | 24 ms | 0.0016 | 784 MB |
| 500  (5s)  | ANE |  8 ms | 0.0016 | 765 MB |

End-to-end with TDT decode (release build of `speech transcribe` on the bundled 20s fixture, default `-30s` model, ANE): warm transcribe 0.10 s (RTF 0.005), peak memory footprint 857 MB.

Cold start (first call) includes BNNS graph compile on ANE — 5 s for the 5s variant, ~11 s for 30s. CoreML caches the compile to disk so subsequent process launches skip it.

## Model Architecture Reference

See [docs/models/parakeet-asr.md](../models/parakeet-asr.md) for detailed architecture documentation including FastConformer layer dimensions, TDT vocabulary, and CoreML conversion details.
