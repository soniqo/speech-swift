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

## CLI

```bash
# Transcribe with Parakeet
.build/release/audio transcribe --engine parakeet recording.wav

# Default engine (Qwen3-ASR)
.build/release/audio transcribe recording.wav
```

## Performance (M2 Max)

| Metric | Value |
|--------|-------|
| RTF (after warmup) | ~0.03 |
| Cold start | ~3s (CoreML compilation) |
| Warm inference (20s audio) | ~0.6s |
| Backend | Neural Engine (CoreML) |

## Model Architecture Reference

See [docs/models/parakeet-asr.md](../models/parakeet-asr.md) for detailed architecture documentation including FastConformer layer dimensions, TDT vocabulary, and CoreML conversion details.
