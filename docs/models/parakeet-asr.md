# Parakeet TDT ASR

CoreML-based automatic speech recognition using NVIDIA's Parakeet-TDT 0.6B v3.

## Architecture

**FastConformer Encoder**: 24-layer conformer with depthwise-separable convolutions and 8x subsampling. Input: 128-dim mel spectrogram at 16kHz. Output: 1024-dim encoded representations.

**TDT Decoder**: Token-and-Duration Transducer — an extension of RNN-T that adds a duration prediction head alongside the standard token prediction head.

- **Prediction network**: 2-layer LSTM (640 hidden) predicts from previously emitted tokens
- **Joint network**: Combines encoder and prediction outputs, produces dual logits:
  - Token logits (8193 classes: 8192 SentencePiece tokens + 1 blank)
  - Duration logits (5 classes: `[0, 1, 2, 3, 4]` frames to advance)

### TDT Decoding Algorithm

Standard RNN-T always advances 1 encoder frame on blank. TDT enhances this:

```
t = 0
while t < encoded_length:
    (token_logits, dur_logits) = joint(encoder[t], decoder_state)
    token = argmax(token_logits)
    if token == blank:
        t += 1
    else:
        emit(token)
        duration = duration_bins[argmax(dur_logits)]
        t += max(duration, 1)
        decoder_state = lstm(token, decoder_state)
```

This variable-rate alignment is more accurate and efficient than fixed-rate RNN-T.

## CoreML Models

The model is split into 3 CoreML sub-models for optimal compute unit placement:

| Model | Compute | Quantization | Purpose |
|-------|---------|--------------|---------|
| `encoder.mlmodelc` | CPU + Neural Engine | INT8 palettized | Mel → 1024-dim encoded features |
| `decoder.mlmodelc` | CPU + Neural Engine | None | LSTM prediction network |
| `joint.mlmodelc` | CPU + Neural Engine | None | Dual-head token + duration logits |

Default model (`aufklarer/Parakeet-TDT-v3-CoreML-INT8-30s`):
- **INT8** palettized encoder — ~50% size reduction, best quality/speed balance
- **Single fixed shape** (3000 mel frames = 30s), not EnumeratedShapes — so it loads on any CoreML compute unit (CPU-only, GPU, or Neural Engine) without the heavy multi-shape compile. Audio longer than 30s is window-chunked in `transcribeAudio`. The `aufklarer/Parakeet-TDT-v3-CoreML-INT8-iOS-5s` variant (fixed 500 frames = 5s) trades a smaller window for lower runtime memory on iOS.

Mel preprocessing (pre-emphasis, STFT, mel filterbank, normalization) is done in Swift using Accelerate/vDSP — no CoreML preprocessor model needed. Decoder and joint are small enough that quantization isn't necessary.

## Audio Preprocessing

- Sample rate: 16kHz
- Mel bins: 128
- n_fft: 512, hop: 160, win: 400
- Window: Hanning
- Pre-emphasis: 0.97 (applied in Swift via vDSP)

## CLI Usage

```bash
# Transcribe with Parakeet (CoreML, fast on Neural Engine)
speech transcribe --engine parakeet audio.wav

# Transcribe with Qwen3 (MLX, default)
speech transcribe audio.wav
speech transcribe --engine qwen3 audio.wav
```

## Performance

On Apple Silicon with Neural Engine (M2 Max, 20s audio):
- Cold: RTF ~0.12 (first inference includes CoreML compilation)
- Warm: RTF ~0.03 (subsequent inferences, ~32x real-time)
- Mel preprocessing uses vectorized Accelerate/vDSP (pre-emphasis, normalization)
- Decode loop uses memcpy, memset, and vDSP argmax to minimize scalar overhead
- Encoder runs on Neural Engine for maximum throughput

## Model Weights

- [aufklarer/Parakeet-TDT-v3-CoreML-INT8-30s](https://huggingface.co/aufklarer/Parakeet-TDT-v3-CoreML-INT8-30s) — default (single fixed 30s shape)
- [aufklarer/Parakeet-TDT-v3-CoreML-INT8-iOS-5s](https://huggingface.co/aufklarer/Parakeet-TDT-v3-CoreML-INT8-iOS-5s) — iOS (single fixed 5s shape, lower memory)

## Thread Safety

- `ParakeetConfig` — `Sendable` (immutable value type)
- `ParakeetVocabulary` — `Sendable` (immutable dictionary)
- `ParakeetASRModel` — NOT thread-safe (LSTM decoder state). Create separate instances for concurrent use.
- CoreML `MLModel.prediction()` is thread-safe for stateless models, but the TDT decode loop manages LSTM h/c state as local variables (safe per-call, not re-entrant).
