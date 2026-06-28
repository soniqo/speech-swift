# Silero VAD: Streaming Voice Activity Detection

## Overview

Silero VAD is a lightweight (~309K params) voice activity detection model designed for real-time streaming. speech-swift defaults both the MLX and CoreML backends to v6.2.1 exports. Both paths process 512-sample audio chunks (32ms @ 16kHz), output a speech probability between 0 and 1, and carry LSTM state across chunks.

```
┌─────────────────────────────────────────────────────────────────┐
│  SileroVADModel                                                 │
│                                                                 │
│  Input: 512 samples (32ms @ 16kHz)                              │
│    │                                                            │
│    ├── Prepend 64-sample context from previous chunk → 576      │
│    │                                                            │
│    ├── ReflectionPad(right=64) → 640 samples                    │
│    ├── STFT Conv1d(1→258, k=256, s=128) → 4 frames × 258       │
│    ├── Magnitude: √(real² + imag²) → 4 × 129                   │
│    │                                                            │
│    ├── Encoder:                                                 │
│    │   Conv1d(129→128, k=3, s=1, p=1) → 4 × 128   ─ ReLU      │
│    │   Conv1d(128→ 64, k=3, s=2, p=1) → 2 ×  64   ─ ReLU      │
│    │   Conv1d( 64→ 64, k=3, s=2, p=1) → 1 ×  64   ─ ReLU      │
│    │   Conv1d( 64→128, k=3, s=1, p=1) → 1 × 128   ─ ReLU      │
│    │                                                            │
│    ├── LSTM(128→128, 1 layer) ─── h,c state carried ──→        │
│    │                                                            │
│    ├── Decoder: ReLU → Conv1d(128→1, k=1) → Sigmoid            │
│    │                                                            │
│  Output: speech probability [0, 1]                              │
│  State: h (1,1,128), c (1,1,128), context (64 samples)         │
└─────────────────────────────────────────────────────────────────┘
```

## STFT

The STFT is implemented as a Conv1d with a pre-computed DFT basis matrix stored as weights (no learnable parameters). At inference it's a single convolution followed by magnitude extraction:

1. **Pad**: Reflect 64 samples on the right only (`F.pad(x, [0, 64], "reflect")`)
2. **Conv1d**: 258 output channels (129 real + 129 imaginary), kernel=256, stride=128
3. **Magnitude**: Split channels at 129, compute `√(real² + imag²)` → 129 frequency bins × 4 frames

The asymmetric right-only padding (not symmetric) is critical — it produces 4 STFT frames from 576 input samples, which the encoder reduces to a single feature vector.

## Encoder

Four Conv1d layers with ReLU activations progressively reduce the temporal dimension from 4 frames to 1:

| Layer | Channels | Kernel | Stride | Padding | Output Frames |
|-------|----------|--------|--------|---------|---------------|
| 0 | 129 → 128 | 3 | 1 | 1 | 4 |
| 1 | 128 → 64 | 3 | 2 | 1 | 2 |
| 2 | 64 → 64 | 3 | 2 | 1 | 1 |
| 3 | 64 → 128 | 3 | 1 | 1 | 1 |

The weights use reparameterized convolutions (`reparam_conv`) — at inference, the separate conv+bn branches are fused into a single convolution.

## LSTM

A single-layer unidirectional LSTM with hidden size 128. Receives a single timestep from the encoder and updates its hidden/cell state. The state is carried across chunks for streaming — this is what gives the model temporal context beyond the 32ms window.

The decoder uses only the LSTM hidden state `h` (not the full sequence output), reshaped to `[B, 1, 128]` for the final Conv1d.

## Decoder

Applies ReLU, then a 1×1 convolution (`Conv1d(128→1, k=1)`) followed by sigmoid to produce a probability in [0, 1].

## StreamingVADProcessor

The `StreamingVADProcessor` wraps `SileroVADModel` with a four-state machine for event-driven speech detection:

```
                onset crossed
    ┌─────────┐ ─────────────→ ┌──────────────┐
    │ silence │                │ pendingSpeech │
    └─────────┘ ←───────────── └──────────────┘
                 offset crossed       │
                 (too brief)          │ duration ≥ minSpeechDuration
                                      │ → emit speechStarted
                                      ▼
    ┌────────────────┐         ┌─────────┐
    │ pendingSilence │ ←────── │ speech  │
    └────────────────┘ offset  └─────────┘
           │          crossed         ▲
           │                          │
           │ onset crossed            │
           │ (speech resumed) ────────┘
           │
           │ silence ≥ minSilenceDuration
           │ → emit speechEnded
           ▼
    ┌─────────┐
    │ silence │
    └─────────┘
```

**States:**
- **silence** — Waiting for probability to cross onset threshold
- **pendingSpeech** — Onset crossed, waiting for `minSpeechDuration` before confirming
- **speech** — Confirmed speech, `speechStarted` event emitted
- **pendingSilence** — Offset crossed, waiting for `minSilenceDuration` before ending

## Configuration

```swift
// Default Silero thresholds (VADConfig.sileroDefault)
onset:            0.5    // Speech starts when prob ≥ 0.5
offset:           0.35   // Speech ends when prob < 0.35
minSpeechDuration: 0.25  // Ignore speech shorter than 250ms
minSilenceDuration: 0.1  // Ignore silence gaps shorter than 100ms
```

## Weight Conversion

Conversion scripts in `scripts/` handle weight format differences (PyTorch → MLX channels-last transpose, LSTM bias fusion). Both MLX and CoreML weights are hosted on HuggingFace and downloaded automatically.

The CoreML v6.2.1 repository also publishes a `silero_vad_256ms.mlmodelc` aggregate bundle for separate benchmarking. The Swift runtime currently loads the streaming `silero_vad.mlmodelc` bundle so chunk timing and existing APIs remain unchanged.

## CoreML Backend

Silero VAD supports a CoreML backend (`engine: .coreml`) that runs on the Neural Engine + CPU while freeing the GPU.

```swift
let vad = try await SileroVADModel.fromPretrained(engine: .coreml)
// Same API — processChunk(), detectSpeech(), resetState()
```

The CoreML model uses float16 I/O with LSTM h/c state carried as `MLMultiArray` between chunks. Input shape: `[1, 1, 576]` (context + chunk).

| Backend | Per-chunk Latency | Hardware | Model |
|---------|------------------|----------|-------|
| MLX | ~0.50ms | GPU (Metal) | [aufklarer/Silero-VAD-v6.2.1-MLX](https://huggingface.co/aufklarer/Silero-VAD-v6.2.1-MLX) |
| CoreML | ~0.06ms | Neural Engine + CPU | [aufklarer/Silero-VAD-v6.2.1-CoreML](https://huggingface.co/aufklarer/Silero-VAD-v6.2.1-CoreML) |

## CoreML v6.2.1 Benchmark

On the Mini50 VAD benchmark, the new 32ms CoreML export improves span recall while keeping very high precision. It is the runtime default because it preserves the existing 32ms streaming API.

| Model | File F1 | Span F1 | Precision | Recall | FAR | Speed |
|-------|---------|---------|-----------|--------|-----|-------|
| v5 CoreML baseline | 100.00% | 86.30% | 100.00% | 75.91% | 0.00% | 505.0x RT |
| v6.2.1 CoreML 32ms | 98.04% | 87.47% | 99.98% | 77.74% | 3.44% | 412.4x RT |
| v6.2.1 CoreML 256ms | 92.59% | 93.88% | 99.97% | 88.49% | 7.80% | 1442.2x RT |

## Comparison with Pyannote VAD

| | Pyannote (PyanNet) | Silero VAD |
|---|---|---|
| **Parameters** | ~1.49M | ~309K |
| **Download** | ~5.7 MB | ~1.2 MB |
| **Processing** | 10s sliding windows (batch) | 32ms chunks (streaming) |
| **Architecture** | SincNet → BiLSTM(4L) → Linear → Softmax | STFT → Conv encoder → LSTM → Sigmoid |
| **Output** | 7-class powerset → speech probability | Direct speech probability |
| **Streaming** | No (requires full windows) | Yes (LSTM state across chunks) |
| **Latency** | ~seconds (window aggregation) | ~0.06ms/chunk (CoreML), ~0.50ms/chunk (MLX) |
| **Use case** | Offline/batch processing | Real-time microphone input |

Both models conform to `VoiceActivityDetectionModel` and can be used interchangeably for the `detectSpeech(audio:sampleRate:)` batch API.
