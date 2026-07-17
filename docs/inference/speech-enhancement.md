# Speech Enhancement — DeepFilterNet3

## Overview

DeepFilterNet3 (Interspeech 2023) removes background noise from speech in real-time. Runs on Apple Neural Engine via Core ML (~2.1M params, 8-bit palettized weights with FP16 compute, ~2.2 MB).

```
Audio 48kHz → STFT (960-pt, 480 hop, Vorbis window, 1/960 analysis scale) → 481 complex bins
  → Encoder (Conv2d + SqueezedGRU) → ERB mask + DF coefficients
  → ERB mask applied to full spectrum (broadband noise removal)
  → Deep filtering on lowest 96 bins (fine-grained enhancement)
  → iSTFT → Enhanced audio 48kHz
```

Neural network runs on **Neural Engine** (Core ML). Signal processing (STFT, ERB filterbank, deep filtering) runs on **CPU** (Accelerate/vDSP).

The STFT follows the reference `libdf` normalization convention: the forward
DFT is scaled by `1 / fftSize` and the inverse DFT is unscaled. Moving that
factor to the inverse preserves a time-domain STFT round trip, but changes the
spectral magnitudes presented to the neural network and degrades enhancement
quality.

The remaining DSP conventions also match the official implementation: the
normalization decay is the upstream rounded value (`0.99`), out-of-range deep
filter taps are zero-padded, and NumPy C/Fortran array order is honored when
loading the ERB matrices. The published `erb_inv_fb` matrix is Fortran-ordered;
interpreting its bytes as row-major applies the predicted mask to the wrong FFT
bins.

## Quality parity

The reference-compatible pipeline was measured on a fixed 20-clip subset of
the VoiceBank-DEMAND test set. Enhancement ran at 48 kHz; PESQ-WB, STOI, and
SI-SDR were evaluated against the clean references at 16 kHz.

| Implementation | PESQ-WB | STOI | SI-SDR |
|----------------|---------|------|--------|
| Noisy input | 2.244 | 0.948 | 9.19 dB |
| Previous speech-swift pipeline | 1.839 | 0.911 | 12.66 dB |
| speech-swift Core ML | **3.097** | **0.964** | **19.12 dB** |
| Official Python DeepFilterNet3 | 3.029 | 0.962 | 18.31 dB |

The reference-compatible Swift path improved all three metrics over the
pre-fix pipeline on 20/20 clips. It exceeded the official result on 16/20
clips by PESQ and 18/20 by both STOI and SI-SDR. The public model ID, bundle,
and API remain unchanged.

## Parameters

| Parameter | Value |
|-----------|-------|
| FFT / hop | 960 / 480 (10ms frames @ 48kHz) |
| ERB bands | 32 |
| DF bins / order | 96 / 5 taps |
| GRU hidden | 256 |
| Params | ~2.1M (~2.2 MB, 8-bit palettized weights / FP16 compute) |
| Single-shot cap | ~60 s (6000 frames @ 10 ms hop, `RangeDim(1, 6000)` in the exported CoreML graph) |
| Long-form | `enhanceChunked(...)` auto-windows above the cap (see below) |

## Latency (M2 Max)

| Duration | Time | RTF |
|----------|------|-----|
| 5s | 0.65s | 0.13 |
| 10s | 1.2s | 0.12 |
| 20s | 4.8s | 0.24 |

Core ML GRU cost scales ~O(n²) due to sequential hidden state processing. Short audio is proportionally faster.

## Long-form audio

The single-shot `enhance(audio:sampleRate:)` API is capped at ~60 s by the
exported CoreML graph's `RangeDim(1, 6000)` on the time axis — feeding
longer input throws a CoreML prediction error. Two paths handle long inputs:

```swift
// Recommended: auto-chunks above the cap, bit-identical for shorter inputs.
let clean = try enhancer.enhanceChunked(
    audio: longAudio, sampleRate: 48000,
    chunkSeconds: 45.0,  // default; must be ≥ 1 s
    overlapMs: 500       // default; must be < chunkSeconds*500
)
```

The chunker splits into ~`chunkSeconds`-second windows with `overlapMs` of
overlap and stitches with an equal-power sin/cos crossfade. Between chunks
the STFT analysis/synthesis overlap buffers AND the EMA normalization
state are preserved (carried across without `resetState()`) — those seams
are *exact*. The GRU hidden state inside the CoreML graph re-zeros at
each chunk (it's not exposed as a model input/output), so the first
~100-200 ms of each non-leading chunk are mildly degraded while the GRU
re-converges; the default 500 ms crossfade masks this for
speech-dominated content. Stationary low-SNR noise may show a brief
noise-floor flicker at boundaries.

A reference 90 s synthetic test (sine + low-level noise) checks that retained
audio stays within 1.5 dB of adjacent windows, or that a signal classified as
noise remains below -50 dBFS across the seam. See
`Tests/SpeechEnhancementTests/E2EDeepFilterNet3ChunkingTests.swift`.

For seamless GRU streaming we'd need a model re-export with the hidden
state threaded as explicit CoreML inputs/outputs (see
`speech-models/.../convert.py` notes). Tracked as a follow-up; not
required for the common speech-denoise case.

## CLI

```bash
swift run speech denoise noisy.wav
swift run speech denoise noisy.wav --output clean.wav

# Long-form audio (auto-chunks transparently above 45 s):
swift run speech denoise long_meeting.wav

# Tune the chunk window or disable chunking entirely:
swift run speech denoise long.wav --chunk-seconds 30 --overlap-ms 300
swift run speech denoise short.wav --no-chunk     # error if > 60 s
```

## Model bundle

The default HuggingFace repository publishes the Core ML model and auxiliary
DSP constants. speech-swift downloads the compiled bundle directly:

- `DeepFilterNet3.mlmodelc` (~2.2 MB) — pre-compiled Core ML model with 8-bit palettized weights and FP16 compute
- `auxiliary.npz` (~126 KB) — ERB filterbank, Vorbis window, normalization states

## References

- [DeepFilterNet3 Paper](https://arxiv.org/abs/2305.08227) (Interspeech 2023)
- [GitHub: Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
- MIT/Apache-2.0 dual license
