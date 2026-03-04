# Speech Enhancement — DeepFilterNet3

## Overview

DeepFilterNet3 (Interspeech 2023) is a real-time speech enhancement model that removes background noise from speech. It operates in the frequency domain, using separate ERB-band masking for broadband noise and deep filtering for fine-grained enhancement of low-frequency speech bins.

```
Audio 48kHz
  → STFT (960-pt, 480 hop, Vorbis window) → 481 complex bins

Encoder:
  ERB stream: [B,1,T,32] → 4× Conv2d(64ch) → downsampled features
  Spec stream: [B,2,T,96] → 2× Conv2d(64ch) → DF context
  → GroupedLinear(groups=32) → SqueezedGRU(256, 1 layer) → embedding + LSNR

ERB Decoder:
  SqueezedGRU(256, 2 layers) → 3× ConvTranspose2d with skip connections
  → Conv2d → sigmoid → ERB mask [B,1,T,32]
  → Apply mask to full 481-bin spectrum via ERB inverse filterbank

DF Decoder:
  SqueezedGRU(256, 2 layers) → GroupedLinear + pathway conv
  → Deep filter coefficients: 96 bins × 5 taps × 2 (real/imag)
  → Multi-frame complex filtering on lowest 96 bins

→ iSTFT → Enhanced audio 48kHz
```

## Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `fft_size` | 960 | STFT window size |
| `hop_size` | 480 | Frame shift (10ms @ 48kHz) |
| `erb_bands` | 32 | ERB frequency bands |
| `df_bins` | 96 | Deep filtering frequency bins |
| `df_order` | 5 | Filter taps (multi-frame) |
| `df_lookahead` | 2 | Lookahead frames |
| `conv_ch` | 64 | Convolution channels |
| `emb_hidden` | 256 | GRU hidden dimension |
| `sample_rate` | 48000 | Native sample rate |
| **Total params** | **~2.1M** | **~8.5 MB float32** |

## Component Details

### STFT / iSTFT

Uses Vorbis window for perfect reconstruction:
```
w[n] = sin(π/2 · sin²(π · (n + 0.5) / N))
```

The window satisfies COLA (Constant Overlap-Add) for 50% overlap:
`w[n]² + w[n + hop]² = 1`

Window normalization: `wnorm = 1 / (N² / (2 · hop))`

### ERB Filterbank

The ERB (Equivalent Rectangular Bandwidth) scale maps 481 FFT bins to 32 perceptual bands. Lower bands are narrow (2 bins each at low frequencies), higher bands are wider (up to ~67 bins).

**Forward filterbank** `[481, 32]`: each column has `1/width` for its bins (normalized average).

**Inverse filterbank** `[32, 481]`: each row has `1.0` across its bins (broadcast).

### Feature Extraction

**ERB features** `[B, 1, T, 32]`:
1. Per-band power: `|X|² @ erb_fb`
2. Convert to dB: `10 · log10(power + 1e-10)`
3. Exponential mean normalization: `state = x·(1-α) + state·α; x = (x - state) / 40`

**Spec features** `[B, 2, T, 96]`:
1. First 96 complex bins (0–4750 Hz)
2. Exponential unit normalization: `state = |x|·(1-α) + state·α; x = x / √state`

### Encoder

1. **ERB conv chain**: 4× Conv2d with fused BatchNorm
   - Conv0: `(3,3)` kernel, 1→64 channels, input conv
   - Conv1-3: depthwise separable `(1,3)` kernel, 64→64 channels
   - Conv1-2 downsample frequency by 2× each → 32→16→8

2. **Spec conv chain**: 2× Conv2d
   - Conv0: `(3,3)` kernel, 2→64 channels
   - Conv1: separable `(1,3)` kernel, stride 2 → 96→48

3. **Combine**: Flatten + GroupedLinear(3072→512, groups=32) + Add

4. **SqueezedGRU_S**: GroupedLinear→GRU(1 layer)→GroupedLinear

5. **LSNR head**: Linear(512→1) + Sigmoid, scaled to [-15, 35] dB

### ERB Decoder

1. **SqueezedGRU_S**: 2 GRU layers
2. **Reshape**: embedding → spatial `[B, T, 8, 64]`
3. **Decode with skip connections**:
   - e3 skip → ConvTranspose2d → e2 skip → ConvTranspose2d(↑2) → e1 skip → ConvTranspose2d(↑2)
4. **Output**: Conv2d(64→1) + Sigmoid → ERB mask `[B, T, 32, 1]`
5. **Apply**: mask × inverse_filterbank → full 481-bin gain mask

### DF Decoder

1. **SqueezedGRU_S**: 2 GRU layers + GroupedLinear skip connection
2. **Pathway**: Conv2d(64→10, kernel 5×1) from encoder spec stream
3. **Output**: GroupedLinear(256→960) + Tanh → reshape to `[B, T, 96, 5, 2]`
4. **Add pathway** residual from encoder

### Deep Filtering

Multi-frame complex FIR filter with 5 taps:

```
Y(t, f) = Σ_{n=0}^{4} X(t + n - 2, f) · W(n, t, f)
```

where multiplication is complex. Uses 2 past frames, current frame, and 2 future frames (lookahead).

### Output Assembly

- Bins 0–95: deep-filtered output (fine-grained enhancement)
- Bins 96–480: ERB-masked output (broadband noise removal)

## Implementation

The neural network (encoder + ERB decoder + DF decoder) runs on the **Apple Neural Engine** via Core ML.
Signal processing (STFT, ERB filterbank, normalization, deep filtering, iSTFT) runs on **CPU** via Accelerate/vDSP.

GRU states are managed internally by Core ML — the model processes all frames at once (batch mode).

State maintained between calls:
- STFT analysis/synthesis memory
- Normalization running states (mean norm, unit norm)

## Conversion

```bash
python scripts/convert_deepfilternet3.py [--output OUTPUT_DIR]
```

Downloads the DeepFilterNet3 checkpoint from GitHub releases, wraps the model to handle lookahead padding (replacing negative ConstantPad2d with slice+pad for Core ML compatibility), traces with TorchScript, and converts to Core ML `.mlpackage` format.

Output files:
- `DeepFilterNet3.mlpackage` (~4.2 MB) — Core ML model targeting Neural Engine
- `auxiliary.npz` (~126 KB) — ERB filterbank matrices, Vorbis window, normalization initial states

## CLI Usage

```bash
# Basic usage
swift run audio denoise noisy.wav

# Custom output path
swift run audio denoise noisy.wav --output clean.wav

# Custom model
swift run audio denoise noisy.wav --model aufklarer/DeepFilterNet3-CoreML
```

## Performance

| Metric | Value |
|--------|-------|
| RTF (M2 Max) | 0.24 (4x real-time) |
| 20s audio | ~4.8s processing |
| Compute target | Neural Engine (Core ML) |

## References

- [DeepFilterNet3 Paper](https://arxiv.org/abs/2305.08227) (Interspeech 2023)
- [GitHub: Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
- MIT/Apache-2.0 dual license
