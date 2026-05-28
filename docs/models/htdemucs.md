# HTDemucs (Demucs v4) — Hybrid Transformer Music Source Separation

State-of-the-art music source separation. Splits stereo music into 4 stems
(drums, bass, other, vocals) — substantially higher separation quality than
Open-Unmix. On a directional benchmark (3 MUSDB *sample* clips, museval /
BSSEval v4) `htdemucs_ft` averages **+3.01 dB SDR** over UMX-HQ, with the
largest gains on bass (**+5.75 dB**) and drums (**+2.40 dB**). Reproduce with
`benchmark.py` in the `speech-models` repo.

## Architecture

A hybrid model with two parallel branches merged by a cross-domain transformer:

```
mix [B,2,T]
 ├─ freq branch:  STFT(4096/1024) → complex-as-channels [B,4,2048,F]
 │                 → 4× HEncLayer (Conv2d over freq, GLU rewrite, DConv) → [B,384,8,F']
 └─ time branch:  raw waveform → 4× HEncLayer (Conv1d) → [B,384,F']
                  ↓ (channel upsample 384→512)
            CrossTransformerEncoder (5 layers, dim 512, 8 heads):
              self-attention per branch + cross-attention between branches,
              sinusoidal pos-emb (2D for freq, 1D for time), LayerScale, GroupNorm
                  ↓ (channel downsample 512→384)
 ┌─ freq decoder: 4× HDecLayer (ConvTranspose2d) → CaC mask → iSTFT
 └─ time decoder: 4× HDecLayer (ConvTranspose1d)
        x = iSTFT(freq) + time   →   stems [B,4,2,T]
```

Key points:
- **Complex-as-channels (cac)**: complex STFT carried as 2 real channels; the
  model output *is* the spectrogram (no Wiener filtering at inference).
- **No frequency reduction to 1**: at depth 4 the freq axis stays at 8 bins; the
  branches are merged by the transformer, not by a 1-D bottleneck.
- **DConv** residual branch in every enc/dec layer: dilated Conv1d + GroupNorm +
  GELU + GLU + LayerScale.

## `htdemucs_ft` (the shipped variant)

A **bag of 4 fine-tuned sub-models**, one specialised per source, with diagonal
combine-weights — inference runs each sub-model and keeps its own stem.

| | |
|---|---|
| Params | 4 × 42.0M = 168M |
| Segment | 7.8 s windows, 25% overlap, triangular cross-fade |
| Sample rate | 44,100 Hz stereo |
| Weights | [aufklarer/HTDemucs-FT-MLX](https://huggingface.co/aufklarer/HTDemucs-FT-MLX) (fp16, 320 MB) |

## Precision

| Variant | Parity vs PyTorch | Bundle size |
|---|---|---|
| fp32 | 57.6 dB SNR | 640 MB |
| **fp16** (shipped) | ≈ fp32 | **320 MB** |
| int8 (transformer Linear only) | 42.2 dB SNR | ~240 MB |

The MLX-Swift port matches the PyTorch reference at **57.6 dB SNR** (fp32). int8
quantizes only the transformer Linear layers (convs and packed-attention
`in_proj` aren't MLX-quantizable), so it saves ~25% with a small quality cost.

## Quality vs Open-Unmix

Median SDR (museval / BSSEval v4) over 3 MUSDB *sample* clips — directional, not
the full 50-track MUSDB18-HQ. `htdemucs_ft` is the PyTorch reference; our MLX port
matches it to 57.6 dB SNR.

| Stem | htdemucs_ft | UMX-HQ | Δ |
|---|---|---|---|
| drums | 6.89 | 4.49 | +2.40 |
| bass | 9.50 | 3.75 | +5.75 |
| other | 4.70 | 2.85 | +1.85 |
| vocals | 10.57 | 8.53 | +2.05 |
| **AVG** | **7.92** | **4.91** | **+3.01** |

## Usage

```bash
# Downloads htdemucs_ft from HuggingFace on first run
speech separate song.wav --engine htdemucs

# Specific stems / output dir
speech separate song.wav --engine htdemucs --stems vocals,drums --output-dir stems/
```

```swift
import SourceSeparation
let sep = try await HTDemucsSeparator.fromPretrained()
let stems = sep.separate(mix)   // [1,2,L] → ["vocals": [1,2,L], ...]
```

## Reference

- Rouard, Massa, Défossez, "Hybrid Transformers for Music Source Separation" (ICASSP 2023)
- [facebookresearch/demucs](https://github.com/facebookresearch/demucs) (MIT)
