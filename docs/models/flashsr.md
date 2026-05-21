# FlashSR

MLX Swift port of [FlashSR](https://arxiv.org/abs/2501.10807) (ICASSP 2025) —
one-step distilled audio super-resolution. A student model that compresses
50-step AudioSR into a **single forward pass** while keeping AudioSR's
versatility (speech + music + SFX).

Module: `Sources/FlashSR/` · Library: `FlashSR` · CLI: `speech upsample`

## What it is

FlashSR upsamples any 48 kHz mono audio (low-bandwidth, lossy compression,
phone-call grade) into 48 kHz mono audio with full-band detail restored. Best
applied as a per-window 5.12 s job; longer inputs are processed window-by-
window.

| | |
|---|---|
| Input | 48 kHz mono, ≥1 sample (padded internally to 5.12 s windows) |
| Output | 48 kHz mono, same length as input |
| Window | 5.12 s = 245760 samples |
| Bundle (INT4) | 363 MB on disk |
| Bundle (INT8) | ~720 MB on disk |
| Runtime memory | weights dequantised to FP at load — int4 is download-size only |
| Licence | MIT (upstream FlashSR, KAIST) |

## Architecture

Three cooperating MLX modules — all bundled into a single `model.safetensors`:

| Component | Role |
|---|---|
| **VAE** (AudioLDM-style AutoencoderKL) | Encodes a 256-bin log-mel spectrogram to a 16-channel latent, then decodes the diffusion output back to mel. 8× spatial downsample/upsample, `ch_mult=[1,2,4,8]`, `num_res_blocks=2`, no attention (resolutions=[]). |
| **AudioSRUnet** (distilled student) | Single-step diffusion UNet. `in_channels=32` (16 cond + 16 noisy concatenated), `model_channels=128`, `channel_mult=[1,2,3,5]`, `attention_resolutions=[8,4,2]`, transformer depth 1, num_head_channels=32. With `extra_sa_layer=True` for the self-attention pre-pass. |
| **SR Vocoder** (BigVGAN-style with audio injection) | mel → 48 kHz waveform. BigVGAN multi-receptive-field blocks with SnakeBeta activations, anti-aliased FIR up/down, and an `audio_block` that injects the LR waveform as a per-level conditioning pyramid. Upsample rates `[10, 6, 2, 2, 2]`. |

## Pipeline (per 5.12 s window)

```
lr_audio (48 kHz)
   │
   ▼  normalize: mean-center + max-abs scale to ±0.5 → (norm, mean, scale)
   │
   ▼  log-mel spec (n_fft=2048, hop=480, 256 mels) on `norm`
   │
   ▼  VAE.encode → cond_z = posterior.mean × 0.3342
   │
   ▼  single DPM-Solver step (v_prediction, cosine α̅(t=999)):
   │     noise = N(0, 1)
   │     v     = AudioSRUnet(noise, cond_z, t=999)
   │     z_0   = √α̅ · noise − √(1−α̅) · v
   │
   ▼  VAE.decode(z_0 / 0.3342) → reconstructed mel
   │
   ▼  SRVocoder(mel, norm) → hr_norm (48 kHz waveform)
   │
   ▼  denormalize → output (48 kHz, same length as input)
```

## Quantization scheme

The bundle uses a custom **`mlx_affine_flat`** scheme: every weight with
`ndim ≥ 2` whose `prod(shape[1:]) % group_size == 0` gets quantised to int4
or int8 with `group_size = 64`. The flat (O, fan_in) reshape lets `mx.quantize`
operate on any conv layout uniformly.

At load time the loader calls `mx.dequantize(weight, scales, biases, ...)`
and reshapes back to the original conv/linear shape via `cfg.quantized_shapes`.
**Runtime weights are FP** — the quantisation is purely a download-size
optimisation. Linear / quantised-conv kernels are not invoked.

## Weight-loading quirk: PyTorch `nn.Sequential` slots

PyTorch `nn.Sequential([norm, SiLU, conv])` saves under keys `…0.weight`,
`…2.weight`. mlx-swift's `update(parameters:)` interprets numeric
sub-segments as list indices, so a Swift module with `@ModuleInfo(key: "0")`-
style named slots fails to load.

The fix used here: every PyTorch `nn.Sequential` group is represented in
Swift as a `FlashSRSeqLayers` (a tiny wrapper around `[Module]`) with
parameterless slots (SiLU, Dropout) filled by `FlashSRNoop()` placeholders.
A bundle-key remap (`remapTSeqKeys`) inserts the implied `.layers.` segment:
`input_blocks.5.1.transformer_blocks.…` → `input_blocks.5.layers.1.transformer_blocks.…`.
The same remap covers `output_blocks`, `middle_block`, `time_embed`, `out`,
`in_layers`, `emb_layers`, `out_layers`, `to_out`, `net`, `downsamples`, `ups`.

## Variants

| Variant | On-disk | Notes |
|---|---|---|
| `int4` | 363 MB | Default. Same quality as int8 at runtime (weights dequantised). |
| `int8` | ~720 MB | Slightly smaller dequant rounding error. |

Both produce identical runtime memory footprint after dequant (~1.4 GB peak
RSS for the loaded model).

## Performance (M-series)

Release build, single 5.12 s window, M2-class GPU:

| | Wall | RTF |
|---|---|---|
| Cold load + first forward | ~3–5 s | first window is slower (JIT shader compile) |
| Subsequent windows | ~1–2 s | RTF 0.2–0.4× (faster than realtime) |

Debug build is ~10× slower than release because MLX kernels aren't optimised.

## References

- Paper: [FlashSR: One-Step Versatile Audio Super-Resolution via Diffusion Distillation](https://arxiv.org/abs/2501.10807) (ICASSP 2025)
- Upstream PyTorch code: [jakeoneijk/FlashSR_Inference](https://github.com/jakeoneijk/FlashSR_Inference)
- Upstream weights: [jakeoneijk/FlashSR_weights](https://huggingface.co/datasets/jakeoneijk/FlashSR_weights) (PyTorch)
- MLX bundles: [aufklarer/FlashSR-MLX-4bit](https://huggingface.co/aufklarer/FlashSR-MLX-4bit), [aufklarer/FlashSR-MLX-8bit](https://huggingface.co/aufklarer/FlashSR-MLX-8bit)
- AudioSR teacher: [versatile_audio_super_resolution](https://github.com/haoheliu/versatile_audio_super_resolution)
- License: MIT

See [docs/inference/flashsr.md](../inference/flashsr.md) for CLI usage.
