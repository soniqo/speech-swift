# MAGNeT Music Generation

MLX Swift port of Meta's [MAGNeT](https://arxiv.org/abs/2401.04577) text-to-music
model. Generates 30 s clips of 32 kHz mono audio conditioned on a free-form
English prompt (e.g. *"happy rock with electric guitar"*).

Module: `Sources/MAGNeTMusicGen/` · Library: `MAGNeTMusicGen` · CLI: `speech compose`

## Architecture

Three components, loaded together on first call to `fromPretrained`:

| Component | Role | Source repo | Format |
|---|---|---|---|
| **Decoder LM** | Masked non-autoregressive transformer | `aufklarer/MAGNeT-{Small,Medium}-30secs-MLX-{4,8}bit` | MLX safetensors (quantized) |
| **T5-base encoder** | Text → cross-attention conditioning | `t5-base` | safetensors (fp32) |
| **EnCodec 32 kHz decoder** | RVQ codes → 32 kHz waveform | `mlx-community/encodec-32khz-float32` | MLX safetensors (fp32) |

### Decoder LM

| Variant | Layers | Hidden | Heads | FFN | Params | On-disk |
|---|---|---|---|---|---|---|
| Small | 24 | 1024 | 16 | 4096 | 300M | 287 MB (int4) / 425 MB (int8) |
| Medium | 48 | 1536 | 24 | 6144 | 1.5B | 1.36 GB (int4) / 2.10 GB (int8) |

The transformer body is shared across the 4 RVQ codebooks. Each step:

1. Embed the 4 codebook token grids and **sum** along the K axis (so the
   transformer always sees a single `[B, T, D]` hidden state).
2. Add sinusoidal position embedding (no learned positions, no RoPE).
3. N × `{LayerNorm → self-attn → LayerNorm → cross-attn(T5) → LayerNorm → FFN}`.
4. Final `LayerNorm` then 4 separate per-codebook output linear heads
   (`linears.0..3`) produce logits `[B, T, 4, card]`.

`Q/K/V/O` projections and `linear1/linear2` are quantized (mlx_affine,
group_size=64). Embeddings, layer norms, and the per-codebook output heads
stay in FP for quality.

### Local-context attention (stages > 0)

| Stage | Self-attention mask | Why |
|---|---|---|
| 0 | Full (T × T) | First codebook needs global structure to coordinate. |
| 1, 2, 3 | Local: `|q − k| ≤ subcodes_context` (5) | Higher codebooks only refine details, so a ±5-frame window is enough — and dramatically cheaper. |

The masks are precomputed once at construction (one `[1, 1, T, T]` array per
stage) and reused every forward.

### EnCodec 32 kHz

SEANet decoder (`Conv1d` → `LSTM×2` → 4 stages of `{ELU + ConvTranspose1d + ResnetBlock}` → `ELU + Conv1d`) with upsampling ratios `[8, 5, 4, 4]` (total ×640, matching 50 Hz frame rate × 32 000 Hz sample rate). Reflect-padded
Conv1d, no GroupNorm (`norm_type=weight_norm`, weights pre-merged in the MLX
safetensors), 4 RVQ codebooks of 2048 entries × 128 dims.

## Masked parallel decoding

For each of the 4 codebooks (stages) we run `K` iterations of cosine-scheduled
masking + classifier-free guidance:

```
for stage in 0..<4:
    stage_seq = [MASK] * T
    for t in 0..<K[stage]:
        mask_p   = cos(t / (K - 1) * π/2)                  # 1.0 → 0.0
        num_mask = max(int(mask_p * num_units), 1)
        # remask the `num_mask` highest-score (most uncertain) positions
        ...
        logits = uncond + cfg_coef * (cond - uncond)        # batched CFG
        cfg_coef = mask_p * cfg_max + (1 - mask_p) * cfg_min
        sampled = top_p(logits / T_anneal(t), p=0.9)
        # rewrite only masked positions
        scores[mask] = -log(p_sampled)                       # for next iter
```

Default schedule `decoding_steps = (20, 10, 10, 10)` = 50 forward passes total
for a 30 s clip.

`span_arrangement="nonoverlap"`: `span_len=3` consecutive frames are masked /
unmasked together as a unit (matches Meta's reference). Stride-1 spans are
not yet ported.

## Loading flow

```
MAGNeTMusicGen.fromPretrained(variant: .smallInt4)
  ├── download aufklarer/MAGNeT-Small-30secs-MLX-4bit  (model.safetensors + config.json)
  ├── download t5-base                                  (model.safetensors + config.json + spiece.model + tokenizer.json)
  ├── download mlx-community/encodec-32khz-float32      (model.safetensors + config.json)
  ├── decode bundle config.json → MAGNeTConfig
  ├── build MAGNeTLM(config) + load lm.* weights
  ├── build T5Encoder(t5Config) + load sanitized encoder.* weights
  ├── build EncodecModelMLX(encConfig) + load decoder.* / quantizer.* weights
  └── tokenizer = AutoTokenizer.from(modelFolder: t5Dir)
```

The bundle's `compression_state_dict.bin` is a passthrough of the original
audiocraft EnCodec checkpoint, kept for offline reproducibility but **not**
used at runtime — we load the MLX EnCodec safetensors from `mlx-community`.

## Variants

| Variant | LM bits | LM disk | Peak RSS | Wall (M-series, 30 s) | RTF |
|---|---|---|---|---|---|
| `small-int4`  | 4 | 287 MB | ~1.4 GB | ~10.8 s | **0.36×** |
| `small-int8`  | 8 | 425 MB | ~1.5 GB | ~11 s   | **0.37×** |
| `medium-int4` | 4 | 1.36 GB | ~2.2 GB | ~36 s   | **1.20×** |
| `medium-int8` | 8 | 2.10 GB | ~3.0 GB | ~36 s   | **1.20×** |

Numbers measured in this repo's Swift port. The small upstream-Python figure
(0.28×) is from `mlx-examples/musicgen` and excludes the T5 encode pass that
our pipeline performs every call.

Quantization affects memory far more than latency — SDPA attention dominates
wall-clock, not the linear projections. Quality (CLAP score) is essentially
unchanged for the small model and drops modestly for medium.

## Weight loading note (project-wide gotcha)

T5 attention and MAGNeT LM both use `@ModuleInfo(key: "...")` on Linear /
QuantizedLinear submodules nested inside transformer blocks. mlx-swift's
`module.update(parameters: ModuleParameters.unflattened(deepDict))` path
**silently fails to route** these nested keys — leaves projection weights at
their random init, with no error. The audio sounds noisy-but-recognisable
because embeddings + norms + output heads load correctly, while every
attention projection is effectively random.

Fix: load per-leaf via `MLXCommon.CommonWeightLoader.apply*Weights(to:prefix:from:)`
(the pattern VoxCPM2 and MADLAD already use). The MAGNeT module follows it
for both T5 and the LM. If you add a new module with `@ModuleInfo(key: ...)`
overrides nested behind module arrays, prefer per-leaf loading or your
weights will silently not load.

## References

- Paper: [MAGNeT: Masked Audio Generation using a Single Non-Autoregressive Transformer](https://arxiv.org/abs/2401.04577) (Meta, 2024)
- Upstream weights: [`facebook/magnet-small-30secs`](https://huggingface.co/facebook/magnet-small-30secs), [`facebook/magnet-medium-30secs`](https://huggingface.co/facebook/magnet-medium-30secs)
- License: **CC-BY-NC 4.0** (non-commercial, inherited from upstream)

See [docs/inference/magnet-music-gen.md](../inference/magnet-music-gen.md) for
CLI usage and tuning.
