# CSM (Conversational Speech Model)

## Overview

CSM-1B is Sesame's Conversational Speech Model — a Moshi-family, codec-token
text→audio model. A **Llama-1B backbone** predicts Mimi codebook 0 and a small
**Llama-100M decoder** predicts the remaining acoustic codebooks (1..31), which a
**Mimi codec** decodes to a 24 kHz waveform. Given a text and a short reference
clip, CSM synthesizes the text in the reference voice (zero-shot cloning).

This runtime is a native Swift/MLX port. It reuses PersonaPlex's Mimi codec, RoPE,
KV cache, and quantized-linear helpers, and loads our exported MLX weights.

- **Model ID:** [`aufklarer/CSM-1B-MLX-8bit`](https://huggingface.co/aufklarer/CSM-1B-MLX-8bit) (default)
- **License:** Apache-2.0 (Sesame CSM-1B)
- **Sample rate:** 24 kHz mono
- **Language:** English

## Status

- Backbone + decoder + heads, Mimi decode, Llama BPE tokenizer, reference voice
  cloning, and temperature/top-k sampling are implemented and covered by tests.
- `CSMPipeline.fromPretrained("aufklarer/CSM-1B-MLX-8bit")` downloads the published
  model and synthesizes end-to-end (validated: non-silent output, RMS ≈ 0.13).

## Bundle Contract

`fromPretrained` downloads these files into the shared model cache:

| File | Purpose |
|---|---|
| `model.safetensors` | Backbone + decoder + heads (HF-style `model.` keys) |
| `config.json` | Tower configs, vocab sizes, codebook count, quantization |
| `mimi.safetensors` | Mimi codec weights (32 codebooks) |
| `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` | Llama BPE tokenizer |

`config.json` carries two tower configs (`backbone`, `decoder`), each with
`dim`, `num_layers`, `num_heads`, `num_kv_heads` (GQA), `head_dim`,
`intermediate_dim`, `rope_base`, and `norm_eps`, plus `text_vocab_size`,
`audio_vocab_size`, `audio_num_codebooks`, and an optional `quantization`
(`bits`, `group_size`). A missing `quantization` block means fp16.

## Architecture

```
[text] --Llama BPE--> text tokens ─┐
                                   ├─► [Backbone: Llama-1B, GQA + RoPE + RMSNorm(f32)] ─► codebook 0
[reference audio] --Mimi encode--> ┘                                   │
                                                                       ▼
                                        [Decoder: Llama-100M] ─► codebooks 1..31 (per-frame)
                                                                       │
                                                                       ▼
                                        autoregressive frame loop (EOS-terminated, ≤ maxFrames)
                                                                       │
                                        [Mimi decode] ─► 24 kHz mono waveform
```

- **Backbone / decoder:** grouped-query attention (separate Q/K/V/O projections),
  traditional RoPE (`rope_base` default 500000), RMSNorm computed in float32.
- **Mimi codec:** 32 codebooks, 12.5 Hz frame rate, 24 kHz — shared with
  PersonaPlex / Moshi.
- **Quantization:** int8 (group size 64) by default; fp16 when no quantization
  block is present.

## Text & Voice Cloning

- Text is tokenized as `[speaker] text` with the Llama BPE tokenizer (`speaker`
  defaults to 0).
- Voice cloning is "voice-match": the reference transcript and the target text are
  encoded as a single text segment placed over the reference audio frames, so the
  model continues in the same voice. Supply an accurate transcript for the
  reference clip — a clean 10–15 s sample works well.

## Sampling

Temperature + top-k categorical sampling (mirrors `mlx_lm.sample_utils`): logits
outside the top-k are masked to −∞, then sampled from `categorical(logits / temp)`.
`temperature = 0` is greedy argmax. Defaults: `temperature = 0.9`, `topK = 50`,
`maxFrames = 1024`.

## Performance

The int8 model is the recommended default. On an M5 Pro it runs faster than
real-time (≈ 0.5 real-time factor — roughly 0.5 s of compute per second of audio)
in a Release build. Debug builds are several times slower because MLX kernels are
unoptimized. Call `MLX.GPU.clearCache()` between repeated generations to keep GPU
memory from growing.

## Related

- Inference & CLI usage: [`docs/inference/csm.md`](../inference/csm.md)
- Shared Mimi / primitives: [`docs/models/personaplex.md`](personaplex.md)
