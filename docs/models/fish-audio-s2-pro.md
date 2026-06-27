# Fish Audio S2 Pro

Experimental Apple-Silicon MLX port of `aufklarer/Fish-Audio-S2-Pro-MLX-fp16`.
This module covers the text-to-codebook runtime, Fish DAC encode for raw
reference audio, and Fish DAC decode from generated codebooks to 44.1 kHz
waveform audio.

## At A Glance

| | |
|---|---|
| Source model | `fishaudio/s2-pro` |
| MLX bundle | `aufklarer/Fish-Audio-S2-Pro-MLX-fp16` |
| Runtime target | `FishAudioTTS` |
| Architecture | Fish Qwen3-Omni dual-AR transformer |
| Text model | 36 layers, d=2560, 32 query heads / 8 KV heads, QK norm |
| Fast audio decoder | 4 layers, d=2560, 10 codebooks, 4096-token logits |
| Codec | Semantic + residual VQ, 8-layer post transformer, causal DAC decoder |
| Control markers | `[pause]`, `[emphasis]`, `[laughing]`, `[excited]`, `[angry]`, `[whisper]`, `[screaming]`, `[shouting]`, `[surprised]`, `[sad]` |
| Current output | 44.1 kHz mono PCM from generated or cloned-reference codebooks |
| License | Research/non-commercial bundle; do not expose as a commercial engine without a Fish Audio license |

## Runtime Layout

The published bundle ships split model weights:

- `text_model.model.*` maps to the slow transformer (`embeddings`, `layers`, `norm`).
- `audio_decoder.codebook_embeddings.*` maps to the shared slow codebook embedding table.
- all other `audio_decoder.*` keys map to `fast_*` modules (`fast_embeddings`, `fast_layers`, `fast_norm`, `fast_output`).
- `codec.safetensors` is loaded by `FishAudioCodec` for semantic/residual VQ
  decode, post-transformer upsampling, and causal DAC waveform decode.

Prompt rows follow upstream Fish inference:

```
[primary token row]
[codebook 0 row]
...
[codebook 9 row]
```

Text tokens fill only the primary row. Reference VQ spans fill the primary row
with `semantic_start_token_id + codebook0` and fill rows 1...10 with the raw
reference codebook ids.

The text and fast transformers use Fish's adjacent-dimension RoPE convention
(`traditional: true` in MLX). Fish Qwen3-Omni also scales semantic-token
embeddings by `1 / sqrt(num_codebooks + 1)` and feeds the normalized slow
hidden state into the fast decoder. The local parity E2E checks the first
prefill semantic token against the PyTorch reference path.

## Generation

`FishAudioDualARModel.generateCodebooks`:

1. Prefills the slow transformer with the prompt rows.
2. Samples the primary next token constrained to semantic ids plus `<|im_end|>`.
3. Uses the primary semantic code as codebook 0.
4. Runs the fast decoder sequentially for the remaining codebooks.
5. Feeds the generated frame back into the slow transformer.
6. Stops on `<|im_end|>` and returns only generated DAC codebooks.

Sampling is CPU-side top-k/top-p over evaluated logits for now. That keeps the
initial port simple and deterministic for greedy tests; a later optimization can
move sampling fully onto MLX arrays. Runtime sampling includes Fish's
repetition-aware semantic fallback and a configurable `minNewTokens` floor so
the model cannot terminate at frame zero.

## Codec Encode/Decode

`FishAudioCodec` encodes raw reference audio to `[10][frames]` Fish codebooks:

1. Resamples Float PCM to 44.1 kHz when needed.
2. Pads to the Fish frame size of 2048 samples.
3. Runs the causal DAC encoder with rates `[2, 4, 8, 8]`.
4. Runs the codec downsampler and pre-transformer.
5. Quantizes the semantic stream and the 9 residual streams with nearest-code
   vector quantization.

The same codec decodes generated `[10][frames]` Fish codebooks:

1. Clamps codebook 0 to the 4096-way semantic VQ and codebooks 1...9 to the
   1024-way residual VQ, matching upstream Fish decode.
2. Projects semantic and residual codebook embeddings back to 1024 channels and
   sums them.
3. Runs the 8-layer causal window-limited post transformer.
4. Upsamples by `[2, 2]`.
5. Runs the causal DAC decoder with rates `[8, 8, 4, 2]` and final tanh.

Each generated Fish frame decodes to 2048 samples at 44.1 kHz.

## Remaining Work

The runtime still needs:

- CLI integration under `speech speak --engine fish-audio`.
- Broader round-trip ASR coverage for Hindi emotional markers after the engine
  is exposed through the CLI/runtime surface.

The gated E2E test `E2EFishAudioTTSTests` already loads the real local bundle
and verifies one greedy codebook-generation step, codec encode/decode smoke,
PyTorch prefill parity, raw-reference Hindi cloning, codec ASR reconstruction,
and text-only plus cloned Hindi ASR round-trips with `FISH_AUDIO_E2E=1` /
`FISH_AUDIO_ASR_E2E=1`.
