# Voxtral Mini 3B 2507

`VoxtralASR` is the native MLX Swift implementation of
`mistralai/Voxtral-Mini-3B-2507`. It combines a Whisper-style audio encoder
with a Llama text decoder for multilingual offline transcription.

## Model contract

| Property | Value |
|---|---|
| License | Apache 2.0 |
| Input | mono Float32 PCM, resampled internally to 16 kHz |
| Languages | en, fr, de, es, it, pt, nl, hi |
| Audio window | 30 seconds, 3,000 mel frames |
| Audio tokens | 375 per padded 30-second chunk |
| MLX variants | FP16, affine INT5, affine INT8 |

## Architecture

The audio tower has 32 Transformer layers, 1,280 hidden dimensions, 20 heads,
and a 5,120-dimensional feed-forward block. Two convolution layers reduce
3,000 mel frames to 1,500 encoder states. Four adjacent states are packed and
projected to the language dimension, producing 375 audio embeddings per
chunk.

The language model is a 30-layer Llama decoder with 3,072 hidden dimensions,
32 query heads, eight key/value heads, 128-dimensional heads, an 8,192-wide
SwiGLU block, and a 131,072-token Tekken vocabulary. Audio embeddings replace
`[AUDIO]` placeholders in the transcription prompt before causal decoding.

## Audio and prompt parity

The frontend uses the checkpoint's 128-bin Slaney mel setup: a 400-point
periodic Hann window, 160-sample hop, reflect padding, log10 compression,
dynamic-range clipping, and Whisper scaling. Audio is zero-padded to complete
30-second chunks.

The prompt follows Mistral's transcription request format:

```text
<s> [INST] [BEGIN_AUDIO] [AUDIO]... [/INST] lang:<code> [TRANSCRIBE]
```

Omitting the language hint omits the `lang:<code>` tokens. The decoder stops
on any of the three checkpoint EOS-equivalent IDs: 2, 4, or 32,000.

FP16, INT5, and INT8 performance and memory measurements are documented in
[`cohere-voxtral-asr.md`](../benchmarks/cohere-voxtral-asr.md).
