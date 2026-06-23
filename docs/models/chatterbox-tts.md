# Chatterbox Multilingual TTS Architecture

## Overview

Chatterbox Multilingual (Resemble AI, MIT) is a zero-shot voice-cloning TTS that
synthesizes a target text in the timbre of a short reference clip across 23
languages — the broadest language coverage of any TTS in this package. The MLX
bundle is a genuine fp16
conversion (~1.3 GB) of the three upstream checkpoints, published at
[`aufklarer/Chatterbox-Multilingual-MLX-fp16`](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16).

The Swift port is built component-by-component, with token ids, mel features, and
speaker embeddings checked against a known-good reference at each stage.

## Languages

23 languages: Arabic, Chinese, Danish, Dutch, English, Finnish, French, German,
Greek, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Norwegian, Polish,
Portuguese, Russian, Spanish, Swahili, Swedish, Turkish.

Text encodes through a grapheme path: NFKD normalization → a per-language
`[lang]` token → BPE.

## Pipeline Stages

### Stage 1: T3 (text → speech tokens)

A 30-layer Llama backbone (hidden 1024, 16 heads, head-dim 64, SwiGLU, RoPE,
536M params). Text tokens (from the multilingual grapheme BPE) plus a
perceiver-resampled speaker conditioning (`cond_enc` over the VoiceEncoder
embedding + emotion control) are fed autoregressively, with a KV cache and
classifier-free guidance (batch-of-2), to emit discrete speech tokens.

### Stage 2: S3Gen (speech tokens → mel)

A CosyVoice-derived flow stack: an UpsampleConformer encoder with relative
positional attention, then a **Matcha-style U-Net conditional flow-matching
(CFM) estimator** solved with an ODE. This is the one decoder that differs from
CosyVoice (which uses a DiT), so it is the headline new component in the port.

### Stage 3: HiFi-GAN / HiFTGenerator (mel → waveform)

A HiFi-GAN vocoder with NSF source excitation and an F0 predictor, the same
family as the CosyVoice HiFTGenerator, producing 24 kHz audio.

### Speaker encoder (VoiceEncoder)

A resemblyzer-style 3-layer LSTM x-vector over a 40-mel slaney front-end
(n_fft=400, hop=160, power=2.0), producing a 256-d L2-normalised embedding that
conditions T3. Reuses the shared `MLXCommon/SlaneyMel` mel and native
`MLXNN.LSTM`.

## Voice Cloning

Zero-shot: a single reference clip is encoded to a speaker embedding (VoiceEncoder)
and to S3Gen reference features (S3TokenizerV2 + CAM++). No fine-tuning. The
reference is resampled to 16 kHz and silence-trimmed before the speaker encoder.

## References

- [Chatterbox](https://github.com/resemble-ai/chatterbox) (Resemble AI, MIT)
