# Cohere Transcribe 2B

`CohereTranscribeASR` is the native MLX Swift implementation of
`CohereLabs/cohere-transcribe-03-2026`. It is an offline encoder-decoder ASR
model intended for multilingual transcription on Apple Silicon.

## Model contract

| Property | Value |
|---|---|
| License | Apache 2.0 |
| Input | mono Float32 PCM, resampled internally to 16 kHz |
| Vocabulary | 16,384 SentencePiece and control tokens |
| Languages | en, fr, de, es, it, pt, nl, pl, el, ar, ja, zh, vi, ko |
| MLX variants | FP16, affine INT5, affine INT8 |

MLX has affine kernels for 2, 3, 4, 5, 6, and 8-bit weights. There is no
INT7 kernel, so INT8 is the high-quality replacement for the requested INT7
variant. The Swift loader rejects INT7 explicitly.

## Architecture

The acoustic encoder is a 48-layer Conformer with 1,280 hidden dimensions,
eight attention heads, and 8x convolutional subsampling. The text decoder is
an eight-layer, 1,024-dimensional Transformer with self-attention and
cross-attention over the acoustic states.

The checkpoint names separate query, key, and value projections. The Swift
runtime fuses each compatible triplet into one `qkv_proj` tensor at load time,
including packed weights and scale/bias companions for quantized exports. The
remaining aliases map the source Conformer, decoder, bridge, and classifier
names onto the Swift module tree before strict weight verification.

## Audio frontend

The frontend applies 0.97 pre-emphasis, a centered 512-point STFT with a
400-sample Hann window and 160-sample hop, 128 Slaney mel filters, natural-log
compression, and per-feature mean/variance normalization. It emits
`[batch, mel, frames]` features for the Conformer subsampler.

Normalization uses only complete hop frames, applies sample variance, and
zeros the trailing centered frame, matching the pinned Python reference.

FP16, INT5, and INT8 performance and memory measurements are documented in
[`cohere-voxtral-asr.md`](../benchmarks/cohere-voxtral-asr.md).
