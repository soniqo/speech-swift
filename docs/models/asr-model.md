# ASR Model Architecture ([paper](https://arxiv.org/abs/2601.21337))

## Overview

Qwen3-ASR is an encoder-decoder model: audio encoder extracts features, text decoder generates transcription tokens autoregressively.

Two model sizes are supported, each available in 4-bit, 5-bit, and 8-bit quantization:

| Variant | Repo | File size | Peak RSS (run) |
|---------|------|-----------|---------------|
| 0.6B 4-bit | `aufklarer/Qwen3-ASR-0.6B-MLX-4bit` | 0.65 GB | 976 MB |
| 0.6B 5-bit | `aufklarer/Qwen3-ASR-0.6B-MLX-5bit` | 1.01 GB | 1057 MB |
| 0.6B 8-bit | `aufklarer/Qwen3-ASR-0.6B-MLX-8bit` | 0.93 GB | 1272 MB |
| 1.7B 4-bit | `aufklarer/Qwen3-ASR-1.7B-MLX-4bit` | 2.07 GB | — |
| 1.7B 5-bit | `aufklarer/Qwen3-ASR-1.7B-MLX-5bit` | 2.27 GB | — |
| 1.7B 8-bit | `aufklarer/Qwen3-ASR-1.7B-MLX-8bit` | 2.29 GB | — |

Note: on 0.6B the 5-bit file is slightly larger than 8-bit because the 4/8-bit paths use a hand-rolled packer (one uint32 per 8/4 elements with no waste), while 5-bit routes through `mx.quantize` whose layout carries a small fixed overhead. Runtime peak RSS still scales as expected (4-bit < 5-bit < 8-bit) — that's the cost users actually feel.

Quality on LibriSpeech test-clean (100 utterances, M-series, 4-bit/5-bit/8-bit on 0.6B):

| Variant | WER% | RTF | xRT | Peak RSS |
|---------|------|-----|-----|----------|
| 0.6B 4-bit | 2.33 | 0.013 | 75.3× | 976 MB |
| 0.6B 5-bit | **1.74** | 0.014 | 70.5× | 1057 MB |
| 0.6B 8-bit | 1.65 | 0.016 | 62.1× | 1272 MB |

5-bit cuts WER by 25% relative over 4-bit and lands within 0.09 pp of 8-bit, with ~215 MB less peak RSS than 8-bit. 1.7B numbers TBD.

```
Audio (16kHz mono)
    |
    v
+-------------------+
|  Mel Spectrogram  |   WhisperFeatureExtractor
|  128 bins, 8ms    |   Accelerate FFT (vDSP_fft_zrip)
+--------+----------+
         |
         v
+-------------------+
|  Audio Encoder    |   Conv2D (3 layers, stride-2) + Transformer
|                   |   Block attention, sinusoidal pos embeddings
+--------+----------+
         |
         v
+-------------------+
|  Projector        |   2-layer MLP (dModel -> outputDim)
+--------+----------+
         |  audio embeddings injected into decoder (no variance scaling)
         v
+-------------------+
|  Text Decoder     |   Qwen3 LLM (28 layers, quantized)
|                   |   GQA, RoPE, SwiGLU, KV cache
+--------+----------+
         |
         v
    Transcription
```

## Audio Encoder

| Parameter | 0.6B | 1.7B |
|-----------|------|------|
| Hidden size (d_model) | 896 | 1024 |
| Layers | 18 | 24 |
| Attention heads | 14 | 16 |
| FFN dim | 3584 | 4096 |
| Output dim (projector) | 1024 | 2048 |
| Conv2D layers | 3 (stride 2 each = 8x downsample) | 3 |
| Downsample hidden size | 480 | 480 |
| Position encoding | Sinusoidal (cached) | Sinusoidal |
| Attention type | Block attention (chunked) | Block attention |
| Chunk size | 100 frames (configurable) | 100 frames |

**Block attention:** Audio is split into fixed-size chunks. Attention is restricted within each chunk via a block diagonal mask, reducing complexity from O(T^2) to O(T * chunk_size).

**Conv2D frontend:** Three Conv2D layers with GELU activation downsample the mel spectrogram 8x in the time dimension before the transformer layers.

## Text Decoder (Qwen3)

| Parameter | 0.6B | 1.7B |
|-----------|------|------|
| Hidden size | 1024 | 2048 |
| Layers | 28 | 28 |
| Attention heads (Q) | 16 | 16 |
| KV heads (GQA) | 8 | 8 |
| Head dimension | 128 | 128 |
| Intermediate size (MLP) | 3072 | 6144 |
| Vocab size | 151936 | 151936 |
| RoPE base | 1,000,000 | 1,000,000 |
| RoPE type | MRoPE [24,20,20]* | MRoPE [24,20,20]* |
| Quantization | 4/5/8-bit (group=64) | 4/5/8-bit (group=64) |
| Activation | SwiGLU | SwiGLU |
| Norm | RMSNorm (eps=1e-6) | RMSNorm |
| Q/K normalization | RMSNorm per head | RMSNorm per head |

*Both models use MRoPE config in HuggingFace, but for ASR (single-modal, no image input) all 3 position dimensions are identical, so it reduces to standard 1D RoPE at inference time.

**Transformer block:**
```
x -> RMSNorm -> Attention(Q/K/V projections, Q/K RMSNorm, RoPE, GQA via SDPA) -> + residual
  -> RMSNorm -> SwiGLU MLP(gate_proj, up_proj, down_proj)                      -> + residual
```

**GQA (Grouped Query Attention):** 16 query heads share 8 KV heads (2:1 ratio). SDPA handles this natively without manual tiling.

**Audio injection:** Audio embeddings from the projector are concatenated into the token embedding sequence at designated positions (marked by special tokens). No variance scaling is applied — direct dtype cast only.

## Tokenizer

- Qwen2 BPE tokenizer (vocab size 151936)
- Special tokens: `<|im_start|>`, `<|im_end|>`, `<asr_text>`, language tags
- Byte-level BPE decoding with GPT-2 byte-to-unicode mapping
- Language auto-detection via model output

## Weight Files

| File | Purpose |
|------|---------|
| `model-00001-of-00002.safetensors` | Audio encoder + text decoder weights (part 1) |
| `model-00002-of-00002.safetensors` | Text decoder weights (part 2) |
| `vocab.json` | Token-to-ID mapping |
| `tokenizer_config.json` | Tokenizer settings + added tokens |

See the variant table at the top of this document for actual file sizes per variant.
