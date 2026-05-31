# ASR Inference Pipeline ([Qwen3-ASR](https://arxiv.org/abs/2601.21337))

## Overview

```
Audio (16kHz) → [Preprocessing] → [Audio Encoding] → [Text Generation] → Transcription
                 ~5% time          ~20% time           ~75% time
```

## Stage 1: Preprocessing (AudioPreprocessing.swift)

Converts raw audio to a mel spectrogram `[128, T]` using Accelerate framework.

- STFT via `vDSP_fft_zrip` (in-place real FFT, zero-padded 400→512)
- Mel filterbank via `vDSP_mmul`, bin frequencies use padded FFT size (`k * fs / 512`)
- Log-mel via `vDSP_vclip` + `vvlog10f` (vForce vectorized)
- Hann window and FFT setup precomputed once in `init()`
- All temporary buffers preallocated outside the frame loop

## Stage 2: Audio Encoding (AudioEncoder.swift)

18-layer transformer with block attention over chunked mel features.

- Self-attention via `MLXFast.scaledDotProductAttention` (Metal kernel)
- Sinusoidal position embeddings cached by sequence length
- Block attention mask via MLXArray broadcast (`blockIds .== blockIds^T`)

## Stage 3: Text Generation (QuantizedTextDecoder.swift)

28-layer quantized Qwen3 decoder with GQA and RoPE.

- RoPE via `MLXFast.RoPE` (fused Metal kernel)
- GQA via `MLXFast.scaledDotProductAttention` (native GQA support, no manual tiling)
- Causal mask: `nil` for autoregressive steps (seqLen=1), broadcast for prefill
- **Prefill** (seqLen > 1): all prompt tokens in one forward pass
- **Decode** (seqLen = 1): SDPA uses optimized T_q=1 Metal kernel
- **Greedy fast path** (default options): decode loop is double-buffered via `MLX.asyncEval` — step N+1's forward pass is queued *before* step N's token syncs to CPU, so host-side EOS check and append overlap with GPU work. Bit-identical to the legacy loop (snapshot-tested); ~5 % faster on long-form audio (1.7B-8bit, 71 s clip)

## Decoder Options

`transcribe(audio:sampleRate:options:)` accepts a `Qwen3DecodingOptions`
struct that exposes the HuggingFace-style decoding knobs:

| Field | Default | Notes |
|---|---|---|
| `maxTokens` | `448` | Cap on decoder output per chunk. |
| `language` | `nil` | Hint; `nil` → auto-detect. |
| `context` | `nil` | Prefix prepended to the decoder prompt. |
| `repetitionPenalty` | `1.0` | HF divisor; `1.1`–`1.3` typical. Positive logits divide, negative logits multiply — matches the HF sign-aware branch so the penalty always reduces the probability of the already-generated token. |
| `noRepeatNgramSize` | `0` | Masks tokens that would form a repeated n-gram of this size. `0` disables. |
| `temperature` | `0.0` | `0` = greedy (argmax). `> 0` = sample via Gumbel-max (`argmax(logits/T + Gumbel(0,1)) ~ softmax(logits/T)`). |

All defaults take the asyncEval greedy fast path described above; any
non-default option falls back to a per-token-sync loop because the
sampler pulls full logits to CPU and defeats the overlap. Output is
byte-identical to the legacy loop in either mode.

The canonical defence against "percent percent percent..." loops on silence
or ambiguous audio is `repetitionPenalty = 1.15`:

```swift
let text = model.transcribe(
    audio: samples,
    sampleRate: 16000,
    options: Qwen3DecodingOptions(repetitionPenalty: 1.15)
)
```

The legacy overload `transcribe(audio:sampleRate:language:maxTokens:context:)`
remains available and forwards into the new path with default options.

## CoreML Encoder Backend (CoreMLEncoder.swift)

The CoreML port of the audio encoder (`aufklarer/Qwen3-ASR-CoreML`) ships a model exported with the same **chunked block attention** the upstream PyTorch model is trained with (100-frame mel chunks → 13 tokens after 3× stride-2 conv, attention restricted to 800-frame / 8-chunk windows). The export uses a **single fixed mel shape** instead of `EnumeratedShapes` and signals real audio length through a dedicated input/output pair so the downstream consumer can slice the padded embeddings to the un-padded count.

### Interface

```
Inputs
  mel          MultiArray<Float32>  shape [1, 128, 3000]
  mel_length   MultiArray<Int32>    shape [1]
                  Number of real mel frames in mel; remaining frames
                  are zero-padded and masked out by the in-graph
                  block-attention bias.

Outputs
  audio_embeddings  MultiArray<Float16>  shape [1, 390, 1024]
  output_length     MultiArray<Int32>    shape [1]
                  Number of real audio tokens. Callers should iterate
                  only embeddings[:, :output_length, :].
```

The 8× conv stride downsamples 3000 mel frames to up to 390 audio tokens. For shorter audio the in-graph mask discards the trailing chunks; `output_length` returns the precise real count (e.g. 64 tokens for a 4.9 s clip).

### Swift consumer

`CoreMLEncoder.encode(_:)` returns `(embeddings: MLXArray, outputLength: Int)`. `CoreMLASRModel.transcribe` uses `outputLength` directly as `numAudioTokens` when chunking the audio prefill into the decoder; this replaces the previous `ceil(realMelFrames / 8)` heuristic with the model-reported truth.

### Why the rebuild

The previous export ran **unmasked global self-attention** over mel padded to enumerated buckets `[100, 200, 400, 600, 800, 1000, 1500, 2000, 3000]`. Trailing zero-padded mel frames contaminated the real audio embeddings via attention, and the text decoder emitted `<|im_end|>` right after the first sentence-final period — 24.88% WER on LibriSpeech test-clean vs 1.82% for the same 0.6B-8bit weights via MLX. The rebuilt export brings WER on the same fixture to **3.02%** and is ~4.7× faster (24 ms vs 113 ms per call on M5 Pro).

## Performance

| Model | Framework | WER% | RTF | xRT |
|-------|-----------|------|-----|-----|
| Qwen3-ASR 1.7B (MLX, 8-bit) | MLX Swift | 1.52 | 0.033 | 30.5× |
| Qwen3-ASR 0.6B (MLX, 8-bit) | MLX Swift | 1.82 | 0.015 | 66.0× |
| Qwen3-ASR 0.6B (MLX, 4-bit) | MLX Swift | 2.20 | 0.012 | 85.6× |
| Qwen3-ASR 0.6B (CoreML INT8) | CoreML (ANE+CPU) | 3.02 | 0.098 | 10.2× |

M5 Pro, 48 GB; LibriSpeech test-clean n=200; isolated per-engine. See [docs/benchmarks/asr-wer.md](../benchmarks/asr-wer.md) for the full cross-engine table including WhisperKit, Parakeet, Omnilingual, and Nemotron.

## Streaming / Partial Transcription

Qwen3-ASR operates in batch mode only. The entire audio input is processed in a single forward pass — there is no streaming or partial transcription support. The audio encoder uses block attention over the full mel spectrogram, and the text decoder generates tokens autoregressively conditioned on the complete encoder output.

For real-time transcription use cases where partial results are needed, consider chunking the audio externally and running separate inference passes on each chunk.

## Language Detection

The model automatically detects the spoken language from the audio content. No language hint or locale parameter is required. The text decoder emits a language token at the start of generation, followed by the transcribed text. Supported languages include English, Chinese, Japanese, Korean, and many European languages.

## Model Architecture Reference

See [docs/models/asr-model.md](../models/asr-model.md) for detailed architecture documentation including layer dimensions, weight formats, and quantization details.
