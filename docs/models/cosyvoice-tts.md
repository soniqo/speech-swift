# CosyVoice3 TTS Architecture

## Overview
Three-stage pipeline: LLM → DiT Flow Matching → HiFi-GAN Vocoder → 24kHz audio

## Languages
9 languages supported: Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian

## Pipeline Stages

### Stage 1: LLM (Qwen2.5-0.5B)
- Speech token generator based on Qwen2.5-0.5B
- 24 transformer layers, 896 hidden, 14Q/2KV heads
- FSQ vocabulary: 6561 tokens at 25 Hz
- Input: [sos, text_tokens..., task_id, speech_tokens...]
- Autoregressive with KV cache

### Stage 2: DiT Flow Matching
- 22-layer Diffusion Transformer (1024 hidden, 16 heads)
- Conditional flow matching with Euler ODE solver (10 steps)
- Classifier-free guidance (rate=0.7)
- AdaLN (Adaptive Layer Norm) for timestep conditioning
- Converts speech tokens → 80-band mel spectrogram
- Token upsampling: 25 Hz → 50 Hz via linear interpolation

### Stage 3: HiFi-GAN Vocoder
- Neural Source Filter (NSF) with 8 harmonics
- F0 prediction from mel spectrogram
- 3-stage upsampling [8, 5, 3] = 120x + ISTFT (hop=4) = 480x total
- Snake activation in residual blocks
- ISTFT reconstruction (n_fft=16) → 24kHz waveform

## Voice Cloning
- CAM++ speaker encoder (CoreML, Neural Engine) extracts 192-dim embedding from reference audio
- Affine projection (192 → 80) conditions DiT flow model on target voice
- Model: `aufklarer/CamPlusPlus-Speaker-CoreML` (~14 MB, FP16)
- CLI: `--voice-sample reference.wav`

## Multi-Speaker Dialogue
- `DialogueParser` parses `[S1] text [S2] text` speaker tags into `DialogueSegment` structs
- `DialogueSynthesizer` orchestrates per-segment synthesis with per-speaker embeddings
- Configurable silence gaps (`--turn-gap`, default 0.2s) and linear crossfade (`--crossfade`)
- CLI: `--speakers s1=alice.wav,s2=bob.wav` loads CAM++ once, extracts per-speaker embeddings

## Emotion / Style Tags
- Inline `(emotion)` tags map to instruction prefixes before `<|endofprompt|>` token
- 8 built-in: happy/excited, sad, angry, whispers/whispering, laughs/laughing, calm, surprised, serious
- Unknown tags pass through as freeform instructions: `(Speak like a pirate)`
- `--cosy-instruct` sets the global default instruction (replaces "You are a helpful assistant.")
- Text format: `{instruction}<|endofprompt|>(token 151646){text_to_synthesize}`
- For zero-shot cloning, a non-default instruction uses CosyVoice's `instruct2`
  layout: the LLM receives the framed instruction and target text, while the
  reference audio remains attached to the Flow model as the voice anchor. The
  unstyled path continues to use the reference transcript plus speech-token
  context for maximum speaker fidelity.

## Streaming
- Chunk-aware causal masking in DiT
- 25-token chunks (~1 second of audio)
- Target: ~150ms latency to first chunk

## Weight Conversion
- Source: FunAudioLLM/Fun-CosyVoice3-0.5B-2512 (PyTorch .pt)
- HiFi-GAN: float32 (weight-norm folded)
- Conv1d weights transposed: PyTorch [out,in,k] → MLX [out,k,in]

### Available bundles

| Variant | LLM | DiT | speech_tokenizer | Total | HF repo |
|---|---|---|---|---:|---|
| **4-bit** (default) | int4, group=64 | bf16 | bf16 | ~1.2 GB | [aufklarer/CosyVoice3-0.5B-MLX-4bit](https://huggingface.co/aufklarer/CosyVoice3-0.5B-MLX-4bit) |
| **8-bit** | int8, group=64 | bf16 | bf16 | ~1.4 GB | [aufklarer/CosyVoice3-0.5B-MLX-8bit](https://huggingface.co/aufklarer/CosyVoice3-0.5B-MLX-8bit) |
| **8-bit-full** | int8, group=64 | int8, group=64 | bf16 | ~1.6 GB | [aufklarer/CosyVoice3-0.5B-MLX-8bit-full](https://huggingface.co/aufklarer/CosyVoice3-0.5B-MLX-8bit-full) |
| **bf16** | bf16 | bf16 | bf16 | ~2.1 GB | [aufklarer/CosyVoice3-0.5B-MLX-bf16](https://huggingface.co/aufklarer/CosyVoice3-0.5B-MLX-bf16) |

The runtime declares every weight-bearing matmul in the LLM and the DiT
as `Linear`. Bundles that ship `.scales` for a given projection get a
per-path `Linear → QuantizedLinear` swap at load time; bundles that omit
the `quantization` block (bf16) stay in plain `Linear` form. This lets a
single runtime serve all four variants from the same module hierarchy.

Pick via the CLI:

```bash
speech speak "Hello, world" --engine cosyvoice --cosyvoice-variant bf16 -o hi.wav
```

`--model-id` still wins if you point it at a custom HF repo.

## Configuration
Key parameters from cosyvoice3.yaml:
- Sample rate: 24000 Hz
- Mel: 80 bins, n_fft=1920, hop=480
- Token frame rate: 25 Hz, mel frame rate: 50 Hz

## References
- CosyVoice 3 paper: https://arxiv.org/abs/2505.17589
- Model: https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
