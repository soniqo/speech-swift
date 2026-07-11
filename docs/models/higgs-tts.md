# Higgs TTS 3

Higgs TTS 3 is a ~4B conversational text-to-speech model from Boson AI
(`bosonai/higgs-tts-3-4b`, formerly `higgs-audio-v3-tts-4b`), built for voice
chat: expressive delivery, zero-shot voice cloning, 100+ languages, and inline
control tokens for emotion, style, prosody, and sound effects. This package
ships a native Swift/MLX runtime for the `aufklarer/Higgs-TTS-3-4B-MLX-bf16`
bundle.

## Status

| Item | Value |
|---|---|
| Module | `HiggsTTS` |
| Upstream | `bosonai/higgs-tts-3-4b` |
| Default bundle | `aufklarer/Higgs-TTS-3-4B-MLX-bf16` |
| Backend | MLX (bf16 backbone, float32 codec) |
| Sample rate | 24 kHz output, 25 fps frames |
| Runtime | Qwen3-4B decoder + fused 8-codebook head + Higgs codec |
| License | Boson Higgs TTS 3 Research and Non-Commercial License |
| Streaming | Not supported |

## Architecture

One checkpoint carries the full system:

- **Backbone** (`body.*`): a standard Qwen3-4B dense decoder — 36 layers,
  hidden 2560, GQA 32/8 with per-head q/k RMSNorm before RoPE, SwiGLU MLPs.
- **Fused codebook interface** (`tied.embedding.modality_embeddings.0.embedding`):
  a single `[8 × 1026, 2560]` table. Input audio frames embed each codebook's
  code at `code + codebook × 1026` and sum the eight vectors; output logits
  reuse the same table as a tied linear head reshaped to `[8, 1026]`.
- **Delay pattern**: MusicGen-style offsets `[0..7]` with `BOC = 1024` leading
  the upper codebooks and `EOC = 1025`; generation stops `N − 2` steps after
  codebook 0 emits EOC.
- **Codec** (`tied.embedding.modality_embeddings.0.model.*`): 8-layer RVQ over
  a 1024-dim latent, a 1024→256 projection, and a DAC-style decoder (Snake
  activations, transposed-conv upsampling by `[8, 5, 4, 2, 3]`) producing 960
  samples per frame. The encode side adds a DAC-style acoustic encoder, an
  embedded HuBERT semantic model, and a semantic/acoustic fusion projection.

## Prompt protocol

`<|tts|>` [ `<|ref_text|>` transcript ] `<|ref_audio|>` delay-patterned
reference codes `<|text|>` target text `<|audio|>` → autoregressive audio
frames. Control tags (`<|emotion:elation|>`, `<|style:whispering|>`,
`<|sfx:laughter|>`, `<|prosody:pause|>`, …) are ordinary added tokens placed
inside the target text.

## Validation Status

The Swift runtime is cross-validated against the reference implementations
(mlx-audio primary; vllm-omni and sglang-omni agree on every mechanic — see
the exporter's REFERENCES.md in speech-models):

- **LM parity** (teacher-forced replay, identical inputs): logits agree to
  bf16 accumulation tolerance (max |Δ| 1.0 over ~tens-scale logits), 62/64
  greedy argmax agreement; the two misses are near-tie flips.
- **Codec decode parity** (identical codes in): waveform diff RMS ≈ 0.1% of
  signal RMS (max |Δ| 7.5e-4).
- **Codec encode**: per-codebook agreement with the reference decays
  82% → 36% from book 0 to book 7 — the expected residual-quantization
  cascade, where a near-tie flip in an early book decorrelates all later
  books. The functional gate below is what matters; encoding a 12 s
  reference takes ~0.6 s.
- **End-to-end native gates** (Swift encoder → LM → decoder, cloned voice,
  Qwen3-ASR roundtrips at temperature 0.8): English lexical overlap 1.00,
  Mandarin CER 0.000.
- **Speed**: RTF 0.78 (release build, Apple M5 Pro) for both cloned and
  reference-free synthesis after the pipelined decode loop — on-device
  sampling with a GPU-masked delay ramp and `asyncEval` overlap; the
  remaining cost is the 4B bf16 memory-bandwidth roofline. See
  `docs/benchmarks/voice-cloning-tts.md`.
- The CLI defaults to temperature 0.8: at the reference's 1.0, single-sentence
  roundtrips wobble (measured across two full reference gate runs).

Remaining work:

- Streaming synthesis (chunked codec decode as frames arrive) and per-voice
  prefix KV caching for repeated same-voice generations.
- Broader language QA beyond the en/zh/es/de/ja gate set.

Quantized variants are intentionally not planned: synthesis stays at
reference precision because quantization audibly degrades speech.
