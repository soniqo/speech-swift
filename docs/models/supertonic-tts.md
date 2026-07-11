# Supertonic TTS Architecture

SupertonicTTS-3 is a 99M-parameter **non-autoregressive flow-matching** text-to-speech model from
Supertone Inc. ([arXiv:2503.23108](https://arxiv.org/abs/2503.23108)). It runs on the Neural Engine
/ GPU via four CoreML graphs the host orchestrates, producing 44.1 kHz speech across 31 languages.
Its defining trait is a **G2P-free** front-end: no espeak, no phonemizer, no lexicon — text is
NFKD-normalized and mapped codepoint→id through a Unicode index table.

## Overview

- **Parameters**: 99M
- **Backend**: CoreML (ANE/GPU), four graphs
- **Output**: 44.1 kHz mono Float32 PCM
- **Inference**: Non-autoregressive flow-matching (fixed-step ODE; no AR loop, no KV cache)
- **Voices**: 10 presets (F1–F5, M1–M5)
- **Languages**: 31 (+ a neutral `na` tag)
- **License**: OpenRAIL-M (weights), MIT (code) — use restrictions pass through
- **Model**: [`aufklarer/Supertonic-3-CoreML`](https://huggingface.co/aufklarer/Supertonic-3-CoreML) ·
  FP16 ANE variant: [`aufklarer/Supertonic-3-CoreML-FP16`](https://huggingface.co/aufklarer/Supertonic-3-CoreML-FP16)

## Pipeline

```
Text → Tokenizer (NFKD + <lang> wrap + index)
          │  text_ids[1,128] + text_mask[1,1,128]
          ├──────────────► DurationPredictor ──► duration (sec) ÷ speed
          └──────────────► TextEncoder ───────► text_emb[1,256,128]
                                                       │
   noise: randn[1,144,L] × latent_mask  (L = ceil(dur·44100 / 3072), dynamic) 
                                                       ▼
        for step in 0..<total_step:  VectorEstimator(noisy, text_emb, style, …) ──► denoised  (ODE)
                                                       ▼
                                  Vocoder(latent[1,144,L]) ──► wav[1, 3072·L]  →  trim to ⌊44100·dur⌋
```

1. **Tokenizer** (`SupertonicTokenizer`, Swift) — NFKD via `decomposedStringWithCompatibilityMapping`,
   cleanup, `<lang>…</lang>` wrap, then per-codepoint lookup in `unicode_indexer.json` (-1 for OOV).
   No grapheme-to-phoneme step; the NFKD pass decomposes accents and Hangul into in-vocab components,
   which is why German and Korean need no special-casing.
2. **DurationPredictor** — predicts utterance seconds from `text_ids` + the `style_dp` voice array.
3. **TextEncoder** — encodes `text_ids` + `style_ttl` into a 256-dim sequence embedding.
4. **VectorEstimator** — one flow-matching ODE step; the host runs it `total_step` times (default 8),
   feeding the denoised latent back in. The latent axis **L is a `RangeDim` (17–512)**, so the graph
   runs at the *true* utterance length — no fixed-window truncation (unlike the fixed-L LiteRT path).
5. **Vocoder** — turns the final latent into 44.1 kHz PCM; the host trims to `⌊44100·duration⌋`.

## Model I/O

| Graph | Inputs | Output |
|---|---|---|
| `DurationPredictor` | `text_ids`[1,128] i32, `style_dp`[1,8,16], `text_mask`[1,1,128] | `duration`[1] |
| `TextEncoder` | `text_ids`, `style_ttl`[1,50,256], `text_mask` | `text_emb`[1,256,128] |
| `VectorEstimator` | `noisy_latent`[1,144,L], `text_emb`, `style_ttl`, `latent_mask`[1,1,L], `text_mask`, `current_step`[1], `total_step`[1] | `denoised_latent`[1,144,L] |
| `Vocoder` | `latent`[1,144,L] | `wav`[1,3072·L] |

`text_ids` is **int32** on CoreML (the export wraps it); CoreML binds every input by name, so feature
order is irrelevant.

## Voices

10 precomputed style presets in `voice_styles/<id>.json`, each holding a `style_ttl` (`[1,50,256]`) and
`style_dp` (`[1,8,16]`) array. On-device voice cloning is out of scope (the style extractor is not
released).

## Usage

```swift
import SupertonicTTS

let tts = try await SupertonicTTSModel.fromPretrained()        // aufklarer/Supertonic-3-CoreML
let pcm = try tts.synthesize(text: "Hello there", voice: "F1", language: "en")  // [Float] @ 44.1 kHz
```

It also conforms to `SpeechGenerationModel` (`generate(text:language:)` → `[Float]`), so it drops into
the shared pipeline / CLI / server like the other TTS models.

## Source

Exported first-party from [`Supertone/supertonic-3`](https://huggingface.co/Supertone/supertonic-3)
ONNX → PyTorch → `coremltools` (`mlprogram`, `RangeDim` on the latent/time axes). The Swift front-end
is a faithful port of the reference `helper.py::UnicodeProcessor`.

## Responsible use

OpenRAIL-M use restrictions pass through: no non-consensual voice impersonation / deepfakes, no
undisclosed machine-generated content, no fraud or disinformation.
