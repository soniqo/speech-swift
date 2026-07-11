# Chatterbox Multilingual TTS Architecture

## Overview

Chatterbox Multilingual (Resemble AI, MIT) is a zero-shot voice-cloning TTS that
synthesizes a target text in the timbre of a short reference clip. The upstream
model publishes 23 language tags; the Swift runtime enables all 23, with Hebrew
requiring pre-diacritized text (niqqud) until automatic Dicta ONNX
diacritization is bundled. The MLX bundle is a genuine fp16 conversion
(~1.3 GB) of the three upstream checkpoints, published at
[`aufklarer/Chatterbox-Multilingual-MLX-fp16`](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16).

The Swift port is built component-by-component, with token ids, mel features, and
speaker embeddings checked against a known-good reference at each stage.

## Languages

Runtime-enabled languages: Arabic, Chinese, Danish, Dutch, English, Finnish,
French, German, Greek, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Norwegian,
Polish, Portuguese, Russian, Spanish, Swahili, Swedish, Turkish.

Text encodes through a grapheme path: NFKD normalization → a per-language
`[lang]` token → BPE. Frontend-heavy scripts are handled before that path:

- Chinese uses word segmentation plus the published Cangjie-5 token map.
- Japanese converts kanji readings to hiragana through `CFStringTokenizer`.
- Korean decomposes Hangul syllables to Jamo.
- Hebrew accepts pre-diacritized input with niqqud. Raw Hebrew is rejected with a
  clear error until the runtime includes automatic Dicta ONNX diacritization.

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

The MLX clone path uses `ChatterboxMemoryOptions.balanced` by default. During
`clone(...)` and standalone `ChatterboxS3Gen.synthesize(...)`, the runtime
temporarily caps MLX cache memory at up to 512 MB without raising a stricter
caller cap, clears reusable buffers between T3, S3Gen flow, and vocoder stages,
then restores the caller's previous cache limit. This keeps long-form synthesis
from retaining large stage-local Metal buffers. A 72 s Russian clone on the
local M-series test machine dropped from ~39.0 GB process footprint to ~7.4 GB
with this policy. Pass
`memoryOptions: .unrestricted` to preserve MLX's default cache behavior for
maximum-throughput experiments.

## Chatterbox Flash Core ML

`ChatterboxFlashCoreMLModel` loads the published
[`aufklarer/Chatterbox-Flash-CoreML`](https://huggingface.co/aufklarer/Chatterbox-Flash-CoreML)
bundle from the `ChatterboxTTS` product. It exports the Flash T3 block decoder
and the S3Gen audio back half as compiled Core ML graphs:

- `t3/ConditioningEncoder.mlmodelc`
- `t3/TextPrefill.mlmodelc`
- `t3/NullTextPrefill.mlmodelc` (optional; required only for `cfgScale > 0`)
- `t3/BlockDecoder.mlmodelc`
- `audio/FlowSpeakerProjector.mlmodelc`
- `audio/FlowEncoder.mlmodelc`
- `audio/FlowEstimator.mlmodelc`
- `audio/HiFTVocoder.mlmodelc`

The Swift runtime owns the host-side Flash loop: BPE text tokenization,
compact T3 prefix planning, optional classifier-free guidance through
`NullTextPrefill`, PMI ranking against `uncond_block_prior.npy`, block unmask
scheduling, EOS trimming, and S3Gen waveform synthesis.

```swift
import ChatterboxTTS

let flash = try await ChatterboxFlashCoreMLModel.fromPretrained()

// Reference-audio encoding is supplied by the MLX Chatterbox model.
let mlx = try await ChatterboxTTSModel.fromPretrained()
let conditioning = try mlx.prepareFlashConditioning(
    referenceSamples: referenceAudio,
    sampleRate: referenceSampleRate
)

let audio24k = try flash.generate(
    text: "Core ML speech test.",
    conditioning: conditioning
)
```

Voice cloning is supported when the caller provides reference conditioning
tensors. The convenience bridge above uses the existing MLX VoiceEncoder,
S3Tokenizer, and S3Gen reference encoder to create those tensors from a
reference waveform. `ChatterboxTTSModel.fromPretrained(localDir:
s3TokenizerWeights:)` accepts either the standalone converted S3TokenizerV2
weights or Flash `s3gen.safetensors`; for the latter it extracts and converts
the `tokenizer.*` tensors.

Fully Core ML `ref.wav -> cloned wav` is not complete yet because the
reference-audio encoders are not included in the Flash Core ML bundle.
Generated speech defaults to `cfgScale: 0.0` for compatibility with bundles
that do not include a null-branch graph. Passing a positive `cfgScale` requires
`t3/NullTextPrefill.mlmodelc` in the bundle.

Voice-cloning quality also depends on the exported S3Gen audio bucket. The
audio graph consumes `prompt_token + generated_speech_tokens`; the full MLX
path gives S3Gen up to 10 seconds of prompt audio, while Flash Core ML caps the
reference prompt to 6 seconds so the public 192-token audio export still leaves
room for generated speech. A 192-token audio export is enough for smoke tests,
but it forces short utterances and limited prompt audio. Use a larger audio
export, for example `token_len=512` and `mel_len=1024`, when validating speaker
similarity.

Opt-in E2E tests:

```bash
CHATTERBOX_FLASH_COREML_PATH=/path/to/Chatterbox-Flash-CoreML \
CHATTERBOX_REFERENCE_WAV=/path/to/reference.wav \
  swift test --filter E2EChatterboxFlashCoreMLTests
```

Regression tests that do not require model files:

```bash
swift test --filter ChatterboxFlashCoreMLTests
swift test --skip E2E
```

## References

- [Chatterbox](https://github.com/resemble-ai/chatterbox) (Resemble AI, MIT)
- [Chatterbox Flash](https://github.com/resemble-ai/chatterbox-flash) (Resemble AI, MIT)
