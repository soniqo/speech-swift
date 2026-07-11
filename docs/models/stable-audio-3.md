# Stable Audio 3 Model

MLX text-to-audio backend exposed by the `StableAudio3MusicGen` target and by
the default `speech compose --engine sa3` CLI path.

## Overview

The runtime currently supports the Stable Audio 3 Medium DiT family:

- `aufklarer/Stable-Audio-3-DiT-Medium-MLX-8bit` (default)
- `aufklarer/Stable-Audio-3-DiT-Medium-MLX-4bit`

The public variant enum also names Small Music and Small SFX bundles, but the
current Swift generator intentionally rejects those families until the small
DiT/SAME-S path is wired end to end.

## Pipeline

```text
Prompt
  -> T5Gemma text encoder
  -> Stable Audio 3 DiT-Medium rectified-flow sampler
  -> SAME-L decoder
  -> 44.1 kHz stereo Float32 PCM
```

The generator uses an 8-step ping-pong sampler by default. Output length is
variable; the CLI defaults to 30 seconds and writes a 44.1 kHz stereo WAV.

## Model Components

| Component | Shape |
|---|---|
| DiT-Medium | 24 layers, 1536 dim, 24 heads, differential attention |
| T5Gemma text encoder | 12 layers, 768 dim, 12 heads |
| SAME-L decoder | 12 blocks, 1536 dim, stereo audio decoder |
| Audio | 44.1 kHz stereo, 4096 samples per latent step |

## Swift API

```swift
import StableAudio3MusicGen

let model = try await StableAudio3MusicGen.fromPretrained(
    variant: .mediumInt8
)

let audio = model.generate(
    prompt: "cinematic synthwave intro, wide stereo, bright drums",
    params: StableAudio3GenerationParams(
        seconds: 15,
        steps: 8,
        cfgScale: 1.0,
        seed: 42
    )
)

// audio.left and audio.right are Float32 channel arrays at 44.1 kHz.
```

## Source Files

```text
Sources/StableAudio3MusicGen/
  StableAudio3MusicGen.swift  Main loader and generation pipeline
  Configuration.swift         Variants, architecture constants, component names
  DiTMedium.swift             Medium DiT implementation
  T5GemmaEncoder.swift        Text encoder
  SAMELDecoder.swift          SAME-L audio decoder
  Downloader.swift            HuggingFace bundle downloader
```
