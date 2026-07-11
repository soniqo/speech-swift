# VoxCPM2 - Tokenizer-Free Multilingual TTS

> Reference for the Swift MLX port. Converted from [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) into [aufklarer/VoxCPM2-MLX-{bf16,int8,int4}](https://huggingface.co/aufklarer).

## Overview

VoxCPM2 is a 2B-parameter multilingual TTS model with 30 languages, 48 kHz output, voice design, controllable voice cloning, and prompt-audio continuation. The Swift port exposes the model as `VoxCPM2TTSModel`.

Key properties:

- Input reference audio is accepted at 16 kHz and resampled internally
- Output audio is produced at 48 kHz
- The upstream model supports zero-shot TTS, voice design, controllable cloning, and ultimate cloning
- The Swift port keeps the same operational modes through a single `generateVoxCPM2(...)` API

## Pipeline

```
Text + optional instruct + optional reference/prompt audio
    |
    v
Tokenizer / prompt formatting
    |
    v
MiniCPM-4 backbone
  - base LM
  - residual LM
    |
    v
LocEnc + feature projection
    |
    v
FSQ + UnifiedCFM / LocDiT
    |
    v
AudioVAE V2 decoder
    |
    v
48 kHz waveform
```

## Mode Matrix

| Mode | Inputs | Swift entry point |
|---|---|---|
| Zero-shot | Text only | `generate(text:language:)` |
| Voice design | Text + style instruction | `generateVoxCPM2(..., instruct:)` |
| Controllable cloning | Text + reference audio | `generateVoxCPM2(..., refAudio:)` |
| Ultimate cloning | Text + reference audio + prompt audio + prompt text | `generateVoxCPM2(..., refAudio:, promptAudio:, promptText:)` |

The CLI mirrors the same modes through `speech speak --engine voxcpm2`.

## Model Details

| Property | Value |
|---|---|
| Parameters | ~2B |
| Backbone | MiniCPM-4 |
| Languages | 30 |
| Input reference sample rate | 16 kHz |
| Output sample rate | 48 kHz |
| Architecture | LocEnc -> TSLM -> RALM -> LocDiT |
| Voice design | Supported |
| Controllable cloning | Supported |
| Ultimate cloning | Supported |
| License | Apache-2.0 |

## Special Tokens

The current Swift implementation follows the upstream VoxCPM2 control tokens:

| Token | ID |
|---|---:|
| `audio_start_token` | 101 |
| `audio_end_token` | 102 |
| `ref_audio_start_token` | 103 |
| `ref_audio_end_token` | 104 |

## Swift Implementation Notes

- `VoxCPM2TTSModel.fromPretrained()` defaults to `aufklarer/VoxCPM2-MLX-bf16`
- `generateVoxCPM2(...)` accepts optional `refAudio`, `promptAudio`, `promptText`, and `instruct`
- `language` is accepted for protocol compatibility, but the upstream model auto-detects supported languages
- `AudioVAE` and the LocDiT stack are loaded from the same model directory as the base LM weights
- On Apple Silicon, the Swift runtime promotes the low-precision VoxCPM2 parameters to `float32` by default to mirror the upstream MPS safety policy; `AudioVAE` decode always runs in `float32`

## Weight Bundles

| Bundle | Format | Notes |
|---|---|---|
| `openbmb/VoxCPM2` | PyTorch / HF | Upstream reference model (conversion source) |
| `aufklarer/VoxCPM2-MLX-bf16` | MLX / safetensors | Full-precision Apple Silicon port (default) |
| `aufklarer/VoxCPM2-MLX-int8` | MLX / safetensors | 8-bit group quantization, ~3 GB |
| `aufklarer/VoxCPM2-MLX-int4` | MLX / safetensors | 4-bit group quantization, ~1.9 GB |


## Source Files

```
Sources/VoxCPM2TTS/
  Configuration.swift   ModelArgs, LMConfig, AudioVAEConfig, DiTConfig
  MiniCPM4.swift        MiniCPM-4 backbone layers, LocEnc, LocDiT, UnifiedCFM
  AudioVAE.swift        AudioVAE V2 encoder/decoder
  VoxCPM2TTS.swift      Public model API, loading, generation, memory management
```

## Official Sources

- [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)
- [OpenBMB VoxCPM repository](https://github.com/OpenBMB/VoxCPM)
