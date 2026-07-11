# VoxCPM2 - Inference Pipeline

VoxCPM2 is the native 48 kHz multilingual TTS backend in `speech-swift`. It supports:

- zero-shot synthesis
- voice design from an instruction string
- controllable cloning from a reference clip
- ultimate cloning from reference audio plus transcript

## Quick Start

```swift
import Foundation
import VoxCPM2TTS
import AudioCommon

let model = try await VoxCPM2TTSModel.fromPretrained()
let audio = try await model.generate(text: "Hello from VoxCPM2.")
let outputURL = URL(fileURLWithPath: "hello.wav")
try WAVWriter.write(samples: audio, sampleRate: model.sampleRate, to: outputURL)
```

## CLI Usage

```bash
speech speak "Hello from VoxCPM2." \
  --engine voxcpm2 \
  --output hello.wav

speech speak "Hello from VoxCPM2." \
  --engine voxcpm2 \
  --voxcpm2-instruct "Young female voice, warm and gentle" \
  --output voice-design.wav

speech speak "This is a cloned voice." \
  --engine voxcpm2 \
  --voxcpm2-ref-audio speaker.wav \
  --output clone.wav

speech speak "This is an ultimate cloning demo." \
  --engine voxcpm2 \
  --voxcpm2-ref-audio speaker.wav \
  --voxcpm2-prompt-audio speaker.wav \
  --voxcpm2-prompt-text "reference transcript" \
  --output hifi-clone.wav
```

## CLI Flags

| Flag | Purpose |
|---|---|
| `--voxcpm2-variant` | Quantization variant: `bf16` (default), `int8`, `int4`. Resolves to `aufklarer/VoxCPM2-MLX-<variant>`. |
| `--voxcpm2-instruct` | Style instruction for voice design |
| `--voxcpm2-ref-audio` | Reference audio for cloning |
| `--voxcpm2-prompt-audio` | Prompt audio for continuation |
| `--voxcpm2-prompt-text` | Transcript for the prompt audio |
| `--voxcpm2-cfg-value` | Classifier-free guidance scale |
| `--voxcpm2-timesteps` | Diffusion timesteps per generated patch |
| `--voxcpm2-max-tokens` | Maximum generated patches |
| `--voxcpm2-min-tokens` | Minimum patches before early stop |
| `--voxcpm2-streaming-prefix-len` | Number of prompt patches retained for continuation |
| `--voxcpm2-warmup-patches` | Prefill warmup patches before emission |

## Behavior Notes

- VoxCPM2 auto-detects the supported language from the text, so `--language` is optional
- Reference audio is loaded at 16 kHz and resampled internally when needed
- Output samples are written at 48 kHz
- If `--voxcpm2-prompt-audio` is set, `--voxcpm2-prompt-text` must be provided too
- The HF snapshot includes `tokenization_voxcpm2.py` and `tokenizer_config.json` with `VoxCPM2Tokenizer`; the Swift backend refreshes stale tokenizer snapshots and mirrors the upstream multi-character Chinese token split before encoding
- On Apple Silicon, the Swift backend promotes VoxCPM2 parameters to `float32` by default to match the upstream MPS safety policy, and the `AudioVAE` decode path always receives `float32` latents

## Swift API

```swift
import Foundation
import AudioCommon
import VoxCPM2TTS

let refURL = URL(fileURLWithPath: "speaker.wav")
let promptURL = URL(fileURLWithPath: "prompt.wav")
let refSamples = try AudioFileLoader.load(url: refURL, targetSampleRate: 16000)
let promptSamples = try AudioFileLoader.load(url: promptURL, targetSampleRate: 16000)

let model = try await VoxCPM2TTSModel.fromPretrained(
    modelId: "aufklarer/VoxCPM2-MLX-bf16"
)

let audio = try await model.generateVoxCPM2(
    text: "Hello from VoxCPM2.",
    language: "english",
    refAudio: refSamples,
    promptText: "reference transcript",
    promptAudio: promptSamples,
    cfgValue: 2.0,
    inferenceTimesteps: 10,
    streamingPrefixLen: 4,
    warmupPatches: 0,
    instruct: "Young female voice, warm and gentle"
)
```

## Implementation Checklist

- Model config comes from `config.json` in the HF snapshot
- Weights are loaded from the same snapshot root as the tokenizer and auxiliary assets
- The model returns `SpeechGenerationModel.sampleRate == 48000`
- `unload()` releases all module parameters so the model can be reloaded cleanly
