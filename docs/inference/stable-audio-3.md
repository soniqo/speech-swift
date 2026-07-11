# Stable Audio 3 Inference

Stable Audio 3 is the default engine for `speech compose`. It generates
44.1 kHz stereo music/audio from a text prompt using MLX.

## CLI

```bash
speech compose "lofi house loop with warm bass" -o loop.wav
```

Useful options:

```bash
speech compose "cinematic trailer drums" \
  --engine sa3 \
  --sa3-variant medium-int8 \
  --seconds 15 \
  --sa3-steps 8 \
  --sa3-cfg 1.0 \
  --seed 42 \
  -o trailer.wav
```

| Flag | Meaning |
|---|---|
| `--engine sa3` | Select Stable Audio 3. This is the default. |
| `--sa3-variant` | `medium-int8` or `medium-int4`. |
| `--seconds` | Output duration in seconds. |
| `--sa3-steps` | Ping-pong sampler steps. Default: 8. |
| `--sa3-cfg` | Classifier-free guidance. `1.0` disables CFG. |
| `--sigma-max` | Initial noise level. Default: `1.0`. |
| `--sa3-bundle-dir` | Load a local exported bundle instead of downloading. |
| `--seed` | Reproducible generation seed. |

The command writes a stereo WAV at 44.1 kHz. Use `--engine magnet` when you
want the older MAGNeT path, which writes 32 kHz mono fixed-length clips.

## Swift

```swift
import StableAudio3MusicGen
import AudioCommon

let model = try await StableAudio3MusicGen.fromPretrained(
    variant: .mediumInt8
)

let (left, right) = model.generate(
    prompt: "ambient piano, soft tape hiss, slow evolving pads",
    params: StableAudio3GenerationParams(seconds: 20, seed: 7)
)

try WAVWriter.writeStereo(
    left: left,
    right: right,
    sampleRate: StableAudio3MusicGen.sampleRate,
    to: URL(fileURLWithPath: "ambient.wav")
)
```

## Current Limits

- Only the Medium family is wired for generation today.
- Small Music / Small SFX variants are named in `StableAudio3Variant` but throw
  `unsupportedFamily` until their DiT/SAME-S path is implemented.
- Generation is local and model-heavy; use the INT4 bundle when memory pressure
  matters more than quality.
