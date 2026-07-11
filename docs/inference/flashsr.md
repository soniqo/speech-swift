# FlashSR Inference

CLI command: `speech upsample`. Audio super-resolution at 48 kHz mono.

## Quick start

```bash
# Build (release + metallib for best speed)
make build

# Upsample a low-bandwidth recording
.build/release/speech upsample noisy_lowres.wav -o clean_hr.wav

# Pick the int8 variant (slightly more accurate quantisation, ~720 MB on disk)
.build/release/speech upsample input.wav --variant int8 -o out.wav

# Reproducible noise (DPM-Solver init)
.build/release/speech upsample input.wav --seed 42 -o out.wav
```

## Flags

```
--variant {int4,int8}   # default int4; both run at FP after load
--timestep N            # diffusion timestep (default 999 for the 1-step student)
--seed N                # random seed for the noise init
--output, -o PATH       # output WAV path (default hr.wav)
```

## Input handling

- Input can be any sample rate; it's resampled internally to 48 kHz mono.
- Inputs are processed in non-overlapping 5.12 s windows.
- Output length matches the input length (trimmed after windowed processing).
- A single window is the most common case for music / SFX clips.

## Programmatic use

```swift
import FlashSR

let model = try await FlashSR.fromPretrained(variant: .int4)
let hr = try model.upsample(audio: lowRes, sampleRate: 48_000)
try WAVWriter.write(samples: hr, sampleRate: 48_000,
                    to: URL(fileURLWithPath: "hr.wav"))
```

The model also conforms to `SpeechEnhancementModel`, so it slots into any
`enhance(audio:sampleRate:)` consumer (e.g. an ASR pre-processor) — same
signature as DeepFilterNet3 even though the semantics differ (super-resolution
vs noise suppression). Pick by import: `import FlashSR` vs `import SpeechEnhancement`.

## Pipeline

1. **Normalize**: mean-center + max-abs scale to ±0.5.
2. **Log-mel spectrogram**: HiFi-GAN-style (reflection pad, Hann window, 256 mels,
   Slaney filterbank, log(clamp(., 1e-5))).
3. **VAE encode**: 8× downsample to a 16-channel latent, scaled by 0.3342.
4. **One-step DPM-Solver** (v-prediction, cosine α̅): one UNet forward pass.
5. **VAE decode**: mel back from the denoised latent.
6. **SR Vocoder**: BigVGAN-style with SnakeBeta activations + audio
   conditioning pyramid; produces 48 kHz waveform.
7. **Denormalize**: re-apply mean + scale.

## License

Inherits FlashSR's upstream license: **MIT** (KAIST research). Output is free
to use commercially; the weights themselves carry no use restrictions.
