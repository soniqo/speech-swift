# Sidon Inference

CLI command: `speech restore`. On-device speech restoration (joint denoise +
dereverb) with Sidon. 16 kHz in → **48 kHz** mono out. Also exposed as
`speak --clean-reference` for cleaning a voice-cloning reference before TTS.

## Quick start

```bash
# Build (release + metallib for best speed)
make build

# Restore a noisy / reverberant recording
.build/release/speech restore noisy.wav -o clean.wav

# Smaller, lower peak RAM (int8-palettized predictor)
.build/release/speech restore noisy.wav --variant int8 -o clean.wav
```

## Flags

```
--variant {fp16,int8}   # precision variant (default fp16)
--model, -m REPO        # HuggingFace repo id (default aufklarer/Sidon-CoreML)
--output, -o PATH       # output WAV path (default <input>_restored.wav, 48 kHz)
```

## Input handling

- Input can be any sample rate; it's resampled internally to 16 kHz mono.
- Audio is processed in non-overlapping 10 s windows (160000 samples each).
- Each window is padded to the fixed graph length, restored, and concatenated.
- Output is 48 kHz mono, trimmed to the input's true duration on the 48 kHz
  timeline (each window emits a fixed number of samples regardless of how much
  of it was real audio, so partial-window padding is removed).

## Clean a voice-cloning reference (`speak --clean-reference`)

`--clean-reference` runs the voice-cloning reference through Sidon (denoise +
dereverb) before the cloner extracts speaker conditioning. It is **opt-in** — it
changes the audio materially and shifts speaker similarity slightly (per the
upstream caveat), so it never runs implicitly. Applies to the qwen3 / cosyvoice /
voxcpm2 reference paths.

```bash
# VoxCPM2 cloning with a cleaned reference
speech speak "Hello there" --engine voxcpm2 --voice-sample noisy_ref.wav --clean-reference -o out.wav

# Pick the variant used for the cleanup pass
speech speak "Hello there" --engine cosyvoice --voice-sample ref.wav \
    --clean-reference --clean-reference-variant int8 -o out.wav
```

Flags:

```
--clean-reference                       # enable Sidon cleanup of the reference (opt-in)
--clean-reference-variant {fp16,int8}   # variant for the cleanup pass (default fp16)
```

Sidon's pipeline is 16 kHz in / 48 kHz out; the cleaned 48 kHz result is resampled
to whatever the TTS engine wants (24 kHz for qwen3, 16 kHz for cosyvoice/voxcpm2).

## Programmatic use

```swift
import SpeechRestoration
import AudioCommon

let restorer = try await SpeechRestorer.fromPretrained(variant: .fp16)
let clean = try restorer.restore(audio: noisy, sampleRate: 16_000)  // → 48 kHz
try WAVWriter.write(samples: clean, sampleRate: SpeechRestorer.outputSampleRate,
                    to: URL(fileURLWithPath: "clean.wav"))
```

Load locally-converted artifacts directly (no HuggingFace fetch):

```swift
let restorer = try SpeechRestorer.fromLocalBundle(
    directory: URL(fileURLWithPath: "/path/to/sidon-coreml"), variant: .fp16)
```

`SpeechRestorer` also conforms to `SpeechEnhancementModel`, so `enhance(audio:sampleRate:)`
is an alias for `restore`. Note the **output is 48 kHz** (the DAC vocoder),
unlike DeepFilterNet3 which returns audio at its input rate — callers that care
should use `restore` directly and read `SpeechRestorer.outputSampleRate`.

## Pipeline

1. **Resample** input to 16 kHz mono.
2. **SeamlessM4T log-mel front-end** (DSP): 16-bit scale → frame → DC-remove →
   pre-emphasis → Povey window → 512-pt rFFT → power → kaldi mel → log →
   per-bin normalize → stride-2 stacking → `input_features[1, T, 160]`.
3. **Predictor** (w2v-BERT 2.0 + merged LoRA): `features[1, T, 1024]`.
4. **Vocoder** (DAC decoder, ×960): 48 kHz waveform per window.
5. **Stitch + trim** windows to the input's true 48 kHz duration.

See [docs/models/sidon.md](../models/sidon.md) for the model architecture and
variant details.

## License

Inherits Sidon's upstream license (sarulab-speech). The w2v-BERT 2.0 front-end
matches HuggingFace's `SeamlessM4TFeatureExtractor`.
