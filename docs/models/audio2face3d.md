# Audio2Face-3D Avatar Motion

`Audio2Face3D` is the avatar-motion boundary for speech-driven faces. The
public output target is `Audio2Face3DFrame`: a timestamp plus the full NVIDIA
Audio2Face-3D coefficient vector for downstream avatar renderers. This module
does not synthesize viseme or energy-derived controls.

## Architecture

Audio2Face-3D is a two-branch audio-to-coefficient regressor. Per 8320-sample
window of 16 kHz mono audio (hop 4160) it emits one frame of geometry
coefficients, conditioned on a 26-value emotion vector (16 implicit + 10
explicit; the explicit part is projected to 8 dims by a learned linear layer).

```text
audio window ──┬─ audio branch: Wav2Vec2 feature extractor (7 convs, GELU)
               │    -> feature projection -> grouped positional conv
               │    -> transformer encoder layers -> learned layer blend
               │    -> linear audio head
               │
               └─ frequency branch: 25 Hann-windowed frames
                    x 32 autocorrelation lags -> 5 conv stages

fusion:  concat both branches -> 4 strided conv stages, with the emotion
         vector broadcast onto the channels before each stage -> dense (256)
outputs: [features, emotion] -> face head (skin + jaw + eyes, two stacked
         linears) and tongue head (two stacked linears)
```

- The audio branch is a trimmed Wav2Vec2 encoder; per-layer transformer
  outputs are blended with the learned `combine_weights` vector.
- The frequency branch computes windowed autocorrelation (pitch/periodicity
  cues) directly in the graph — no FFT dependency.
- Input audio is pre-scaled by `input_strength` from `model_config.json`.

Coefficient layouts and encoder depth are identity-specific:

| Identity | Version | Encoder layers | Skin | Tongue | Jaw | Eyes | Total |
|----------|---------|----------------|------|--------|-----|------|-------|
| Mark | v2.3 | 1 | 272 | 10 | 15 | 4 | 301 |
| Claire | v2.3.1 | 4 | 140 | 10 | 15 | 4 | 169 |
| James | v2.3.1 | 4 | 140 | 10 | 15 | 4 | 169 |

Skin coefficients live in the identity's own geometry basis, so renderers need
the matching rig or a retarget projection for the chosen identity.

## Current Runtime

The only target backend is the full NVIDIA Audio2Face-3D model exported to MLX
tensors. The hand-written MLX graph port runs the real model forward pass and
is parity-checked against an ONNX fixture generated from the official NVIDIA
model, which is distributed under the NVIDIA Open Model License.

Published bundles, downloaded on first use:

- [`aufklarer/Audio2Face-3D-v2.3.1-James-MLX`](https://huggingface.co/aufklarer/Audio2Face-3D-v2.3.1-James-MLX) (default)
- [`aufklarer/Audio2Face-3D-v2.3.1-Claire-MLX`](https://huggingface.co/aufklarer/Audio2Face-3D-v2.3.1-Claire-MLX)
- [`aufklarer/Audio2Face-3D-v2.3-Mark-MLX`](https://huggingface.co/aufklarer/Audio2Face-3D-v2.3-Mark-MLX)

Generate coefficient frames from a WAV file:

```bash
swift run speech avatar-motion input.wav --output motion.jsonl --verbose
```

Pass `--model` to select another identity bundle, or `--model-dir` to run a
locally exported bundle.
