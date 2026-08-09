# SpeechBrain ECAPA Language Identification

`SpeechLanguageID` runs the compact 21.25M-parameter SpeechBrain VoxLingua107
classifier with either native MLX or compiled Core ML. Both FP16 exports come
from the same pinned upstream checkpoint, preserve its classifier order, and
ship source hashes plus byte-level artifact manifests.

## Architecture

```
16 kHz mono audio
  → SpeechBrain-compatible 60-bin log-mel frontend
  → temporal mean normalization
  → ECAPA-TDNN encoder (256-dimensional embedding)
  → BatchNorm / LeakyReLU classifier
  → 107 log probabilities
```

The ECAPA encoder contains an initial TDNN block, three SE-Res2Net blocks,
multi-layer feature aggregation, and attentive statistics pooling. Temporal
convolutions use SpeechBrain's reflected edge padding. The native frontend also
matches SpeechBrain's centered 400-sample DFT, periodic Hamming window,
160-sample hop, symmetric triangular filters, and 80 dB log-mel clipping.

## Model files

| Backend | Model | Artifact | Minimum output cosine |
|---------|-------|---------:|----------------------:|
| MLX | [`aufklarer/SpeechBrain-ECAPA-VoxLingua107-21M-MLX`](https://huggingface.co/aufklarer/SpeechBrain-ECAPA-VoxLingua107-21M-MLX) | 40.6 MiB | 0.99999987 |
| Core ML | [`aufklarer/SpeechBrain-ECAPA-VoxLingua107-21M-CoreML`](https://huggingface.co/aufklarer/SpeechBrain-ECAPA-VoxLingua107-21M-CoreML) | 40.8 MiB | 0.99996739 |

Each export includes `config.json`, the exact 107-label `labels.json`, license,
conversion validation, a runnable reference frontend, and an artifact manifest.
The Core ML repository contains a compiled `.mlmodelc` bundle; the MLX
repository contains safetensors and the reference MLX graph.

The source checkpoint is pinned to
`speechbrain/lang-id-voxlingua107-ecapa@0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9`.
Export validation compares 10, 101, 301, 1,001, and 3,001 mel-frame inputs
against the official PyTorch graph. All representative top-1 predictions match.

## Constraints

- Input is mono audio and is resampled to 16 kHz when needed.
- The minimum accepted input is approximately 90 ms, but very short clips are
  not expected to classify reliably. Prefer several seconds of voiced speech.
- A single inference window supports up to approximately 30 seconds.
- Longer recordings are split into non-overlapping windows and probabilities
  are combined with duration weighting.
- This is a closed-set classifier. It always ranks one of its 107 labels and
  does not provide calibrated unknown-language or code-switch detection.
- The upstream label codes are preserved exactly, including legacy `iw` for
  Hebrew and `jw` for Javanese.

The upstream model card reports 6.7% error on a manually verified development
set containing 1,609 clips from only 33 of the 107 languages. That result is
not a complete 107-language benchmark.

Product thresholds and unknown-language rejection should be calibrated with
representative accents, noise, short clips, and out-of-inventory languages.
