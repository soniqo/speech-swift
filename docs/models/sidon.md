# Sidon

Core ML port of [Sidon](https://arxiv.org/abs/2509.17052) (sarulab-speech,
ICASSP 2026) — on-device **speech restoration**: joint denoise + dereverb that
**preserves speaker identity**. A w2v-BERT 2.0 predictor cleans degraded speech
features, and a DAC decoder vocoder reconstructs a full-band 48 kHz waveform.

Module: `Sources/SpeechRestoration/` · Library: `SpeechRestoration` · CLI: `speech restore` (+ `speak --clean-reference`)

## What it is

Sidon takes degraded speech (noise, reverberation, codec/channel artifacts) and
restores it while keeping the speaker's timbre intact. Because identity is
preserved, it is well suited to cleaning a noisy/reverberant **voice-cloning
reference** before TTS — unlike a generic noise suppressor, which can scrub away
the very cues a cloner relies on. The pipeline runs window-by-window so any
length of audio is supported.

| | |
|---|---|
| Input | 16 kHz mono (any sample rate accepted; resampled internally) |
| Output | **48 kHz** mono |
| Window | 10 s = 160000 input samples → 499 stacked frames |
| Output per window | 479014 samples (≈ 9.98 s @ 48 kHz, DAC ×960 minus the conv-stack trim) |
| Predictor | w2v-BERT 2.0, 8 layers + merged LoRA (`input_features[1,T,160]` → `features[1,T,1024]`) |
| Vocoder | DAC decoder, `rates=[8,5,4,3,2]` (×960) |
| Variants | `fp16` (default), `int8` (palettized predictor) |
| Licence | upstream Sidon (sarulab-speech) |

## Architecture

Two Core ML graphs plus a DSP front-end:

```
audio (16 kHz)
   │
   ▼  SeamlessM4T log-mel front-end (DSP, Swift/Accelerate)
   │     → input_features[1, T, 160]
   │
   ▼  predictor (w2v-BERT 2.0, 8 layers + merged LoRA)
   │     → features[1, T, 1024]   (cleansed last_hidden_state)
   │
   ▼  vocoder (DAC decoder, rates [8,5,4,3,2], ×960)
   │     → audio (48 kHz)[1, M]
```

The CoreML graphs are exported at a **fixed** sequence length (`SidonConfig.frames`
= 499 ≈ 10 s), so the runtime chunks longer inputs into 160000-sample windows,
restores each, concatenates, and trims the result to the input's true duration
mapped onto the 48 kHz timeline.

### SeamlessM4T front-end

The predictor was exported starting from w2v-BERT 2.0's `input_features`, so the
log-mel extraction is DSP done in the runtime (same pattern as DeepFilterNet3's
STFT). `SeamlessM4TFrontEnd` is a Swift/Accelerate port of HuggingFace's
`SeamlessM4TFeatureExtractor` (the one `facebook/w2v-bert-2.0` ships). The recipe,
validated against the reference extractor (max-abs error ≈ 2.6e-4 on a 1 s clip):

1. Scale to 16-bit: `waveform *= 2^15` (Kaldi compliance; no peak normalization).
2. Frame: `frame_length=400` (25 ms), `hop=160` (10 ms), `center=false`.
3. Per frame: remove DC offset → pre-emphasis (0.97) → Povey window
   (`hann(400)^0.85`) → 512-point rFFT → power spectrum.
4. Mel projection with the kaldi-scale triangular filterbank (80 filters,
   20–8000 Hz, triangularized in mel space), then natural `log()`.
5. Per-mel-bin normalization over time, sample variance (ddof = 1), `eps = 1e-7`.
6. Stride-2 frame stacking: `[T, 80] → [T/2, 160]` (drops a trailing odd frame).

## Variants

| Variant | Predictor weights | Peak RSS (single-artifact process) | Notes |
|---|---|---|---|
| `fp16` | FP16 | ≈ 1711 MB | Default, higher fidelity |
| `int8` | k-means palettized | ≈ 1321 MB | ~half the disk, lower peak RAM, small naturalness cost |

The vocoder is FP16/FP32 in both variants; only the predictor's on-disk precision
differs.

## Model resolution

The published repo is `aufklarer/Sidon-CoreML`, laid out per variant subfolder
(`fp16/`, `int8/`), each holding `Sidon-Predictor.*` and `Sidon-Vocoder.*`.
`SidonModelResolver` prefers a precompiled `.mlmodelc` bundle and falls back to
compiling a `.mlpackage` on device (cached next to the package so the slow compile
runs once per install). `SpeechRestorer.fromLocalBundle(directory:)` loads
locally-converted artifacts directly, skipping HuggingFace.

## Performance (M-series)

Single 10 s window, Core ML `.all` compute units:

| | Wall | Notes |
|---|---|---|
| First window | slower | on-device `.mlpackage` compile (if no precompiled bundle) + cold load |
| Subsequent windows | ~2 s | faster than realtime for the 10 s window |

Validated against the Python Core ML reference: waveform cosine ≈ 0.9994 (fp16)
and ≈ 0.9990 (int8) on the benchmark clip (neural-vocoder phase noise keeps this
below 1.0).

## References

- Paper: [Sidon: Fast and Robust Open-Source Speech Restoration](https://arxiv.org/abs/2509.17052) (sarulab-speech, ICASSP 2026)
- Front-end reference: HuggingFace `SeamlessM4TFeatureExtractor` (`facebook/w2v-bert-2.0`)
- Core ML bundles: [aufklarer/Sidon-CoreML](https://huggingface.co/aufklarer/Sidon-CoreML)

See [docs/inference/sidon.md](../inference/sidon.md) for CLI usage.
