# F5-TTS

F5-TTS is a zero-shot voice-cloning TTS model from `SWivid/F5-TTS`. This
package ships a native Swift/MLX runtime for the expanded
`aufklarer/F5TTS-v1-Base-MLX-fp16` bundle.

## Status

| Item | Value |
|---|---|
| Module | `F5TTS` |
| Upstream | `SWivid/F5-TTS` |
| Default bundle | `aufklarer/F5TTS-v1-Base-MLX-fp16` |
| Source checkpoint | `F5TTS_v1_Base/model_1250000.safetensors` |
| Backend | MLX |
| Sample rate | 24 kHz output |
| Runtime | DiT flow-matching sampler + Vocos vocoder |
| License | `cc-by-nc-4.0`; non-commercial bundle |
| Streaming | Not supported |

`F5TTSModel` conforms to `SpeechGenerationModel` so generic callers can
identify it as a TTS model. The protocol-only `generate(text:language:)` method
throws a reference-required error because F5-TTS requires both reference audio
and a transcript. Use the model-specific
`generate(text:referenceAudio:referenceText:options:)` overload.

## Bundle Contract

The expanded bundle contains:

- `config.json` with model metadata, architecture dimensions, mel settings,
  converted file names, precision, and license posture.
- `model.safetensors` with the F5 DiT/text-conditioning weights.
- `vocos.safetensors` with the Vocos decoder weights.
- `vocos_config.yaml` copied from the source vocoder.
- `vocab.txt` for F5 character/pinyin token ids.
- `README.md` for the published bundle.

`F5TTSBundleLoader` validates the model type and required files, then reports
`memoryFootprint` as the converted DiT plus Vocos safetensor bytes.

## Native Graph

The Swift runtime runs:

1. Reference audio loading and optional resampling to 24 kHz.
2. RMS normalization for quiet references.
3. Vocos-style 100-band mel extraction for the reference audio.
4. Reference transcript normalization and tokenizer encoding.
5. Duration estimation from reference and target text byte lengths.
6. DiT flow matching with classifier-free guidance, sway timesteps, text
   ConvNeXt conditioning, rotary attention, and convolutional positional
   embedding.
7. Vocos mel-to-waveform decoding.
8. RMS restoration when the reference was boosted for conditioning.

The sampler exposes `F5TTSSynthesisOptions` for flow steps, CFG strength, sway
coefficient, speaking-rate scaling, deterministic seed, and target RMS.

## Text Surface

The native tokenizer supports English/ASCII plus Mandarin (and mixed EN/ZH)
text, preserving F5's character/pinyin ids from the published `vocab.txt`. The
loader normalizes CRLF vocab files before splitting; without that, every ASCII
symbol misses the vocab and collapses to token `0` (space), producing fluent
but content-incorrect speech.

Mandarin goes through `F5TTSPinyinConverter`, which replaces upstream's
rjieba + pypinyin frontend with longest-match lookup over the bundle's
`pinyin_lexicon.tsv` (declared as `files.pinyin_lexicon` in `config.json`).
The lexicon is generated at export time: multi-character entries are jieba
dictionary words and pypinyin phrases whose TONE3 reading differs from the
per-character defaults, with tone sandhi baked in per word — so no tone rules
run in Swift. Characters outside the lexicon fall back to the system
Mandarin-Latin transform. Measured against upstream `convert_char_to_pinyin`
on a 67-sentence polyphone/sandhi/code-switch corpus the converter reproduces
99.2% of pinyin tokens exactly, with every residual difference tone-only
(word-segmentation quirks); the corpus and lexicon subset ship as unit-test
fixtures.

Bundles that predate the lexicon stay English-only and reject CJK input with a
clear error.

## Validation Status

Current validation covers config parsing, CRLF vocab loading, pinyin lexicon
loading and parity fixtures, reference-required API behavior, DiT fixed-input
parity, Vocos mel decode parity, full local-bundle synthesis, and optional
Qwen3-ASR roundtrips for English and Mandarin. A local English clone test with
the exported bundle transcribed exactly as:

```text
This is a short F five TTS voice cloning test running locally on Apple Silicon.
```

Remaining work:

- Broaden subjective listening and ASR roundtrips across longer zh/en corpora.
- Add release benchmarks for memory and RTF on representative Apple Silicon
  devices.
- Publish a quantized (int8) bundle variant behind the same roundtrip gate.
