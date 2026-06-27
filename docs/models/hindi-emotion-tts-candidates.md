# Hindi Emotion TTS Candidates

This note tracks Hindi-capable TTS models with explicit emotion or style marker
control before and during runtime promotion. Runtime support is not implied for
candidate models unless a Swift/MLX or CoreML implementation is listed below.

The matching source-of-truth catalog is
`Sources/AudioCommon/HindiEmotionTTSCatalog.swift`.

## Current Runtime Status

Indic-Mio has an exported MLX fp16 bundle at
`aufklarer/Indic-Mio-MLX-fp16`. The bundle includes the Qwen3 speech-token LM
and the required MioCodec companion weights under `miocodec/`, so speech-swift
does not need a separate MioCodec model download.

`IndicMioTTS` is now a Swift target with the first runtime layer:

- downloads and opens the exported bundle;
- loads the Qwen3 config, tokenizer, and fp16 safetensors weights;
- validates the documented Indian-language emotion tags;
- generates MioCodec content token IDs from Hindi text plus tags;
- decodes content token IDs into MioCodec content embeddings;
- decodes content embeddings through the MioCodec wave decoder to 24 kHz
  waveform audio;
- accepts an explicit 128-dim MioCodec global embedding, or uses a
  deterministic default embedding for non-cloned synthesis;
- extracts the 128-dim MioCodec global embedding from raw reference audio via
  WavLM-base-plus when the WavLM safetensors companion bundle is available.

It is exposed as the `indic-mio` TTS engine in the model registry, realtime
server dispatch, one-shot synthesis dispatch, and `speech speak --engine
indic-mio`. The CLI accepts `--indic-mio-global-embedding` for callers that
already have a 128-dim MioCodec global speaker embedding.

Indic-Mio's full zero-shot clone path requires a global speaker embedding. The
bundled MioCodec weights include the global encoder and decoder path. Raw
reference waveform cloning now runs the WavLM-base-plus SSL feature extractor,
averages the first two hidden layers, and feeds the result through MioCodec's
global ConvNeXt encoder. The runtime resolves the WavLM companion from
`aufklarer/WavLM-Base-Plus-MLX-fp16`; `INDIC_MIO_WAVLM_BUNDLE` can override it
for local export testing.

E2E coverage currently loads `aufklarer/Indic-Mio-MLX-fp16`, runs Hindi text
with an emotion tag through Qwen3 speech-token generation, decodes MioCodec
content embeddings, synthesizes waveform audio, and verifies intelligibility
with a Hindi Qwen3ASR keyword-recovery roundtrip. This is a runtime sanity
check, not an exact-transcript quality gate.

## Decision

Port **Indic-Mio** first. It is Apache-2.0, Hindi-capable, documents explicit
Indian-language emotion tags, and describes zero-shot voice cloning through
speaker embeddings.

Use **Fish Audio S2 Pro** as an experimental quality benchmark unless a
commercial license is secured. Its marker control is the strongest of the
reviewed models, and the Swift port now covers programmatic generation,
raw-WAV zero-shot reference conditioning, and Fish DAC decode. The public model
card still requires non-commercial use, so it should stay outside default
product paths.

## Candidates

| Model | HF repo | License posture | Hindi | Explicit emotion control | Voice path | speech-swift status |
|---|---|---|---:|---|---|---|
| Indic-Mio | `SPRINGLab/Indic-Mio` / `aufklarer/Indic-Mio-MLX-fp16` | Apache-2.0 | Yes | End-of-utterance tags: `<happy>`, `<sad>`, `<angry>`, `<disgust>`, `<fear>`, `<surprise>` | Speaker embeddings / zero-shot clone path | Runtime: bundle download, token generation, WavLM raw-reference embedding, wave decode, CLI/server exposure, Hindi ASR roundtrip |
| Svara-TTS v1 | `kenpath/svara-tts-v1` | Apache-2.0 | Yes | Tags: `<happy>`, `<sad>`, `<anger>`, `<fear>` | Adaptation / speaker identity path; clone quality needs validation | Secondary port candidate |
| Fish Audio S2 Pro | `fishaudio/s2-pro` / `aufklarer/Fish-Audio-S2-Pro-MLX-fp16` | Research/non-commercial public weights | Yes | Inline bracket tags, including `[angry]`, `[sad]`, `[whisper]`, `[shouting]`, `[surprised]` | Raw-WAV reference conditioning through Fish DAC | Programmatic runtime implemented; CLI/server exposure and broader quality characterization remain |
| Indic Parler-TTS | `ai4bharat/indic-parler-tts` | Apache-2.0 | Yes | Caption/descriptive prompt emotions; Hindi emotion is not officially tested | Preset/descriptive voices | Secondary comparison |
| IndicF5 | `ai4bharat/IndicF5` | MIT | Yes | None documented | Reference cloning baseline | Tracked only |
| Orpheus TTS Hindi | `SachinTelecmi/Orpheus-tts-hi` | Apache-2.0, needs review | Yes | Emotion/prosody tokens listed as future work | Not clear enough | Tracked only |

## Marker Mapping

Speech Studio markers should map by syntax, not by a single global prompt
format:

- Indic-Mio and Svara use suffix angle tags: `text <happy>`.
- Fish S2 Pro uses inline bracket tags: `[whisper] text`, `[angry] text`.
- Indic Parler uses a descriptive caption rather than strict inline tags.
- IndicF5 and Orpheus Hindi must not be selected for marker-driven acting until
  explicit emotion controls are documented and validated.

## Porting Order

1. **Indic-Mio**
   - Export script and bundle layout are in `speech-models`.
   - Text LM, tokenizer, content-token range, MioCodec FSQ path, and decode
     timing are identified.
   - Swift target is present for bundle loading, token generation, and content
     embedding decode.
   - MioCodec wave decoder is implemented.
   - Model registry, realtime server dispatch, one-shot server synthesis, and
     CLI synthesis are wired under `indic-mio`.
   - E2E coverage is present for Hindi text plus `<happy>`, including Qwen3ASR
     keyword recovery over synthesized audio.
   - Raw reference cloning is wired through the published WavLM companion bundle.
   - Next: add Studio clone selection copy that explains the extra first-run
     download.

2. **Svara-TTS v1**
   - Validate whether its speaker/adaptation path can satisfy Studio's
     reference-clip clone workflow.
   - If speaker similarity is weak, keep it as a Hindi emotion fine-tuning base.

3. **Fish Audio S2 Pro**
   - Keep outside default product paths.
   - Use the programmatic runtime for benchmark samples and marker-quality
     comparisons only unless a separate commercial license is obtained.
   - Add CLI/server exposure only if the product path remains explicitly
     experimental or a commercial license is secured.

## References

- [Indic-Mio](https://huggingface.co/SPRINGLab/Indic-Mio)
- [Svara-TTS v1](https://huggingface.co/kenpath/svara-tts-v1)
- [Fish Audio S2 Pro](https://huggingface.co/fishaudio/s2-pro)
- [Indic Parler-TTS](https://huggingface.co/ai4bharat/indic-parler-tts)
- [IndicF5](https://huggingface.co/ai4bharat/IndicF5)
- [Orpheus TTS Hindi](https://huggingface.co/SachinTelecmi/Orpheus-tts-hi)
