# Voice Cloning Candidate Runtime Bundles

`IndexTTS2TTS`, `HiggsAudioTTS`, and `F5TTS` are first-pass runtime surfaces for
candidate local voice-cloning engines requested for benchmark follow-up. They
share `VoiceCloneTTSCommon`, which loads `soniqo_manifest.json`, validates the
exported safetensors bundle, exposes model metadata, and tracks the local weight
footprint.

This page is the source of truth for bundle shape, architecture coverage, and
remaining native-port work. Runtime usage, CLI flags, and E2E commands live in
[`docs/inference/voice-cloning-candidate-runtimes.md`](../inference/voice-cloning-candidate-runtimes.md).

IndexTTS2, Higgs Audio, and F5-TTS are unrelated upstream model families. They
are grouped here only because this package exposes them through the same
candidate voice-cloning bundle contract while their native Swift ports mature.

## Status

| Module | Upstream | Default bundle | Params | Runtime status |
|---|---|---|---|---|
| `IndexTTS2TTS` | `IndexTeam/IndexTTS-2` | `aufklarer/IndexTTS2-MLX-fp16` | 1.5B-class | Native reference conditioning + semantic GPT + S2Mel flow + BigVGAN synthesis + opt-in ASR roundtrip |
| `HiggsAudioTTS` | `bosonai/higgs-audio-v3-tts-4b` | `aufklarer/Higgs-Audio-v3-TTS-4B-MLX-fp16` | 4B | Bundle loader only |
| `F5TTS` | `SWivid/F5-TTS` (`F5TTS_v1_Base`) | `aufklarer/F5-TTS-v1-Base-MLX-fp16` | 335M-class | Bundle loader only |

The modules conform to `SpeechGenerationModel` so generic callers can identify
them as TTS models. `IndexTTS2TTSModel` also exposes a voice-cloning
`generate(text:referenceAudio:emotionReferenceAudio:emotionControl:...)` entry
point that runs the expanded Swift/MLX graph. `emotionReferenceAudio` uses a
separate style reference, while `IndexTTS2EmotionControl` applies upstream-style
8-value emotion-vector presets such as `.eager`. `IndexTTS2SynthesisOptions`
adjusts the S2Mel frame expansion separately from emotion and can optionally cap
long internal low-energy pauses after vocoding, so callers can make speech
faster without overdriving the emotion vector and losing speaker identity. The
protocol-only `generate(text:language:)` method still throws a
reference-required error because IndexTTS2 is a zero-shot voice-cloning model.
Higgs Audio and F5-TTS keep clear unsupported-runtime errors until their native
inference graphs are ported.

## Bundle Contract

The matching exporter lives in `speech-models` under
`models/voice-cloning-candidates/export/`. Each bundle includes:

- `soniqo_manifest.json` with model key, source repo, sample rate, parameter
  class, license posture, converted files, and runtime status.
- fp16 `*.safetensors` files with upstream tensor names preserved.
- tokenizer/config files copied from the upstream repository.

`VoiceCloneBundleLoader` validates that the manifest model key matches the
runtime module and that all converted and copied files listed in the manifest
exist. It reports `memoryFootprint` as the total bytes of the converted
safetensors.

Publishing is a release operation outside this package. The exported bundle must
include the generated model card, `soniqo_manifest.json`, fp16 safetensors, and
copied tokenizer/config/license files.

## IndexTTS2 Expanded Bundle

The Swift module exposes `IndexTTS2TTSModel.auxiliaryModels` so callers and tests
can see the complete upstream runtime surface. The expanded export stores
GPT/S2Mel/Qwen artifacts at the bundle root plus auxiliary model weights under
`aux/`.

| Component | Upstream repo | Purpose |
|---|---|---|
| w2v-BERT 2.0 | `facebook/w2v-bert-2.0` | SeamlessM4T features and semantic hidden states for reference audio |
| MaskGCT semantic codec | `amphion/MaskGCT` | Quantizes reference features and maps generated semantic codes to embeddings |
| CAMPPlus | `funasr/campplus` | 192-d global style vector from the speaker reference |
| BigVGAN | `nvidia/bigvgan_v2_22khz_80band_256x` | Vocoder from generated 80-band mels to waveform |

The native port instantiates and validates those weights, then runs:

1. Reference audio feature extraction with SeamlessM4T-style features.
2. w2v-BERT hidden-state extraction from layer 17.
3. MaskGCT semantic quantization for the reference prompt.
4. S2Mel prompt length regulation and 22.05 kHz prompt mel extraction.
5. CAMPPlus style embedding extraction.
6. GPT semantic-code generation.
7. S2Mel CFM decoding.
8. BigVGAN waveform assembly.

Explicit emotion vectors use the exported `feat1.safetensors` speaker-style
matrix and `feat2.safetensors` emotion matrix. For each emotion bucket the
runtime picks the speaker-style row closest to the CAMPPlus style embedding,
sums the matching emotion rows by vector weight, and keeps the remaining weight
on the reference emotion.

`IndexTTS2Tokenizer` provides a native SentencePiece Unigram path for the
published `bpe.model`. It mirrors the upstream Latin uppercasing before BPE and
uses the model-compatible Viterbi tokenization path. Full parity still needs the
upstream Chinese normalization and glossary handling.

The current validation proves graph wiring, finite waveform output, longer
English ASR roundtrips, explicit emotion control, speaking-rate control, and
internal-pause capping. Upstream PyTorch parity, subjective listening quality,
and the full benchmark language set still need broader checks before treating
the port as benchmark-grade.

## Remaining Native Runtime Work

- IndexTTS2: Chinese normalizer parity, upstream numerical parity checks,
  longer-form generation validation, public sampling controls, and subjective
  listening plus ASR round-trip checks across the benchmark languages.
- Higgs Audio v3: conversational prompt builder, multi-codebook generation loop,
  inline control handling, and audio decoder path.
- F5-TTS v1: text frontend, reference transcript alignment, flow-matching
  sampler, and vocoder integration.
