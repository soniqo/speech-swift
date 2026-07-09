# Voice Cloning Candidate Runtimes

`IndexTTS2TTS`, `HiggsAudioTTS`, and `F5TTS` are first-pass runtime surfaces for
candidate local voice-cloning engines requested for benchmark follow-up. They
share `VoiceCloneTTSCommon`, which loads `soniqo_manifest.json`, validates the
exported safetensors bundle, exposes model metadata, and tracks the local weight
footprint.

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

## Publishing

The intended published bundle IDs are:

| Bundle | Repo |
|---|---|
| IndexTTS2 fp16 | `aufklarer/IndexTTS2-MLX-fp16` |
| Higgs Audio v3 fp16 | `aufklarer/Higgs-Audio-v3-TTS-4B-MLX-fp16` |
| F5-TTS v1 Base fp16 | `aufklarer/F5-TTS-v1-Base-MLX-fp16` |

Publishing is a release operation outside this package. The exported bundle must
include the generated model card, `soniqo_manifest.json`, fp16 safetensors, and
copied tokenizer/config/license files.

The IndexTTS2 export is an expanded synthesis bundle: GPT/S2Mel/Qwen artifacts
at the root plus w2v-BERT, MaskGCT semantic codec, CAMPPlus, and BigVGAN under
`aux/`. The native port instantiates and validates those weights, runs reference
audio through SeamlessM4T features, w2v-BERT hidden-state 17, MaskGCT semantic
quantization, S2Mel length regulation, CAMPPlus style embedding, 22.05 kHz
prompt mel extraction, GPT semantic-code generation, S2Mel CFM decoding, and
BigVGAN waveform assembly. Explicit emotion vectors use the exported
`feat1.safetensors` speaker-style matrix and `feat2.safetensors` emotion matrix:
for each emotion bucket the runtime picks the speaker-style row closest to the
CAMPPlus style embedding, sums the matching emotion rows by vector weight, and
keeps the remaining weight on the reference emotion. The current validation
proves graph wiring, finite waveform output, longer English ASR roundtrips, and
the emotion-control path. Upstream PyTorch parity, subjective listening quality,
and the full benchmark language set still need broader checks before treating
the port as benchmark-grade.

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

## IndexTTS2 Native Port Requirements

The Swift module exposes `IndexTTS2TTSModel.auxiliaryModels` so callers and tests
can see the complete upstream runtime surface:

| Component | Upstream repo | Purpose |
|---|---|---|
| w2v-BERT 2.0 | `facebook/w2v-bert-2.0` | SeamlessM4T features and semantic hidden states for reference audio |
| MaskGCT semantic codec | `amphion/MaskGCT` | Quantizes reference features and maps generated semantic codes to embeddings |
| CAMPPlus | `funasr/campplus` | 192-d global style vector from the speaker reference |
| BigVGAN | `nvidia/bigvgan_v2_22khz_80band_256x` | Vocoder from generated 80-band mels to waveform |

`IndexTTS2Tokenizer` provides a native SentencePiece Unigram path for the
published `bpe.model`. It mirrors the upstream Latin uppercasing before BPE and
uses the model-compatible Viterbi tokenization path. Full parity still needs the
upstream Chinese normalization and glossary handling.

`E2EIndexTTS2BundleTests` validates either a local expanded bundle
(`INDEXTTS2_E2E_BUNDLE`) or the published Hugging Face download path
(`INDEXTTS2_E2E_DOWNLOAD=1`). The test confirms that the full bundle loads,
native reference conditioning produces the expected tensor shapes, short
semantic-code generation returns valid code IDs, and bounded synthesis produces a
finite waveform through S2Mel and BigVGAN. The full synthesis test can also run
an ASR roundtrip (`INDEXTTS2_E2E_ROUNDTRIP=1`) and reports generated duration,
synthesis RTF, pipeline RTF, model footprint, resident-memory deltas, transcript,
and WER. The default semantic decoder uses beam sampling with `beams=3`,
`top_k=30`, `top_p=0.8`, `temperature=0.8`, `repetition_penalty=10`, and seed
`11`; the E2E exposes environment overrides for seed, beam width, sampling
parameters, emotion preset/vector control, supplied semantic codes, and seed
sweeps. The E2E also exposes `INDEXTTS2_E2E_SPEAKING_RATE` for tempo checks.

## Remaining Native Runtime Work

- IndexTTS2: Chinese normalizer parity, upstream numerical parity checks,
  longer-form generation validation, public sampling controls, and subjective
  listening plus ASR round-trip checks across the benchmark languages.
- Higgs Audio v3: conversational prompt builder, multi-codebook generation loop,
  inline control handling, and audio decoder path.
- F5-TTS v1: text frontend, reference transcript alignment, flow-matching
  sampler, and vocoder integration.
