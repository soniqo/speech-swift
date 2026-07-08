# Voice Cloning Candidate Runtimes

`IndexTTS2TTS`, `HiggsAudioTTS`, and `F5TTS` are first-pass runtime surfaces for
candidate local voice-cloning engines requested for benchmark follow-up. They
share `VoiceCloneTTSCommon`, which loads `soniqo_manifest.json`, validates the
exported safetensors bundle, exposes model metadata, and tracks the local weight
footprint.

## Status

| Module | Upstream | Default bundle | Params | Runtime status |
|---|---|---|---|---|
| `IndexTTS2TTS` | `IndexTeam/IndexTTS-2` | `aufklarer/IndexTTS2-MLX-fp16` | 1.5B-class | Bundle loader + CLI validation + tokenizer scaffold |
| `HiggsAudioTTS` | `bosonai/higgs-audio-v3-tts-4b` | `aufklarer/Higgs-Audio-v3-TTS-4B-MLX-fp16` | 4B | Bundle loader only |
| `F5TTS` | `SWivid/F5-TTS` (`F5TTS_v1_Base`) | `aufklarer/F5-TTS-v1-Base-MLX-fp16` | 335M-class | Bundle loader only |

The modules conform to `SpeechGenerationModel` so generic callers can identify
them as TTS models, but synthesis is not available yet. Calling `generate` throws
`AudioModelError.inferenceFailed` with a native-inference-not-ported message.
`speech speak --engine indextts2` is wired to the same loader so an uploaded
IndexTTS2 artifact can be downloaded, cache-validated, and inspected through the
normal CLI path before the full Swift/MLX graph port is complete.

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
`aux/`. Native synthesis still needs Swift/MLX graph ports for those components.

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
published `bpe.model`. It currently covers the model-compatible Viterbi
tokenization used by English text. Full parity still needs the upstream Chinese
normalization and glossary handling.

`E2EIndexTTS2BundleTests` validates either a local expanded bundle
(`INDEXTTS2_E2E_BUNDLE`) or the published Hugging Face download path
(`INDEXTTS2_E2E_DOWNLOAD=1`). The test confirms that the full bundle loads and
that synthesis currently fails with the explicit native-port-pending error.

## Remaining Native Runtime Work

- IndexTTS2: Chinese normalizer parity, w2v-BERT encoder, MaskGCT semantic
  codec, GPT2/Conformer/Perceiver generation loop, S2Mel diffusion decoder,
  CAMPPlus style encoder, and BigVGAN vocoder path.
- Higgs Audio v3: conversational prompt builder, multi-codebook generation loop,
  inline control handling, and audio decoder path.
- F5-TTS v1: text frontend, reference transcript alignment, flow-matching
  sampler, and vocoder integration.
