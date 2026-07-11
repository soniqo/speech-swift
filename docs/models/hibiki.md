# Hibiki — Streaming Speech-to-Speech Translation (Kyutai)

Hibiki is Kyutai's streaming speech-to-speech translation model, built on the
Moshi/Mimi stack (same RVQ codec + delay-pattern decoding as PersonaPlex). This
repo currently ships the **Zero-3B** variant.

## Variants and language coverage

| Variant | Source → Target | Params | Status |
|---|---|---|---|
| Hibiki 1B | FR → EN | 1.7 B | converter only (`models/hibiki/export/convert.py --variant 1b`) |
| Hibiki 2B | FR → EN | 2.7 B | converter only (`models/hibiki/export/convert.py --variant 2b`) |
| **Hibiki Zero-3B** | **FR / ES / PT / DE → EN** | **3.1 B** | **shipped** (`Sources/HibikiTranslate/`) |

Pre-converted MLX weights (CC-BY-4.0):
- `aufklarer/Hibiki-Zero-3B-MLX-4bit` (~2.7 GB)
- `aufklarer/Hibiki-Zero-3B-MLX-8bit` (~3.9 GB)

## Architecture

```
Source-language audio (24 kHz)
        │
        ▼  Mimi streaming encoder (12.5 Hz, 16 codebooks, RVQ)
        │
        ▼  Source codebooks → temporal audio embeddings (streams 1..16)
        │
[Temporal Transformer · GQA · 28 layers · dim=2048]
        │  (text + 32 audio streams summed; 16 KV heads with kv_repeat=2)
        ▼
[Depformer · 6 layers · 16-step scheduled MultiLinear]
        │  (9 unique slice weights, schedule = [0..8, 8×8])
        ▼  Target codebooks (streams 17..32)
        │
        ▼  Mimi streaming decoder (12.5 Hz, 16 codebooks)
        │
Target-language audio (24 kHz)
```

### Architectural deltas vs PersonaPlex

| Component | PersonaPlex 7B | Hibiki Zero-3B |
|---|---|---|
| Temporal dim / layers | 4096 / 32 | **2048 / 28** |
| Heads / KV heads | 32 / 32 (MHA) | **16 / 8 (GQA, kv_repeat=2)** |
| Hidden scale (FFN) | 4.125 → 11264 intermediate | **6 → 8192 intermediate** |
| RoPE | interleaved (`traditional: true`) | **split-half (`rope_concat`, `traditional: false`)** |
| RoPE max period | 10000 | **20000** |
| Audio codebooks (n_q / dep_q) | 16 / 8 (8 user + 8 agent) | **32 / 16 (16 source + 16 target)** |
| Streams | 17 (1 + 8 + 8) | **33 (1 + 16 + 16)** |
| Max delay | 1 | **2** |
| Conditioner | none (system prompt) | **none (Zero is unconditional)** |
| Voice presets | 18 | **none** |
| Depformer schedule | one slice per step (16 unique) | **9 unique slices over 16 steps** |
| Depformer dim_feedforward | 2816 (depformer.dim×2/3×4.125) | **4096 (depformer.dim×2/3×6)** |
| Tokenizer | SPM 32k (tokenizer_spm_32k_3.model) | **SPM 48k (tokenizer_spm_48k_multi6_2.model)** |

## Decode loop

Hibiki streams source Mimi frames (12.5 Hz / 80 ms each) into the temporal
transformer. At each step the model samples one text token and 16 target
audio codes (via the depformer), and feeds them back as autoregressive
input on the next step. There is no separate prefill phase (no voice
prompt, no system prompt).

Hibiki emits text-PAD tokens (id 3) while it accumulates enough source
context to translate, then begins emitting content text tokens and the
matching target audio, and finally samples a text-EOS (id 2) to signal
end of utterance. The Swift driver runs **until EOS is sampled past the
source window**, with a `max(tSrc * 5/2, tSrc + 20)`-step safety cap.

Empirically the output runs ~1.5× the source duration on FLEURS-style
inputs (e.g. 3.54 s FR source → 4.96 s EN output). Callers can no longer
assume `output_duration == input_duration`; expect output length up to
~2.5× source.

Three pieces of the decode loop are non-obvious and were the cause of the
quality bug fixed in PR #238:

1. **Uniform `step` read with init-token substitution.** All 33 streams
   (text + 16 target audio + 16 source audio) are read at `cache[step]`
   each iteration, with the init token substituted when
   `step <= delays[k]`. Mirrors upstream Moshi `lm.py:698-702`
   (`positions = offsets % CT`).
2. **Write generated text + target codes at `step + 1`.** Upstream
   increments the offset *before* the cache scatter (`lm.py:759-772`).
   Writing at the same `step` index leaves the autoregressive read-slot
   at init forever — the model then runs effectively unconditioned on
   its own previous output and produces fluent English that has no
   relationship to the source.
3. **`text_emb` row-2 (EOS) is aliased to row-3 (PAD)** at weight-load
   time (`HibikiWeightLoading.swift`), mirroring Kyutai's
   `loaders.py:312` "implicitly replace early EOS with PAD" patch. Any
   EOS sampled during the audio-streaming window is harmless via this
   alias; EOS sampled after the source ends terminates the loop.

## Files

```
Sources/HibikiTranslate/
  Configuration.swift              HibikiConfig.zero3B + JSON loader
  HibikiTemporalTransformer.swift  GQA + rope_concat (28 layers, dim=2048)
  HibikiDepformer.swift            ScheduledMultiLinear (9 unique slices)
  HibikiTranslateModel.swift       Module shell + fromPretrained()
  HibikiTranslate.swift            translate() / translateStream() driver
  HibikiWeightLoading.swift        4-file safetensors loader
```

## Usage

```swift
import HibikiTranslate
import AudioCommon

let model = try await HibikiTranslateModel.fromPretrained(
    modelId: HibikiTranslateModel.defaultModelId  // 4-bit
)

let pcm = try AudioFileLoader.load(url: input, targetSampleRate: 24000)
let (englishAudio, textTokens) = model.translate(
    sourceAudio: pcm, sourceLanguage: .fr, verbose: true
)
try WAVWriter.write(samples: englishAudio, sampleRate: 24000, to: output)
```

CLI:

```bash
speech audio-translate input_fr.wav --output translated_en.wav --source-lang fr
speech audio-translate input.wav --quantization 8bit --verbose --transcript

# Deterministic / reproducible runs (matches the CI canaries):
HIBIKI_GREEDY=1 speech audio-translate input_fr.wav -o out.wav --source-lang fr
```

## Conversion

The Python converter at `models/hibiki/export/convert.py` (in the speech-models
repo) handles all three Hibiki variants:

```bash
python convert.py --variant 3b-zero --bits 4 \
    --upload --repo-id aufklarer/Hibiki-Zero-3B-MLX-4bit
python convert.py --variant 3b-zero --bits 8 \
    --upload --repo-id aufklarer/Hibiki-Zero-3B-MLX-8bit
```

It downloads the upstream PyTorch bf16 weights from `kyutai/hibiki-{1b,2b}-pytorch-bf16`
or `kyutai/hibiki-zero-3b-pytorch-bf16` and produces MLX-compatible safetensors:

- `temporal.safetensors` (quantized)
- `depformer.safetensors` (quantized; per-step slices packed by step index)
- `embeddings.safetensors` (BF16; text + 32 audio + per-codebook output heads)
- `mimi.safetensors` (Mimi codec, copied as-is)
- `tokenizer_spm_48k_multi6_2.model`
- `config.json`

## Translation quality

Greedy outputs (`HIBIKI_GREEDY=1`) on the canary E2E test fixtures:

| Source | Reference EN | Hibiki output | Keywords hit |
|---|---|---|---|
| FR — fleurs_fr.wav (3.54 s) | "Think of the ski route as a similar hiking route." | "so it's a ski route." | `ski`, `route` |
| ES — hibiki_official_es_5s.wav (5.00 s) | "Gentlemen, the data is worrying." | "gentlemen, the data is worrying." | `gentlemen`, `data`, `worrying` |
| PT — fleurs_pt.wav (5.16 s) | "It is the fifth CEP for Martelly in four years." | "the fifth c is p of the martyr." | `fifth` |
| DE — fleurs_de.wav (5.40 s) | "It didn't seem sensible to me; it certainly wasn't fair." | "that didn't seem to me to be useful." | `seem` |

FR and ES are **strict** in CI — `testFrenchToEnglishTranslation` and
`testSpanishToEnglishTranslation` fail if zero keywords match. PT and DE
are warn-only; promote with `HIBIKI_STRICT_ALL=1`, demote FR/ES with
`HIBIKI_LENIENT=1`.

Sampled mode (default, `HIBIKI_GREEDY` unset) is noticeably noisier than
greedy. Reproducible runs and CI canaries use greedy.

## Known limitations

- **FLEURS Spanish is out-of-distribution.** FLEURS recordings are 16 kHz
  human-narrated news clips; Hibiki Zero was trained on 24 kHz TTS-generated
  speech (11labs, cartesia, gradium). Both Python upstream and the Swift
  port produce degenerate output on FLEURS-ES — Python emits 1643 steps
  (~131 s) of broken audio without sampling EOS. The ES test fixture is a
  5 s trimmed excerpt from Kyutai's official samples space
  (`europarl_st/5dc1d533`, 24 kHz TTS) which matches the training
  distribution and produces clean English.
- **`translateStream()` is single-chunk** — The streaming entry point
  currently wraps `translate()` and emits one final `AudioChunk` once
  full-utterance generation completes. True per-chunk Mimi streaming
  decode is a v2 follow-up.
- **No SentencePiece decoder** — `translate()` returns text token IDs but
  doesn't decode them through `tokenizer_spm_48k_multi6_2.model`. The CLI
  `--transcript` flag prints raw token IDs.
- **Quantization-only** — The repo currently exposes Zero-3B 4-bit and 8-bit
  only. The 1B and 2B converters exist (`models/hibiki/export/convert.py`)
  but the Swift driver targets Zero-3B's GQA + rope_concat + non-conditioned
  layout. Adding 1B/2B variants to the Swift side is a follow-up.

## References

- [Hibiki paper (Kyutai, 2025)](https://arxiv.org/abs/2502.03382)
- [Kyutai Hibiki repo](https://github.com/kyutai-labs/hibiki)
- [Moshi-swift reference](https://github.com/kyutai-labs/moshi-swift) (lib-level Hibiki support)
- [PersonaPlex doc](personaplex.md) (shared Mimi/Depformer stack)
