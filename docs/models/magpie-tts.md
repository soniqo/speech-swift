# Magpie-TTS Multilingual 357M

Apple-Silicon port of NVIDIA's [Magpie-TTS Multilingual 357M](https://huggingface.co/nvidia/magpie_tts_multilingual_357m),
a 9-language autoregressive multi-codebook text-to-speech model over NeMo's
22.05 kHz / 1.89 kbps / 21.5 fps [Nano-Codec](https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps).

## At a glance

| | |
|---|---|
| Total params | 357 M (text encoder 99 M + decoder 90 M + LocalTransformer 1 M + NanoCodec 62 M + audio embeddings) |
| Architecture | Causal Transformer encoder (6L, d=768) + Causal Transformer decoder (12L, d=768, with cross-attention) + LocalTransformer codebook AR head (1L, d=256) + Causal HiFi-GAN decoder |
| Audio | 8 codebooks × 2024 codes, 22.05 kHz mono, 21.5 fps |
| Languages | EN, ES, DE, FR, IT, VI, ZH, HI, JA — all 9 round-trip via ASR (`testMultilingualRoundTrip`) |
| Speakers | 5 baked (Sofia, Aria, Jason, Leo, John Van Stan) — no voice cloning |
| Bundle layout | 4-bundle MLX (text_encoder / decoder_prefill / decoder_step / nanocodec_decoder) |
| Variants on HF | `aufklarer/Magpie-TTS-Multilingual-357M-MLX-{4bit,8bit}` — 247 MB INT4 / 411 MB INT8 |
| License | NVIDIA Open Model License (commercial use permitted, see linked PDF) |

## Inference pipeline

```
text → MagpieTokenizer (per-language G2P, see below)
     → MagpieTextEncoder (6 causal layers, NLC) → encoder_output (B, T, 768)
     → MagpieDecoder.prefill (110 baked speaker frames + BOS) → KV cache + h_0
     → repeat:
          MagpieLocalTransformer per-codebook AR sampling → 8 codes per frame
          → MagpieDecoder.step (single AR frame)
          → break on EOS
     → MagpieNanoCodec (FSQ inverse + causal HiFi-GAN) → 22.05 kHz waveform
```

### Why 4 bundles?

The four sub-bundles share weights between `decoder_prefill` and `decoder_step`
but are kept as separate directories so the layout stays compatible with
[FluidInference's CoreML pipeline](https://github.com/FluidInference/FluidAudio).
For MLX the split is cosmetic; both decoder bundles load the same parameters.

### LocalTransformer head

After the main 12-layer decoder produces a hidden state for each frame, the
1-layer LocalTransformer samples the 8 codebooks **sequentially within the
frame**, conditioning each codebook on the previously sampled codebook's
embedding plus the decoder hidden state. This gives the AR model parallel
codebook coverage without the cost of a fully autoregressive multi-codebook
decoder.

## Per-language G2P

NeMo's `MagpieMultiTokenizer.AggregatedTTSTokenizer` concatenates each
sub-tokenizer's vocab end-to-end into a single 2360-entry table and adds a
per-tokenizer offset when encoding. `MagpieSubVocab` records the offsets we
derived from `<pad>` boundary scanning of the shipped `tokenizer/en.json`
(which exposes the full aggregated vocab):

| Language | Sub-vocab offset | Size | Tokeniser |
|---|---|---|---|
| English (`en`)   | 0    | 96  | IPATokenizer + CMU IPA dict (NeMo `cdd41953...`) |
| Spanish (`es`)   | 96   | 103 | IPATokenizer + Spanish IPA dict (NeMo `9a6b090b...`) |
| German (`de`)    | 199  | 150 | IPATokenizer + German IPA dict (NeMo `bafa5b4c...`) |
| Mandarin (`zh`)  | 349  | 109 | ChinesePhonemesTokenizer + pinyin → IPA + `#N` tone |
| Japanese (`ja`)  | 458  | 175 | JapanesePhonemeTokenizer + `[pitch, mora]` katakana |
| French (`fr`)    | 633  | 384 | byT5 byte tokeniser (HuggingFace `google/byt5-small`) |
| Hindi (`hi`)     | 1017 | 191 | HindiCharsTokenizer (codepoint-level Devanagari) |
| Italian (`it`)   | 1208 | 384 | byT5 byte tokeniser |
| Vietnamese (`vi`)| 1592 | 384 | byT5 byte tokeniser |

Always-appended `eos_id = vocab_size + 1 = 2361` per `magpietts.py:441`
(`num_tokens = vocab_size + 2`; bos/eos rows past the shared vocab).

### English / Spanish / German — dict-based IPA

`MagpieDictG2P` lazily loads a CMU-style IPA dict (`Resources/cmudict_ipa_*.txt`)
from the resource bundle. Pipeline:
1. Split input on letter runs, keep punctuation verbatim.
2. Look up each word's lowercased form in the dict (handles tab + space
   separators, mixed-case keys — covers the EN/ES vs DE dict formats).
3. OOV words fall back to lowercased graphemes (NeMo `use_chars=True`
   semantics).
4. Join phonemised words with single-space separators.

### French / Italian / Vietnamese — byT5 bytes

`MagpieByT5Encoder` mirrors HuggingFace's `ByT5Tokenizer`: encode text to
UTF-8 bytes, map each byte X to native byT5 id `X + 3` (pad=0/eos=1/unk=2
occupy 0–2), then offset by the sub-tokeniser's start in the aggregated
vocab. NeMo's text preprocessing lowercases first
(`italian_text_preprocessing` etc.) — we replicate.

### Hindi — Devanagari char-level

The Hindi sub-vocab covers Devanagari graphemes + ASCII fallback +
punctuation. We iterate `unicodeScalars` (not `Character`) so that
Devanagari conjuncts like `स्ते` (4 codepoints clustered into one grapheme
by Swift) decompose into individual vocab entries — Magpie was trained on
per-codepoint inputs.

> **Last-occurrence dedup.** `HindiCharsTokenizer` emits **duplicate**
> Devanagari entries inside its own vocab (the IPA punctuation list
> overlaps the char set). NeMo's `_token2id = {l: i for i, l in
> enumerate(tokens)}` dict-comp keeps the **last** insertion; we mirror
> that here. The earlier first-occurrence map landed every char in the
> English region of the aggregated vocab and produced nonsense audio.

### Chinese — `NLTokenizer` + pinyin → IPA dict

`MagpieChineseG2P` uses Apple's `NLTokenizer(unit: .word)` with
`.simplifiedChinese` for word segmentation (the jieba equivalent built
into NaturalLanguage). For each word `applyingTransform(.mandarinToLatin)`
emits tone-marked pinyin (CLDR table picks the *contextual* reading, e.g.
`世界` → `shì jiè` as a unit instead of char-by-char). Each pinyin
syllable is then split into its IPA components via the bundled
`Resources/cmudict_pinyin_zh.txt`, with `#1`–`#5` tone markers extracted
from the combining diacritic.

### Japanese — `CFStringTokenizer` + heiban pitch

`MagpieJapaneseG2P` reads kanji via `CFStringTokenizer`'s
`kCFStringTokenizerAttributeLatinTranscription` attribute (gives proper
Japanese readings, e.g. `世界` → `sekai`, not the Mandarin reading you'd
get from `applyingTransform(.toLatin)`). Romaji → katakana via
`applyingTransform(.latinToKatakana)`, then NFC-compose so Japanese
dakuten (デ vs テ, ゴ vs コ) stay attached to the base glyph before
stripping the Mandarin tone combining marks (̀ ́ ̄ ̌) leftover from kanji
borrowings.

The Magpie JA vocab expects per-mora pitch markers `0`/`1` interleaved
with katakana. Without `pyopenjtalk` we can't look up real per-word
pitch-accent values, so we apply the **heiban** (acc=0) pattern
`[L-H-H-…-H]` to every word chain — the most common Japanese accent
class. Particle / fixed-greeting overrides (`は→ワ`, `へ→エ`, `を→オ`,
`こんにちは→コンニチワ`) close the gap on Apple's literal kana readings.
Stochastic decoding (`temperature=0.6, top_k=80`) is recommended for JA
to avoid greedy decoding getting stuck on the first phrase.

## Quantisation

Both bundles use MLX's flat affine quantisation (`mlx_affine_flat`, group
size 64) applied to every 2-D+ tensor whose fan-in is divisible by 64. The
remaining tensors (snake alphas, biases, small embedding tables) ship as
bfloat16. INT8 produces ~411 MB, INT4 ~247 MB; both dequantise to FP32 at
load time, so runtime activations are full precision.

## Round-trip verification

`testMultilingualRoundTrip` (in `Tests/MagpieTTSTests/E2EMagpieTests.swift`)
synthesises a canonical sentence per language, transcribes via Qwen3-ASR,
and asserts content-word coverage. All 9 languages currently pass strict
or near-identical round-trip on the bundled test prompts; ASR transcripts
for each case are logged for spot-checking.

## Bundled resources

- `Sources/MagpieTTS/Resources/cmudict_ipa_en.txt` (3.0 MB, 125 k entries)
- `Sources/MagpieTTS/Resources/cmudict_ipa_es.txt` (2.2 MB)
- `Sources/MagpieTTS/Resources/cmudict_ipa_de.txt` (4.3 MB)
- `Sources/MagpieTTS/Resources/cmudict_pinyin_zh.txt` (~6 KB)
- `Sources/MagpieTTS/Resources/heteronyms_en.txt` (~2 KB; reserved for a
  future heteronym disambiguator)

All extracted from the upstream `magpie_tts_multilingual_357m.nemo`
bundle.

## Source

- Upstream weights: [nvidia/magpie_tts_multilingual_357m](https://huggingface.co/nvidia/magpie_tts_multilingual_357m) (NVIDIA Open Model License)
- Codec: [nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps](https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps)
- Paper: [NanoCodec: Towards High-Quality Ultra Fast Speech LLM Inference (2025)](https://arxiv.org/abs/2508.05835v1)
- Reference CoreML port: [FluidInference/mobius](https://github.com/FluidInference/mobius/tree/main/models/tts/magpie/coreml)
- MLX export script: `speech-models/models/magpie-tts/export/`
