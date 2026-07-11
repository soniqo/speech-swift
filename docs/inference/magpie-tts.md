# Magpie-TTS Multilingual — inference

## CLI

Magpie ships as an engine of the unified `speak` command:

```bash
# English greedy synthesis (the default for non-JA languages)
speech speak "Hello, world." --engine magpie --magpie-speaker aria \
    --magpie-temperature 0 -o out.wav

# Sampled output (default temperature 0.6 / top-k 80)
speech speak "Hello, world." --engine magpie -o out.wav

# Streaming synthesis with playback
speech speak "Streaming test" --engine magpie --stream --play

# Japanese — needs stochastic decoding (greedy gets stuck on the first phrase)
speech speak "こんにちは世界、これは音声合成システムです。" \
    --engine magpie --language ja --magpie-temperature 0.6 \
    --magpie-top-k 80 --seed 42 -o out.wav

# Pre-phonemised IPA bypasses the per-language G2P
speech speak "həˈloʊ" --engine magpie --magpie-prephonemized -o out.wav
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--engine magpie` | — | Select Magpie. Required. |
| `--magpie-variant {int4\|int8}` | `int4` | Quantisation variant. |
| `--magpie-speaker {sofia\|aria\|jason\|leo\|john}` | `sofia` | Baked speaker identity. |
| `--magpie-temperature FLOAT` | `0.6` | Sampling temperature (0 = greedy). |
| `--magpie-top-k INT` | `80` | Top-k filter for sampling. |
| `--magpie-max-frames INT` | `500` | Hard cap on codec frames (~23 s). |
| `--magpie-min-frames INT` | `4` | Minimum frames before EOS allowed. |
| `--magpie-prephonemized` | off | Treat input as IPA / phoneme stream, skip per-language G2P. |
| `--language en\|es\|...` | `english` | Selects the per-language tokeniser. |
| `--stream` | off | Emit `AsyncStream<AudioChunk>` instead of one final WAV. |
| `--seed INT` | none | Reproducible Gumbel sampling. |

### Languages

All nine languages have a dedicated G2P pipeline and round-trip through
Qwen3-ASR in `testMultilingualRoundTrip`. See `docs/models/magpie-tts.md`
for implementation detail.

| Language | Code | G2P pipeline | Round-trip status |
|---|---|---|---|
| English    | `en` | CMU IPA dict (125 k entries, bundled) | ✅ all content words |
| Spanish    | `es` | Spanish IPA dict (bundled) | ✅ identical or near-identical |
| German     | `de` | German IPA dict (bundled) | ✅ all content words |
| French     | `fr` | byT5 UTF-8 byte encoder | ✅ identical or near-identical |
| Italian    | `it` | byT5 UTF-8 byte encoder | ✅ all content words |
| Vietnamese | `vi` | byT5 UTF-8 byte encoder | ✅ all content words |
| Hindi      | `hi` | Devanagari codepoint lookup + last-wins sub-vocab | ✅ all content words (model renders English loanwords as Latin script) |
| Mandarin   | `zh` | `NLTokenizer(.simplifiedChinese)` word seg → Apple `.mandarinToLatin` → bundled pinyin → IPA dict + `#N` tone | ✅ key content words |
| Japanese   | `ja` | `CFStringTokenizer` kanji reading → katakana with NFC-preserved dakuten + heiban pitch markers + particle/greeting overrides | ✅ all content words **with stochastic sampling** |

> JA prefers stochastic decoding. The reference test uses
> `temperature=0.6, top_k=80, seed=42` (matching NeMo's
> `test_round_trip.py`); greedy decoding can get stuck on the first phrase
> because we don't have pyopenjtalk pitch-accent dictionary access.

## Programmatic use

```swift
import MagpieTTS

let model = try await MagpieTTS.fromPretrained(variant: .int4)

// Batch synthesis (English/Spanish/etc. — greedy works)
let audio = try model.synthesize(
    text: "Hello, world.",
    speaker: .aria,
    language: .english,
    params: MagpieTTSParams(temperature: 0, topK: 1, maxSteps: 500))

// Japanese — use stochastic sampling
let audioJA = try model.synthesize(
    text: "こんにちは世界、これは音声合成システムです。",
    speaker: .aria,
    language: .japanese,
    params: MagpieTTSParams(temperature: 0.6, topK: 80,
                              maxSteps: 300, seed: 42))

// Streaming
let stream = model.synthesizeStream(
    text: "Streaming text",
    speaker: .aria,
    language: .english,
    firstChunkFrames: 8,
    framesPerChunk: 25)
for try await chunk in stream {
    // chunk.samples is 22.05 kHz mono Float32
}
```

## Performance (M4 Pro, Apple Silicon)

| Setting | Frames | Wall-clock | RTF |
|---|---|---|---|
| Batch, INT4, greedy, 2.8 s output (EN) | 60  | 0.88 s | 0.32 |
| Batch, INT4, greedy, 5.8 s output (EN) | 125 | 1.35 s | 0.23 |
| Batch, INT4, greedy, 23 s output       | 500 | 5.59 s | 0.24 |
| Streaming, INT4, greedy, 23 s output   | 500 | 21.6 s | 0.93 |

> Streaming RTF is higher because the codec is re-invoked on the full code
> buffer at every chunk emission (causal codec, easy correctness, but not
> the cheapest cadence). A future revision can cache codec state.

First-packet latency in streaming mode is ≈120 ms after model load.

## Memory

| Variant | Disk | RAM (load + decode) |
|---|---|---|
| INT4 | 247 MB | ~ 1.3 GB |
| INT8 | 411 MB | ~ 1.6 GB |

## Voice cloning — not supported

Magpie has no zero-shot speaker conditioning in the model. Only the 5
baked identities (Sofia / Aria / Jason / Leo / John Van Stan) ship in
the bundle. The CLI rejects the shared `--voice-sample` / `--speaker` /
`--instruct` flags with an actionable error pointing users at the
`--magpie-speaker` flag or the engines that do support cloning
(`qwen3`, `cosyvoice`, `voxcpm2`).

## CoreML backend (`--engine magpie-coreml`)

`Sources/MagpieTTSCoreML/` ships a CoreML-driven pipeline backed by
[`aufklarer/Magpie-TTS-Multilingual-357M-CoreML-8bit`](https://huggingface.co/aufklarer/Magpie-TTS-Multilingual-357M-CoreML-8bit)
(~342 MB on disk, INT8 weights). The four `.mlmodelc` packages
(`text_encoder`, `decoder_prefill`, `decoder_step`,
`nanocodec_decoder`) handle the heavy lifting on ANE / GPU; a
Swift-side FSQ inverse turns the sampled codes into the 32-dim
latents the codec window consumes.

```bash
speech speak "Hello world." --engine magpie-coreml --magpie-speaker aria -o hi.wav
speech speak "Hola mundo." --engine magpie-coreml --language es --magpie-speaker leo -o hola.wav
```

Caveats vs `--engine magpie`:

- **Pure-Swift LocalTransformer.** The 1-layer LT and 8 audio embedding
  tables are extracted from the MLX bundle once at `fromPretrained` and
  cached to disk (`extracted_weights.bin`, ~70 MB). The first invocation
  pays the MLX module load + extraction (~3–4 s); every subsequent
  process load reads the cache directly in ~15 ms. The AR loop itself
  runs pure CoreML + Accelerate — no MLX dispatch per frame. ASR
  round-trip matches the MLX backend bit-for-bit.
- **Streaming is supported.** The bundle ships two nanocodec variants:
  `nanocodec_decoder.mlmodelc` (64-frame batch, used by
  `synthesize()`) and `nanocodec_decoder_streaming.mlmodelc` (8-frame
  window, used by `synthesizeStream()`). Per-chunk steady-state is
  ~65 ms; first-packet latency is ~6 s cold (ANE compile dominates),
  drops to ~150–250 ms after `model.prewarm()`. Call `prewarm()`
  once at app start for low-latency streaming.
- **No Japanese.** The CoreML bundle doesn't ship a JA tokenizer JSON
  (the model supports it; we just haven't exported the assets).
  `--language ja` with `--engine magpie-coreml` auto-routes to the
  MLX backend and logs a stderr note about the fallback.

### Performance (M4 Pro, release build)

| Mode | RTF | First-packet | Note |
|---|---|---|---|
| Batch (warm) | 0.87 IT / 0.99 EN | n/a | `synthesize()` |
| Streaming (warm, after `prewarm()`) | ~1.26 | ~150–250 ms | `synthesizeStream()`, 8-frame chunks (~370 ms each) |
| Streaming (cold, no prewarm) | ~1.26 | ~6 s | ANE compile pays once; not session-amortised |

MLX backend remains the fastest path on macOS (RTF 0.23–0.26); the
CoreML pipeline's main value is ANE residency for iOS deployment.

Speaker ordering matches the bundle's `speaker_info.json`
(`0=John, 1=Sofia, 2=Aria, 3=Jason, 4=Leo`), which is a different
ordering than the MLX bundle. The CLI accepts the same five names —
the speaker enum maps internally.

## Known limitations / follow-ups

- **JA pitch accent** — without `pyopenjtalk` we use the heiban (acc=0)
  pattern for every word. Real per-word pitch accents would need MeCab +
  the pyopenjtalk accent dictionary (~30 MB UniDic). The audio is
  intelligible and matches via ASR but is not pitch-faithful.
- **ZH word segmentation** — `NLTokenizer` handles most common compounds
  correctly but occasionally misreads polyphones (rare characters with
  multiple readings). A bundled jieba-style dictionary would close the
  gap.
- **Streaming codec** is full-buffer-replay per chunk; a streaming-friendly
  codec state cache can drop streaming RTF below 0.3.
- **FP16 bundle** is not on HuggingFace yet (export available locally in
  `speech-models/models/magpie-tts/export/output/Magpie-TTS-Multilingual-357M-MLX-FP16`).
- **CoreML LT + audio embeddings** — the CoreML bundle currently
  delegates the LocalTransformer + audio embedding averaging back to
  MLX. Shipping `local_transformer/*.npy` and `audio_embedding_*.npy`
  inside the CoreML bundle (and a Swift LT) would let `--engine
  magpie-coreml` run without the MLX runtime at all — the prerequisite
  for ANE-only iOS deployment.
