# Voice-Cloning TTS Benchmarks

Single-sentence synthesis benchmark for the zero-shot voice-cloning engines,
cloned from the same 12 s reference clip (with its transcript where the engine
accepts one). Roundtrip = Qwen3-ASR transcription of the generated audio
compared against the input text.

Sentence: "The quick brown fox jumps over the lazy dog and rests in the
afternoon sun."

## Results

| Engine | Params | Peak RSS | Audio | Synth | RTF | Roundtrip |
|---|---|---:|---:|---:|---:|---|
| Higgs TTS 3 (clone) | 4B bf16 | 8.6 GB | 6.04 s | 4.73 s | **0.78** | exact |
| Higgs TTS 3 (no reference) | 4B bf16 | 8.3 GB | 4.80 s | 3.77 s | **0.78** | exact |
| F5-TTS (clone, 16 steps, default) | 336M fp16 | 0.8 GB | 5.09 s | 2.91 s | **0.57** | 1-word sub |
| F5-TTS (clone, 32 steps) | 336M fp16 | 0.8 GB | 5.09 s | 5.75 s | 1.13 | 1-word sub |
| F5-TTS (clone, 12 steps) | 336M fp16 | 0.8 GB | 5.09 s | 2.19 s | 0.43 | 1-word sub |
| IndexTTS2 (clone) | 1.5B-class fp16 | 3.0 GB | 5.80 s | 5.8 s | **1.0** | exact |

**Machine**: Apple M5 Pro, 48 GB, release build with compiled metallib.
Synth time excludes model load (Higgs/F5 report it directly); RSS from
`/usr/bin/time -l`, includes weights.

IndexTTS2 stage timing (via `INDEXTTS2_TIMING=1`): reference conditioning
0.6 s, semantic GPT 3.4 s, S2Mel flow 1.1 s (15 steps, the default),
BigVGAN 0.7 s. The semantic GPT decodes against a preallocated KV cache
grown in 256-token chunks and written in place (no O(history) concat
copies), with the history stored in **float32** and attention run in
float32: the Metal fp16 SDPA vector kernel degrades ~2.5x per step under
the buffer states real decoding produces (measured on M5 Pro; synthetic
caches don't trigger it), while the fp32 kernel is immune — and float32
attention is strictly higher precision. Step eval: 48.8 ms mean before
the cache rework, 21.7 ms with the in-place fp16 cache, ~10 ms with
fp32 attention. Same seed now takes a marginally different sample path
than the fp16 build (275 → 291 codes on the benchmark sentence);
roundtrip stays exact. Wall-clock runs vary with thermal state, so
per-step minima and same-day interleaved pairs are the comparable
metrics.
S2Mel step sweep (`--indextts2-s2mel-steps`, word-identical roundtrips,
ear-validated): 25 steps 1.7 s, 15 steps 1.0 s, 10 steps 0.7 s —
**default is now 15**; use 25 for upstream-exact flow.

## Notes

- **Higgs RTF includes the decode-loop pipelining** landed alongside this
  benchmark (previously 1.04 clone / 0.82 plain): sampling stays on-device
  with the delay ramp masked on GPU, and `asyncEval` overlaps the next
  forward pass with the CPU-side EOC state machine. Cloning now costs the
  same RTF as plain synthesis; the remaining decode cost is the bf16
  memory-bandwidth roofline of the 4B backbone (~32 frames/s at 25 fps).
- **Higgs reference encoding** (12 s clip → codes) adds ~0.6 s once per
  voice; `encodeReference` returns reusable codes for repeated generations.
- **F5's flow-step count is the whole speed lever** (CFG is already batched):
  the ASR roundtrip is word-identical at 32/16/12 steps and a listening A/B
  found them indistinguishable, so the default moved from 32 to **16**
  (`--f5-steps 32` remains for maximum fidelity). Steps run over the full
  reference+target sequence, so RTF also improves on longer utterances.
- **IndexTTS2 dropped from ~9 to 1.0 RTF** in four passes: the semantic GPT
  gained a KV cache (decode was quadratic — every token re-ran the full
  sequence) with the three sampling beams batched into one forward per step;
  BigVGAN's anti-aliased FIR resamplers were rewritten from materialized
  sliding windows (a `[B, T, K, C]` gather around every Snake activation) to
  dense channels-as-batch `conv1d` — bit-equivalent output (diff 0.03% of
  signal RMS), 3.9 s → 0.7 s; the KV cache moved to preallocated in-place
  chunks (14.9 s → 9.4 s synth); and attention moved to a float32 KV
  history after step-level probes isolated the dominant remaining cost as
  a ~2.5x slowdown inside the fp16 SDPA vector kernel that only real
  decode buffer states trigger (9.4 s → 5.8 s; see stage timing above).
  The step is now within ~2 ms of its ~8 ms weight-bandwidth floor; what's
  left is host-side beam sampling (~1.5 ms/step) and the S2Mel/BigVGAN
  tail, so the practical next levers are on-device sampling and the
  10-step S2Mel option. Still the heavyweight path; prefer it for its
  emotion-control surface.
- Cloned-voice quality gates for Higgs and F5 (ASR roundtrips en/zh, plus
  es/de/ja for the Python reference) live in the E2E suites; see
  `docs/models/higgs-tts.md` and `docs/models/f5-tts.md`.
