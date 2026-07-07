# iPhone on-device CoreML benchmarks

**Device**: iPhone 16 Pro (A18 Pro), iOS 26.5, release build. Measured with
[`Examples/iOSBenchmark`](../../Examples/iOSBenchmark) — an on-device harness that
loads each CoreML model, runs one warm-up pass, then takes the median of 5 timed
runs. Peak memory is `phys_footprint` (matches the Xcode Memory Report).

RTF = wall-time ÷ audio-seconds — **lower = faster; < 1.0 is faster than real time**
(the standard ASR convention). The LLM row reports decode throughput in tokens/s.

| Task | Model | RTF / throughput | Peak memory |
| ---- | ----- | ---------------- | ----------- |
| Streaming ASR + EOU | Parakeet-EOU 120M · CoreML INT8 | **0.04 RTF** | 297 MB |
| Multilingual ASR | Omnilingual 300M · CoreML INT8 | **0.28 RTF** | 495 MB |
| TTS | Supertonic-3 99M · CoreML | **0.15 RTF** | 956 MB |
| TTS | Kokoro-82M · CoreML | **0.08 RTF** | 676 MB |
| LLM tokens/s | FunctionGemma 270M · CoreML ANE | **128 tok/s** | 236 MB |

Every model runs faster than real time on the iPhone Neural Engine / GPU. The
figures are slower than the M5 Pro Mac results in the other docs here — that is
the expected iPhone-class ANE envelope, not a regression. These same numbers are
published on the [soniqo.audio](https://soniqo.audio) landing page alongside the
M5 Pro (CoreML/MLX) and Galaxy S23 (ONNX/LiteRT) columns.

## Reproducing

See [`Examples/iOSBenchmark/README.md`](../../Examples/iOSBenchmark/README.md).
On a slow or flaky phone connection the first-run HuggingFace downloads can stall;
side-load the CoreML bundles from a Mac's `~/Library/Caches/qwen3-speech/models/`
cache into the app container to skip them (documented in the harness README).
