# TTS Round-Trip WER Benchmark

## Method

Synthesize text → transcribe audio back → compute WER vs original text. Measures TTS intelligibility end-to-end.

## Results (smoke test, 2 sentences)

| TTS Engine | ASR Engine | Round-trip WER% |
|------------|-----------|----------------|
| Qwen3-TTS (base) | Qwen3-ASR 0.6B | 0.00 |

Full benchmark (30 sentences) pending. Expected to show TTS stage latency breakdown (embed/generate/decode).

## TTS Latency Breakdown

Qwen3-TTS outputs detailed per-stage timing:
- **embed**: Text embedding preparation
- **generate**: Autoregressive codec token generation (ms/step)
- **decode**: Codec decoder → waveform

The benchmark script parses these automatically.

## Reproduction

```bash
make build
python scripts/benchmark_tts.py --num-sentences 10
python scripts/benchmark_tts.py --compare  # all TTS engines
```
