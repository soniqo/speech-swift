# Parakeet Ultra Core ML benchmark

On an Apple M5 Pro with macOS 26.6.2, Parakeet Ultra reduced word errors by about 10% on clean speech and 11% on the noise mixtures. Both 30-second INT8 bundles decoded at about 100× real time in this run.

| Audio | Clips | Reference words | v3 WER | Ultra WER |
| --- | ---: | ---: | ---: | ---: |
| Clean English | 80 | 1,702 | 5.29% | 4.64% |
| Clean French | 80 | 2,171 | 5.16% | 4.74% |
| **Clean total** | **160** | **3,873** | **5.22%** | **4.70%** |
| Noisy English | 24 | 396 | 3.54% | 3.03% |
| Noisy French | 24 | 576 | 8.85% | 7.99% |
| **Noisy total** | **48** | **972** | **6.69%** | **5.97%** |

| Metric over 2,006 seconds of audio | v3 | Ultra |
| --- | ---: | ---: |
| Warm inference time | 20.33 s | 20.09 s |
| Real-time factor (lower is faster) | 0.0101 | 0.0100 |
| Peak CLI process footprint | 98.6 MB | 98.7 MB |

Both models used the release `speech transcribe-batch` command, fixed 30-second encoders, and Core ML configured for CPU and Neural Engine. The Core ML compute plan selected the Neural Engine for 1,374 encoder operations and CPU for nine in each bundle. Each model ran in a separate process on the same audio. The command warms up the model before measuring inference time. Peak process footprint comes from `/usr/bin/time -l`.

The clean set uses FLEURS English and French test rows 0–79. The noisy set uses eight evenly spaced clips from each language, with white noise at 20 and 10 dB signal-to-noise ratio and pink noise at 10 dB. Word error rate uses the package's `Normalizer` and `WER` implementations. [FLEURS test set](https://huggingface.co/datasets/google/fleurs/tree/168de341b3db6859a9bac1c50a2ef5e3b47647e0) · [v3 benchmark bundle](https://huggingface.co/aufklarer/Parakeet-TDT-v3-CoreML-INT8-30s/tree/8054d9f680cfa4c0012876ed71e854fb20dd563b)

The Ultra end-to-end test passed with CPU-only and Neural Engine configurations. It checks silence, language steering, confidence, and absolute word timestamps across a 30-second boundary using English and German fixtures. [Ultra benchmark bundle](https://huggingface.co/aufklarer/Parakeet-Ultra-CoreML-INT8-30s/tree/6c88a36e5751e7ef6c4d233dfcec5643767cbfc1)
