# Speaker Embedding Benchmark

## Models

| Model | Architecture | Embedding Dim | Params | Backend | Weights |
|-------|-------------|---------------|--------|---------|---------|
| WeSpeaker ResNet34-LM | ResNet34 + Stats Pooling | 256 | 6.6M | MLX (GPU) | 25 MB |
| WeSpeaker ResNet34-LM | ResNet34 + Stats Pooling | 256 | 6.6M | CoreML (ANE) | 25 MB |
| CAM++ (3D-Speaker) | CAM++ | 192 | ~7M | CoreML (ANE) | 14 MB |

## Extraction Latency (M2 Max, 64 GB)

20s audio clip, 10 iterations after warmup.

| Engine | Dim | Mean | Std | Min |
|--------|-----|------|-----|-----|
| WeSpeaker MLX | 256 | 64 ms | 4.3 ms | 55 ms |
| WeSpeaker CoreML | 256 | 143 ms | 2.7 ms | 140 ms |
| CAM++ CoreML | 192 | 12 ms | 0.3 ms | 11 ms |

CAM++ is **5x faster** than WeSpeaker MLX and **12x faster** than WeSpeaker CoreML. The 192-dim model runs almost entirely on the Neural Engine with minimal overhead.

WeSpeaker CoreML is slower than MLX because the ResNet34 architecture maps poorly to the Neural Engine — convolutions with statistics pooling favor GPU execution.

## Embedding Quality (VoxConverse)

Cosine similarity between segment-level embeddings extracted from VoxConverse test set (5 multi-speaker recordings). Measures how well embeddings discriminate speakers in real conversational audio.

- **Intra-speaker**: cosine similarity between different segments of the **same** speaker
- **Inter-speaker**: cosine similarity between segments of **different** speakers
- **Separation**: intra - inter (higher = more discriminative)

| Engine | Intra (mean ± std) | Inter (mean ± std) | Separation |
|--------|-------------------|-------------------|------------|
| WeSpeaker MLX | 0.929 ± 0.056 | 0.920 ± 0.050 | 0.008 |
| WeSpeaker CoreML | 0.896 ± 0.057 | 0.892 ± 0.055 | 0.005 |
| CAM++ CoreML | 0.693 ± 0.162 | 0.436 ± 0.132 | **0.257** |

### Key findings

- **WeSpeaker has near-zero separation** on conversational audio (0.008). Both same-speaker and different-speaker segments produce very similar embeddings (>0.92 cosine). This explains the poor clustering in diarization (#145) — agglomerative clustering cannot reliably distinguish speakers when embeddings are this similar.

- **CAM++ has 30x better separation** (0.257 vs 0.008). The inter-speaker mean drops to 0.44, creating clear separation from the intra-speaker mean of 0.69. This makes it viable for speaker clustering.

- **WeSpeaker's high inter-speaker similarity** (0.92) on VoxConverse is an artifact of the evaluation context: all speakers share the same recording environment, microphone, and acoustic conditions. WeSpeaker captures channel characteristics as much as speaker identity. On VoxCeleb (diverse recording conditions), WeSpeaker achieves EER < 1%.

- **CAM++ is more robust to channel effects** — its lower absolute similarity values indicate it focuses more on speaker-specific features and less on shared acoustic conditions.

### Implications for diarization

The WeSpeaker separation gap of 0.008 means standard agglomerative clustering (cosine distance threshold) cannot reliably separate speakers in same-channel audio. Options:
1. **Use CAM++ for diarization clustering** — 30x better separation
2. **Per-segment embedding extraction** instead of per-window (#145)
3. **Spectral clustering** which can handle smaller margins

## VoxCeleb1-O Speaker Verification

Standard speaker verification benchmark (37,720 trial pairs from 40 speakers). Requires VoxCeleb1 test audio download.

| Engine | EER% | minDCF (p=0.01) | Positive Mean Sim | Negative Mean Sim |
|--------|------|-----------------|-------------------|-------------------|
| WeSpeaker MLX | — | — | — | — |

*Results pending VoxCeleb1 audio download.*

Published WeSpeaker ResNet34-LM numbers: **EER 0.56%** on VoxCeleb1-O (from WeSpeaker paper). Our MLX port should match within quantization tolerance.

## Reproduction

```bash
make build

# Latency benchmark
python scripts/benchmark_speaker.py --latency --engine mlx
python scripts/benchmark_speaker.py --latency --engine coreml
python scripts/benchmark_speaker.py --latency --engine camplusplus

# VoxConverse embedding quality
python scripts/benchmark_speaker.py --voxconverse --engine mlx

# All engines comparison
python scripts/benchmark_speaker.py --compare

# VoxCeleb1-O verification (requires audio download)
python scripts/benchmark_speaker.py --download-voxceleb
python scripts/benchmark_speaker.py --voxceleb --engine mlx
```
