# Speaker Embedding Benchmark

## Models

| Model | Architecture | Embedding Dim | Params | Backend | Weights |
|-------|-------------|---------------|--------|---------|---------|
| WeSpeaker ResNet34-LM | ResNet34 + Stats Pooling | 256 | 6.6M | MLX (GPU) | 25 MB |
| WeSpeaker ResNet34-LM | ResNet34 + Stats Pooling | 256 | 6.6M | CoreML (ANE) | 25 MB |
| CAM++ (3D-Speaker) | CAM++ | 192 | ~7M | CoreML (ANE) | 14 MB |

## Extraction Latency (M2 Max, 64 GB)

20s audio clip, 10 iterations after warmup.

| Engine | Dim | Mean | Min |
|--------|-----|------|-----|
| WeSpeaker MLX | 256 | 65 ms | 60 ms |
| WeSpeaker CoreML | 256 | 148 ms | 141 ms |
| CAM++ CoreML | 192 | 12 ms | 11 ms |

## Speaker Verification (LibriSpeech test-clean)

40 speakers, 2780 trial pairs. Cosine similarity scoring.

| Engine | EER% | minDCF (p=0.01) |
|--------|------|-----------------|
| WeSpeaker MLX | **0.98** | **0.084** |
| CAM++ CoreML | 7.27 | 0.288 |

Published VoxCeleb1-O: WeSpeaker 0.72% EER (WeSpeaker VoxSRC2023), CAM++ 0.65% EER (3D-Speaker, Interspeech 2023). LibriSpeech is easier than VoxCeleb1-O (read speech, fewer speakers). CAM++ uses a fixed 500-frame CoreML input (short audio tiled, long audio center-cropped).

## Embedding Quality (VoxConverse)

Cosine similarity between segment-level embeddings from 5 multi-speaker recordings. Separation = intra-speaker mean - inter-speaker mean.

| Engine | Intra-Speaker | Inter-Speaker | Separation |
|--------|--------------|--------------|------------|
| WeSpeaker MLX | 0.726 | 0.142 | **0.584** |
| WeSpeaker CoreML | 0.726 | 0.143 | 0.582 |
| CAM++ CoreML | 0.723 | 0.395 | 0.328 |

All implementations verified against Python pyannote reference (cosine >0.96).

## Reproduction

```bash
make build

# Speaker verification (LibriSpeech test-clean)
python scripts/benchmark_speaker.py --librispeech --engine mlx

# VoxConverse embedding quality
python scripts/benchmark_speaker.py --voxconverse --engine mlx

# All engines comparison
python scripts/benchmark_speaker.py --compare
```
