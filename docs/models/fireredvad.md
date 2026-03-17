# FireRedVAD: DFSMN Voice Activity Detection

## Architecture

FireRedVAD uses DFSMN (Deep Feedforward Sequential Memory Network) — a purely feedforward architecture with depthwise 1D convolutions for temporal context. No recurrence (unlike Silero's LSTM), making it ideal for CoreML/Neural Engine.

```
Audio (16kHz)
  │
  ├── Kaldi Fbank: 80-dim log Mel (25ms window, 10ms shift, Povey window)
  ├── CMVN normalization (baked into CoreML model)
  │
  ├── Input Layer:
  │   Linear(80→256) + ReLU
  │   Linear(256→128) + ReLU
  │   FSMN: depthwise Conv1d(128, k=20, groups=128) + residual
  │
  ├── 7× DFSMN Blocks:
  │   Linear(128→256) + ReLU
  │   Linear(256→128, no bias)
  │   FSMN: depthwise Conv1d(128, k=20, groups=128) + skip connection
  │
  ├── DNN: Linear(128→256) + ReLU
  │
  └── Output: Linear(256→1) → sigmoid → speech probability per frame
```

## DFSMN Block

Each FSMN layer uses depthwise 1D convolution for temporal context:
- **Lookback**: `k=20, stride=1, dilation=1` (causal, 200ms context)
- **Lookahead**: `k=20, stride=1, dilation=1` (non-streaming, 200ms future context)
- **Depthwise**: `groups=P` (128), each channel has independent temporal filter
- **Residual**: input added to FSMN output (skip connection)

## Specifications

| Property | Value |
|----------|-------|
| Parameters | 588,417 |
| Size (float32) | 2.2 MB |
| Size (CoreML float16) | 1.2 MB |
| Input | 80-dim log Mel fbank |
| Output | Speech probability [0,1] per frame |
| Frame rate | 100 Hz (10ms shift) |
| Sample rate | 16 kHz |
| Temporal context | 400ms (200ms lookback + 200ms lookahead) |

## Feature Extraction

Kaldi-compatible log Mel filterbank:
- 25ms Povey window (Hann^0.85), 10ms hop
- 0.97 pre-emphasis, DC offset removal
- 512-point DFT (zero-padded from 400 samples)
- 80 mel bins (20Hz–8kHz, Hz-domain triangular filters)
- Log energy with FLT_EPSILON floor

CMVN (Cepstral Mean and Variance Normalization) is baked into the CoreML model — Swift passes raw fbank features directly.

## Weight Files

- **Source**: [FireRedTeam/FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S)
- **CoreML**: [aufklarer/FireRedVAD-CoreML](https://huggingface.co/aufklarer/FireRedVAD-CoreML)
- **Conversion**: `scripts/convert_fireredvad.py` (PyTorch → CoreML, bakes CMVN)

## References

- Paper: "FireRedASR2S: A State-of-the-Art Industrial-Grade All-in-One ASR System" (arXiv:2603.10420)
- FLEURS-VAD-102 benchmark: 97.57% F1, 2.69% FAR, 3.62% MR
