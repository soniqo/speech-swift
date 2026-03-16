# FireRedVAD Inference Pipeline

## Overview

FireRedVAD is a lightweight VAD model based on DFSMN (Deep Feedforward Sequential Memory Network). It runs on CoreML (Neural Engine + CPU) with 588K parameters (~1.2 MB).

## Architecture

```
Audio (16kHz) → Kaldi Fbank (80-dim, 25ms/10ms) → CMVN → DFSMN (8 blocks) → DNN → sigmoid
```

- **Input**: 80-dim log Mel filterbank features (Kaldi-compatible)
- **CMVN**: Baked into CoreML model (subtract mean, multiply inverse std)
- **DFSMN**: 8 blocks, hidden=256, projection=128, depthwise Conv1d (k=20) for temporal context
- **DNN**: 1 feedforward layer (128→256, ReLU)
- **Output**: sigmoid → speech probability per frame (10ms resolution)

## Post-processing

1. Moving-average smoothing (window=5 frames)
2. Threshold at 0.4 (speech vs non-speech)
3. Minimum speech duration filter (0.2s)
4. Minimum silence gap merging (0.2s)

## Usage

### CLI

```bash
.build/release/audio vad audio.wav --engine firered
```

### Swift API

```swift
let vad = try await FireRedVADModel.fromPretrained()
let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
for seg in segments {
    print("Speech: \(seg.startTime)s - \(seg.endTime)s")
}
```

### Configuration

```swift
let vad = try await FireRedVADModel.fromPretrained()
vad.speechThreshold = 0.5      // default 0.4
vad.smoothWindowSize = 3       // default 5
vad.minSpeechDuration = 0.3    // default 0.2s
vad.minSilenceDuration = 0.1   // default 0.2s
```

## Model

- **Source**: [FireRedTeam/FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S) (Xiaohongshu)
- **Weights**: [aufklarer/FireRedVAD-CoreML](https://huggingface.co/aufklarer/FireRedVAD-CoreML)
- **Parameters**: 588,417 (2.2 MB float32, 1.2 MB CoreML float16)
- **Conversion**: `scripts/convert_fireredvad.py`

## Performance

On test audio (20s, single speaker):
- **Detection**: [5.17s - 8.37s] — matches Python reference exactly
- **Latency**: 0.12s for 20s audio (RTF 0.006)
- **Model load**: ~0.5s (CoreML cached)

Paper results (FLEURS-VAD-102, 102 languages):
- F1: 97.57%, FAR: 2.69%, MR: 3.62%
- Outperforms Silero VAD (95.95% F1) and all other open-source baselines
