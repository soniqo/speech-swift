# FireRedVAD Inference Pipeline

## Pipeline

```
Audio → Resample to 16kHz → Kaldi Fbank (80-dim) → CoreML (ANE) → Post-processing → Segments
```

1. **Feature extraction**: Kaldi-compatible 80-dim log Mel fbank (vDSP_mmul DFT basis)
2. **CoreML inference**: DFSMN model on Neural Engine (CMVN baked in)
3. **Post-processing**: Smoothing → threshold → duration filter → gap merging

## CLI

```bash
# Basic usage
.build/release/audio vad audio.wav --engine firered

# Custom threshold
.build/release/audio vad audio.wav --engine firered --onset 0.5
```

## Swift API

```swift
let vad = try await FireRedVADModel.fromPretrained()
let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
for seg in segments {
    print("Speech: \(seg.startTime)s - \(seg.endTime)s")
}
```

## Configuration

```swift
let vad = try await FireRedVADModel.fromPretrained()
vad.speechThreshold = 0.5      // default 0.4
vad.smoothWindowSize = 3       // default 5
vad.minSpeechDuration = 0.3    // default 0.2s
vad.minSilenceDuration = 0.1   // default 0.2s
```

## Post-processing

1. **Moving-average smoothing**: 5-frame window reduces frame-level noise
2. **Threshold**: 0.4 (speech if probability >= threshold)
3. **Minimum speech duration**: 0.2s (discard short bursts)
4. **Minimum silence gap merging**: 0.2s (bridge short pauses)

## Chunking

For audio longer than 60s, features are processed in 6000-frame chunks (CoreML input limit). Chunks are processed independently — no cross-chunk state.

## Performance

| Metric | Value |
|--------|-------|
| RTF | 0.007 (135x real-time) |
| Cold start | ~0.5s (CoreML cached) |
| FLEURS F1 | 99.12% (vs Python reference) |
| VoxConverse F1 | 94.21% |

## VoxConverse Performance Gap (#146)

FLEURS F1 is 99.12% but VoxConverse F1 drops to 94.21% with a 69% false alarm rate (FAR). The root cause is a model-level limitation on conversational audio with overlapping speakers and background noise. This is not a feature extraction issue — verified by feeding exact Kaldi-compatible features (matching the Python reference pipeline frame-for-frame) to the CoreML model. The model itself produces high false alarm rates on this type of audio.

## Threshold Tuning Results

Grid-searched threshold and smoothing window on VoxConverse (5 files):

| Threshold | Smooth | F1% | FAR% | MR% |
|-----------|--------|-----|------|-----|
| 0.3 | 11 | 94.2 | 74.4 | 4.7 |
| 0.4 | 5 | 93.5 | 68.9 | 6.4 |
| 0.5 | 5 | 93.1 | 64.0 | 7.5 |
| 0.8 | 5 | 91.9 | 47.5 | 11.2 |

Smoothing window has minimal effect (+/-0.3% F1). The threshold controls the FAR/MR tradeoff but cannot fix the underlying gap — raising the threshold reduces false alarms at the cost of increased miss rate.

## Alternative Models (local testing)

- **Stream-VAD (N2=0, causal)**: Best F1 94.3%, lowest MR 4.9%. Removing the lookahead context (N2=0) helps on conversational audio where future context contains overlapping speakers.
- **AED speech channel**: Best FAR 48.4% but higher MR 9.1%. The 3-class model (speech/singing/music) provides a more discriminative speech channel than the binary VAD output.
- **Energy pre-filter**: Harmful — F1 drops to 81%, kills real speech. Energy-based gating is too coarse for conversational audio with varying loudness.

## CLI Tuning

```bash
# Custom threshold and smoothing
audio vad audio.wav --engine firered --threshold 0.5 --smooth-window 7

# Local model variant (e.g., Stream-VAD N2=0)
audio vad audio.wav --engine firered -m /path/to/local/model
```
