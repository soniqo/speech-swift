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
2. **Threshold**: 0.4 (speech if probability ≥ threshold)
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
