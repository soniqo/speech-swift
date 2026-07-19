# Speaker Diarization & Speaker Embedding

## Overview

Speaker diarization identifies **who spoke when** in an audio recording. Three engines are available:

1. **Pyannote** (default) — two-stage pipeline: segmentation + activity-based speaker chaining → post-hoc WeSpeaker embedding
2. **Community-1** (CoreML) — Pyannote segmentation + masked WeSpeaker embeddings + native PLDA/VBx clustering
3. **Sortformer** (CoreML) — NVIDIA's end-to-end neural diarization model, runs on Neural Engine

Named voice identity is a separate operation. `ReDimNet2SpeakerModel` compares
clean voice samples across recordings; it does not replace the embedding and
clustering stages inside any diarization engine.

## Architecture

### Engine Selection

```bash
speech diarize meeting.wav                    # Pyannote (default)
speech diarize meeting.wav --engine community1  # Community-1 (CoreML + VBx)
speech diarize meeting.wav --engine sortformer  # Sortformer (CoreML)
```

### Community-1 (CoreML + Native VBx)

Community-1 is the accuracy-oriented, embedding-capable CoreML pipeline. Its
two neural stages run through Core ML; powerset decoding, PLDA transforms, VBx,
constrained assignment, and timeline reconstruction are native Swift. It does
not load MLX or require a Metal shader library.

```
Audio → 10 s / 1 s-hop PyanNet → hard powerset tracks
      → overlap-masked WeSpeaker embeddings → centroid AHC
      → 256→128 PLDA transform → VBx → constrained assignment
      → overlap-add speaker counting → diarized segments + 256-d centroids
```

- **Segmentation context**: 10 seconds, advanced by 1 second
- **Local speakers**: up to three powerset tracks per chunk
- **Embedding masks**: clean single-speaker frames, with the upstream
  overlap-inclusive fallback for very short tracks
- **AHC initialization**: centroid linkage over L2-normalized embeddings,
  Euclidean threshold 0.6
- **VBx defaults**: `Fa=0.07`, `Fb=0.8`, at most 20 iterations
- **Assignment**: maximum-weight one-to-one matching inside each chunk
- **Speaker bounds**: inferred by default; exact, minimum, and maximum counts
  can be supplied by API or CLI

The bundle is
[`aufklarer/Pyannote-Community-1-CoreML`](https://huggingface.co/aufklarer/Pyannote-Community-1-CoreML),
derived from `pyannote/speaker-diarization-community-1` and distributed under
CC BY 4.0. Preserve its attribution when redistributing the weights.

### Sortformer (End-to-End, CoreML)

NVIDIA Sortformer predicts per-frame speaker activity for up to 4 speakers directly from mel features. No separate embedding or clustering stages.

```
Audio → [128-dim Mel] → [Chunk Sliding Window] → [CoreML Neural Engine] → [Sigmoid + Binarize] → Segments
                             ↕ streaming state
                        (spkcache + fifo buffers)
```

- **Input**: 128-dim log-mel features (Hann window, nFFT=400, hop=160, 16kHz)
- **Chunking**: 112 mel frames per chunk (6 core + 1 left + 7 right context × 8 subsampling)
- **CoreML model**: `[1,112,128]` chunk + `[1,188,512]` spkcache + `[1,40,512]` fifo → `[1,242,4]` speaker preds
- **State management**: FIFO overflow → spkcache (streaming state carried across chunks)
- **Post-processing**: Sigmoid → hysteresis binarization (onset=0.5, offset=0.3) → segment merging
- **Frame duration**: 0.08s per prediction frame

No speaker embeddings are produced — `--target-speaker` and `--embedding-engine` are not available with Sortformer.

### Pyannote Pipeline

```
Audio → [Segmentation] → [Per-Window Embedding] → [Constrained Clustering] → Diarized Segments
```

**Stage 1 — Segmentation**: Pyannote processes 10s sliding windows with 50% overlap. The `PowersetDecoder` extracts per-speaker probabilities from the 7-class powerset output:
- spk1 = P(class 1) + P(class 4) + P(class 5)
- spk2 = P(class 2) + P(class 4) + P(class 6)
- spk3 = P(class 3) + P(class 5) + P(class 6)

Hysteresis binarization (onset/offset) produces per-speaker speech segments within each window.

**Stage 2 — Per-Window Embedding**: For each local speaker in each window, non-overlapping speech frames (where only this speaker is active) are extracted and passed through WeSpeaker ResNet34-LM to produce a 256-dim embedding. Speakers with < 0.5s of non-overlapping speech are skipped.

**Stage 3 — Constrained Agglomerative Clustering**: Embeddings are clustered using centroid linkage with cosine distance. A **same-window constraint** ensures that speakers from the same window are never merged (they are known to be different). Merging stops when the minimum cosine distance between unconstrained pairs exceeds the threshold (default 0.715). Cluster IDs are mapped back to segments, clipped to center zones, and merged.

### WeSpeaker ResNet34-LM

~6.6M params, 256-dim output, ~25 MB.

```
Input: [B, T, 80, 1] log-mel spectrogram (80 fbank, 16kHz)
  │
  ├─ Conv2d(1→32, k=3, p=1) + ReLU           (BN fused)
  ├─ Layer1: 3× BasicBlock(32→32)
  ├─ Layer2: 4× BasicBlock(32→64, s=2)
  ├─ Layer3: 6× BasicBlock(64→128, s=2)
  ├─ Layer4: 3× BasicBlock(128→256, s=2)
  │
  ├─ Statistics Pooling: mean + std → [B, 5120]
  ├─ Linear(5120→256) → L2 normalize
  │
  Output: 256-dim L2-normalized speaker embedding
```

BatchNorm is **fused into Conv2d at conversion time** — no BN layers in the Swift model. This simplifies the model and avoids train/eval mode differences.

### CoreML Backend

WeSpeaker supports a CoreML backend (`engine: .coreml`) that runs on the Neural Engine, freeing the GPU for concurrent workloads.

```swift
let model = try await WeSpeakerModel.fromPretrained(engine: .coreml)
// Same API — embed(), cosineSimilarity()

let pipeline = try await DiarizationPipeline.fromPretrained(embeddingEngine: .coreml)
```

The CoreML model uses EnumeratedShapes for variable mel lengths (20–2000 frames, covering ~0.3s–32s audio) and float16 I/O. Input: `[1, 1, T, 80]` NCHW mel spectrogram. Output: `[1, 256]` L2-normalized embedding.

| Backend | Latency (20s audio) | Hardware | Model |
|---------|-------------------|----------|-------|
| MLX | ~310ms | GPU (Metal) | [aufklarer/WeSpeaker-ResNet34-LM-MLX](https://huggingface.co/aufklarer/WeSpeaker-ResNet34-LM-MLX) |
| CoreML | ~430ms | Neural Engine + CPU | [aufklarer/WeSpeaker-ResNet34-LM-CoreML](https://huggingface.co/aufklarer/WeSpeaker-ResNet34-LM-CoreML) |

MLX and CoreML embeddings are **not interchangeable** — NHWC vs NCHW layout causes stats pooling to flatten features in different orders. Each backend is self-consistent (cosine sim 1.0 for same input) but cross-backend similarity is low (~0.15). Use the same backend for enrollment and comparison.

### ReDimNet2-B6 Named Voice Identity

ReDimNet2-B6 is the separate Core ML encoder for recording-local and persistent
named voice matching. It accepts clean speaker-only PCM rather than a mixed
meeting waveform and returns one 192-dimensional, L2-normalized embedding.

```swift
let identity = try await ReDimNet2SpeakerModel.fromPretrained()
try identity.prewarm()

let enrollment = try identity.embed(audio: enrollmentAudio, sampleRate: 16_000)
let candidate = try identity.embed(audio: candidateAudio, sampleRate: 16_000)
let similarity = ReDimNet2SpeakerModel.cosineSimilarity(enrollment, candidate)
```

- Input is fixed at six seconds / 96,000 mono samples for the fast Core ML path.
- Clean clips from two to six seconds are repeated to fill the window.
- Longer clips are center-cropped; clips shorter than two seconds fail explicitly.
- The model uses 192-dimensional embeddings. They cannot be compared with
  WeSpeaker's 256-dimensional embeddings or Community-1 centroids.
- Matching thresholds must be calibrated for ReDimNet2; do not reuse WeSpeaker
  thresholds.
- Voice embeddings support labeling, not biometric authentication or spoofing
  resistance.

The fixed-shape compiled model is approximately 25 MiB and measured about
13.6 ms per warm six-second inference on an Apple M2 Max. This encoder remains
outside Community-1: Community-1's masked WeSpeaker, PLDA, and VBx stages stay
in their original shared embedding space.

### Mel Feature Extraction

80-dim log-mel spectrogram via vDSP (same pipeline as WhisperFeatureExtractor but with different parameters):
- **Hamming window** (not Hann): `0.54 - 0.46 * cos(2π*i/N)`
- nFFT=400, hop=160, 16kHz
- 80 mel bins with Slaney normalization
- Simple `log(max(mel, 1e-10))` — no Whisper-specific normalization

## Usage

### Speaker Diarization

```swift
let pipeline = try await DiarizationPipeline.fromPretrained()
let result = pipeline.diarize(audio: samples, sampleRate: 16000)

for seg in result.segments {
    print("Speaker \(seg.speakerId): [\(seg.startTime)s - \(seg.endTime)s]")
}
print("\(result.numSpeakers) speakers detected")
```

### Community-1 Swift API

```swift
let diarizer = try await Community1DiarizationPipeline.fromPretrained()
try diarizer.prewarm()

let result = try diarizer.diarize(
    audio: samples,
    sampleRate: 16_000,
    speakerBounds: Community1SpeakerBounds(minimum: 2, maximum: 6)
)

for segment in result.segments {
    print("Speaker \(segment.speakerId): \(segment.startTime)s–\(segment.endTime)s")
}
// result.speakerEmbeddings contains one 256-d centroid per returned speaker.
```

Use `Community1SpeakerBounds(exact: 2)` when the number of speakers is known.
Speaker IDs are local to one result; use the returned centroids to match them
to a recording-local or persistent voice registry.

#### Progress Reporting & Cancellation

For long audio files, use the `progressHandler` overload to track progress.
The handler returns a `Bool`: `true` to continue, `false` to cancel immediately.

```swift
// Progress only (never cancel)
let result = pipeline.diarize(audio: samples, sampleRate: 16000) { progress, stage in
    print("[\(Int(progress * 100))%] \(stage)")
    return true
}

// With cancellation support
let result = pipeline.diarize(audio: samples, sampleRate: 16000) { progress, stage in
    print("[\(Int(progress * 100))%] \(stage)")
    return !isCancelled  // return false to stop early
}
```

When the handler returns `false`, `diarize()` stops at the next window boundary and returns an empty `DiarizationResult`. The worst-case cancellation latency is one window's inference time (~50–200ms on Apple Silicon).

Progress is based on completed work units (segmentation windows + embedding windows). The `stage` string indicates the current processing step (e.g. "Segmenting 5/12", "Embedding 3/12").

### Speaker Embedding

```swift
let model = try await WeSpeakerModel.fromPretrained()
let embedding = model.embed(audio: samples, sampleRate: 16000)
// embedding: [Float] of length 256, L2-normalized
```

For named identity across recordings, use the throwing ReDimNet2 API instead:

```swift
let model = try await ReDimNet2SpeakerModel.fromPretrained()
let embedding = try model.embed(audio: cleanSpeakerAudio, sampleRate: 16_000)
// embedding: [Float] of length 192, L2-normalized
```

### Speaker Extraction

Given a reference audio of a target speaker, extract only their segments:

```swift
let pipeline = try await DiarizationPipeline.fromPretrained()

// Get target speaker embedding from enrollment audio
let targetEmb = pipeline.embeddingModel.embed(audio: enrollmentAudio, sampleRate: 16000)

// Extract target speaker's segments from the main audio
let segments = pipeline.extractSpeaker(
    audio: meetingAudio, sampleRate: 16000,
    targetEmbedding: targetEmb
)
```

### Sortformer Swift API

```swift
let diarizer = try await SortformerDiarizer.fromPretrained()
let result = diarizer.diarize(audio: samples, sampleRate: 16000, config: .default)

for seg in result.segments {
    print("Speaker \(seg.speakerId): [\(seg.startTime)s - \(seg.endTime)s]")
}
// result.speakerEmbeddings is empty (end-to-end model)
```

Sortformer also supports the same `progressHandler` pattern as Pyannote for progress reporting and cancellation:

```swift
let result = diarizer.diarize(audio: samples, sampleRate: 16000) { progress, stage in
    print("[\(Int(progress * 100))%] \(stage)")
    return !isCancelled  // return false to stop early
}
```

### Incremental streaming session

`SortformerStreamingSession` runs the same model incrementally for live
audio. Feed 16 kHz PCM in arbitrary sizes; every 480 ms of accumulated audio
triggers one Neural Engine step (~8 ms), and speaker IDs are Arrival-Order
Speaker Cache slots — stable for the whole session, with no window
re-clustering and no renumbering. Each push returns a whole-stream snapshot;
the model's 560 ms right context means the newest ~1 s stays pending until
the next chunk (or `finish()`).

```swift
let session = try await SortformerStreamingSession.fromPretrained()
while let samples = captureNextBuffer() {           // any push size
    let snapshot = try session.push(audio: samples)
    render(snapshot.segments)                       // stable speaker IDs
}
let final = try session.finish()                    // flushes the tail
```

An existing `SortformerDiarizer` loaded with the `.streaming` preset can open
sessions without reloading the model via `makeStreamingSession()`. The
session reproduces whole-buffer `diarize` output for the same preset —
chunk mel is extracted with reflect-padding margins that keep every consumed
frame bit-identical to whole-file extraction.

Measured on VoxConverse-dev samples (M-series ANE): DER on par with the
offline Community-1 pipeline, 12 ms median latency per 500 ms push, ~39×
realtime end to end.

### CLI Commands

```bash
# Pyannote diarization (default)
speech diarize meeting.wav

# Community-1 (CoreML + native PLDA/VBx)
speech diarize meeting.wav --engine community1
speech diarize meeting.wav --engine community1 --num-speakers 2
speech diarize meeting.wav --engine community1 --min-speakers 2 --max-speakers 6

# Sortformer diarization (CoreML, Neural Engine)
speech diarize meeting.wav --engine sortformer

# CoreML embeddings (Neural Engine, pyannote only)
speech diarize meeting.wav --embedding-engine coreml

# JSON output
speech diarize meeting.wav --json

# Speaker extraction (pyannote only)
speech diarize meeting.wav --target-speaker enrollment.wav

# Embed a speaker's voice
speech embed-speaker enrollment.wav
speech embed-speaker enrollment.wav --engine coreml --json
speech embed-speaker enrollment.wav --engine redimnet2 --json
```

Community-1 compute units can be selected with
`--community1-compute-units ane|cpu|gpu|all` (`ane` is the default).

## Community-1 Parity Check

On the fixed five-file VoxConverse release subset (1,057.49 seconds, 0.25 s
collar, overlap included), the Swift output rescored with `pyannote.metrics`
measured **4.66% DER / 21.43% JER**. The published CoreML export measured
**4.65% / 21.42%** with the same scorer. The residual difference is 0.02
percentage points of DER and comes from floating-point/tie behavior; detected
speaker counts match on all five files.

This is a small release parity check, not a dataset-wide quality claim. Do not
compare these percentages directly with the package's frame-grid benchmark,
whose collar and boundary accounting are intentionally different.

## Model Weights

- **Community-1 bundle (CoreML + PLDA/VBx)**:
  [`aufklarer/Pyannote-Community-1-CoreML`](https://huggingface.co/aufklarer/Pyannote-Community-1-CoreML)
  (~32 MiB)
- **Segmentation**: `aufklarer/Pyannote-Segmentation-MLX` (~5.7 MB)
- **Speaker Embedding (MLX)**: `aufklarer/WeSpeaker-ResNet34-LM-MLX` (~25 MB)
- **Speaker Embedding (CoreML)**: `aufklarer/WeSpeaker-ResNet34-LM-CoreML` (~13 MB)
- **Named Voice Identity (CoreML)**: `aufklarer/ReDimNet2-B6-CoreML` (~25 MiB)
- **Sortformer (CoreML)**: `aufklarer/Sortformer-Diarization-CoreML` (~240 MB)
- Cache: `~/Library/Caches/qwen3-speech/`

### Weight Conversion

Both backends fuse BatchNorm into Conv2d at conversion time: `w_fused = w * γ/√(σ²+ε)`, `b_fused = β - μ·γ/√(σ²+ε)`. MLX additionally transposes to channels-last `[O,H,W,I]`. Conversion scripts are in `scripts/`.

## Protocols

The module provides protocol conformances in `AudioCommon`:

```swift
// SpeakerEmbeddingModel
extension WeSpeakerModel: SpeakerEmbeddingModel {}

// SpeakerDiarizationModel
extension DiarizationPipeline: SpeakerDiarizationModel {
    func diarize(audio: [Float], sampleRate: Int) -> [DiarizedSegment]
}

// SpeakerDiarizationModel (end-to-end, CoreML)
extension SortformerDiarizer: SpeakerDiarizationModel {
    func diarize(audio: [Float], sampleRate: Int) -> [DiarizedSegment]
}
```

## File Structure

```
Sources/SpeechVAD/
├── MelFeatureExtractor.swift          80-dim fbank via vDSP (WeSpeaker)
├── WeSpeakerModel.swift               ResNet34 network (BN-fused Conv2d)
├── WeSpeakerWeightLoading.swift       Weight loading from safetensors
├── WeSpeaker.swift                    Public API: embed(), fromPretrained(), engine selection
├── CoreMLWeSpeakerInference.swift     CoreML inference (EnumeratedShapes, float16)
├── ReDimNet2Speaker.swift             Persistent identity encoder (fixed-waveform CoreML)
├── PowersetDecoder.swift              7-class powerset → per-speaker probs
├── DiarizationHelpers.swift            Merge segments, compact IDs, constrained clustering
├── DiarizationPipeline.swift          Pyannote pipeline (embedding clustering + speaker extraction)
├── Community1Configuration.swift      Published pipeline constants + speaker-count bounds
├── Community1CoreML.swift             PyanNet + masked WeSpeaker CoreML wrappers
├── Community1Clustering.swift         PLDA loading, centroid AHC, VBx, assignment
├── Community1DiarizationPipeline.swift  Community-1 orchestration + reconstruction
├── SortformerConfig.swift             Sortformer model configuration
├── SortformerMelExtractor.swift       128-dim log-mel for Sortformer (Hann window)
├── SortformerModel.swift              CoreML wrapper for Sortformer inference
├── SortformerDiarizer.swift           End-to-end Sortformer pipeline (streaming)
└── SpeechVAD+Protocols.swift          Protocol conformances

Sources/AudioCommon/Protocols.swift    DiarizedSegment, SpeakerEmbeddingModel, SpeakerDiarizationModel
Sources/AudioCLILib/DiarizeCommand.swift       `speech diarize` (--engine, --embedding-engine)
Sources/AudioCLILib/EmbedSpeakerCommand.swift  `speech embed-speaker` (--engine)
scripts/convert_wespeaker.py                    MLX weight conversion
scripts/convert_wespeaker_coreml.py             CoreML weight conversion
```


## Ultra-Sortformer: eight streaming speaker slots

`aufklarer/Ultra-Sortformer-Diarization-CoreML` hosts the
[Ultra-Sortformer](https://github.com/mago-research/Ultra-Sortformer)
8-speaker fine-tune of the same streaming architecture, converted with the
same pipeline and validated by the same NeMo streaming-parity gate. The
head widens 4 → 8; chunk shape, cache geometry, and the session protocol
are unchanged.

```swift
let session = try await SortformerStreamingSession.fromPretrained(
    modelId: SortformerDiarizer.ultraStreamingModelId,
    config: .streamingUltra8)
```

Benchmark it against the 4-speaker base with the diarization bench:

```bash
swift run -c release diarization-bench \
  --manifest suite.tsv \
  --engines sortformer-session sortformer-session-ultra8
```

The fine-tune trained on synthetic multi-speaker sessions and its authors
publish real-corpus rankings only for the 4-speaker base, so treat the
extra capacity as something to validate on your own data: crowded scenes
are the target, and 2–4 speaker behavior should be regression-checked
before switching defaults.
