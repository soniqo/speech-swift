# Parakeet-EOU-120M — Streaming Inference Pipeline

Runtime pipeline for real-time microphone dictation with
`ParakeetStreamingASRModel`. Covers the session API, chunking math,
VAD integration, and the force-finalize pattern used by DictateDemo.

For model architecture see [parakeet-streaming-asr.md](../models/parakeet-streaming-asr.md).

## Quick start — batch

```swift
import ParakeetStreamingASR

let model = try await ParakeetStreamingASRModel.fromPretrained()
let text = try model.transcribeAudio(audioSamples, sampleRate: 16000)
```

Conforms to `SpeechRecognitionModel`, so it drops into any code path that
accepts a generic STT model (voice pipelines, diarization + ASR, etc.).

## Preinstalled and offline models

Use `cacheDir` to control where `fromPretrained()` stores the model and
`offlineMode` to reuse an existing cache without a network request:

```swift
let model = try await ParakeetStreamingASRModel.fromPretrained(
    cacheDir: appModelsDirectory,
    offlineMode: true
)
```

Apps that install and verify the model themselves can bypass the downloader
entirely. The directory must contain `encoder.mlmodelc/`, `decoder.mlmodelc/`,
`joint.mlmodelc/`, `vocab.json`, and optionally `config.json`:

```swift
let model = try await ParakeetStreamingASRModel.fromLocal(
    bundleDir: installedModelDirectory
)
```

## Quick start — AsyncSequence streaming

```swift
for await partial in model.transcribeStream(audio: samples, sampleRate: 16000) {
    if partial.isFinal { print("FINAL: \(partial.text)") }
    else               { print("... \(partial.text)") }
}
```

`transcribeStream` creates an internal `StreamingSession`, chunks the input
in session-native chunk sizes, and yields `PartialTranscript` values as the
model emits them. Each value carries:

| Field | Meaning |
|---|---|
| `text` | Decoded text for the current utterance so far |
| `isFinal` | `true` when this partial ends the current utterance (EOU or force) |
| `confidence` | `exp(mean(logProb))` of the tokens since the last EOU |
| `eouDetected` | `true` if the joint's EOU head fired (vs force-finalized) |
| `segmentIndex` | Monotonic utterance index, increments on each final |

## Long-lived session API (mic input)

For live dictation, create a session once and feed it chunks as they come
from the mic:

```swift
let session = try model.createSession()

// each mic chunk:
let partials = try session.pushAudio(float32Chunk16kHz)
for p in partials {
    if p.isFinal { commit(p.text) }
    else          { showPartial(p.text) }
}

// when the stream ends:
let trailing = try session.finalize()
```

### Chunk sizing

`pushAudio` buffers audio internally and runs the encoder once enough samples
accumulate. The math:

- `samplesPerChunk = streaming.melFrames * hopLength` — total input samples
  per encoder call (default `64 * 160 = 10240` = 640 ms)
- `shiftSamples   = streaming.outputFrames * subsamplingFactor * hopLength` —
  how far the buffer advances between calls (default `20 * 4 * 160 = 12800`)

The encoder chunk is wider than the shift by `(melFrames - outputFrames * subsamplingFactor) * hopLength`
samples of future-context overlap. The session keeps the overlap internally
and only drops `shiftSamples` of the buffer per call — callers don't need to
manage the overlap themselves.

Practical consequence: push audio in arbitrary sizes. The session processes
whenever enough samples have accumulated. For lowest latency, push every
~300 ms so each timer tick triggers 0–2 encoder calls.

## End-of-utterance: two signals, one decision

The session has two ways to decide an utterance is over:

### 1. Joint EOU head (model-internal)

The joint network has a dedicated EOU class. When the greedy decoder emits
the EOU token, `result.eouDetected` is true for that chunk. The session
debounces over `eouDebounceMs` (default 1280 ms) — EOU must be sustained
across multiple chunks without new tokens before a final is emitted.

**Why debounce:** the joint sometimes emits EOU mid-word on disfluencies,
so firing immediately would split utterances. Waiting for sustained EOU
prevents this.

**Failure mode:** room tone, keyboard clicks, mouse movement during a
"silent" pause will make the joint occasionally emit a non-blank token,
resetting the debounce timer. In practice this can delay committal by
several seconds (see the DictateDemo regression tests).

### 2. External VAD force-finalize (DictateDemo pattern)

When you have a Silero VAD already running in your pipeline, use it to
drive force-finalize:

```swift
if hasPendingUtterance && !vadSpeechActive && vadSilentChunks >= 30 {
    // ~960 ms of sustained silence per Silero
    if let forced = session.forceEndOfUtterance() {
        commit(forced.text)
    }
    hasPendingUtterance = false
}
```

`forceEndOfUtterance()` commits whatever tokens have accumulated since the
last segment boundary, advances internal state, and keeps the encoder /
decoder caches alive so the next utterance continues streaming. It is
independent of the joint's EOU debounce timer.

**Guardrail:** don't also force-finalize if the joint already fired an EOU
for the same chunk, or you'll get duplicate sentences:

```swift
if partials.contains(where: { $0.isFinal }) {
    hasPendingUtterance = false   // joint handled it
}
```

## VAD chunking for Silero

Silero VAD requires **exactly** 512-sample chunks at 16 kHz. Audio arriving
from the mic does not line up with that boundary, so accumulate a leftover
buffer between calls and carry it forward:

```swift
var leftover: [Float] = []

// per mic batch:
var vadInput = leftover + micBatch
var offset = 0
while offset + 512 <= vadInput.count {
    let prob = vad.processChunk(Array(vadInput[offset..<offset+512]))
    // ... track speechActive / silenceCount
    offset += 512
}
leftover = Array(vadInput[offset...])
```

Dropping the trailing sub-512 remainder would starve Silero of audio that it
needs for LSTM state continuity. Call `vad.resetState()` when starting a new
recording so prior-session LSTM state doesn't leak.

## Complete pipeline reference

`Examples/DictateDemo/DictateDemo/DictateViewModel.swift` has the full
implementation used by the menu-bar demo:

- `ASRProcessor` — off-main audio sink with lock-protected buffer
- 300 ms timer tick on a background queue, calls `processBuffered`
- Silero VAD with leftover carry-over + persistent silence counter
- Guarded VAD force-finalize (skipped if joint already finalized)
- `RunLoop.main.perform(inModes: [.common, .default, .eventTracking, .modalPanel])`
  for UI updates — required for MenuBarExtra popovers (default-mode
  dispatch is starved while the popover holds the run loop in tracking mode)

The matching E2E tests are in `Examples/DictateDemo/Tests/DictateDemoTests.swift`:

- `testMultiUtteranceForceFinalize` — two utterances with silence → two finals
- `testSecondUtteranceNotPrematurelyFinalized` — stuck-EOU regression
- `testNoiseInSilenceDoesNotBlockFinalize` — noisy-silence regression using
  synthetic background noise in the gap

## Performance

| Metric | Value |
|---|---|
| Weight memory | ~120 MB (INT8) |
| Peak inference | ~200 MB |
| Chunk latency (M-series) | ~30 ms / 340 ms-of-audio (RTF ~0.09) |
| End-to-end partial latency | ~340 ms (one chunk) |
| End-to-end commit latency (VAD path) | ~1 s after speech stops |

## See also

- [parakeet-streaming-asr.md](../models/parakeet-streaming-asr.md) — model architecture
- [silero-vad.md](silero-vad.md) — Silero VAD streaming details
- [shared-protocols.md](../shared-protocols.md) — `SpeechRecognitionModel` protocol
