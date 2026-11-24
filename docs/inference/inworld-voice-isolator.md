# Inworld Voice Isolator v1

The offline checkpoint isolates foreground speech in recordings. It accepts
mono Float32 PCM and returns aligned audio at 48 kHz. The CLI averages input
channels, resamples to 48 kHz, and writes a mono Float32 WAV. The package runs
the learned network in Core ML and the selective state-space recurrence in
Metal on Apple Silicon. [Inworld model card](https://github.com/inworld-ai/voice-isolator/blob/c73b3fab81a870410cbac50d2f05f25edc4d10aa/MODEL_CARD.md)

## Core ML bundle

Pass a local offline export directory to the CLI or Swift initializer. Keep
its compiled stage directories, scan
coefficients, format manifest, license, and notices together. The runtime
also accepts stages in the `.mlpackage` format. The bundle uses weights from
the pinned Apache-2.0 checkpoint. [Upstream weights](https://huggingface.co/inworld/voice-isolator/tree/9b0285b3fe8403aa8e2f1dbb2165a80a4294c970)

## CLI

```bash
swift build --product speech --disable-sandbox
.build/debug/speech denoise noisy.wav --engine inworld \
  --model InworldVoiceIsolator-CoreML -o voice.wav
```

Recordings use eight-second windows with a two-second linear crossfade. The
last full window ends at the end of the recording. `--chunk-seconds` accepts
0.01 through 8 seconds. `--overlap-ms` is capped at half the window. Use
`--no-chunk` for a single input of at most eight seconds. DeepFilterNet3 is
the default engine for `speech denoise`.

Bundles with an eight-second Core ML specialization write the Mamba scan
input directly into shared Metal memory. Shorter windows use their exact
frame count through the flexible Core ML function.

The lower-memory bundle processes the independent rows of each axis in
smaller groups. Every time-attention row still sees the full eight-second
window. Recent exports include fixed shapes for full tiles, which reduce Core
ML prediction time and temporary memory. When the bundle has these fixed
shapes, adjacent windows share the stage loads. Both bundles use the same CLI
and Swift API. [Performance profile](../benchmarks/inworld-voice-isolator.md#offline-tile-sizes)

## Swift

```swift
import Foundation
import SpeechEnhancement

let model = try InworldVoiceIsolator(
    modelDirectory: URL(fileURLWithPath: "InworldVoiceIsolator-CoreML")
)
let voice = try model.enhance(audio: samples, sampleRate: inputSampleRate)
```

`InworldVoiceIsolator` conforms to `SpeechEnhancementModel`. Its initializer
defaults Core ML to `.cpuAndGPU`. `enhanceChunked` accepts custom windows and
crossfades. For a background residual, subtract `voice` sample by sample from
the same 48 kHz mono input.

### Streaming

Use a Core ML export of the realtime checkpoint with
`InworldVoiceIsolatorStream`. Feed it mono Float32 frames at 48 kHz and
serialize calls to each session. Each push returns the samples that became
final. Flush at the end to retrieve the buffered tail.

```swift
let stream = try InworldVoiceIsolatorStream(
    modelDirectory: URL(fileURLWithPath: "InworldVoiceIsolator-Realtime-Fast-CoreML")
)
try stream.prepare()
var pending: [Float] = []
for chunk in pcmChunks {
    pending.append(contentsOf: chunk)
    while pending.count >= stream.preferredPushSamples {
        let count = stream.preferredPushSamples
        consume(try stream.push(Array(pending.prefix(count))))
        pending.removeFirst(count)
    }
}
consume(try stream.push(pending))
consume(try stream.flush())
```

The fast export processes up to eight frames per graph call by default.
Call `prepare()` before capture to warm a prediction and reset the
stream state. Use `preferredPushSamples` to determine how many samples
to collect before the next push. With aligned capture at 48 kHz, the
first push takes 4,384 samples (91.3 ms) to fill the analysis window and
later pushes take 3,840 samples (80 ms). This places crossfade boundaries
between predictions. Capture buffers smaller than the model blocks
let the caller deliver the first block as soon as it is ready.
Export with `--streaming-block-frames 2` for 20 ms blocks or `4` for
40 ms blocks. Partial pushes are accepted.

Use `lookaheadFrames: 0`, `2`, or `4` for 0, 20, or 40 ms of model
look-ahead. The default is 20 ms. `latencySamples` includes the analysis
window and model look-ahead. Input buffering and processing time depend
on the caller's push cadence.

The 80 ms export runs both crossfade lanes through the same one-lane
graph and shares the spectrum encoder output. The 20 ms and 40 ms fast
exports batch both lanes during a crossfade. Fast exports use Float16 for
selected large linear layers and run the band decoder in Metal. The
low-memory export keeps learned operations in Float32 and uses one
two-frame graph per stage for both lane counts.

The 80 ms fast export packs attention keys at 10 bits and values at
12 bits. The smaller fast blocks and the low-memory export use 12 bits
for both. The runtime pads partial blocks and saves state at the last
real frame. The session also accepts earlier specialized bundles, older
Float16 cache bundles, and the seven-stage stateful export.

The 80 ms path allocates attention cache pages as audio arrives on
devices with Tier 2 Metal argument buffers. Short sessions use fewer
pages. [Streaming benchmarks](../benchmarks/inworld-voice-isolator.md#streaming-cache-pages)

## Reference fixtures

The end-to-end test compares Float32 WAV output from `speech denoise` against
Inworld's three- and eleven-second reference renders. The eleven-second clip
exercises the end-anchored window and crossfade. Use the `tests/fixtures`
directory from the pinned upstream source. [Upstream fixtures and attribution](https://github.com/inworld-ai/voice-isolator/tree/c73b3fab81a870410cbac50d2f05f25edc4d10aa/tests/fixtures)

```bash
INWORLD_VOICE_ISOLATOR_MODEL_DIR="$PWD/InworldVoiceIsolator-CoreML" \
INWORLD_VOICE_ISOLATOR_FIXTURES=/path/to/voice-isolator/tests/fixtures \
  swift test --disable-sandbox --no-parallel --filter E2EInworldDenoiseTests
```

The streaming test uses incremental input with fixed and uneven pushes and
compares the complete output to the corresponding reference renders.

```bash
INWORLD_VOICE_ISOLATOR_FAST_MODEL_DIR="$PWD/InworldVoiceIsolator-Realtime-Fast-CoreML" \
INWORLD_VOICE_ISOLATOR_FIXTURES=/path/to/voice-isolator/tests/fixtures \
  swift test --disable-sandbox --no-parallel \
    --filter E2EInworldVoiceIsolatorStreamTests/testFastGoldenRendersWithIncrementalInput
```

```bash
INWORLD_VOICE_ISOLATOR_LOW_MEMORY_MODEL_DIR="$PWD/InworldVoiceIsolator-Realtime-LowMemory-CoreML" \
INWORLD_VOICE_ISOLATOR_FIXTURES=/path/to/voice-isolator/tests/fixtures \
  swift test --disable-sandbox --no-parallel \
    --filter E2EInworldVoiceIsolatorStreamTests/testLowMemoryGoldenRendersWithIncrementalInput
```

See the [model architecture](../models/inworld-voice-isolator.md) for the
Core ML and Metal boundary.
