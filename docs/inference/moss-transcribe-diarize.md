# MOSS Transcribe Diarize Inference

## Overview

```text
Audio
  -> 16 kHz mono
  -> native Whisper log-mel frontend
  -> Core ML audio encoder
  -> audio-aware MOSS prompt
  -> stateful Core ML decoder
  -> timestamped speaker segments + plain text
```

The `MossTranscribe` target is a native Swift/Core ML runtime. It does not
invoke Python, ONNX Runtime, or MLX during inference.

## CLI

INT8 is the default MOSS variant:

```bash
.build/release/speech transcribe recording.wav --engine moss
```

Select the FP16 reference or a custom compatible repository:

```bash
.build/release/speech transcribe recording.wav \
  --engine moss --model fp16

.build/release/speech transcribe recording.wav \
  --engine moss \
  --model organization/custom-moss-coreml \
  --max-tokens 768
```

The command prints the plain transcript, timestamped speaker segments, elapsed
time, real-time factor, and realtime throughput. `--stream` is rejected because
this runtime currently performs batch decoding.

## Swift API

```swift
import MossTranscribe

let model = try await MossTranscribeModel.fromPretrained(
    variant: .int8)
try model.warmUp()

let result = try model.transcribeDetailed(
    audio: samples,
    sampleRate: 48_000)

print(result.text)
for segment in result.segments {
    print(segment.startTime, segment.endTime, segment.speaker, segment.text)
}
print(result.metrics.realTimeFactor)
```

Use `.fp16` for the FP16 decoder:

```swift
let reference = try await MossTranscribeModel.fromPretrained(
    variant: .fp16)
```

`fromDirectory` loads a freshly exported or already downloaded bundle without
contacting Hugging Face:

```swift
let local = try await MossTranscribeModel.fromDirectory(
    URL(fileURLWithPath: "/path/to/moss-coreml"))
```

## Structured output

`MossTranscription` contains:

- `rawText`: the decoded upstream timestamp/speaker wire format;
- `text`: plain text formed from valid segments;
- `segments`: `[MossTranscriptSegment]`;
- `metrics`: preprocessing, encoder, prefill, decode, total time, token counts,
  RTF, throughput, and stop reason.

The default instruction requests `[start][Sxx] text [end]` segments. A custom
instruction can be supplied to `transcribeDetailed`, but changing the output
contract may leave `segments` empty; `rawText` and `text` still preserve the
model output.

## Warm-up

`warmUp()` specializes the audio encoder, 128-token prefill, and one-token
decode paths. It also preloads the fixed prompt embeddings. Call it once during
application initialization when first-request latency matters.

Core ML caches compiled execution plans on disk. The first run after changing a
model or compute-unit policy may therefore take longer than subsequent process
launches.

## Audio and context limits

- Input is resampled to 16 kHz mono Float32.
- The frontend emits one fixed `[80, 3000]` feature tensor per at-most-30-second
  encoder chunk.
- Audio produces 12.5 decoder embeddings per second.
- The stateful decoder has a fixed 1024-token cache shared by the prompt and
  generated output.

Longer input consumes more of the decoder cache and leaves fewer generation
tokens. If the audio prompt itself exceeds 1024 tokens, the call fails with
`promptTooLong` instead of truncating audio silently.

## Cache and offline use

Models use the standard speech-swift cache. Once downloaded:

```swift
let model = try await MossTranscribeModel.fromPretrained(
    variant: .int8,
    offlineMode: true)
```

The runtime checks the compiled Core ML files, decoder configuration, tokenizer,
host contract, and exact Whisper frontend configuration before loading.

## Benchmarking

Both variants are registered in `asr-bench`:

```bash
.build/release/asr-bench \
  --dataset english.tsv \
  --engines moss-coreml-int8 moss-coreml-fp16 \
  --isolated \
  --output moss.json
```

Use an isolated release build for publishable RSS numbers. The current matched
results and reproduction details are in
[MOSS CoreML benchmark](../benchmarks/moss-coreml.md).

## Current scope

- Greedy decoding only; no beam search or temperature fallback.
- Timestamped speaker segments are model-generated, not a separate diarization
  pipeline.
- Batch inference only.
- The 1024-token decoder context is the hard combined prompt/output limit.

See [MOSS model architecture](../models/moss-transcribe-diarize.md) for the
export layout and quantization profile.
