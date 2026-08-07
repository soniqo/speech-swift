# VoiceChat inference

`VoiceChat` runs NVIDIA NemotronLabs VoiceChat 11B as a continuous native MLX
Swift duplex speech-to-speech session. It accepts user speech as 16 kHz mono
Float32 PCM and directly returns agent speech, plus the corresponding
text/function decision, as one 22.05 kHz audio frame for every complete 80 ms
input frame. The upstream architecture reference is
[SALM-Duplex](https://arxiv.org/abs/2505.15670).

## Requirements

- Apple Silicon with macOS 15 or newer.
- The `VoiceChat` package product.
- A complete VoiceChat MLX bundle with `encoder/`, `llm/`, and `tts/`.
- Enough unified memory for the selected export (about 12.11 GB on disk for
  INT8 or 8.56 GB for INT5, plus runtime allocations). The optimized controlled
  INT5 run peaked at about 8.70 GB RSS and 15.64–15.66 GB macOS physical
  footprint; the latter includes file-backed MLX mappings that RSS can
  undercount.

An older bundle containing only `encoder/` and `llm/` can still be loaded by
the low-level perception/language APIs, but `VoiceChatModel.load` rejects it
because it cannot produce speech.

## Load the model

The default Hub snapshot is the complete INT8 export:

```swift
let model = try await VoiceChatModel.loadFromHub()
let session = try await model.startSession()
```

Use `aufklarer/VoiceChat-11B-Perception-MLX-int5` explicitly for the smaller
protected-head INT5 export. The historical `Perception` name is retained for
URL compatibility; current snapshots contain the complete pipeline.

To load an already-downloaded bundle:

```swift
import VoiceChat

let root = URL(fileURLWithPath: "/path/to/complete-voicechat-bundle")
let model = try await VoiceChatModel.load(from: root)
let session = try await model.startSession()
```

`VoiceChatModel.loadFromHub(...)` uses the same strict loader after downloading
a snapshot. The repository must contain the full three-directory bundle.

The default system prompt is neutral for timing. To reproduce the model's
greeting behavior explicitly:

```swift
let session = try await model.startSession(
    systemPrompt: VoiceChatSession.greetingSystemPrompt)
```

Prompt wording is part of turn-taking behavior. Do not add “wait for the user
to finish” to a system prompt used for onset measurement.

## Stream microphone audio

Feed mono Float32 samples already resampled to 16 kHz. Partial frames stay
buffered until 1,280 samples are available.

```swift
for try await microphoneChunk in microphoneChunks {
    let events = try await session.pushAudio(microphoneChunk)
    for event in events {
        // One 80 ms, 22.05 kHz mono frame (1,764 samples).
        playback.enqueue(
            event.audio,
            sampleRate: VoiceChatSession.outputSampleRate)

        if event.speaking {
            print(event.text, terminator: "")
        }
    }
}
```

Every `VoiceChatFrameEvent` includes:

- the text and function token selected for that frame;
- whether the text channel is speaking;
- its position on the model's 80 ms conversation clock;
- perception, language-decision, and speech-synthesis wall latency; and
- the matching 1,764 output samples.

For finite files, push a silent tail so an answer that begins near EOF can
finish. This helper accepts at most 60 seconds per call; feed live silence as
ordinary bounded audio chunks instead.

```swift
let finalEvents = try await session.pushSilence(seconds: 6)
for event in finalEvents {
    playback.enqueue(event.audio, sampleRate: VoiceChatSession.outputSampleRate)
}
```

## Read text, timing, and exact audio

```swift
let transcript = await session.userTranscript()
let response = await session.reply()
let metrics = await session.summary()
let exactWaveform = await session.renderedAudio()

print("input: \(transcript)")
print("response: \(response)")
print("perception p95: \(metrics.perceptionP95Milliseconds) ms")
print("decision p95: \(metrics.decisionP95Milliseconds) ms")
print("synthesis p95: \(metrics.synthesisP95Milliseconds) ms")
print("total p95: \(metrics.totalP95Milliseconds) ms")
print("real time: \(metrics.realTime)")
```

Per-event audio uses a bounded causal codec window for immediate playback.
`renderedAudio()` decodes the entire generated code sequence and is the exact
artifact to save or compare offline. Both return the checkpoint's native Float
PCM amplitude without peak or loudness normalization; apply playback gain in
the application if a device's output is too quiet.

`turnTakingLatencyMilliseconds(userStoppedAtMilliseconds:)` measures the gap
on the model timeline between a known end-of-user-speech time and the first
later speaking frame. It is not the same as wall compute latency. Only call it
with a defensible speech-end timestamp; a forced turn is a test control, not a
natural latency measurement.

## Sampling

Greedy text generation is the default. Speech uses the checkpoint's eight-step
MaskGIT schedule by default.

```swift
let session = try await model.startSession(
    sampling: .init(
        temperature: 0,
        topP: 1,
        repetitionPenalty: 1,
        presencePenalty: 0),
    speech: .init(
        guidance: 0.2,
        topP: 0.95,
        noise: 0.001,
        iterations: 8))
```

Changing speech iterations changes the codebook assignment schedule and is
not checkpoint parity mode.

## Command-line E2E run

Build the benchmark product and its Metal library:

```bash
swift build -c release --product voicechat-bench --disable-sandbox
./scripts/build_mlx_metallib.sh release
```

Run a real file through the complete path and save the spoken response:

```bash
./.build/release/voicechat-bench \
  --model /path/to/complete-bundle \
  --e2e-audio question.wav \
  --tail-seconds 6 \
  --response-output response.wav
```

The command prints the input transcript, text response, model onset position,
first spoken text-token and playable-audio compute latency, stage and total
p50/p95, peak RSS and physical footprint, wall-time RTF, and whether the p95
complete-frame path fits within one 80 ms model frame.

Wall timing starts after model loading, system-prompt prefill, and EAR-TTS
speaker-prompt warmup. Startup latency and bundle download time are therefore
reported separately from per-frame inference.

For reference, release builds on an M5 Pro (48 GB), using the controlled
120-frame fixture and the default one-frame/80 ms live chunks, completed three
independent protected-head INT5 runs in 9.039 s, 8.833 s, and 8.786 s for a
9.600 s model timeline. Their whole-pipeline RTF values were `0.94`, `0.92`,
and `0.92`. Perception/decision/synthesis p50 values ranged over
9.0–9.2/35.6–36.6/28.3–29.0 ms, and total/frame p50 was 72.9–74.8 ms. Peak RSS
was about 8.70 GB, the physical-footprint peak was 15.64–15.66 GB, and MLX
reported 9.53 GB peak GPU allocation. Across the three runs, first spoken text
arrived in 44.3–46.7 ms and first playable audio in 74.0–76.4 ms.

Whole-pipeline RTF below `1` establishes sustained throughput, while the
session's `realTime` flag deliberately requires the stricter total/frame p95
to remain below `80 ms`. All three runs met that deadline at 76.2–78.6 ms,
although the remaining per-frame headroom is narrow. Measure on the deployment
machine and avoid concurrent GPU work when comparing results. Current INT8
correctness is verified, but its post-optimization release performance has not
yet been remeasured.

The first-token/audio values printed by the command are hot compute latency
after the controlled turn opens, not learned turn-taking latency.
`--chunk-frames` can test larger input batches, but those are not the default
80 ms live cadence.

`--force-turn-at-end` injects BOS at the end of the input. It exists for
controlled regression tests; do not use its onset as a natural turn-taking
measurement.

Encoder-only and language-parity modes remain available:

```bash
./.build/release/voicechat-bench --model /path/to/bundle --durations 1 5 15 30

./.build/release/voicechat-bench \
  --model /path/to/bundle --llm \
  --parity-input in.safetensors --parity-output out.safetensors
```

## Concurrency

Each `VoiceChatSession` owns its language, EAR-TTS, and character-embedding
caches. Sessions can therefore share one loaded `VoiceChatModel`; applications
still need to budget GPU scheduling and per-session cache memory when advancing
several conversations concurrently.
