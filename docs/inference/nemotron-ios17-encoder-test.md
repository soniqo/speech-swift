# Nemotron iOS 17 encoder export test

This experiment tests whether converting the Nemotron encoder for the iOS 17
operation set avoids the ANE specialization failure reported in
[speech-swift issue 503](https://github.com/soniqo/speech-swift/issues/503).
It preserves the 320 ms input/output interface and all 296 published encoder
palettes. The decoder, joint network, configuration, vocabulary, language map,
and tokenizer come from the published revision
`447095fe87b480b5e6a15367f135303d479de8ac` without changes.

Only the encoder targets iOS 17. The unchanged decoder and joint still target
iOS 18; the bundle and Swift SDK require iOS 18 or later. This is an export
experiment for current devices, not an iOS 17 SDK release.

## Download both bundles

Install `huggingface_hub` and download the published revision and the candidate
into separate directories:

```python
from huggingface_hub import snapshot_download

published = snapshot_download(
    "aufklarer/Nemotron-3.5-ASR-Streaming-0.6B-CoreML-INT8",
    revision="447095fe87b480b5e6a15367f135303d479de8ac",
    local_dir="nemotron-published",
)
candidate = snapshot_download(
    "aufklarer/Nemotron-3.5-ASR-Streaming-0.6B-CoreML-INT8-iOS17-Encoder-Test",
    revision="606bc51c5cb0232aee5a136a93bb0bb6587efcb5",
    local_dir="nemotron-ios17-encoder-test",
)
```

Copy each complete directory into the test app's accessible storage, retaining
the three `.mlmodelc` directories and runtime files together. Record the
candidate revision and retain `experiment.json` with the device results.

## Compare on the affected iPhone

Use a separate directory for the candidate bundle. Point the existing Core ML
loading reproducer at that directory; it accepts the same model names and
features as the published bundle. No Swift runtime changes are required.

Run one configuration per app process:

| Bundle | Compute units | Purpose |
|---|---|---|
| Published | `.cpuAndNeuralEngine` | Reproduce the reported failure |
| Candidate | `.cpuAndNeuralEngine` | Test the changed encoder operation set |
| Candidate | `.cpuOnly` | Check the CPU control path |

For each row, terminate the app and repeat the same row in a new process.
Record the exact OS build, encoder load time, and Core ML/ANE compiler errors.
The first observed load can use an existing system cache. Compare it with the
second process rather than assuming every first-in-process load is cold.

## Use the included load probe

Add `testing/LoadProbe.swift` to an iOS test target and invoke the following
once per launch, with `bundleURL` pointing at either bundle directory:

```swift
import CoreML
import Foundation

let report = try await NemotronExportProbe.run(
    bundleURL: bundleURL,
    computeUnits: .cpuAndNeuralEngine)
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
let json = try encoder.encode(report)
print(String(decoding: json, as: UTF8.self))
```

The probe records per-model load and reload times, synthetic-input prediction
times, feature shapes/types, and compute-plan preferences. It loads each model
sequentially. Compute-plan preferences describe planned placement; they do not
prove that the ANE executed a prediction. Preserve the console logs alongside
the JSON report.

For a macOS cross-check, compile and run the same source:

```bash
swiftc -O -parse-as-library -D LOAD_PROBE_CLI testing/LoadProbe.swift -o /tmp/nemotron-load-probe
/tmp/nemotron-load-probe /path/to/published ane published-ane.json
/tmp/nemotron-load-probe /path/to/candidate ane candidate-ane.json
/tmp/nemotron-load-probe /path/to/candidate cpu candidate-cpu.json
```

## Check transcription and streaming caches

Synthetic inputs establish load and encoder timing only. Run the included
16 kHz `testing/test_audio.wav` and representative real speech through both
bundles. The reference clip says:

> Can you guarantee that the replacement part will be shipped tomorrow

Use the same language, audio, and decoding settings for each comparison. Check
batch output, streaming final text, and word boosting. Include silence, a
partial final chunk, and enough chunks to fill the 56-frame attention cache.
Record actual-speech real-time factor and chunk latency after one warm-up.

With speech-swift v0.0.27 or later, select the candidate without changing the
loader default:

```swift
import CoreML
import NemotronStreamingASR

let model = try await NemotronStreamingASRModel.fromLocal(
    bundleDir: candidateBundleURL,
    computeUnits: .cpuAndNeuralEngine)
```

The bare Core ML probe also works with the reporter's vendored v0.0.25 code.
Run each bundle in a separate process to avoid retaining both encoders.

For a speech-swift real-speech check, load the included WAV after selecting
the bundle and record the final text and elapsed time:

```swift
import AudioCommon

let audio = try AudioFileLoader.load(url: fixtureURL, targetSampleRate: 16_000)
_ = try model.transcribeAudio(audio, sampleRate: 16_000, language: "en-US")
let started = ProcessInfo.processInfo.systemUptime
let text = try model.transcribeAudio(audio, sampleRate: 16_000, language: "en-US")
let elapsed = ProcessInfo.processInfo.systemUptime - started
print("Batch:", text, "seconds:", elapsed, "RTF:", elapsed / (Double(audio.count) / 16_000))
for await partial in model.transcribeStream(audio: audio, sampleRate: 16_000, language: "en-US") {
    if partial.isFinal { print("Streaming:", partial.text) }
}
```

The included fixture is 20 seconds long, resampled from the package's test
fixture to mono 16 kHz. Its source and output hashes are in
`testing/fixture-provenance.json`.

## Local validation

On M5 Pro, macOS 26.6.2 (25G83), the candidate loaded with CPU+ANE in two
separate processes without the reported compiler failure. Its input/output
names, shapes, and types match the published bundle.

All six encoder outputs matched the published CPU encoder exactly across
67 calls, covering the included speech, silence, partial chunks, and a full
56-frame attention cache. Four SDK tests passed in each of the published
CPU+ANE, candidate CPU+ANE, and candidate CPU-only configurations: batch,
streaming, phrase boosting, and a check that boosting changes decoder
decisions. The final batch and streaming transcripts matched in all three
configurations. Detailed results are in `testing/mac-validation.json`.

## Report the result

Share the JSON reports and the Core ML/ANE error window for the published
and candidate CPU+ANE cases. Include the second-process load time and
real-speech transcripts/timings. Success requires the candidate to avoid the
repeated compiler failure and preserve transcription and streaming behavior.
Keep the production bundle until the affected-device comparison passes.
