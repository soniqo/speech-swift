# Acoustic Echo Cancellation — LocalVQE v1.4-AEC

## What problem this solves

Acoustic echo is playback from the Mac speakers leaking back into the
microphone. The microphone signal and the playback signal are different inputs:

```text
microphone: local voice + room noise + leaked speaker playback
reference:  the clean audio sent to the speakers
```

`LocalVQEEchoCanceller` uses the reference to remove only the predictable
playback component from the microphone. It does not mix the streams, and it is
not a general noise suppressor.

```text
microphone ───────────────┐
                         ├─ delay estimate ─ adaptive filter ─ residual mask ─ clean mic
playback reference ───────┘                     CPU              Core ML
```

The complete chain is required. The Core ML model is a learned residual mask,
not a standalone waveform model; feeding raw microphone/system spectra into it
without the adaptive front end is unsupported.

## Streaming API

Both streams must describe the same capture interval. Keep them separate from
capture through cancellation:

```swift
import SpeechEnhancement

let aec = try await LocalVQEEchoCanceller.fromPretrained()

// Each input is exactly 256 mono Float32 samples at 16 kHz (16 ms).
let cleanMicrophone = try aec.processFrame(
    microphone: microphoneFrame,
    reference: playbackReferenceFrame
)
```

Create one instance per recording and serialize calls. Call `reset()` when a
recording ends, when either input drops samples, or when capture is restarted.
Each `processFrame` call drains its temporary Core ML objects before returning,
so a dedicated capture thread does not need to provide its own autorelease
pool for long-running streams.
Resetting clears:

- online delay estimation;
- adaptive-filter and controller memory;
- STFT history and overlap-add buffers;
- Core ML temporal state.

The spectral stage adds one 256-sample hop, or 16 ms, of algorithmic latency.
The online delay estimator initially reports zero and locks only after it has
enough correlated playback/microphone evidence. `currentDelaySamples` and
`delayConfidence` expose its current decision for diagnostics.

## Complete clips

For a synchronized file pair, the batch helper can estimate bulk delay from the
whole clip before processing. This avoids the online acquisition period:

```swift
let clean16k = try aec.process(
    microphone: microphoneSamples,
    reference: playbackSamples,
    sampleRate: 48_000,
    primeDelay: true
)
```

Both inputs are resampled identically to 16 kHz. The returned audio is mono
Float32 at 16 kHz and has the resampled microphone's sample count. Unequal
input lengths fail explicitly because silently padding one live stream would
break synchronization.

## Capture requirements

Robust cancellation depends more on the reference contract than on a threshold:

- Use the exact signal sent to the physical output, before acoustic leakage.
- Timestamp microphone and reference frames from the same clock when possible.
- Do not use a system-audio transcription mix as the reference if it omits,
  adds, or time-shifts audible playback.
- Preserve reference samples during double-talk; local speech and playback can
  be active simultaneously.
- Reset after device changes, discontinuities, or dropped frames rather than
  carrying stale filter state forward.

Apple Voice Processing I/O can still be used when it works for the chosen audio
graph. LocalVQE is the explicit-reference path for capture arrangements where
Apple's processing is unavailable or does not cancel the actual system output.
Do not cascade both blindly: first verify whether the microphone is already
echo-cancelled, since two adaptive cancellers can damage near-end speech.

## Model and host stages

The default model is
`aufklarer/LocalVQE-v1.4-AEC-200K-CoreML` and contains:

| Artifact | Role |
|---|---|
| `LocalVQEAECResidualMask.mlmodelc` | Stateful FP16 residual mask, 200,199 parameters |
| `LocalVQEAECFrontend.npz` | 2,742 adaptive-controller parameters and the exact analysis window |

The host implementation follows LocalVQE v1.4's gated GCC-PHAT delay estimate,
partitioned-block frequency-domain Kalman filter, and learned v2xp controller.
The neural stage uses one persistent Core ML `MLState` per recording.

Export measurements on an M5 Pro put the host adaptive stage at 0.236 ms mean
per 16 ms frame and the Core ML mask at 0.177 ms. Their conservative summed
estimate is 0.413 ms (RTF 0.0258). These were separate-stage measurements;
the native release Swift path processed a four-second synchronized pair in a
median 0.108 seconds (RTF 0.0270, 37.1× real-time throughput) across three
runs. The direct XCTest process peaked at 86.7 MiB RSS, 27.8 MiB above the
empty-test baseline. The timing covers delay priming, the C++ adaptive filter,
spectral codec, and stateful Core ML mask, but excludes model download and
initial load.
These measurements are performance checks, not a claim about real-room echo
quality; the E2E test separately enforces finite output and at least 6 dB of
late-window attenuation on a deterministic synthetic echo path.

## Failure behavior

The runtime fails closed when the compiled model or frontend archive is absent,
when a frame has the wrong size, when streams are unequal, or when samples are
non-finite. It never falls back to subtracting the system mix or muting the
microphone. A silent reference leaves microphone-only speech on the microphone
path, delayed only by the documented spectral hop.

## References

- [LocalVQE source](https://github.com/localai-org/LocalVQE)
- [Core ML model](https://huggingface.co/aufklarer/LocalVQE-v1.4-AEC-200K-CoreML)
- [Export and benchmark source](https://github.com/soniqo/speech-models/tree/main/models/localvqe-aec/export)

The upstream implementation and weights are Apache-2.0 licensed.
