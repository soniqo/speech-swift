# System Audio Tap

`SystemAudioTap` (in `AudioCommon`) captures the system output mix — what the
Mac is currently playing — as mono Float32 samples, resampled to a target rate.
It complements `AudioIO`, which captures the microphone; together they cover
both sides of a conversation happening on the machine.

```swift
import AudioCommon

let tap = SystemAudioTap()                 // excludes this process by default
try tap.start(targetSampleRate: 16000) { samples in
    pipeline.pushAudio(samples)            // mono Float32 @ 16 kHz
}
// ...
tap.stop()
```

## How it works

The capture path is the Core Audio process-tap API (macOS 14.4+):

1. A **mono global tap** (`CATapDescription`) taps the mix of all processes,
   minus an exclusion list. The current process is excluded by default so the
   app's own playback — for example a TTS voice — is not re-captured.
2. The tap is wrapped in a **private aggregate device that contains only the
   tap**. No output sub-device is aggregated: including the output device can
   import its input streams (for example a Bluetooth headset microphone) and
   duplicate audio into the capture.
3. An IO proc on the aggregate device delivers the tap buffers. They are
   downmixed to mono if needed, counted for the silence diagnostics, resampled
   with a stateful `AVAudioConverter`, and handed to `onSamples`.

Because a global tap follows processes rather than a device, capture survives
default-output-device switches. The delivered rate can change on such a switch;
property listeners re-read the tap format and the resampler is rebuilt on the
fly, so `onSamples` always receives `targetSampleRate`.

## Permission

The host app must declare `NSAudioCaptureUsageDescription` in its Info.plist.
macOS shows the system prompt on first tap creation, and the grant appears
under Privacy & Security → Screen & System Audio Recording.

**Known trap:** if the permission is denied, tap creation can still succeed and
deliver pure silence. Consumers that must fail closed should watch the
counters: `framesCaptured` growing while `nonSilentFrames` stays 0 is the
denied-permission signature (assuming audio is actually playing).

## Diagnostics

| Property | Meaning |
|---|---|
| `captureState` | `stopped` / `running` / `error(String)` |
| `tapSampleRate` | Rate the tap currently delivers (before resampling) |
| `framesCaptured` | Total frames delivered since `start` |
| `nonSilentFrames` | Frames above the silence threshold since `start` |
| `audioLevel` | RMS (0–1) of the latest buffer, for UI meters |
| `deviceChanges` | Default-output-device switches observed while running |

## Testing

Unit tests (`SystemAudioTapTests`) cover the pure logic: exclusion-list
assembly, the tap-only aggregate composition, mono mixdown, silence counting,
and OSStatus rendering. The hardware path is covered by
`E2ESystemAudioTapTests`, which plays a tone through the default output,
asserts the tap hears it, and then asserts that excluding the current process
silences the capture. The E2E test needs the audio-recording permission and a
quiet system (no other audio playing).
