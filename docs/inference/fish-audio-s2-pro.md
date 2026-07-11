# Fish Audio S2 Pro — Inference

`FishAudioTTS` is currently an experimental programmatic runtime, not a
complete `speech speak` engine. It can load the published MLX bundle, build
Fish chat prompts, generate DAC codebook rows, and decode those rows to 44.1 kHz
mono PCM. It also supports zero-shot cloning from a raw reference waveform by
encoding the waveform through the Fish DAC encoder before prompt construction.

## Programmatic Use

```swift
import FishAudioTTS

let bundle = URL(fileURLWithPath: "/path/to/Fish-Audio-S2-Pro-MLX-fp16")
let model = try await FishAudioTTSModel.fromBundle(bundle)

let samples = try await model.generate(
    text: "नमस्ते [excited]",
    sampling: FishAudioSamplingConfig(maxNewTokens: 256))
// samples are mono Float PCM at 44.1 kHz.
```

For lower-level tests or custom scheduling, generate and decode separately:

```swift
let codes = try model.generateCodebooks(
    text: "नमस्ते [excited]",
    sampling: FishAudioSamplingConfig(maxNewTokens: 256))

let samples = try model.decode(codes)
```

## Reference Prompts

Reference prompts can be built directly from a raw WAV reference:

```swift
let samples = try await model.generate(
    text: "आज मैं बहुत खुश हूँ। [excited]",
    referenceAudioURL: URL(fileURLWithPath: "/path/to/reference.wav"),
    referenceText: "नमस्ते, यह संदर्भ आवाज है।",
    sampling: FishAudioSamplingConfig(
        maxNewTokens: 256,
        temperature: 1.0,
        topK: 30,
        topP: 0.9,
        minNewTokens: 48))
```

For lower-level integrations, reference prompts also accept already-encoded
Fish DAC codebooks:

```swift
let reference = try FishAudioReferencePrompt(
    text: "नमस्ते, यह संदर्भ आवाज है।",
    codes: referenceCodes) // [10][frames]

let codes = try model.generateCodebooks(
    text: "आज मैं बहुत खुश हूँ। [excited]",
    references: [reference],
    sampling: FishAudioSamplingConfig(maxNewTokens: 256))
```

`FishAudioTTSModel.encodeReferencePrompt(audio:sampleRate:text:)` exposes the
same DAC encode path when the caller already has Float PCM samples in memory.

## Codec

`FishAudioCodec` loads `codec.safetensors` from the same bundle, encodes raw
reference audio to Fish codebooks, and decodes generated Fish codebooks:

```swift
let codec = try FishAudioCodec.load(from: bundle)
let referenceCodes = try codec.encode(audio: referenceSamples, sampleRate: 24_000)
let samples = try codec.decode(codes)
```

The codec expects exactly 10 codebook rows. Codebook 0 is semantic VQ
(`0..<4096`); codebooks 1...9 are residual VQ and are clamped to `0..<1024`,
matching upstream Fish decode.

## Tests

Fast unit tests:

```bash
swift test --filter FishAudioTTSTests
```

The MLX forward/generation shape test runs when `mlx.metallib` is available;
otherwise it skips with an explicit message. Build the metallib with:

```bash
./scripts/build_mlx_metallib.sh debug
```

Local model-loading E2E:

```bash
FISH_AUDIO_E2E=1 swift test \
  --filter E2EFishAudioTTSTests/testLocalBundleLoadsAndGeneratesCodebooks
```

Local codec decode E2E:

```bash
FISH_AUDIO_E2E=1 swift test \
  --filter E2EFishAudioTTSTests/testLocalBundleLoadsCodecAndDecodesOneFrame
```

Local raw-reference clone smoke:

```bash
FISH_AUDIO_E2E=1 swift test \
  --filter E2EFishAudioTTSTests/testLocalBundleVoiceCloningFromHindiReferenceSmoke
```

Local ASR round-trip for Hindi cloning and an `[excited]` marker:

```bash
FISH_AUDIO_ASR_E2E=1 swift test \
  --filter E2EFishAudioTTSTests/testVoiceCloningHindiRoundTripWithASR
```

Local codec reconstruction ASR:

```bash
FISH_AUDIO_ASR_E2E=1 swift test \
  --filter E2EFishAudioTTSTests/testCodecHindiReferenceRoundTripWithASR
```

Set `FISH_AUDIO_BUNDLE=/path/to/Fish-Audio-S2-Pro-MLX-fp16` to override the
default local export path.
