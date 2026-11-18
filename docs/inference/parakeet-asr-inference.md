# Parakeet TDT ASR Inference Pipeline

## Overview

```
Audio (16kHz) → Mel (vDSP) → CoreML Encoder (Neural Engine) → TDT Greedy Decode → Text
```

Parakeet TDT is a transducer ASR model running entirely on CoreML. The encoder runs on the Neural Engine, freeing the GPU for concurrent MLX workloads (e.g., TTS generation or LLM inference). This makes it well-suited for voice pipeline scenarios where ASR and TTS run in parallel.

## Pipeline Stages

### 1. Audio Preprocessing

Raw 16kHz audio is converted to 128-bin log mel features using vDSP (Accelerate framework): pre-emphasis, a symmetric 400-sample Hann window centered in a 512-point STFT, a 128-bin Slaney mel filterbank, logarithm, and per-feature normalization.

The real forward FFT in vDSP returns twice the mathematical Fourier coefficients. The power spectrum is multiplied by `0.25` before the mel filterbank. The default log features use `log(mel + 2^-26)`: the additive guard is scaled with power to preserve the package's existing power-to-floor ratio. In exact arithmetic, `log(4E + epsilon) = log(4) + log(E + epsilon/4)`; per-feature centering removes the constant offset. Float32 rounding can still produce small differences.

This default preserves legacy feature calibration rather than claiming exact NeMo feature parity. The internal NeMo reference calibration uses the same corrected FFT power with a `2^-24` guard. Paired scaling itself does not require another encoder pass or automatic waveform gain.

Audio longer than the loaded encoder's window is split in PCM sample space before preprocessing. Each primary window receives its own pre-emphasis, centered STFT with zero padding, log mel features, and per-feature normalization over valid frames. This keeps a quiet window independent of louder audio elsewhere in the recording. Constant windows and single-frame tails normalize to zero. The extra STFT padding frame is removed before selecting the encoder shape, so it cannot displace the first audio frame of a full window.

Encoders limited to five seconds retain the original cuts and add context windows halfway between them. The runtime matches words near the same absolute time and keeps one copy of an overlapping phrase. Original windows are preferred away from the cut; context can fill a missing passage or complete a word at the join. An empty hypothesis does not erase the previous words or advance their coverage. A successful window reaching the end of the recording is not replaced by a shorter suffix. Inputs within one encoder window and encoders with longer context retain consecutive-window decoding.

A matching word later in the overlap does not establish that the earlier phrases agree. Before accepting a join, the runtime checks the competing phrases and protects a more confident original phrase from being discarded. Stronger context can still repair an uncertain original phrase. Fallback cuts also respect the span of a retained original word, avoiding a second hypothesis for that word on the other side. These decisions use the model's word confidence as a local heuristic, not a guarantee of correct recognition; a five-second encoder can still misrecognize speech even when the join is correct.

An empty original short-encoder window can retry using a wider feature context. Its statistics and centered STFT are computed from up to 30 seconds of audio, and the requested frames are copied into the same short encoder shape. The feature cache is bounded and reused across retries. Zero PCM and tails shorter than the maximum predicted token span do not trigger this retry. Primary and context predictions run sequentially, with CoreML temporaries released after each window. This adds encoder/decoder passes; no automatic input gain or predictor-state carry is applied.

CPU-only tests compare the default with immutable pre-correction Swift features at normal and quiet amplitudes. Separate tests compare the explicit NeMo calibration with its reference features. Missing fixtures fail the tests. These checks run in CI without model downloads:

```bash
swift test --no-parallel --disable-sandbox --filter 'ParakeetASRTests.(MelContractTests|MelCalibrationTests)'
```

### 2. CoreML Encoder

The FastConformer encoder runs on the Apple Neural Engine via CoreML. The encoder produces per-frame audio representations. The joint network combines these with the predictor state to produce token and duration logits.

- **First inference**: ~3s due to CoreML model compilation and ANE scheduling. The compiled model is cached by CoreML for subsequent runs.
- **Subsequent inference**: ~0.6s for 20s of audio (RTF ~0.03).

### 3. TDT Greedy Decode

Token-and-Duration Transducer (TDT) decoding interprets the encoder output. Unlike standard CTC which emits one token per frame, TDT predicts both the token and a duration (number of frames to skip). This reduces redundant computation and improves accuracy on long-form audio.

Greedy decoding selects the highest-probability token and duration at each step. A non-blank token with duration zero stays on the current encoder frame, allowing several tokens to share it. Blank tokens advance by at least one frame and otherwise follow their predicted duration. Ten non-blank emissions on the same frame force advancement to bound decoding. Predictor state is initialized for each audio window. No beam search or language model rescoring is applied.

The vocabulary classifies angle-bracket entries as model-control tokens and omits them from the
returned transcript without suppressing them during decoding. Ordinary tokens are not classified
by ID range, so model-emitted digits `0`–`9` are preserved even though their IDs sit between
control-token ranges. Hidden control tokens still update the recurrent decoder state.

### 4. Confidence Score

The pipeline returns an overall confidence score (0.0 to 1.0) computed as the exponential of the mean log probability of retained text tokens. Duplicate overlap hypotheses do not contribute to the score. This provides a rough estimate of transcription reliability — useful for deciding whether to retry with a different model or prompt the user.

### 5. Word Timestamps

`transcribeWithLanguage` returns `TranscriptionResult.words`. Each `WordConfidence` carries `startTime` and `endTime` in seconds from the start of the input audio, taken from the encoder frame at which the decoder emitted the word's tokens. One encoder frame is 80 ms (hop 160 × subsampling 8 at 16 kHz). Audio longer than the encoder window is decoded in windows, and the times are converted to absolute seconds through mel frames, so they stay exact even when the window length is not a multiple of the subsampling factor (the 5 s encoder is 62.5 encoder frames).

The times are emission-aligned, not forced-aligned. The offline encoder sees the whole window, so a word can be stamped before its sound begins: on the bundled test clip the first word of an utterance starts about 0.3 s before the acoustic onset and the last word ends about 0.2 s after the acoustic end. `endTime` uses the last text token's predicted TDT duration, including zero; attached punctuation keeps the preceding text token's end. Several words can begin on the same encoder frame, so duration-zero words can have zero-length spans. For merged short-encoder output, word starts are kept monotonic and all times are clamped to the input audio duration. Use `Qwen3ForcedAligner` when you need acoustic word boundaries.

The public `ParakeetVocabulary.decodeWords` method retains its existing `frameDuration` parameter and uniform-span behavior for caller-provided token streams. Model inference uses predicted token durations internally.

## Language Support

Parakeet TDT supports 25 European languages via its SentencePiece vocabulary. The model handles multilingual audio without explicit language selection.

## Thread Safety

Model instances retain mutable result and language-steering state. Create separate instances for concurrent transcription rather than sharing one instance across threads or async tasks.

## Warmup

For latency-sensitive applications (voice pipelines), call a warmup inference with a short silent audio buffer during app initialization. This triggers CoreML compilation on the Neural Engine so that the first real inference runs at full speed:

```swift
let model = try await ParakeetASRModel.fromPretrained()
try model.warmUp()
let text = try model.transcribeAudio(samples, sampleRate: 16000)
```

## Encoder Variant Selection (multi-encoder repo)

`aufklarer/Parakeet-TDT-v3-CoreML-INT8` ships multiple single-shape encoders in one repo (`encoder.mlmodelc` = 30s, `encoder_5s.mlmodelc`, `encoder_15s.mlmodelc`). Pick the variant matching your typical input length to cut per-call ANE bandwidth:

```swift
// Short voice-pipeline chunks (≤5s): ~6× less weight bandwidth per call than the 30s encoder
let model = try await ParakeetASRModel.fromPretrained(
    modelId: "aufklarer/Parakeet-TDT-v3-CoreML-INT8",
    encoderVariant: "5s")
```

`encoderVariant: nil` (default) loads `encoder.mlmodelc`, so the single-shape `-30s` and `-iOS-5s` repos work unchanged.

## Compute Units

| Platform | Default | Fallback |
|----------|---------|----------|
| iOS Simulator | `.cpuOnly` | — (only path that returns non-zero tokens; the simulator's Espresso lacks an MPSGraph backend) |
| iOS device | `.cpuAndNeuralEngine` | `.cpuAndGPU` |
| macOS | `.cpuAndNeuralEngine` | `.cpuAndGPU` |

macOS was historically pinned to `.cpuAndGPU` because the iOS17-target **EnumeratedShapes** INT8 encoder SIGSEGV'd in `bnns::GraphCompile` on Mac ANE (surfaced on M5, see issue #313). Single-shape builds aren't affected: the shipped `-30s` and `-iOS-5s` repos (iOS17 single-shape) ANE-compile cleanly on M5; the legacy multi-encoder `-INT8` repo was rebuilt with iOS18 single-shape variants since EnumeratedShapes can't be re-emitted on iOS18 either (MIL validator rejects the dynamic `tile` reps). So macOS now defaults to ANE and falls back to GPU only if loading throws.

## CLI

```bash
# Transcribe with Parakeet
.build/release/speech transcribe --engine parakeet recording.wav

# Default engine (Qwen3-ASR)
.build/release/speech transcribe recording.wav
```

## Performance (M5 Pro, macOS 26.5)

Encoder-only forward, warm best of 5, swift JIT process:

| Shape | Compute | Warm encode | Encoder RTF | Peak RSS |
|-------|---------|------|-----|----------|
| 3000 (30s) | ANE | 74 ms | 0.0025 | 820 MB |
| 1500 (15s) | ANE | 24 ms | 0.0016 | 784 MB |
| 500  (5s)  | ANE |  8 ms | 0.0016 | 765 MB |

End-to-end with TDT decode (release build of `speech transcribe` on the bundled 20s fixture, default `-30s` model, ANE): warm transcribe 0.10 s (RTF 0.005), peak memory footprint 857 MB.

Cold start (first call) includes BNNS graph compile on ANE — 5 s for the 5s variant, ~11 s for 30s. CoreML caches the compile to disk so subsequent process launches skip it.

## Model Architecture Reference

See [docs/models/parakeet-asr.md](../models/parakeet-asr.md) for detailed architecture documentation including FastConformer layer dimensions, TDT vocabulary, and CoreML conversion details.
