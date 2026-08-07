# VoiceChat 11B

`VoiceChat` is a native MLX Swift implementation of the complete duplex
speech-to-speech (S2S) path in
`nvidia/NVIDIA-NemotronLabs-VoiceChat-11B`: continuous user speech perception,
the Nemotron-H duplex text/function channels, direct EAR-TTS agent-speech
generation, and the 22.05 kHz neural audio-codec decoder.

The [upstream NVIDIA model card](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B)
cites [SALM-Duplex: Efficient and Direct Duplex Modeling for Speech-to-Speech
Language Model](https://arxiv.org/abs/2505.15670) (Interspeech 2025) as its
primary architecture reference. This documentation uses “speech-to-speech”
for the model task and “audio-to-audio” only when describing the concrete PCM
input/output API.

Audio arrives as 16 kHz mono PCM. For every complete 80 ms input frame the
session makes a text/function-channel decision, generates 31 codec tokens with
EAR-TTS, and returns a 1,764-sample output-audio frame. The same session can
also return the user's transcript, the model's text response, timing events,
and an exact offline rendering of the generated waveform.

The runtime requires a **complete** bundle containing `encoder/`, `llm/`, and
`tts/`. It deliberately rejects the earlier understanding-only exports rather
than silently starting a session without speech generation.

Published bundles: [INT8](https://huggingface.co/aufklarer/VoiceChat-11B-Perception-MLX-int8)
(default, 12.11 GB of weights) and
[protected-head INT5](https://huggingface.co/aufklarer/VoiceChat-11B-Perception-MLX-int5)
(8.56 GB of weights). Their legacy `Perception` names are retained for URL
compatibility; current snapshots contain all three components.

## Model contract

| Property | Value |
|---|---|
| License | OpenMDW 1.1 (commercial use permitted) |
| Input | mono Float32 PCM at 16 kHz |
| Conversation clock | one frame every 80 ms (1,280 input samples) |
| Encoder | 24-layer causal FastConformer, 128 log-mel bins, 10 ms hop |
| Language backbone | 56 layers: 27 Mamba2, 25 MLP, 4 attention |
| Language width / vocabulary | 4,480 / 131,072 |
| Duplex channels | user audio ×1 + previous text ×1 + previous function ×2 |
| Speech decoder | 28-layer, 1,152-wide text-conditioned EAR-TTS |
| Speech generation | MaskGIT, 8 progressive unmasking iterations |
| Codec representation | 31 RVQ codes per 80 ms frame |
| Output | mono Float32 PCM at 22.05 kHz, 1,764 samples per frame |
| MLX variants | FP16, affine INT8, protected-head affine INT5 |

## Architecture

### Perception and duplex language model

The perception path applies the checkpoint's own centred-STFT frontend, a
24-layer FastConformer at 1,024 hidden dimensions, and an `IdentityConnector`
whose single 1,024 → 4,480 linear projection places audio in the language
model's embedding space. The encoder attends to 70 frames on the left and none
on the right. Streaming uses bounded recomputation over that attention window,
the causal-convolution receptive field, and a small safety margin.

Each language-model position is the sum of three channels:

```text
fused[t] = userAudio[t] + textEmbedding[t - 1]
          + 2 * functionEmbedding[t - 1]
```

The function weight of `2` comes from the published checkpoint configuration;
using `1` changes the generated conversation despite producing valid tensors.
VoiceChat also overrides the tokenizer metadata at runtime: `<SPECIAL_12>` is
the idle text-channel token, `<s>` opens a spoken turn, and `</s>` closes it.
The loader resolves and round-trips all three strings instead of trusting the
tokenizer's advertised EOS role.

Only 4 of the Nemotron-H backbone's 56 layers use attention. The other 27
mixing layers are Mamba2 and retain fixed recurrent state, so only four KV
caches grow with conversation length.

### EAR-TTS speech decoder

EAR-TTS is a standalone **text-conditioned** model. It does not consume the
4,480-wide language hidden state, and no 4,480 → 1,152 bridge exists. The
checkpoint's 37-frame Aria prompt primes a separate 28-layer Gemma-style
backbone. Each text-channel token then conditions one 80 ms speech frame.

The output head is a 1,024-component low-rank mixture of Gaussians. Generation
is not one-shot: the 31 codebooks are progressively assigned over eight
MaskGIT iterations. For the published schedule, the assignments per iteration
are `[0, 0, 0, 1, 1, 3, 4, 22]`.

### Neural audio codec

The codec sums one 512-dimensional entry from each of 31 residual codebooks,
then upsamples by 9 × 7 × 7. Its final 18 channels are nine magnitudes and nine
phases for a 16-point ISTFT. Treating them as real and imaginary components
produces loud noise while preserving the expected waveform shape, which is why
the loader immediately decodes the checkpoint's canonical silence frame and
requires RMS below `1e-5`.

The implementation also preserves causal left-only ConvNeXt padding, a
periodic Hann window, real-valued DC and Nyquist bins, and the six-sample ISTFT
trim. Codec weights remain dense fp16 in quantized bundles and are promoted to
fp32 for decoding, matching the verified Python path.

## Quiet failure modes kept under test

These errors load successfully and return plausible tensor shapes:

1. Adding biases that the FastConformer checkpoint does not contain.
2. Treating the module named `batch_norm` as BatchNorm instead of LayerNorm.
3. Producing 16 post-subsampling frequency bins instead of the causal path's 17.
4. Omitting the `[70, 0]` attention mask and letting audio attend to the future.
5. Calling Nemotron-H without a cache and accidentally making attention
   bidirectional.
6. Taking PAD/EOS roles from tokenizer metadata instead of VoiceChat's runtime
   overrides.
7. Giving the function feedback channel weight `1` instead of `2`.
8. Running EAR-TTS in one pass instead of its eight-step MaskGIT schedule.
9. Interpreting codec magnitude/phase channels as real/imaginary channels.

## Verification

Swift is compared with independent Python MLX and PyTorch paths on identical
inputs, rather than only against itself:

| Component | Result |
|---|---|
| Encoder | cosine 0.99999934; relative error 0.110% |
| Language backbone | cosine 0.99999833; relative error 0.179% |
| EAR-TTS INT8 | all 1,364 generated codec IDs exactly equal |
| EAR-TTS waveform | cosine 0.99999915; RMSE 1.96e-5 |
| Canonical codec silence | RMS below 1e-5; the historical decoder bug produced 1.60 |
| Full audio → audio test | exact expected response, 50 speaking frames, finite non-silent audio |
| Live-window codec vs exact render | cosine above 0.9999; RMSE below 2e-4 |

The complete controlled E2E test uses a real 3.6-second FLEURS clip and forces
turn opening at the end of the clip so output is deterministic. That forced
onset is a correctness fixture, **not** a natural turn-taking latency result.

## Quantization and bundle sizes

Measured on the text channel against the FP16 understanding bundle:

| Variant | Complete size | Text top-1 agreement | Text KL (nats) |
|---|---:|---:|---:|
| FP16 | 22.19 GB | — | — |
| INT8 | 12.11 GB | 100.00% | 0.00018 |
| INT5 | 8.56 GB | 92.55% | 0.01213 |

The INT5 export holds `lm_head`, `function_head`, and the sensitive speech
output heads at 8 bits. INT8 is the parity/reference variant; INT5 trades some
text-token agreement for a substantially smaller bundle.

## Runtime status and latency

The current implementation is functionally streaming but does not yet sustain
the model's 80 ms frame clock on the tested M5 Pro. `VoiceChatSessionSummary`
reports perception, language-decision, speech-synthesis, and total per-frame
latency separately. `realTime` is true only when the p95 of the complete
per-frame path is below 80 ms; omitting encoder cost can incorrectly classify
the smaller variant as real-time.

Release builds on an M5 Pro (48 GB), over the 120-frame controlled E2E fixture
with one 80 ms frame per live input push, measured:

| Variant | Peak RSS | First spoken text token | First playable audio | Total/frame p50 / p95 | Whole-pipeline RTF |
|---|---:|---:|---:|---:|---:|
| INT8 | 12.21 GB | 68.8 ms | 105.1 ms | 104.6 / 114.0 ms | 1.34 |
| INT5 | 8.73 GB | 57.5 ms | 91.4 ms | 93.0 / 104.7 ms | 1.17 |

The corresponding macOS physical-footprint peaks were 24.42 GB for INT8 and
20.94 GB for INT5; physical footprint includes file-backed MLX mappings that
RSS can undercount. Both variants remain outside the 80 ms frame budget.
Sustained real-time operation also requires whole-pipeline RTF < 1; INT8 at
1.34 and INT5 at 1.17 do not yet meet that threshold. Stateful FastConformer
caching, replacing bounded encoder recomputation, is the next optimization
target.

For INT8, perception/decision/synthesis p50/p95 were 23.4/28.8,
47.0/50.8, and 33.1/36.9 ms. For INT5 they were 22.0/30.3, 36.3/39.5,
and 33.4/36.1 ms. The whole 9.60 s model timeline took 12.86 s for INT8
and 11.20 s for INT5.

Model turn onset and hardware compute latency are different measurements. The
default prompt is intentionally neutral because adding “greet the user” or
"wait for the user to finish" changes the model's chosen onset. Do not cite a
turn-taking number from a forced-BOS run or from a prompt containing either
instruction. The first-token and first-audio values above are hot compute time
for the first spoken response frame after the controlled turn is opened; they
are not learned turn-taking latency.

## Usage

See [VoiceChat inference](../inference/voicechat.md) for streaming Swift and
benchmark examples. The high-level entry points are:

```swift
import VoiceChat

let model = try await VoiceChatModel.load(
    from: URL(fileURLWithPath: "/path/to/complete-bundle"))
let session = try await model.startSession()

for event in try await session.pushAudio(mono16kSamples) {
    playback.enqueue(event.audio, sampleRate: VoiceChatSession.outputSampleRate)
}

// Supply a tail when processing a finite file so a late answer can finish.
_ = try await session.pushSilence(seconds: 6)
print(await session.userTranscript())
print(await session.reply())
```

Language and EAR-TTS generation state belongs to the returned session, so
several conversations can share one loaded `VoiceChatModel`. Concurrent callers
still contend for the same GPU and allocate their own per-session caches.

## Tests

```bash
swift test --filter VoiceChat --disable-sandbox

swift build --build-tests --disable-sandbox
./scripts/build_mlx_metallib.sh debug

E2E_ONLY_FILTER=E2EVoiceChatSpeechGenerationTests E2E_SKIP_UNIT=1 \
  VOICECHAT_BUNDLE=/path/to/complete-bundle scripts/test_e2e_isolated.sh

E2E_ONLY_FILTER=E2EVoiceChatConversationTests E2E_SKIP_UNIT=1 \
  VOICECHAT_BUNDLE=/path/to/complete-bundle scripts/test_e2e_isolated.sh
```

Model-backed tests skip when `VOICECHAT_BUNDLE` is unset. The isolated runner
keeps the multi-gigabyte speech and complete-conversation suites in separate
processes so their peak memory cannot accumulate.
