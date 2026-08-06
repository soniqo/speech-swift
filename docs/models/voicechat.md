# VoiceChat 11B

`VoiceChat` is the native MLX Swift implementation of the verified on-device
pieces of `nvidia/NVIDIA-NemotronLabs-VoiceChat-11B`: a streaming
FastConformer encoder, the projection that bridges it into the language model,
the Nemotron-H hybrid backbone, and the decoder half of the 22.05 kHz neural
audio codec.

The EAR-TTS code generator and the full-duplex orchestration are not yet ported
to Swift. The target can understand speech and produce text, and it can decode
externally generated VoiceChat codec codes into waveform audio; it cannot yet
hold a complete spoken conversation by itself.

## Model contract

| Property | Value |
|---|---|
| License | OpenMDW 1.1 (commercial use permitted) |
| Input | mono Float32 PCM at 16 kHz, 128 log-mel bins, 10 ms hop |
| Encoder output | 80 ms frames, projected to the language model's 4,480 hidden dims |
| Codec input | 31 RVQ codes per 80 ms frame |
| Codec output | mono Float32 PCM at 22.05 kHz, exactly 1,764 samples per frame |
| Language backbone | 56 layers — 27 Mamba2, 25 MLP, 4 attention |
| Vocabulary | 131,072 tokens |
| Context | 131,072 tokens; encoder attends 70 frames left, 0 right |
| MLX variants | FP16, affine INT8, affine INT5 |

## Architecture

The encoder is a 24-layer Conformer at 1,024 hidden dimensions with eight
attention heads, 8x causal subsampling, and relative positional attention with
untied per-layer position biases.

The language backbone comes from `mlx-swift-lm`'s `NemotronHModel`. Only 4 of
its 56 layers are attention; the other 27 mixing layers are Mamba2, which keep
a fixed-size recurrent state instead of a growing key/value cache. That is the
reason long conversations stay affordable: the recurrent state is about 71 MB
and never grows, while the four attention layers add roughly 16 KB per frame.

The modality adapter in the source config is an `IdentityConnector`, so a
single `Linear` of 1,024 to 4,480 is the entire bridge between the two halves.

### Codec decoder

The decoder sums one 512-dimensional entry from each of 31 residual codebooks,
then upsamples by 9 x 7 x 7. Its final 18 channels are nine magnitudes and nine
phases for a 16-point ISTFT; treating them as real and imaginary channels
produces loud noise while still returning a correctly shaped waveform.

The implementation preserves the other checkpoint-specific details that fail
quietly: causal left-only ConvNeXt padding, a periodic Hann window, DC and
Nyquist bins constrained to be real, and a six-sample ISTFT trim. Codec weights
are stored dense fp16 in quantized bundles and promoted to fp32 for decoding,
matching the verified Python path.

## Four things a stock Conformer gets wrong

Each of these loads without complaint and computes nonsense, so all four are
asserted in `VoiceChatTests` and recorded in the exported `encoder/config.json`.

1. **No biases.** NeMo has none on the feed-forward, self-attention or
   convolution linears. A stock Conformer builds 264 that do not exist.
2. **LayerNorm, not BatchNorm.** The module named `batch_norm` in the
   checkpoint is a LayerNorm and carries no running statistics.
3. **Seventeen frequency bins, not sixteen.** Causal subsampling pads
   (K-1, S-1) per spatial axis, so `pre_encode.out` is `[1024, 4352]`, and
   4352 is 256 x 17.
4. **Chunked-limited attention, `[70, 0]`.** The quietest of the four. Without
   the mask the encoder attends to future audio, which a duplex model never
   has, and every output changes with no error raised.

A fifth applies to the language backbone: `NemotronHBackbone` derives its
causal mask from the cache and falls back to no mask when there is none, which
silently makes attention bidirectional. `VoiceChatLanguageModel` therefore
always supplies a cache, and `testAttentionIsCausalAcrossSequenceLength` guards
it.

## Verification

Both halves are checked against the Python reference on identical input rather
than against themselves:

| Component | Cosine similarity | Relative error |
|---|---|---|
| Encoder | 0.99999934 | 0.110% |
| Language backbone | 0.99999833 | 0.179% |
| Codec decoder | canonical silence RMS below 1e-5 | model-backed Swift canary |

For scale, the same weights in Python at float32 versus float16 differ by
0.184%, so both are at the precision floor.

## Quantization

Measured on the text channel against the FP16 understanding bundle:

| Variant | Size | Top-1 agreement | KL (nats) |
|---|---|---|---|
| FP16 | 20.2 GB | — | — |
| INT8 | 10.8 GB | 100.00% | 0.00018 |
| INT5 | 7.5 GB | 92.55% | 0.01213 |

INT8 reproduces greedy output token for token. INT5 diverges on roughly one
token in thirteen, which is fine for conversational text but worth measuring on
your own task before relying on it for structured output such as tool-call
arguments, where one divergent token invalidates the result.

The INT5 build holds `lm_head` and `function_head` at 8 bits. That costs
0.44 GB and buys 2.75 points of agreement.

Complete local exports add EAR-TTS and the dense codec payload: 22.19 GB fp16,
12.11 GB int8, and 8.56 GB protected-head int5. The two existing published
`Perception` repositories still carry the older understanding-only payloads;
do not point `VoiceChatCodec` at them until a full-bundle re-export is
separately approved and published.

MLX has affine kernels for 2, 3, 4, 5, 6 and 8-bit weights. There is no INT7
kernel; INT8 is the high-quality variant.

## Usage

```swift
import VoiceChat

let root = URL(fileURLWithPath: "/path/to/VoiceChat-11B-Perception-MLX-int8")

let perception = try VoiceChatPerception.load(from: root.appending(path: "encoder"))
let embeddings = perception(logMel)          // (B, T, 128) -> (B, T/8, 4480)

let llm = try VoiceChatLanguageModel.load(from: root.appending(path: "llm"))
let logits = llm(tokenIds)

// A complete bundle also contains tts/model.safetensors. Codes are
// [batch, frames, 31] and decode to [batch, frames * 1764].
let codec = try VoiceChatCodec.load(from: root)
let waveform = codec.decode(codes: codes)
let silenceCheck = codec.verifySilence()
precondition(silenceCheck.passed)
```

`VoiceChatLanguageModel` keeps the tool-call head aside as `functionHead`. It
is a separate output channel that `NemotronHModel` has no slot for, carried in
the bundle for a future duplex runtime.

## Benchmark

```bash
swift build -c release --product voicechat-bench
./.build/release/voicechat-bench --model /path/to/bundle --durations 1 5 15 30
```

Encoder throughput on Apple Silicon, FP16:

| Audio | Encoded frames | Mean | Real-time factor |
|---|---|---|---|
| 1 s | 14 | 49.8 ms | 0.0498 |
| 5 s | 64 | 131.8 ms | 0.0264 |
| 15 s | 189 | 102.6 ms | 0.0068 |

Peak GPU memory 2.36 GB. Real-time factor is the number that matters for a
duplex model: the encoder runs continuously while the user speaks, so anything
at or above 1.0 cannot keep up with live audio.

`--llm --parity-input in.safetensors --parity-output out.safetensors` runs a
one-shot forward pass instead of timing, for diffing against the Python
reference.

## Tests

```bash
swift test --filter VoiceChatSpeechTests

# Model-backed codec test, one E2E class per process:
swift build --build-tests --disable-sandbox
./scripts/build_mlx_metallib.sh debug
E2E_ONLY_FILTER=E2EVoiceChatSpeechTests E2E_SKIP_UNIT=1 \
  VOICECHAT_BUNDLE=/path/to/full-bundle scripts/test_e2e_isolated.sh
```

Tests that need weights skip when `VOICECHAT_BUNDLE` is unset. The DSP,
configuration, mask, and subsampling tests need no weights and run everywhere.
