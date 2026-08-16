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

For interactive clients, streaming captions or
`turnTaking: .nvidiaRealtime` carry the RNN-T prediction/LSTM state alongside
the duplex session. Each encoder frame is decoded once before its modality
projection. Captioned events contain the complete append-only
`userTranscript`; turn-taking also retains whether the head's first prediction
was blank and whether that frame emitted a recognized lexical token. Both
options share the encoder result and do not recompute mel features or the
FastConformer. The small RNN-T prediction/joint head stays on the shared MLX
inference stream. Moving it to a CPU stream introduces a per-frame device
synchronization that costs more than the head itself and breaks real-time
throughput. The low-level API's `.modelNative` default keeps existing inference
and benchmark behavior unchanged; the live CLI explicitly selects
`.nvidiaRealtime`.

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
on the right. Each streaming layer retains those projected keys/values plus its
eight-frame causal-convolution state, so only the newly subsampled frame crosses
the FastConformer stack. The frontend keeps two 80 ms input frames, which cover
the centred STFT and the three causal stride-2 subsampling stages without
recomputing the encoder's longer history.

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

The greedy text selection and optional function-head selection share the same
language hidden state. The 8-bit function projection is still roughly 518 MB,
so evaluating all 131,072 rows even intermittently makes tool-enabled capture
fall behind. The runtime precomputes only the PAD and `<SPECIAL_20>` output rows
(about 36 KB) and evaluates that probe with the text decision after RNN-T
confirms user speech. If the start row beats PAD before end-of-utterance, the
runtime retains only that 4,096-value hidden-state candidate. The full head
verifies the global argmax when the learned text head emits BOS or the final
RNN-T safety endpoint expires, not while the user is still speaking. This is
exact: `<SPECIAL_20>` cannot be the full-head argmax unless it first beats PAD.
Once the verified native start token wins, the complete head runs at full
language-model speed on a cached perception embedding generated from one second
of zero PCM. It is no longer coupled to incoming 80 ms frames. Because the text
channel is necessarily PAD inside the open call, these steps skip the unused
131,072-row text projection and evaluate only the shared backbone plus native
function projection. `VoiceChatFunctionCallDecodeMetrics` records this
background phase's elapsed wall time, completed asynchronous decode steps,
completion state, derived tokens per second, shared model/projection compute,
idle EAR-TTS cache work, and the residual live-interleave time used by actor
yields, bookkeeping, or system contention. The initial audio-frame start token
is excluded because it is already charged to foreground audio RTF. These
values are separate from foreground audio RTF, which covers the 80 ms
perception/RNN-T path and cannot describe function-projection work.
`VoiceChatFunctionResponseMetrics` independently records the elapsed time,
token count, causal-prefill batch count, throughput, language-cache work,
speech-cache work, and live-interleave residual for replaying a known tool
result. Keeping response replay
separate prevents a fast MCP call or foreground RTF from hiding expensive 11B
cache synchronization.
While the external provider is pending, the two-phase path freezes the shared
language and EAR-TTS timeline at EOTC, matching NVIDIA's reference wrapper.
Real microphone frames continue through perception and RNN-T for captions and
interruption evidence, but are not inserted as synthetic PAD positions between
the native call and its eventual result. Speech-bearing regions are retained as
evaluated modality embeddings with bounded onset/tail context. Once the result
closes, the session replays at most one retained embedding per microphone
callback into the shared language/TTS timeline without advancing the streaming
encoder twice. The corresponding new microphone embedding remains queued, so
the foreground callback never drains an unbounded backlog. Replay text is
forced to PAD and its elapsed audio is never scheduled for playback. Idle
EAR-TTS cache positions are synchronized later in bounded eight-frame chunks.
The session yields its actor after each asynchronous step, allowing live
FastConformer/RNN-T work to interleave and detect a sustained user interruption.
The live driver reads one coherent function-channel snapshot and one MCP
snapshot every two audio periods (160 ms), instead of making several actor
calls after every frame. Diagnostics therefore cannot repeatedly interleave
with the asynchronous 11B decoder or delay the next microphone read.
The call has independent 256-step and eight-second safety bounds; phase-two
response injection has a separate 512-step/eight-second budget. PAD therefore
cannot keep either phase active indefinitely.

When function calling is enabled, `<SPECIAL_20>` opens a tool call and
`<SPECIAL_21>` closes it. Tokens between them decode to the checkpoint's JSON
`<TOOLCALL>` payload. The text channel is forced to idle for the complete call
and while an external result is pending, so tool JSON cannot leak into speech.
The runtime then feeds the tokenized
`<TOOL_RESPONSE>...</TOOL_RESPONSE>` result back through the function feedback
channel on a second asynchronous cached-silence phase, followed by
`<SPECIAL_22>`. Because the complete result is known, the runtime advances the
language and EAR-TTS caches in bounded 16-token causal-prefill chunks instead of
performing one 11B decode per token. Cache-parity coverage checks this against
the sequential recurrence. This mirrors the two-phase path in NVIDIA's
[`nemotron-labs-voicechat` reference wrapper](https://github.com/NVIDIA-NeMo/NeMo/blob/nemotron-labs-voicechat/nemo/collections/speechlm2/inference/model_wrappers/nemotron_voicechat_inference_wrapper.py).
The closing marker also restores the assistant authorization consumed when the
native call opened. Without that transition, RNN-T self-play suppression would
reject the checkpoint's result-conditioned BOS as an unprompted turn even
though the tool completed successfully. The restored authorization is consumed
by that BOS; fresh RNN-T speech is still required for every later user turn.
System-prompt positions keep both generated feedback channels at PAD, matching
NeMo; feeding unsupervised prompt predictions forward can otherwise begin live
audio midway through a hallucinated call.

The MCP coordinator receives only a completed native function-channel payload;
its API has no transcript argument. It validates the selected tool and the
small JSON-Schema subset used by MCP, applies the configured read/write policy,
executes the provider, and returns compact structured JSON. It never classifies
transcript text, repairs arguments, fills slots, matches spoken reminder names,
or authors assistant speech. Missing or invalid values return through the
trained function-response channel so the checkpoint can clarify in its own
words.

The CLI defaults to `allow`: one complete native write executes immediately and
the provider result returns through the function channel, without a second
confirmation turn. An identical completed write is suppressed until new
acoustic activity, preventing an immediate model retry from duplicating the
side effect. `confirm` remains an explicit opt-in; it returns
`confirmation_required` and requires the checkpoint to emit the identical call
again after fresh RNN-T non-blank activity. No yes/no phrase is interpreted by
the runtime. These are protocol and side-effect guards, not a second
natural-language router.

The Apple Reminders facade exposes a deliberately small native tool surface.
`create_reminder` requires only a name and otherwise uses schema-documented
provider defaults. `list_reminders` performs one flattened EventKit read and
returns stable session references such as `r1` instead of long provider UUIDs.
`update_reminder` accepts one of those model-visible references, which is mapped
exactly back to the provider ID immediately before execution. The runtime does
not resolve a spoken name or cache reminder content. The model alone selects the
tool and produces its arguments. No-op updates and fractional priorities are
rejected instead of being reported as successful provider mutations. Successful
write responses contain only `{"ok":true}`, and successful reads omit the
redundant tool name while preserving their records. This shortens cache
synchronization without changing the native call or fabricating assistant text.

The three expensive phases have separate scheduling and measurements:

1. Native call JSON is decoded on the cached-silence fast path, yielding the
   session actor after every token.
2. External MCP I/O runs in a separate Swift task. The coordinator and MCP
   actors remain reentrant while awaiting the provider, so the capture loop is
   not held by EventKit or stdio latency.
3. A known result is synchronized into the language and EAR-TTS caches in
   bounded 16-token causal-prefill batches, yielding between batches.

Perception and RNN-T continue while phases one and three own the language cache;
captions and sustained-speech interruption evidence therefore remain live.
During the external wait, ordinary real-audio frames do not advance the shared
model timeline. Meaningful captured regions are replayed immediately after the
result, so a captioned follow-up is not lost. The model does not speak a
runtime-authored acknowledgement. Result-conditioned speech resumes only after
`<SPECIAL_22>` closes the response.

This is the important concurrency boundary: the function and text channels
share one Nemotron-H KV cache. They cannot generate two independent causal
futures at once. Fully overlapping arbitrary assistant speech with a pending
tool would require a speculative second cache plus a defined merge/replay
policy beyond the bounded post-result input replay used here, or a separately
trained function decoder. The current path prioritizes live microphone
processing and truthful, model-generated post-result speech over such
speculation. Function-response injection rejects attempts to
overwrite an active native call, and bounded valid error JSON is used if a
provider result cannot be injected. If both normal and fallback injection fail,
the host aborts the function cycle so later speech remains usable. Function
calling remains disabled by
default; without an executor the standard conversation path and its latency are
unchanged.

Prompt conditioning is causally prefetched in chunks of 64 positions. Each
position still receives its prompt token plus PAD feedback from both generated
channels, but the prefill bypasses the unused text and function vocabulary
projections. This is numerically equivalent at the checked INT5 output while
avoiding hundreds of serialized 11B steps for a schema-rich MCP prompt.

### RNN-T turn-taking

The checkpoint has no standalone VAD probability head. Its bundled RNN-T
decoder and joint projection provide the activity signal used by NVIDIA's
realtime wrapper. For every 80 ms encoder frame, the first RNN-T prediction is
classified as blank or non-blank before the remaining label loop runs.
The frontend uses the checkpoint's filterbank and window together with NeMo's
reflection padding. Zero padding changes the spectrogram at both edges of each
rolling live window and can turn a marginal word into a different greedy
RNN-T token sequence.

The `.nvidiaRealtime` session policy ports that control path: two initial
non-blank frames and three on later turns are sustained-activity fallbacks, and
one recognized lexical RNN-T token immediately confirms an idle user turn.
This token-level path matters because a short complete phrase can emit several
labels inside one encoder frame; its caption must not wait for two more speech
frames before it can receive a response. Unknown and punctuation-only labels
do not qualify. Forty blanks force agent BOS after an utterance, and forty
consecutive non-blanks provide the RNN-T agent-EOS safety fallback. This
runtime uses the reference branch's conservative 40-frame profile; learned
BOS/EOS predictions remain the normal low-latency turn-taking path. Agent BOS
clears the consecutive-speech counter, so pre-BOS user activity cannot
terminate the new response. In normal interactive mode,
model-native BOS is suppressed before the first confirmed user turn and after
every completed agent turn; explicit greeting mode permits only the initial
exception. Once content has been followed by the larger of 16 PAD frames or
three PAD frames per content token, the next blank PAD becomes EOS so the
logical turn cannot remain stuck open. This content-scaled budget preserves
delayed EAR-TTS speech that a fixed 1.28-second cutoff can truncate. Audio
frames are never gated or removed; continuous silence remains part of the model
timeline.

Tool-enabled realtime sessions add a narrower endpoint only for an existing
model-native function candidate: eight blank frames (640 ms) permit its full
131,072-row verification, while ordinary replies and barge-in keep the
40-frame safety thresholds. The candidate is scoped to one uninterrupted user
speech segment and is cleared on resumed RNN-T activity, assistant output, or
tool-result injection. This reduces native tool-start latency without turning
the RNN-T activity signal into a general short-pause VAD.

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

The iteration count affects only progressive EAR-TTS refinement for future
frames and can be changed on a live actor with `setSpeechIterations(_:)`.
The CLI begins with eight iterations, uses two as a visible, reversible
realtime fallback after the first 88 ms callback or three queued frames, and
uses one emergency step after a 120 ms callback, six queued frames, or input
resynchronization. Perception, language, RNN-T, turn-taking, codec state, and
the 80 ms model clock are not skipped by this quality adjustment. If even the
emergency setting cannot keep up, the CLI drops stale
queued input at its latency bound and calls `resynchronizeLiveInput()`. The
buffer removes only enough oldest audio to accept fresh capture and coalesces a
sustained overload into one recovery episode. Fresh RNN-T counters and predictor
text reset across the discontinuity, while an already-confirmed user turn stays
armed so repeated overload callbacks cannot erase the whole request. Agent and
language history remain intact.

One-step synthesis is an overload safety mode, not acoustic parity with the
checkpoint's eight-step schedule: all 31 RVQ codebooks are assigned in a single
pass. It may change pronunciation detail, onset behavior, and pauses. The
dashboard therefore labels the active refinement level, and quality comparisons
must record the iteration count. A lower count protects live microphone service;
it does not guarantee that a particular stochastic reply will contain fewer
acoustic pauses.

The realtime path also follows NVIDIA's agent-idle and content-scaled PAD
policy. BOS opens a speech turn and EOS closes it. PAD inside that open turn
continues normal EAR-TTS for the larger of 16 frames or three frames per
emitted content token, because text generation can finish long before the
corresponding audio. Continued PAD beyond that budget, PAD while idle, and EOS
advance the TTS backbone cache but return canonical silence without discarded
MaskGIT refinement or a live codec window. New text resets the consecutive-PAD
counter and resumes normal synthesis immediately. This preserves delayed words,
turn alignment, and compute headroom for microphone input.

Normal synthesis evaluates the generated code and all 28 retained EAR-TTS
attention-cache pairs together on every frame. Extended canonical-silence PAD
advances eight frames at once with a causal mask, then evaluates the same cache
roots. This fences MLX's lazy concatenations before they can accumulate across
a long conversation without paying 28 decoder passes for eight inaudible
frames. A synchronous fence is intentional: queuing unrelated live work first
oversubscribes the shared MLX stream and is slower for this 80 ms pipeline.

Unbounded attention is still the low-level and file-inference default. Live CLI
sessions instead retain the immutable 37-frame Aria speaker prompt plus 250
recent generated frames (20 seconds). When the rolling region fills, its oldest
middle-history frame is removed while the absolute RoPE offset continues to
advance. This bounds all 28 EAR-TTS attention layers without resetting position
or losing speaker conditioning. It does not truncate Nemotron-H's semantic
conversation state. `VoiceChatSpeechGenerationParameters.recentContextFrames`
controls the policy; `nil` preserves the complete TTS history.

Masked codebooks contribute exact zero vectors. During progressive assignment,
the runtime therefore carries the sum of the already selected RVQ embeddings
forward instead of gathering and summing all 31 codebooks before every
MaskGIT head pass. The selected code IDs and resulting latent are unchanged;
the optimization removes redundant GPU work from each generated frame.

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
fp32 for decoding, matching the verified Python path. Live playback uses a
bounded eight-frame decoder window whose fixed-shape graph is compiled during
model warmup; short startup windows and arbitrary-length offline rendering keep
the general decoder path.

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
10. Decoding the RNN-T transcript but discarding its first-prediction blank
    signal, which removes NVIDIA's deterministic EOU and barge-in path.

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
| RNN-T turn policy | unit coverage for lexical short-turn confirmation, sustained activity, EOU, barge-in, silence self-play suppression, post-function turn resumption, and disabled passthrough |
| Reminder MCP adapter | unit coverage for schema-preserving native tools, flattened list results with session-scoped opaque references, exact reference-to-provider-ID resolution, stale-reference rejection, provider-ID compatibility, invalid-argument rejection, immediate writes by default, opt-in model-mediated confirmation, write denial, duplicate suppression, and coordinator reentrancy during suspended MCP I/O |
| Function fast path | three native reminder-list phrases emit the expected calls in 27–31 asynchronous steps and inject a 67-token result in five causal prefills; a real three-cycle list → update confirmation → confirmed update regression completes without a repeated-call loop; realistic update IDs drop from 67 steps / 5.25 s to 37 steps / about 2.29 s with lossless session references |

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

Stateful FastConformer caches and the compiled live codec path let the INT5
bundle sustain the model's 80 ms frame clock on the tested M5 Pro.
`VoiceChatSessionSummary` reports perception, language-decision,
speech-synthesis, and total per-frame latency separately. Two thresholds remain
deliberately distinct:

- whole-pipeline RTF below `1` means aggregate inference keeps pace with live
  audio; and
- total per-frame p95 below `80 ms` means nearly every individual frame meets
  its playback deadline. `realTime` reports this stricter p95 condition.

Release builds on an M5 Pro (48 GB), over the 120-frame controlled E2E fixture
with one 80 ms frame per live input push, produced these three independent INT5
runs after model loading and prompt warmup:

| Run | Wall time | Model timeline | Whole-pipeline RTF |
|---:|---:|---:|---:|
| 1 | 9.039 s | 9.600 s | 0.94 |
| 2 | 8.833 s | 9.600 s | 0.92 |
| 3 | 8.786 s | 9.600 s | 0.92 |

All three runs produced the same transcript and generated response. Typical
perception, decision, and synthesis p50 values ranged over 9.0–9.2 ms,
35.6–36.6 ms, and 28.3–29.0 ms; total/frame p50 was 72.9–74.8 ms. Peak RSS was
about 8.70 GB, MLX reported 9.53 GB peak GPU allocation, and the macOS
physical-footprint peak was 15.64–15.66 GB. Physical footprint includes
file-backed MLX mappings that RSS can undercount. Across the three runs, first
spoken text arrived in 44.3–46.7 ms and first playable audio in 74.0–76.4 ms.

This establishes sustained whole-pipeline RTF below `1` for the tested INT5
configuration, with all three total/frame p95 measurements below 80 ms at
76.2–78.6 ms. The remaining deadline headroom is still narrow. Current INT8
correctness gates pass, but release performance has not been remeasured after
this optimization, so no updated INT8 real-time claim is made here. Run the
benchmark on the target machine because thermal state and concurrent GPU work
materially affect this margin.

A paired live-caption regression on the same M5 Pro exposed why execution
placement matters. Moving the small RNN-T head to a CPU stream produced RTF
1.30 and 99.5 ms total/frame p50. Keeping it on the shared MLX inference stream
produced the identical transcript and response at RTF 0.96 and 76.2 ms p50,
matching the no-caption control at RTF 0.96 and 76.3 ms p50. The opt-in
`E2EVoiceChatRealtimeCaptionTests` gate uses one-frame pushes across a
33.6-second/420-frame session and rejects RTF at or above 1.0 or p50 at or
above 85 ms. The long tail catches retained lazy-cache graphs that a short
smoke test cannot expose.

A separate 795-frame/63.6-second sustained profile exercised the rolling
EAR-TTS boundary. Full speech history produced aggregate RTF `1.06`, final
8-second RTF `1.19`, 47.7 ms/frame late synthesis, and 38.5 GB peak physical
footprint. With the content-safe tail, retaining the prompt plus 250 recent
frames and batching only post-tail PAD produced aggregate RTF `0.87`,
final-window RTF `0.71`, 4.1 ms/frame final-window synthesis, 8.72 GB peak RSS,
and 23.67 GB peak physical footprint. Its speech-active eight-step windows were
RTF `1.05` and `1.12`. The live CLI now lowers refinement from eight to two
steps as soon as one callback reaches 88 ms or three microphone frames queue,
and directly to one step after a 120 ms callback, six queued frames, or input
resynchronization. It restores the requested quality only after 100 stable
frames. The minimum cosine between sequential and eight-frame batched cache
state was `0.9999999`. File inference and the default Swift API do not enable
this realtime-only compaction.

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
let session = try await model.startSession(
    turnTaking: .nvidiaRealtime)

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
