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

## Live command-line conversation

Build the release binary and Metal library, then start the microphone demo:

```bash
make build
./.build/release/speech voice-chat
```

The command defaults to the complete
`aufklarer/VoiceChat-11B-Perception-MLX-int5` bundle. It downloads the model on
first use, enables Apple's acoustic echo cancellation, displays the bundle's
own streaming RNN-T user transcript beside Soniqo's text, and plays each
generated 22.05 kHz frame while continuing to capture 16 kHz microphone audio.
Press Control-C to stop cleanly.

The RNN-T frontend uses NeMo-compatible reflection padding rather than zero
padding at each rolling microphone-window edge. Decoding remains the
checkpoint's greedy 1,024-token RNN-T path; the padding correction removes a
reference mismatch but does not turn it into a general large-vocabulary ASR.

Local and downloaded bundles report tokenizer, perception/RNN-T, 11B language,
EAR-TTS/codec, and codec-verification stages while loading. The model is large,
so the language and speech-weight stages can each take tens of seconds on a
cold filesystem even though the process is active. Session startup causally
prefills long system/tool prompts in 64-token chunks instead of advancing the
11B model once per prompt token.

The default product persona is **Soniqo**. The underlying checkpoint and model
architecture remain NVIDIA NemotronLabs VoiceChat 11B and are identified as
such in the dashboard and model documentation. `--system-prompt` can replace
the persona for an individual session.

In an interactive terminal, VoiceChat uses the alternate screen and redraws one
fixed dashboard in place. Exiting restores the normal terminal and prints one
final snapshot. Use `--plain` for append-only output suitable for logs.

Use a local bundle, change the playback cushion, or disable the updating
terminal display as follows:

```bash
./.build/release/speech voice-chat \
  --model /path/to/complete-voicechat-bundle \
  --prebuffer-frames 3

./.build/release/speech voice-chat --plain

./.build/release/speech voice-chat --debug-timeline

./.build/release/speech voice-chat --no-rnnt-turn-taking
```

## Configurable MCP tools

The live CLI can expose local or remote Model Context Protocol servers through
VoiceChat's trained function-output channel. MCP is off unless
`--mcp-config` is supplied. The included Apple Reminders example launches a
pinned EventKit-backed server through `npx`:

```bash
./.build/release/speech voice-chat \
  --model /path/to/complete-voicechat-bundle \
  --mcp-config Examples/VoiceChatMCP/apple-reminders.json
```

macOS may request permission when the server first accesses Reminders. On its
first launch, the pinned package compiles a small Swift EventKit helper; MCP
startup therefore has a separate 60-second minimum allowance while ordinary
tool calls retain the configured timeout. The deliberately narrow example
exposes `list_reminders`, `create_reminder`, and `update_reminder`; `list_lists`
does not occupy the model's small tool-selection envelope. It does not invoke any
tool during startup; startup only performs the MCP initialize and tool-discovery
exchange. Custom configurations can enable other server tools.

Configuration uses a provider-neutral `mcpServers` map. Each server supplies
`command`, optional `args`, `env`, and `workingDirectory`, plus two safety
lists:

```json
{
  "mcpServers": {
    "my-server": {
      "command": "path-or-command",
      "args": [],
      "env": {},
      "enabledTools": ["read_item", "create_item"],
      "readOnlyTools": ["read_item"]
    }
  }
}
```

`adapter` is optional. The bundled `apple-reminders-eventkit` adapter is a
narrow facade over `mcp-server-apple-events`: it maps the provider's multi-action
tools to flattened `list_reminders`, `create_reminder`, and `update_reminder`
aliases. Provider delete, list-mutation, calendar, and subtask actions are never
exposed. The
facade also converts a model-provided due date to the provider's validated
EventKit timestamp before any write is sent. The provider constructs the whole
reminder and commits it with one EventKit save, avoiding the old AppleScript
sequence that could create an undated reminder and then report failure while
setting its date.

Only names in `enabledTools` are shown to the model. Any enabled tool not also
listed in `readOnlyTools` is treated as a write, regardless of MCP annotation
hints. VoiceChat currently permits at most five enabled tools per session,
matching the checkpoint's reliable tool-selection envelope. Duplicate names
across selected servers and configured names missing from discovery fail at
startup. Use repeated `--mcp-server NAME` options to select a subset from a
larger file; without them, every configured server is selected.

Writes default to `--mcp-write-policy allow`, so a complete model-native write
call executes once and its real provider result returns directly to VoiceChat.
This is the low-friction mode used by the reminders demo: Soniqo does not add a
second confirmation turn before or after a successful action. An identical
completed write is still suppressed until fresh acoustic activity, preventing
an immediate model retry from duplicating the side effect. Use
`--mcp-write-policy deny` to refuse every write.

`--mcp-write-policy confirm` remains an explicit opt-in for applications that
want model-mediated confirmation. In that mode the first write call is held,
the model receives `confirmation_required`, and it must emit the identical call
again after fresh post-prompt RNN-T speech. The runtime does not inspect yes/no
words. If the model answers normally instead, the pending write expires.

There is no transcript router, regular-expression intent matcher, slot
extractor, reminder draft state, argument repair, or scripted assistant reply.
The model's native function channel alone chooses the tool and arguments. The
runtime verifies that the tool is enabled, validates required fields and basic
JSON types, applies the write policy, executes MCP, and returns structured JSON
through the trained function-response channel. Missing or invalid arguments are
reported to the model, which can ask the user or use an available read tool.
Clarification, success, and failure speech are generated by VoiceChat. In the
optional `confirm` mode, confirmation speech is model-generated as well.

Failed provider results, invalid arguments, denied writes, and unavailable
tools also suspend new function starts until the checkpoint opens a spoken
assistant turn. This prevents an immediate model-native retry loop from leaving
the session silently waiting for another tool result. The gate uses only the
structured result and the model's BOS token; retries after later user speech
remain model-controlled.

For the reminder facade, `create_reminder` requires only `name`; `list`, body,
due date, and priority are optional, and an omitted list uses the provider's
documented system default. `list_reminders` performs one flattened EventKit read
and replaces long provider UUIDs with stable, session-scoped references such as
`r1`. `update_reminder` requires one of those references and includes only fields
the model chose to change; the adapter resolves the exact reference back to its
provider UUID immediately before execution. This is opaque-ID compression, not
intent routing: the runtime does not match spoken names, cache reminder content,
or inspect transcript words. Unknown or stale references are rejected and can be
refreshed with another list call. An update with no changed field is rejected,
and priority must be one of the provider's exact integer values rather than a
silently truncated decimal. A bad model argument is never silently replaced
with words from the displayed transcript.

```bash
./.build/release/speech voice-chat \
  --mcp-config /path/to/mcp.json \
  --mcp-write-policy deny
```

`--no-transcript` remains compatible with the optional confirmation mode because
only the RNN-T activity signal is used; no transcript words are inspected. MCP
tool calls use a 15-second timeout by default; change it with
`--mcp-timeout-seconds`.
Initialization allows at least 60 seconds so a pinned `npx` dependency can
perform its one-time local Swift build without extending live tool latency.

Child MCP processes inherit only `PATH`, `HOME`, temporary-directory, and
locale variables. Credentials and other required values must be added
explicitly to that server's `env` map; unrelated parent-process secrets are not
forwarded. The configured `command` still runs locally with the current user's
filesystem and application permissions. Use MCP configuration files and server
packages only from sources you trust.

At runtime, `<SPECIAL_20>` starts a function call and `<SPECIAL_21>` completes
it. VoiceChat then follows NVIDIA's two-phase asynchronous function path rather
than waiting for one function token per 80 ms microphone frame. Session startup
encodes one second of zero PCM and caches its first perception embedding. Call
JSON is generated at full language-model speed from that learned silence input;
after the function head independently selects `<SPECIAL_20>`, the fixed trained
opening literal `<TOOLCALL>[{"name":"` is inserted with one causal prefill.
This skips nine redundant full-vocabulary selections without choosing a tool or
changing any model-generated name or argument. The literal opening and closing
markers are validated as a balanced, parseable JSON payload in real-checkpoint
coverage. After the MCP result arrives, its already-known response tokens are
replayed as bounded 16-token causal-prefill chunks. This preserves the trained
function feedback and both language/speech caches while avoiding one 11B decode
and unused text-vocabulary projection per response token. The actor yields between
generated call tokens and response-prefill chunks, so microphone perception,
streaming RNN-T captions, and sustained-speech interruption detection continue
while the language cache advances. The CLI starts the mapped MCP call in a
separate Swift task, outside the microphone loop. The coordinator actor is
reentrant across the external `await`, so capture, perception, RNN-T captions,
UI updates, and interruption signals continue while the provider is suspended.
Meaningful microphone regions captured while the function channel owns the
shared cache are retained as already-evaluated modality embeddings, with two
frames of onset context and a bounded silence tail. After the tool result is
causally complete, at most one old embedding is replayed into Nemotron-H per
live callback without running perception or RNN-T twice. The current live
embedding is retained during that callback, so catch-up cannot synchronously
drain a long backlog. Replay text is forced to PAD, elapsed replay audio is not
queued to the speaker, and the matching idle EAR-TTS cache is advanced later in
bounded eight-frame chunks. This fixes the former state where a follow-up
appeared in captions during `USING TOOL` but was never understood by the model,
without recreating the multi-second input loss caused by unbounded replay.
The live driver reads one coherent function-channel snapshot and one MCP
snapshot every 160 ms. This replaces several per-frame actor queries that could
repeatedly interleave with the asynchronous 11B decoder and delay microphone
capture; it does not route or match transcript text.
Normal model speech is silenced while the call is pending. The CLI injects a
compact ASCII
`<TOOL_RESPONSE>...</TOOL_RESPONSE>` payload back into the function channel;
`<SPECIAL_22>` closes the injected response. Only then can Soniqo describe the
result. Closing the response explicitly restores the assistant side of that
already-accepted user turn, so RNN-T self-play suppression cannot mistake a
native post-tool answer for an unprompted new turn. The authorization is
consumed by the next assistant BOS and does not carry into later turns. Tool
output is bounded before injection. If a response cannot be injected, the CLI
retries with a short valid error result; it still does not force assistant
wording. If even that bounded recovery result cannot be injected, the session
abandons the broken function cycle and returns control to the model instead of
remaining parked forever. An open native call rejects unrelated response
injection.
Successful reminder writes return only `{"ok":true}` because the immediately
preceding call already identifies the tool and written values. Successful reads
likewise omit the redundant tool name while retaining their result. Local validation
accepts exact local timestamps, RFC 3339 offsets, and exact relative values such
as `tomorrow at 8 PM`. Ambiguous values such as `tomorrow morning` are not
guessed: the terminal reports `needs input`, and a compact
`clarification_required` result asks the model for one missing value instead of
presenting the condition as an MCP/provider failure. Verbose provider/debug
prefixes are not replayed through the 11B and EAR-TTS caches.
Function generation is also bounded to 256 asynchronous positions and eight
seconds; response injection has its own 512-position/eight-second bound. A
malformed PAD-only call therefore releases the accepted turn back to the model
instead of consuming the live microphone clock indefinitely. The narrow demo
exposes no destructive delete operation; completion is available through
`update_reminder`.

Asynchrony has a deliberate boundary. External I/O and microphone/RNN-T work
can overlap, but ordinary result-conditioned agent speech cannot safely run in
parallel with function tokens on the same Nemotron-H KV cache. VoiceChat must
finish `<SPECIAL_22>` before it can speak coherently from the result. It uses a
bounded replay policy for user speech captured during that interval; fully
simultaneous freeform assistant speech would still require a speculative
second cache branch and a defined merge. The runtime therefore prioritizes
live capture, causally replays any user interruption, and then generates
coherent post-tool speech. It does not play a cached or scripted acknowledgement;
supporting a simultaneous model-generated acknowledgement requires a checkpoint
trained with independently synchronized speech and action channels.
The terminal shows `PREPARING TOOL` while the checkpoint decodes the tool name
and arguments,
then `USING TOOL` with the sanitized tool name while the external MCP service is
executing. These are coordinator lifecycle events around validated calls;
the raw function projection does not itself supply a trustworthy UI event.
Arguments and raw JSON remain hidden by default. For local diagnosis,
`--debug-timeline` adds relative timestamps to user and Soniqo phrases and
inserts the exact parsed native call, MCP start/completion, and result-cache
synchronization into the conversation timeline. Each completed assistant line
also receives a `pronunciation ended` timestamp at the end of the final
sustained audible 10 ms generated-PCM window above -50 dBFS. This is the model
audio timeline, not the later instant when the speaker device drains its
playback queue. Control characters and tool names are sanitized before display,
and argument output is bounded. This mode can expose reminder names or other
tool arguments, so it should not be enabled in shared logs. During either
phase, the session holds natural text generation so a guessed answer cannot
race the real result. The fallback phase timer resets between native call
preparation and provider waiting; it no longer attributes JSON decode time to
the connected service.
The interactive dashboard intentionally omits historical native-decode,
provider, and result-cache timing blocks, even in debug mode. Active states such
as `PREPARING TOOL` and `USING TOOL` remain visible, while `--debug-timeline`
records their lifecycle in context. Detailed phase timing, token throughput,
cache work, live interleave, and microphone pressure belong in the structured
`voicechat-bench` JSON report rather than the conversational demo.

Tool-enabled sessions retain RNN-T suppression and barge-in while preserving
the checkpoint's learned BOS/EOS decisions as the normal turn-taking path.
This runtime's 40-frame (3.2-second) RNN-T threshold is used only as the hard
safety fallback. Incomplete tool arguments are returned to the model for
clarification. The session cannot remain silent forever when neither the text
nor function head opens a turn. Finite-file mode
automatically advances pending function/confirmation output for up to 30
additional seconds after the configured silent tail.

The 8-bit function projection is approximately 518 MB in the INT5 bundle.
Evaluating all 131,072 rows even every fourth frame cannot sustain the 80 ms
microphone clock. Instead, the runtime keeps a dequantized 36 KB probe containing
only the function PAD and `<SPECIAL_20>` rows. After RNN-T confirms user speech,
that probe runs with the normal text decision. A winning start row retains only
the 4,096-value hidden-state candidate until learned BOS or the final RNN-T
safety endpoint; the complete projection is not evaluated continuously while
speech is active. It still verifies that `<SPECIAL_20>` is the global argmax
before opening a native call. Once a native call is open, the full head runs on
the cached-silence asynchronous path until the JSON payload is complete. The
text channel is forced to PAD over this interval, so the runtime advances only
the shared backbone and function projection; it does not materialize the unused
131,072-row text projection for each JSON token. After EOTC, the two-phase path
freezes the shared language and EAR-TTS timeline while the external result is
pending, matching NVIDIA's reference wrapper. Incoming microphone frames
continue through perception and RNN-T without evaluating the 11B language or
EAR-TTS stages. They are not inserted as synthetic PAD positions between the
native call and its eventual result. Speech-bearing regions are buffered as
evaluated modality embeddings and inserted immediately after the result, while
long provider-wait silence is compressed. Replaying a typical spoken follow-up
does require catch-up language/TTS compute, but it no longer disappears from
the conversation. The real-checkpoint natural-phrase
regression generated the tested reminder calls in 27–31 asynchronous call steps
and injected each 67-token result in five 16-token-or-smaller prefills, with no
timeout. Cache-parity coverage compares batched response replay with the former
one-token implementation. The minimum language-cache cosine was `0.9999986`
with a maximum absolute difference of `1.2352`; the minimum speech-cache cosine
was `0.9999994`.

On an otherwise idle M5 Pro with 48 GB, a controlled 33-token result after 600
input frames used three prefills: cache synchronization took 291 ms (211 ms
language, 60 ms voice, 19 ms interleave), first speech began at 472 ms, and the
slowest concurrently sampled microphone frame took 146 ms. A 32-token experiment
reduced the number of prefills to two but increased synchronization to 658 ms
and the worst microphone stall to 465 ms, so the bounded 16-token schedule is
retained. This is a tool-response microbenchmark, not general conversation RTF.
The optimized path preserves exact generic tool-start decisions while known
reminder conversation stays on the normal realtime path.

The complete natural-phrase profile on the same machine and release INT5 bundle
used a fixed 200 ms simulated provider:

| Phrase | Native call start | Native JSON | Result sync | First post-tool speech | Mic p95 / max |
|---|---:|---:|---:|---:|---:|
| “What reminders do I have?” | 1,425 ms | 1,147 ms / 27 steps | 577 ms / 67 tokens | 3,557 ms | 127 / 175 ms |
| “List my reminders.” | 1,793 ms | 1,176 ms / 27 steps | 616 ms / 67 tokens | 3,977 ms | 128 / 170 ms |
| “Show my active reminders.” | 1,614 ms | 1,278 ms / 31 steps | 722 ms / 67 tokens | 4,000 ms | 121 / 288 ms |

Repeated runs vary modestly with thermals and unified-memory activity. A final
clean `list-question` replication measured 1,478 ms to call start, 1,093 ms
for 27 native JSON steps, 204 ms provider time, 544 ms for the 67-token result,
and 3,503 ms to first post-tool speech. Microphone actor service was 118 ms p95
and 172 ms maximum; accumulated lateness beyond the 80 ms input deadlines was
139 ms p95 and 231 ms maximum. That deadline-overrun metric exposes short
queueing bursts that foreground audio RTF and per-call actor service time do
not show.

All three listing phrases selected `list_reminders` with the expected native
arguments. “Create a reminder called Phone John in the Reminders list” was
transcribed correctly but selected ordinary speech instead of a function call;
that is a checkpoint tool-selection quality miss, not execution latency. A
separate two-second-provider stress run measured 19.6 ms microphone p95 and
19.8 ms maximum during the external wait. Reply onset increased only by the
additional provider time, confirming that external I/O is additive without
holding live capture. The real local EventKit adapter started in 1,236 ms once
and completed the flattened reminder read in 358 ms.

A focused update-call regression also quantified the cost of opaque provider
identifiers. Replaying a realistic EventKit UUID required 67 native function
steps and 5.25 seconds; exposing the same record as session reference `r1`
required 37 steps and about 2.29 seconds. That is 45% fewer function-token steps
and about 56% less native decode time in this controlled run. The checkpoint
still selected `update_reminder` and generated every argument; only the
losslessly resolved identifier representation changed.

Reproduce the end-to-end natural-phrase profile in isolated release processes
so one 11B session cannot retain cache state into the next phrase:

```bash
for phrase in list-question list-command list-active create; do
  VOICECHAT_BUNDLE=/path/to/complete-bundle \
  VOICECHAT_PERFORMANCE_TEST=1 \
  VOICECHAT_TOOL_PHRASE="$phrase" \
  swift test -c release \
    --filter E2EVoiceChatFunctionCallingTests/testNaturalToolPhraseLatencyProfile \
    --disable-sandbox
done

VOICECHAT_BUNDLE=/path/to/complete-bundle \
VOICECHAT_PERFORMANCE_TEST=1 \
swift test -c release \
  --filter E2EVoiceChatToolResponsePerformanceTests/testLongContextToolSuccessStaysRealtime \
  --disable-sandbox
```

Run these only when no other MLX/Core ML/model benchmark is active. The phrase
profile separates end-of-input to call start, native JSON decode, a fixed 200 ms
provider, result synchronization, first post-tool speech, microphone service
p95/max for the full cycle and each tool phase, and lateness beyond the 80 ms
input deadline. Set
`VOICECHAT_TOOL_PROVIDER_DELAY_MS=2000` to reproduce the slow-provider capture
stress. If the checkpoint answers conversationally instead of opening its
native function channel, the test records that as a tool-selection quality miss
rather than silently routing the transcript.

The live command starts with eight EAR-TTS refinement steps. It switches to two
steps as soon as one callback reaches 88 ms or queued input reaches three 80 ms
frames. A callback at or above 120 ms, six queued frames, or an input
resynchronization goes directly to one emergency step. Only voice refinement
changes; perception, RNN-T transcription, language state, and every accepted
microphone frame remain intact. `--realtime-speech-iterations` changes the
two-step fallback. After 100 consecutive frames with less than one queued frame
and at least 20% compute headroom, it restores the requested
`--speech-iterations` value.

Live mode also retains the fixed 37-frame speaker prompt plus the newest 20
seconds (250 frames) of EAR-TTS attention history. This bounds the TTS work as
the conversation grows while preserving Soniqo's semantic history in the
separate Nemotron-H language state. Set `--live-speech-context-seconds 0` for
unbounded diagnostic runs. File input and the low-level Swift API remain
full-history unless `recentContextFrames` is selected explicitly.

After an answer's last text token, live mode keeps rendering PAD because those
frames can still carry delayed EAR-TTS speech. The acoustic budget is the
larger of 16 frames (1.28 seconds) or three PAD frames per emitted content
token, matching NVIDIA's content-scaled realtime tail. Only PAD beyond that
budget becomes canonical silence and advances the same causal EAR-TTS state in
batches of eight. New text resets the consecutive-PAD count and immediately
returns to normal frame-by-frame synthesis. File mode and the default Swift API
retain the exact published schedule; this realtime compaction is enabled only
when `realtimeIdleOptimization` is selected.

Three 80 ms frames (240 ms) are buffered before initial playback. This absorbs
small inference jitter while retaining scheduled `AVAudioPlayerNode` buffers
that Voice Processing can use as its echo reference. If playback nevertheless
drains, recovery waits for eight frames before restarting instead of repeatedly
restarting with the small startup cushion. This trades one visible recovery
pause for a much lower risk of recurring cuts. `--no-aec` is available
for headphones or diagnostics. `--no-transcript` hides the user caption while
leaving the RNN-T head active for turn-taking. Combine it with
`--no-rnnt-turn-taking` only when benchmarking the learned text-channel policy
without either caption or RNN-T control.

Live mode follows NVIDIA's realtime RNN-T turn-taking policy by default. The
transcript head's first prediction on each 80 ms encoder frame is treated as
speech (non-blank) or silence (blank). Two initial non-blank frames and three on
later turns remain the sustained-activity fallbacks, but one recognized lexical
RNN-T token now arms an idle user turn immediately. A short complete phrase can
emit several tokens inside only one encoder frame; requiring three separate
non-blank frames after displaying that caption could otherwise leave “yes” or
“who are you?” waiting for more speech. Punctuation and unknown tokens do not
arm the shortcut, and it never applies while Soniqo is speaking.

VoiceChat's learned BOS/EOS predictions remain the normal low-latency turn
decisions. Forty blank frames (3.2 seconds) force agent BOS only if the learned
head never opens the turn, and forty consecutive non-blank frames provide this
runtime's matching RNN-T agent-EOS safety fallback. The speech counter is
cleared when agent BOS is accepted, so audio
from the completed user turn cannot immediately terminate the response. Normal
mode replaces an initial model-native BOS with PAD until user speech is
confirmed, so Soniqo waits for the user; `--greet` explicitly permits that
first turn. After any agent turn finishes, an unprompted BOS is again replaced
with PAD until new user speech is confirmed, preventing repeated silence-only
self-play. Once a reply has emitted content, blank PAD remains inside the open
turn for at least 16 frames and for three frames per content token when that is
longer. The first blank PAD beyond that content-scaled acoustic budget becomes
EOS, so it cannot mute delayed speech and the next sentence still begins a new
logical turn.

MCP remains native-first. Ordinary conversation and barge-in retain NVIDIA's
40-frame safety fallback. An already-proposed native function candidate may be
committed after eight blank RNN-T frames (640 ms), because the function probe
has supplied additional semantic evidence; this narrower threshold does not
change pause handling for ordinary speech. The retained candidate is discarded
as soon as RNN-T detects resumed speech, so it cannot leak across two user
speech segments. Tool argument safety comes from structural schema validation
and the selected write policy; argument meaning remains the model's
responsibility.

This is not a separate VAD model. During normal operation every microphone
frame reaches VoiceChat's continuous 80 ms clock because silence is meaningful
input to the duplex model and to the RNN-T blank counters. Use
`--no-rnnt-turn-taking` to leave BOS/EOS decisions entirely to the learned
language head.

Microphone capture and inference are connected through a bounded queue. If the
quality reduction still cannot keep up and the queue reaches its default
eight-frame/640 ms bound, the buffer discards only the minimum oldest audio
needed to admit the newest capture callback. Continued overload is coalesced
into one recovery episode. A single lost 80 ms frame lowers voice refinement
immediately but preserves RNN-T predictor and turn context; resetting the full
decoder would otherwise discard more linguistic context than the short gap.
Two or more frames lost in the same episode are treated as a hard
discontinuity: fresh RNN-T counters and predictor text reset, while an already
confirmed user turn remains armed. Only that hard discontinuity marks the
interrupted transcript `input dropped; please repeat`. Counts and total skipped
duration remain visible for every loss, and dropped audio is never represented
as heard.

`--max-buffered-frames` changes that latency bound. Increasing it adds
interaction delay and does not make a sustained slow workload realtime. A
repeatedly degraded session needs less concurrent MLX/Core ML load, fewer
speech iterations, or faster hardware.

On the tested M5 Pro, a 795-frame/63.6-second release profile reproduced the
reported growth: with full TTS history, aggregate RTF was `1.06` and the last
8-second window reached `1.19`, with synthesis rising to 47.7 ms/frame. After
switching the live path to the content-safe tail above, retaining the prompt
plus 250 recent frames and batching only post-tail PAD measured aggregate RTF
`0.87`, final-window RTF `0.71`, 4.1 ms/frame final-window synthesis, 8.72 GB
peak RSS, and 23.67 GB peak physical footprint. The speech-active eight-step
windows reached RTF `1.05` and `1.12`; the interactive CLI therefore uses the
reversible two-step fallback and one-step emergency protection described above.
Batched and sequential idle-cache states measured minimum cosine `0.9999999`.
These measurements were separate release processes; avoid concurrent rendering,
encoding, or GPU work when comparing them.

The header uses human-facing metrics. Its second row splits live-frame RTF over
one shared rolling window of the latest 120 microphone callbacks:

- `behind` is queued microphone audio, not response latency;
- `normal` RTF covers callbacks outside a tool cycle, including ordinary
  listening and assistant speech;
- `tool` RTF covers callbacks that overlap native call decoding, provider wait,
  or result synchronization. It still measures microphone-frame service only;
  the external provider and background decoder retain their separate latency
  lines;
- `avg` RTF covers every callback in the same window. Replayed events are not
  counted as extra audio denominators. `1.00×` is the boundary, values below it
  provide headroom, and values above it fall behind. Because these are rolling
  values, they can stay high briefly after one expensive operation even when
  `last` has already recovered;
- `last 74 ms for 80 ms audio` means the last fixed 80 ms model frame took 74 ms of
  wall time across perception, language decision, and speech synthesis; and
- `replies` counts assistant turns opened.

Below the conversation, `speaker gaps` counts times generated playback ran
empty, while RNN-T `barge-ins` counts intentional microphone-driven response
interruptions. `microphone audio skipped` is captured audio discarded by
bounded resynchronization and should prompt the user to repeat that sentence.

For a deterministic file run without opening audio hardware:

```bash
./.build/release/speech voice-chat \
  --model /path/to/complete-voicechat-bundle \
  --input question.wav \
  --output response.wav
```

`--force-turn-at-end` is retained for controlled regression fixtures only; it
must not be used to represent natural turn-taking behavior.

## Load the model

The default Hub snapshot is the complete INT8 export:

```swift
let model = try await VoiceChatModel.loadFromHub()
let session = try await model.startSession(
    turnTaking: .nvidiaRealtime)
```

The command-line demo selects `.nvidiaRealtime`. The low-level Swift API keeps
`.modelNative` as its source-compatible default, so applications opt into the
RNN-T token override explicitly as shown above.

Use `aufklarer/VoiceChat-11B-Perception-MLX-int5` explicitly for the smaller
protected-head INT5 export. The historical `Perception` name is retained for
URL compatibility; current snapshots contain the complete pipeline.

To load an already-downloaded bundle:

```swift
import VoiceChat

let root = URL(fileURLWithPath: "/path/to/complete-voicechat-bundle")
let model = try await VoiceChatModel.load(from: root)
let session = try await model.startSession(
    turnTaking: .nvidiaRealtime)
```

`VoiceChatModel.loadFromHub(...)` uses the same strict loader after downloading
a snapshot. The repository must contain the full three-directory bundle.

The default Soniqo prompt does not request an initial greeting, so it remains
neutral for turn-onset timing. It also grounds the capabilities exposed by this
package: Soniqo can answer and converse, but cannot access calendars, reminders,
apps, accounts, devices, or external services. When asked to perform an external
action, it should state that limitation directly, avoid asking for confirmation,
and offer a brief alternative the user can perform. Applications that implement
real tool execution can enable `functionCallingEnabled`, build a prompt with
`toolCallingSystemPrompt`, handle completed `functionCall` events, and return
results through `injectFunctionResponse`. The CLI's MCP path implements this
contract directly.

```swift
let prompt = try VoiceChatSession.toolCallingSystemPrompt(
    availableToolsJSON: toolsJSON)
let session = try await model.startSession(
    systemPrompt: prompt,
    streamUserTranscript: true,
    turnTaking: .functionCallingRealtime,
    functionCallingEnabled: true)

for event in try await session.pushAudio(input) {
    if let callJSON = event.functionCall {
        let resultJSON = try await executeTool(callJSON)
        try await session.injectFunctionResponse(
            resultJSON,
            requireAssistantReplyBeforeNextFunctionCall: resultNeedsConfirmation)
    }
}
```

To reproduce the model's greeting behavior explicitly:

```swift
let session = try await model.startSession(
    systemPrompt: VoiceChatSession.greetingSystemPrompt,
    streamUserTranscript: true,
    turnTaking: .nvidiaRealtime)
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

        // Non-nil only when streamUserTranscript was enabled.
        if let userText = event.userTranscript {
            updateUserCaption(userText)
        }
    }
}
```

Every `VoiceChatFrameEvent` includes:

- the text and function token selected for that frame;
- a completed `functionCall` payload on the end-of-tool-call frame when
  function calling is enabled;
- the complete append-only user transcript when streaming transcription is
  enabled;
- whether the text channel is speaking;
- whether RNN-T forced agent start/end or suppressed an unprompted turn;
- its position on the model's 80 ms conversation clock;
- perception, language-decision, and speech-synthesis wall latency;
- the matching 1,764 output samples; and
- whether those samples should be queued for live playback. Idle audio from a
  deferred-input replay has already elapsed and reports
  `playbackRequired == false`; newly generated assistant audio remains true.

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

Streaming RNN-T decoding reuses the already-produced 1,024-wide encoder frame
and carries its prediction/LSTM state forward. It does not load a second ASR
model or run another FastConformer pass. The head runs whenever captions or
RNN-T turn-taking are enabled; `streamUserTranscript` controls only whether the
running transcript is attached to events. The small head stays on the shared
MLX inference stream because moving it to CPU forces a device synchronization
on every 80 ms frame. Use `.modelNative` with captions disabled to preserve the
original no-RNN-T benchmark path.
The caption is observational: the Nemotron-H conversation and function head
consume the audio embedding directly, not this text. On the fixed FLEURS
fixture, one-frame streaming and offline greedy RNN-T both produce exactly
`fellow wrestlers also paid tribute to luna`, which guards against streaming
token loss. The terminal also keeps punctuation emitted just after assistant
BOS on the preceding user line instead of showing the next line as `? What ...`.
Word substitutions on live accented or noisy speech remain RNN-T/acoustic
errors; the runtime does not rewrite them or use transcript matching to repair
tool arguments.

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
        iterations: 8),
    turnTaking: .nvidiaRealtime)
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
  --stream-user-transcript \
  --response-output response.wav
```

Use `--stream-user-transcript` to include the live demo's RNN-T caption cost in
the whole-pipeline latency and RTF figures. Omit it for the original duplex
generation baseline.

For a model-native tool cycle, pass a versioned scenario JSON containing the
synthesized audio path, reference transcript, available tool schemas, expected
tool, deterministic provider delay, and compact provider result:

```bash
./.build/release/voicechat-bench \
  --model /path/to/complete-bundle \
  --tool-scenario /path/to/list-reminders.json \
  --speech-iterations 1 \
  --output tool-latency.json \
  --response-output tool-response.wav
```

Reminder scenarios may also include `expected_arguments`. `required` checks
that a value is present, `absent` catches optional fields the user never spoke,
`equals` checks exact JSON values such as short IDs or booleans, and
`string_contains_words` checks semantic string content without requiring word
order or capitalization. Argument quality is evaluated only for a valid call to
the expected tool. A wrong tool or wrong argument set receives a synthetic
error rather than the expected tool's success payload, so a selection or
argument miss cannot be masked by an unrelated successful response.

Scenarios may additionally include `expected_reply` with
`contains_all_words`, `contains_any_words`, and `absent_words`. These checks
score the completed assistant reply by normalized words rather than exact text,
so natural phrasing remains valid while missing reminder names, unsupported
claims, or other semantically wrong spoken results are reported as
`wrong_reply`. These checks run only after inference in the benchmark; they do
not route reminder intents or change the live model's response.

Set `reply_required` to `false` for a first-step-only tool scenario. The runner
then stops after the expected provider result has synchronized instead of
waiting for speech. The reminder suite uses this for the lookup prerequisite of
named updates; executing and scoring the subsequent update call requires a
future multi-call fixture.

The first diversified 11-case reminder smoke passed 4 cases end to end: both
no-tool controls, the canonical one-item list request, and missing-ID
clarification. The remaining cases exposed prompt-dependent tool-selection
misses, incorrect date/priority arguments, or incomplete search terms. The
successful canonical list path measured 837 ms native decode, 478 ms result
sync, 3.570 seconds to sustained speech, and normal/tool/overall foreground RTF
of 0.79/0.71/0.77. A name-only create prompt initially showed 1.05 foreground
RTF under transient load; three isolated repeats measured 0.82 p50 and 0.92
maximum, so the earlier overrun was not sustained. These are one-run quality
smokes except for that three-run replication, not publication-grade accuracy
estimates.

The current 13-case suite adds two deterministic hesitation clips: one pauses
after a complete-looking date/time clause and one pauses inside a reminder
title. They distinguish a tool endpoint regression from the checkpoint's own
learned early BOS/function behavior. They do not authorize a transcript router
or host-side semantic completion rule.

This mode feeds one 80 ms microphone frame on each wall-clock deadline and
continues feeding live silence while the native function decoder, provider, and
tool-result synchronization run. The JSON report separates request-end to tool
start/completion, native JSON decode, provider service and actor-queue wait,
result synchronization, first assistant text, first PCM frame ready, first
sustained audible PCM, and reply completion. It also reports microphone service
p50/p95/max, separate normal/tool/overall foreground frame RTF, maximum
accumulated lag, and per-tool-phase microphone pressure. It also reports
wall-clock RTF over the complete paced timeline. Foreground RTF excludes
background function work. Because input is deliberately paced, wall-clock RTF
has a lower bound near `1.00×`; interpret it together with foreground RTF and
backlog rather than as a standalone pass/fail value.

The report also analyzes both the exact frames queued by live playback and one
continuous offline decode of the same generated codec history. It records
non-finite and clipped samples, active speech duration, internal silence runs of
at least 120 ms, isolated onset transients, and discontinuities at 80 ms frame
joins relative to ordinary within-frame sample changes. The onset detector
examines the generated lead-in plus the first 50 ms of sustained speech and
compares its sample jumps with the p99 jump during steady speech. A transient
is suspect only when it exceeds both `0.05` absolute amplitude and six times
that steady-speech baseline. Scenario-level `expected_audio` constraints can
require minimum active speech and cap the longest pause, clipping fraction,
suspect onset-transient count, or suspect join count; a violation is reported
as `audio_quality_regression`.
Comparing live and offline measurements helps isolate a playback-stitching
click from an artifact already present in the generated codes. These signal
checks catch deterministic regressions but are not a substitute for perceptual
listening or a learned speech-quality metric.

The corresponding scenario fields are
`minimum_active_speech_ms`, `maximum_internal_pause_ms`,
`maximum_clipped_sample_fraction`, `maximum_suspect_onset_transients`, and
`maximum_suspect_frame_boundaries`. Setting either suspect-count maximum to
zero turns any detected onset pop or frame-join discontinuity into a failed
quality gate.

When the scenario includes `audio_sha256`, the runner verifies the fixture before
loading the model. This keeps synthesized requests pinned across machines and
prevents an accidentally edited clip from producing a misleading comparison.

On an otherwise idle M5 Pro with 48 GB, three release runs of protected-head
INT5 with one EAR-TTS refinement step completed both the no-tool control and
native `list_reminders` path in every run, with 0% transcript WER. From input
end, the control's first sustained speech was 0.860/0.860 seconds p50/p95.
The tool path measured 0.833/0.843 seconds for native JSON decode,
0.210/0.212 seconds for the fixed provider, 0.561/0.563 seconds for result
synchronization, and 3.567/3.582 seconds to first sustained speech. Its p50
tool-added speech delay was 2.707 seconds; measured from request start, first
sustained tool speech was 5.087/5.102 seconds. Tool-path normal/tool/overall
foreground RTF p50 was 0.77/0.71/0.75, while the paced wall-clock RTF was 1.01.
Sustained throughput therefore remained real time. Result synchronization is
still bursty: the live-frame p95 was 103/114 ms p50/p95 across runs, producing
a maximum-backlog p50/p95 of 0.177/0.219 seconds (0.224 seconds worst case)
before recovery. The rolling tool RTF can therefore cross `1.00×` briefly even
though the complete tool phase stays below it. Run benchmarks without
concurrent GPU work: contention-distorted runs can cross `1.00×` even when the
isolated path does not.

First sustained PCM requires a complete 10 ms window above `-50 dBFS` by
default. That intentionally excludes an isolated codec-edge click from the
latency metric; the separate onset-transient analysis still records that click.
Change the sustained threshold with `--audibility-threshold-dbfs` only when the
threshold is recorded with the result. `--provider-delay-ms` can override the
fixture delay for a stress arm. The built-in provider is deterministic and has
no external side effects, so it measures runtime orchestration rather than
EventKit or network variance. Use `--speech-iterations 4` for the higher-detail
speech comparison; every report records the selected value. Use
`--function-call-endpoint-frames` only for controlled A/B tests of the native
candidate endpoint; interactive tool sessions default to eight frames.

On an M5 Pro with 48 GB, three release runs of protected-head INT5 with live
captions enabled each measured RTF `0.92`. Total/frame p95 was 74.8–75.8 ms,
first spoken text was available in 44.9–45.7 ms, first playable audio in
74.2–74.9 ms, and peak RSS was about 8.74 GB. The input transcript and generated
reply were identical across all three runs.

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
