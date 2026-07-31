# Gemma 4 Chat Model

Swift MLX backend for Gemma 4 text checkpoints in the `Qwen3Chat` module. The public entry point is `Gemma4Chat`, which conforms to the same `Qwen35ChatBackend` protocol used by `Qwen35PipelineLLM`.

## Overview

`Gemma4Chat` loads Gemma 4 text weights from MLX safetensors exports such as:

- `aufklarer/gemma-4-E4B-it-MLX-4bit`
- `aufklarer/gemma-4-E2B-it-MLX-4bit`

The implementation targets the text tower from Gemma 4 multimodal-style configs. It is a hand-written Swift port of the Gemma 4 text architecture, not a wrapper around `mlx-swift-lm`.

## Architecture

Gemma 4 text differs from Qwen-style dense chat models in several important ways:

| Component | Gemma 4 behavior |
|---|---|
| Chat template | `<|turn>role\n...<turn|>\n`, with assistant role rendered as `model` |
| Tokenizer | SentencePiece-style tokenizer with byte fallback |
| Layer input | Per-layer embeddings projected from token ids |
| Norms | Sandwich RMSNorm blocks around attention and feed-forward stages |
| Attention | Sliding-attention and full-attention layers with different head dimensions |
| RoPE | Standard sliding RoPE plus proportional RoPE for full attention |
| KV sharing | Later layers reuse K/V from earlier producer layers of the same attention type |
| MLP | Double-wide MLP on KV-shared layers when configured |
| Logits | Tied LM head followed by final logit softcap |

The config parser reads the nested `text_config` block used by Gemma 4 exports and falls back to root-level fields for standalone text configs.

Quantization is resolved by `ChatQuantization`, searching the root and then `text_config`: the flat `quantization_bits` / `quantization_group_size` fields first, then the nested `"quantization": {"bits", "group_size"}` object, then a `"quantization": "int8"` label (which names no group size), then INT4 / 64 for checkpoints that state nothing. Every quantized layer is built from that answer, so a config stating only the flat fields no longer falls back to INT4 and loads a wider checkpoint into narrow layers.

## Chat Template

Gemma 4 does not use ChatML and does not use the older Gemma `<start_of_turn>` format. The runtime renders:

```text
<bos><|turn>system
...
<turn|>
<|turn>user
...
<turn|>
<|turn>model
```

Assistant messages in history are mapped to the `model` role. Generation starts after the final `<|turn>model\n` prompt.

## Reasoning Channel

Some Gemma 4 checkpoints may emit a thought channel before the user-visible answer:

```text
<|channel>thought
...
<channel|>
```

`Gemma4AnswerFilter` suppresses that channel by exact token id, drops special tokens, and streams only answer text. Text bytes are buffered until they form valid UTF-8, which prevents partial byte-fallback tokens from leaking mojibake into downstream TTS.

## Loading

```swift
import Qwen3Chat

let chat = try await Gemma4Chat.fromPretrained(
    modelId: "aufklarer/gemma-4-E4B-it-MLX-4bit"
)
```

For local or test exports:

```swift
let dir = URL(fileURLWithPath: "/path/to/gemma-4-E4B-it-MLX-4bit")
let chat = try Gemma4Chat.fromDirectory(dir)
```

Required files:

- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `generation_config.json`
- `model.safetensors` and/or `model.safetensors.index.json`

## Streaming

```swift
let messages = [
    ChatMessage(role: .system, content: "You are concise."),
    ChatMessage(role: .user, content: "What is the capital of France?")
]

let sampling = ChatSamplingConfig(
    temperature: 0.3,
    topK: 50,
    topP: 0.9,
    maxTokens: 80,
    repetitionPenalty: 1.05
)

for try await chunk in chat.generateStream(messages: messages, sampling: sampling) {
    print(chunk, terminator: "")
}
```

## Decode Path

Each decode step is one lazy MLX graph: the model forward, end-token suppression, the repetition penalty, top-K/top-P and the draw. Reading the sampled token id is the only point the graph is waited on, and the only thing that crosses the GPU boundary is that one `Int32`.

Sampling has two implementations of one set of rules:

| | `ChatSampler.sample` | `ChatSampler.sampleOnDevice` |
|---|---|---|
| Input | `[Float]` of `vocabSize` | `MLXArray` logits, already on device |
| Cost per token | the whole vocabulary copied to the host, twice more on arrival, then O(V) host scans | a scalar read |
| Used by | `Qwen35MLXChat`, `Qwen3DenseChat`, `Qwen35CoreMLChat`, `MLXGenerator` | `Gemma4Chat` |

Gemma 4's vocabulary is 262,144 entries, so the host form is a megabyte per generated token. The device form applies the same rules — suppression to `-greatestFiniteMagnitude`, the penalty's divide-when-positive/multiply-when-not over the last 64 ids, top-K by logit (equivalent to top-K by probability, since softmax is monotonic), the shortest cumulative prefix reaching top-P, and an inverse-CDF draw.

The win is small and worth stating as such. On an E4B INT4 checkpoint, decode is dominated by streaming 4.2 GB of weights, and sampling is a few percent of a token either way. Best of three runs over a fixed 64-token window with the prefill outside the clock:

| Context | Sampling | Host | Device |
|---:|---|---:|---:|
| 498 | greedy | 13.99 ms/tok | 13.35 ms/tok |
| 498 | temperature 0.7 | 13.85 ms/tok | 13.84 ms/tok |
| 9,990 | greedy | 20.24 ms/tok | 19.68 ms/tok |
| 9,990 | temperature 0.7 | 20.33 ms/tok | 20.30 ms/tok |

Greedy gains 3–5%; sampling breaks even, because ranking 262,144 entries with `argSort` costs about what the copy and the host's top-K heap did. Measure before assuming otherwise — an earlier version of the benchmark divided a total by a token count and reported 1.27x that was entirely prefill variance.

Nothing about the construction makes the two agree, so tests hold them together. `ChatSampler.sample(…uniform:)` takes the nucleus draw as a parameter rather than drawing it, which makes both paths pure functions of the same inputs and lets them be compared token-for-token over a sweep of draws.

The one divergence is inherent: neither path breaks equal probabilities by index — the host ranks candidates with `Array.sort`, the device with `argSort` — so on exactly equal logits they may return different tokens *of equal probability*. The lm_head emits bfloat16, whose eight mantissa bits make exact ties common in the tail of a nucleus; measured over 1,000 sampled steps of a real decode, six landed on such a tie and none differed for any other reason. Greedy decoding is unaffected, because `argMax` and the host's strict `>` scan both take the lowest index among equal maxima.

## Verification

The backend has both deterministic and E2E coverage:

- `ChatModelConfigTests` checks nested Gemma 4 config parsing and derived layer types.
- `ChatSamplerDeviceTests` compares the device sampler against the host one on constructed logits: greedy equality, tie-breaking, the repetition penalty, end-token suppression, and the top-K/top-P reachable sets, including an exactly representable top-P boundary.
- `E2EGemma4ParityTests` verifies next-token argmax against the `mlx_lm` reference.
- `E2EGemma4GenTests` verifies streaming generation, thought-channel suppression, and cache-path first-token parity.
- `E2EGemma4SamplingParityTests` runs the previous host-sampling loop and the current one against the real weights and requires identical greedy token sequences; `GEMMA4_BENCH=1` additionally reports ms/token for both at short and long contexts.

Reference parity prompt:

| Prompt tokens | Expected argmax |
|---|---:|
| `[818, 5279, 529, 7001, 563]` | `7001` |

## Source Files

```text
Sources/Qwen3Chat/
  Gemma4Chat.swift           Public loader, decode loop, streaming generation, answer filter, chat template
  Gemma4Model.swift          Gemma 4 text transformer layers and incremental state
  Gemma4Tokenizer.swift      SentencePiece byte-fallback tokenizer
  Gemma4WeightLoading.swift  MLX safetensors loader for Gemma 4 key layout
  ChatSampler.swift          Host sampler over a [Float] vocabulary
  ChatSamplerMLX.swift       The same rules as MLX ops, sampling without leaving the device
```
