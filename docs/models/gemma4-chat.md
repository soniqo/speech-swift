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

## Verification

The backend has both deterministic and E2E coverage:

- `ChatModelConfigTests` checks nested Gemma 4 config parsing and derived layer types.
- `E2EGemma4ParityTests` verifies next-token argmax against the `mlx_lm` reference.
- `E2EGemma4GenTests` verifies streaming generation, thought-channel suppression, and cache-path first-token parity.

Reference parity prompt:

| Prompt tokens | Expected argmax |
|---|---:|
| `[818, 5279, 529, 7001, 563]` | `7001` |

## Source Files

```text
Sources/Qwen3Chat/
  Gemma4Chat.swift           Public loader, streaming generation, answer filter, chat template
  Gemma4Model.swift          Gemma 4 text transformer layers and incremental state
  Gemma4Tokenizer.swift      SentencePiece byte-fallback tokenizer
  Gemma4WeightLoading.swift  MLX safetensors loader for Gemma 4 key layout
```
