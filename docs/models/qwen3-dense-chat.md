# Qwen3 Dense Chat (Qwen3-4B-Instruct)

On-device LLM for chat using a **standard dense Qwen3 transformer** on `mlx-swift`. It runs a **>1B
instruct model** on Apple Silicon behind the same `Qwen35ChatBackend` protocol the 0.8B hybrid uses,
so it is a drop-in swap.

This is the **dense** counterpart to the recurrent-attention [Qwen3.5-0.8B hybrid](qwen35-chat.md):
where that model trades a hybrid DeltaNet/attention design for tiny memory, this one is a plain,
parity-verified transformer chosen for **coherence and instruction-following** at the cost of more RAM.

> **Not pinned to one model.** The runtime is a generic dense-transformer reader: export any standard
> dense LLM (other Qwen3 sizes, Gemma, Llama, Granite) to int4/int5 MLX and load it the same way.
> `Qwen3-4B-Instruct-2507` is the current default.

## Architecture

A textbook decoder-only transformer (no hybrid layers, no gating):

- **Attention** — grouped-query (GQA), with **RMSNorm on Q and K** (per head) applied *before* RoPE.
- **RoPE** — full (all head dims), base `rope_theta`.
- **MLP** — SwiGLU (`down(silu(gate(x)) * up(x))`).
- **Residuals** — pre-norm RMSNorm; final RMSNorm before the LM head.
- **Embeddings** — tied (`lm_head` shares `embed_tokens`).

### Model dimensions (`Qwen3-4B-Instruct-2507`)

| Parameter | Value |
|-----------|-------|
| Hidden size | 2560 |
| Layers | 36 |
| Attention heads | 32 query / 8 key-value (GQA 4:1) |
| Head dim | 128 |
| MLP intermediate | 9728 (SwiGLU) |
| Vocab | 151,936 |
| RoPE theta | 5,000,000 |
| Context | 262,144 |
| Tied embeddings | Yes |
| RMSNorm eps | 1e-6 |

## Weight formats

### MLX (Mac GPU)

First-party exports (via `speech-models/export_mlx.py` → `mlx_lm.convert`), group size 64:

| Repo | Quant | Bits/weight | Size |
|------|-------|-------------|------|
| [`aufklarer/Qwen3-4B-Instruct-2507-MLX-5bit`](https://huggingface.co/aufklarer/Qwen3-4B-Instruct-2507-MLX-5bit) | int5 | ~5.5 | 2.78 GB |
| [`aufklarer/Qwen3-4B-Instruct-2507-MLX-4bit`](https://huggingface.co/aufklarer/Qwen3-4B-Instruct-2507-MLX-4bit) | int4 | ~4.5 | 2.28 GB |

**int5 is the default** (better quality at modestly more RAM); int4 is the smaller, lower-RAM variant.

Weights are safetensors with quantized linears (`weight` uint32, `scales`/`biases` bfloat16). The tied
embedding is read directly as a pre-quantized table (`PreQuantizedEmbedding`), whose packed last
dimension is `dimensions * bits / 32` — correct for 3/5/6-bit as well as 4/8-bit.

Key naming (HF `model.` prefix stripped on load):
- Attention: `layers.{i}.self_attn.{q_proj, k_proj, v_proj, o_proj, q_norm, k_norm}`
- MLP: `layers.{i}.mlp.{gate_proj, up_proj, down_proj}`
- Norms: `layers.{i}.{input_layernorm, post_attention_layernorm}`, final `norm`
- Embedding: `embed_tokens` (tied → also the LM head)

## Swift API

```swift
import Qwen3Chat

// Download + load (defaults to the int5 export; pass modelId: for int4 or another model)
let chat = try await Qwen3DenseChat.fromPretrained()

// Stream a reply
let messages = [
    ChatMessage(role: .system, content: "You are a helpful assistant."),
    ChatMessage(role: .user, content: "What is the capital of France?")
]
for try await chunk in chat.generateStream(
    messages: messages,
    sampling: ChatSamplingConfig(temperature: 0.6, topK: 50, topP: 0.9, maxTokens: 64)
) {
    print(chunk, terminator: "")
}

// Or load a local export directory (dev)
let dev = try Qwen3DenseChat.fromDirectory(URL(fileURLWithPath: "out/mlx/Qwen3-4B-Instruct-2507-MLX-5bit"))
```

`Qwen3DenseChat` conforms to `Qwen35ChatBackend`, so it slots into the same call sites as the 0.8B
hybrid backend.

## Token IDs (Qwen3 ChatML)

| Token | ID |
|-------|-----|
| `<\|im_start\|>` | 151644 |
| `<\|im_end\|>` | 151645 |
| `<\|endoftext\|>` | 151643 |
| `\n` | 198 |

The chat template is built by token id (`<\|im_start\|>{role}\n{content}<\|im_end\|>\n`), then the
assistant header is appended for generation.

## Numeric parity

The forward pass is validated against the Python `mlx_lm` reference: for a fixed prompt
it produces the **same next-token argmax and top-5 logits** before any generation is trusted
(`Tests/Qwen3ChatTests/Qwen3DenseParityTests`). Each quant has its own reference (int4 → argmax 358,
int5 → argmax 1096 for tokens `[9707, 11, 1879, 0]`).

## Conversion

- **MLX**: `speech-models/export_mlx.py --hf-path Qwen/Qwen3-4B-Instruct-2507 --q-bits {4,5}` —
  quantizes via `mlx_lm.convert`. Publish with `speech-models/publish_mlx.py {4,5}`.
