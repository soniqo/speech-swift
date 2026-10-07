# Clef-flash

`Clef` runs the text-only Clef-flash 9B decision model locally with MLX on Apple Silicon. It scores a caller's allowed answers jointly in one backbone prefill; it does not generate chat text or execute actions.

The default checkpoint is [aufklarer/Clef-flash-9B-MLX-4bit](https://huggingface.co/aufklarer/Clef-flash-9B-MLX-4bit), an affine 4-bit, group-64 conversion of [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash). The initial published revision is `92a850bc10a355a69799e63fe87aa7545c551c74`; `export.json` records source revisions and weight checksums. Download size is approximately 5.3 GB. Both the original and conversion are Apache-2.0; the license is included with the module.

## Architecture

The Qwen3.5 backbone has 32 layers, a 4096-wide hidden state, three gated DeltaNet layers for each full attention layer, 16 linear key heads, and 32 linear value heads. Query/key heads repeat to match value heads. The runtime retains normalized hidden states, without computing vocabulary logits.

The joint head pools question and option spans, attends to the full encoded state, and lets fields interact through transformer decoder layers. A lexical prior uses rows of the **separate output embedding**; input embeddings are not interchangeable with it. Each field produces a softmax distribution over its allowed options. The head runs in float32.

Schema encoding follows the released reference's piece boundaries and option ordering. Choice IDs sort lexicographically. Score options retain array order. Boolean (`noul`) options are true then false. Questions retain caller array order. The backbone processes 512-token chunks with the MLX fused Metal recurrence to bound intermediate memory while preserving states and span pooling across chunks. Clef opts into `Qwen35MLXModel(config:optimizedDeltaNet:)`; existing Qwen3.5 chat callers retain the original execution path by default.

## Scope

- Text states and text descriptions only; no images, video, or audio input.
- Choice, noul (probability of true), and score (expected zero-based criterion index).
- Default limit 4096 encoded tokens, configurable up to 16384. Oversized requests fail; state is never silently truncated.
- One instance must be used serially. No batching or streaming decoder is provided.
- The 27B Clef model and other quantization formats are not supported.
- Scores are model probabilities, not calibrated guarantees or permission to execute a command.
- Vendor-hosted latency is not a local Apple Silicon measurement.

See [usage](../inference/clef.md) and `Tests/ClefTests` for schema, reference-head, grouped-head recurrence, and opt-in local model tests.
