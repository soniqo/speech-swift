# Inkling 2 Mini validation

Selected local checks on an Apple M5 Pro with 48 GB unified memory, dated 2026-11-22. Raw transcripts and probe results remain in the linked model releases; this report records the method, findings and limits.


## INT4 acceptance

Date: 2026-11-22. Hardware: Apple M5 Pro, 48 GB unified memory.
Bundle: affine MLX INT4 experts / INT8 attention and vocabulary weights, group 64,
source `thinkingmachines/Inkling-2-Mini` at
`ed40e7da625f594fc9c4990f5f66be24873c22d7`.

### Result and scope

**25/25 fixed acceptance cases passed.** These selected cases establish that the
export and Swift runtime produce coherent, task-correct final replies on the
specified inputs. They are not a standard benchmark or a general intelligibility,
accuracy, multilingual, safety, or long-context guarantee. No comparison against
the complete original source-precision checkpoint was run.

| Coverage | Cases | Result |
| --- | ---: | ---: |
| Text: comprehension, arithmetic, negation, conversation memory, code reading/repair, JSON, prose, SQL, five non-English prompts | 20 | 20 passed |
| Images: counting, relative position, OCR, labeled chart | 4 | 4 passed |
| Longer document: retrieve city/year across a 2,032-token prompt | 1 | 1 passed |

The German reply translates the train's departure time. The Spanish reply identifies
an umbrella, the French reply identifies Wednesday, and the Chinese reply gives the
correct apple count. The Japanese reply says the library is closed today. This is
one prompt per language and does not establish parity with English.

We checked all final replies for meaning and coherence, alongside the deterministic
rules. The seasons explanation identifies axial tilt and changing sunlight. The
missing-information reply states that the passage does not provide Eva's birthday.
The generated SQL also ran against an isolated in-memory table and returned the
active users in name order. No final reply contained leaked framing markers or
Unicode replacement characters.

### Method

- Debug Swift build; one loaded model and one request at a time.
- Temperature 0, top-k 0, top-p 1, repetition penalty 1, thinking effort 0.
- Maximum 512 generated tokens, including hidden reasoning and framing.
- Context limit 8,192; prefill chunks 128; image budget 1,024 merged patches.
- Rules: expected-answer regex, JSON equality, requested word limit, forbidden-word
  checks, nonempty output, and no leaked framing/replacement characters.
- Four opaque synthetic PNG fixtures isolate counting, position, OCR, and chart
  labels. No natural-image benchmark was run.
- First-visible-text timing includes hidden reasoning. The run took 459.18 seconds;
  the median case took 15.19 seconds. These debug timings are not throughput or
  production latency benchmarks and came from a shared development machine.

Fixtures and rules:
`Tests/Qwen3ChatTests/Fixtures/Inkling2/quality-cases.json` and `quality-*.png`.
The corresponding XCTest class is `E2EInkling2QualityTests`.

```bash
INKLING2_MODEL_DIR=/path/to/Inkling-2-Mini-MLX-4bit \
  INKLING2_QUALITY_OUTPUT=/tmp/inkling-quality/results.json \
  swift test --filter E2EInkling2QualityTests --disable-sandbox
```

Use `INKLING2_QUALITY_FILTER=<id substring>` for a subset. Model tests run
sequentially and require the compiled MLX metallib.

### Evidence

- [Prompts, rules, final replies and timings](https://huggingface.co/aufklarer/Inkling-2-Mini-MLX-4bit/blob/1033fa524242140fcc7168094839cb34fb3549b7/validation/quality-results.json)
- [Semantic review and SQL check](https://huggingface.co/aufklarer/Inkling-2-Mini-MLX-4bit/blob/1033fa524242140fcc7168094839cb34fb3549b7/validation/quality-review.json)
- [CPU Core ML operator probe](https://huggingface.co/aufklarer/Inkling-2-Mini-MLX-4bit/blob/1033fa524242140fcc7168094839cb34fb3549b7/validation/coreml-probe-cpu.json)
- [CPU/GPU-allowed Core ML operator probe](https://huggingface.co/aufklarer/Inkling-2-Mini-MLX-4bit/blob/1033fa524242140fcc7168094839cb34fb3549b7/validation/coreml-probe-gpu.json)
- [CPU/Neural-Engine-allowed Core ML operator probe](https://huggingface.co/aufklarer/Inkling-2-Mini-MLX-4bit/blob/1033fa524242140fcc7168094839cb34fb3549b7/validation/coreml-probe-ane.json)

Earlier targeted validation also passed 20 CPU exporter tests, four source-sample
MLX conversion tests, seven configuration/template/image-layout tests, two CLI
argument tests, three small-reference tests, four full-weight runtime tests, and
two real CLI tests. Small-reference CPU logits match exactly; the GPU bound is
0.004. Full-weight top-20 scores had maximum absolute drift 0.4565 and normalized
RMS drift 1.29% against the Python reference, with the same top token. The Python
reference used MLX 0.32.4; the Swift dependency embeds MLX core 0.31.1.

## INT5 acceptance

Date: 2026-11-22. Hardware: Apple M5 Pro, 48 GB unified memory.
Bundle: affine MLX INT5 experts, INT8 attention/vocabulary, group 64.
The 1,623 output tensors contain 19,198,967,808 bytes in forty shards.

All source tensor hashes match the pinned original source tensors used for INT4:
`thinkingmachines/Inkling-2-Mini` at `ed40e7da625f594fc9c4990f5f66be24873c22d7`.
The INT5 export decodes that source independently.

### Functional result

**25/25 selected acceptance cases passed.** The corpus and generation settings
match the [INT4 evaluation](inkling2-mini-quality.md): twenty text cases,
four synthetic image cases, and a 2,032-token document retrieval case.
Both real CLI tests also passed: arithmetic and a PNG image's color.

We checked all final replies for semantic correctness and intelligibility. The
seasons explanation links axial tilt to sunlight angle and day length. The
missing-information reply avoids inventing a birthday. The tested German,
Spanish, French, Chinese, and Japanese inputs preserve their intended meaning.
The Japanese output says the library is closed today. One prompt per language
does not establish general multilingual quality.

The generated SQL ran on an isolated in-memory fixture and returned only active
users sorted by name. JSON, word-limit, framing, Unicode, and expected-answer
checks passed. The image replies were `3`, `Red square`, `MEET AT 4 PM`, and `B`.

### Numerical checks

Full-model BF16 routing amplifies differences in reductions and graph compilation.
The full-weight acceptance criteria bound maximum score drift relative to the
checkpoint's logit soft cap (3%), aggregate normalized RMS drift (2.5%), and
require the same top token. The small FP32 reference retains its separate,
much tighter equation checks.

The initial maximum absolute threshold of 0.6, calibrated on INT4, failed for
INT5. It was replaced with the cap-relative criterion above; the independent
normalized RMS and top-token checks were retained. This changed the test's
acceptance threshold, not the production inference math. The rebuilt tests
passed against both recorded reference cores, along with all three small-model
checks and all four full-weight generation/vision tests.

| Python reference core | Maximum absolute drift | Fraction of score cap (30) | Normalized RMS drift | Top token |
| --- | ---: | ---: | ---: | --- |
| MLX 0.32.4 | 0.704590 | 2.35% | 1.30% | Same (`4`) |
| MLX 0.31.1, matching Swift's embedded core | 0.870966 | 2.90% | 2.46% | Same (`4`) |

These are numerical tolerances for the tested prompt, not bit-exact full-model
parity or a comparison against the original source-precision model.

### Reproduction and evidence

```bash
INKLING2_MODEL_DIR=/path/to/Inkling-2-Mini-MLX-5bit \
  INKLING2_QUALITY_OUTPUT=/tmp/inkling-int5/quality-results.json \
  swift test --filter E2EInkling2QualityTests --disable-sandbox
```

Run model-loading jobs sequentially. Build the debug metallib first.
The run used a debug build, greedy decoding, effort 0, max 512 generated tokens,
8,192 context tokens, and 128-token prefill chunks. Runtime is recorded in the
transcript; this was a shared development machine with upload overlapping early
checks, so the timings are not a controlled comparison against INT4.

- [Inputs, rules, final replies and timings](https://huggingface.co/aufklarer/Inkling-2-Mini-MLX-5bit/blob/ab5eea1b3f1181dcc52ba89499243b26df167f9a/validation/quality-results.json)
- [Semantic review and SQL execution check](https://huggingface.co/aufklarer/Inkling-2-Mini-MLX-5bit/blob/ab5eea1b3f1181dcc52ba89499243b26df167f9a/validation/quality-review.json)

The selected cases do not establish general accuracy, long-context reliability,
multilingual parity, or performance. No natural-image benchmark was run.

## Local action selection: INT5

All eight fixed cases in `scripts/tests/fixtures/inkling2-decisions.json` passed through the real Swift CLI: two questions, summarization, an image, missing image input, ambiguity, an unsupported action and a request to override the schema. Nine CPU tests cover validation and invocation boundaries, including malformed and duplicate-key JSON, unknown actions, extra fields and process failures. No action was executed. These cases do not establish general routing accuracy or prompt-injection resistance. See [the example and reproduction steps](../inference/inkling2-mini-decisions.md).
