# GLiNER: local structured decisions and entity spans

The `GLiNER` library runs DeBERTa-based GLiNER2 span checkpoints directly in MLX Swift. It exposes single-label classification with all candidate probabilities and entity extraction with scores and UTF-16 source offsets. Inputs are plain text and a caller-supplied schema. No Python process is used during inference. The loader explicitly preserves added schema-token IDs; mapping these markers to the generic Unigram unknown token changes model predictions.

The first target is `fastino/GLiNER2.5-Decide`. The encoder and head equations follow [gliner2-mlx](https://github.com/Andrew-Chen-Wang/gliner2-mlx), copyright 2026 Andrew Chen Wang, MIT license; the license accompanies the module as `LICENSE-reference`. The upstream [GLiNER2 code](https://github.com/fastino-ai/GLiNER2) and [Decide weights](https://huggingface.co/fastino/GLiNER2.5-Decide) are Apache-2.0.

## Supported contract

- Span architecture, `count_lstm` head, first-subtoken word pooling.
- Shared content/position attention keys, relative position buckets, GELU, no convolution or absolute position embeddings.
- Single text at a time, at most 512 encoded tokens. Longer input is rejected instead of silently truncated.
- Nonempty, distinct labels (maximum 255). Entity scores default to threshold 0.5; overlapping spans are removed per label in descending score order.
- Output spans preserve source spelling. Offset units are UTF-16, matching `NSRange`; upstream Python uses Unicode code points.
- One instance must be used serially. Load once and reuse it.

This API does not implement joint constraints, relation graphs, record extraction, multi-label classification, training, or the GLiNER2.5 boundary architecture. Entity mentions do not resolve negation, normalize dates, or execute tools. A valid label can still be wrong; probabilities are model scores rather than guarantees.

## Weights

Three MLX conversions of revision `7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6` are published:

| Variant | Repository | Weights | Routing | Extraction | Peak memory | Max confidence drift |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `int8` (default) | `aufklarer/GLiNER2.5-Decide-340M-MLX-8bit` | 567 MB | 7.6 ms | 8.9 ms | 0.85 GB | 0.006 |
| `fp16` | `aufklarer/GLiNER2.5-Decide-340M-MLX-fp16` | 973 MB | 8.8 ms | 10.0 ms | 1.58 GB | 0.0012 |
| `fp32` | `aufklarer/GLiNER2.5-Decide-340M-MLX` | 1.95 GB | 11.1 ms | 12.6 ms | 2.55 GB | 0.0005 |

Apple M5 Pro, idle machine, one process per variant: median full request including tokenization (16 routing cases with six labels, 8 entity cases with two labels), peak process physical footprint, and the largest confidence difference from upstream PyTorch across 24 reference cases. Every variant makes the same decisions and returns the same spans and offsets as upstream on those cases. For comparison, Python gliner2-mlx 0.1.2 (FP32) measured 13.7 ms routing and 15.0 ms extraction in the same session. Details: [`docs/benchmarks/gliner-decide.md`](../benchmarks/gliner-decide.md).

Each bundle has `weights.safetensors`, `config.json`, `encoder_config/config.json`, the tokenizer files and `export.json` (source revision and weight SHA-256). `int8` quantizes the 145 encoder matrices and the token embeddings with MLX affine 8-bit weights, group size 64; heads, normalization, relative embeddings and activations stay FP16. Its `quantization.json` lists the packed tensors and is validated before loading.

## Swift

```swift
import GLiNER

let model = try await GLiNER.fromPretrained()   // int8; or GLiNER.load(from: directory)
let choices = try model.classify(
    "Remind me to call Dad at six PM.",
    labels: ["create_reminder", "send_message", "other"]
)
let entities = try model.extractEntities(
    "Remind me to call Dad at six PM.",
    labels: ["person", "time"]
)
```

`fromPretrained` downloads into `~/Library/Caches/qwen3-speech/` and reuses the cache; `offlineMode: true` loads only from it. The classification result preserves candidate order. Select its highest probability if your application accepts it. Missing entities return an empty array. Label descriptions may be passed explicitly to both methods.

## CLI

The `speech` binary exposes both tasks. Text may be passed as an argument or piped on stdin. Labels are comma-separated; `--description label=text` may be repeated and must name a listed label. Argument errors (duplicate or empty labels, unknown description labels, an unknown variant, a threshold outside 0...1, both `--model` and `--model-dir`) are rejected before the model loads.

```sh
speech gliner classify "Remind me to call Dad at six PM." \
  --labels create_reminder,send_message,set_timer,other
speech gliner extract "Remind me to call Dad at six PM." --variant int8 \
  --labels person,time \
  --description "time=Time of day or duration mentioned in the command" --json
```

`--variant` selects a published bundle (default `int8`), `--model` overrides its repository, and `--model-dir` loads a local directory instead. `classify` prints probabilities in descending order; `--json` returns `text`, `task`, `label`, `probability`, every entry of `choices` in the supplied order, and `metrics` (`load_ms`, `inference_ms`). `extract --json` returns `text`, `threshold`, `offset_units` (`utf16`), `entities` keyed by label, and `metrics`. `--task` (classify, default `action`), `--threshold` (extract, default 0.5) and `--evaluate-layers` map directly to the library parameters. Runtime errors print `Error: ...` to stdout and exit with status 1.

## Runtime notes

DeBERTa's disentangled attention projects the relative-position embeddings through each layer's query and key weights. Those projections depend only on weights, so the loader computes them once for all layers (about 100 MB at FP32, 50 MB at FP16 and INT8) instead of on every request, where they would cost 48 matrix products of 512 × 1024 × 1024. Each request then scores only the band of position buckets its length can reach. `testCachedPositionProjectionsMatchReference` holds the encoder to the original per-request formulation for short and log-bucketed lengths.

`GLiNER.load(from:evaluateLayers:)` accepts an optional layer-by-layer evaluation mode. The default remains false. Setting true materializes each encoder layer before building the next graph, allowing temporary attention and feed-forward arrays to be released earlier, at some latency cost.

The INT8 path keeps matrices packed and uses `quantizedMM` directly; token embeddings are dequantized only for the requested rows. The full encoder and vocabulary are never expanded to FP16.

## Benchmark

Build the release `gliner-bench` product and compile the MLX metallib before measuring. Supply a JSON array of `{text, action}` or `{text, entities}` cases. The fixed benchmark action labels and extraction schema are recorded in the executable source.

```sh
swift build -c release --product gliner-bench --disable-sandbox -j 4
scripts/build_mlx_metallib.sh release
.build/release/gliner-bench --model /path/to/bundle \
  --cases scripts/tests/fixtures/gliner/cases.json --output results.json
```

Reports contain individual full-request durations, all classification scores, extracted spans, expected-output matches, load duration, process peak RSS, sampled physical footprint, and MLX active, cache and peak allocations at loading, first request, warmed inference, cache clearing and unloading. Five warmups per task precede timed requests.

`scripts/profile_gliner_memory.py` runs every variant in its own process on an otherwise idle machine. It waits for other model jobs and compiler processes, discards a run if one starts, and compares each variant with the 24 upstream reference fixtures: identical labels, spans and offsets, with confidence drift at most 0.001 (FP32), 0.005 (FP16) or 0.02 (INT8). Pass `--output` a scratch directory; it writes per-variant JSON, `summary.json` and `REPORT.md` there. The published figures are in [`docs/benchmarks/gliner-decide.md`](../benchmarks/gliner-decide.md).

## Validation

`scripts/test_gliner.sh` runs a focused release test harness, reusing this worktree's build cache without building the other model test suites. Run it after other builds and model processes have stopped.

```sh
GLINER_MODEL_DIR=/path/to/bundle \
GLINER_REFERENCE_FILE=scripts/tests/fixtures/gliner/reference-fixtures.json \
scripts/test_gliner.sh
```

The E2E test checks exact token IDs and output labels/spans for all 24 fixtures, with a 0.001 absolute tolerance on scores. Without the model environment variable the E2E test is skipped; unit tests still run. The initial tokenizer integration failed parity because the generic Unigram model did not include GLiNER's added marker IDs in its base vocabulary; explicit added-token lookup fixes this.
