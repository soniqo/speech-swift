# Local decisions with Clef-flash

Add the `Clef` library product to the application's dependencies. Requires Apple Silicon and macOS 15+ (MLX); leave several GB of free disk space for the approximately 5.3 GB checkpoint and adequate unified memory for inference.

```swift
import Clef

let model = try await Clef.fromPretrained()
let result = try model.decide(
    state: "Please turn the kitchen lights on.",
    questions: [
        ClefQuestion(id: "action", instructions: "Which action was requested?",
                     kind: .choice(["on": "Turn lights on", "off": "Turn lights off", "other": "Other request"])),
        ClefQuestion(id: "urgent", instructions: "Is this urgent?", kind: .noul())
    ])
for decision in result.decisions {
    print(decision.id, decision.probabilities)
}
```

`Clef.load(from:)` loads an existing local export without network access. `fromPretrained(cacheDir:offlineMode:progressHandler:)` uses the package's model downloader and cache. The default model is `aufklarer/Clef-flash-9B-MLX-4bit`. The downloader follows the repository's current revision; for reproducibility download the pinned revision listed in the model document to a dedicated directory, then use `load(from:)`.

`selectedOption` and `confidence` describe the highest-probability option. For `.noul()`, read `noul` for the probability of true even when false wins. For `.score(["normal", "urgent"])`, `score` is the probability-weighted criterion index. The API never calls a tool.

## Command line

The main CLI accepts a JSON request and downloads the published model on first use:

```sh
speech clef decide request.json
speech clef decide request.json --model-dir /path/to/Clef-flash-9B-MLX-4bit
speech clef decide request.json --offline --max-tokens 4096
```

Output is JSON with `decisions`, `input_tokens`, `load_seconds`, and `decision_seconds`. `--offline` uses only the local cache. `--model-dir` loads an existing directory without downloading. `--max-tokens` accepts 1–16384 and defaults to 4096. The library's `ClefRequest` decodes this same JSON format and exposes `resolvedQuestions()` for applications.

For the smaller standalone build:

```sh
swift build -c release --product clef-decide --disable-sandbox
scripts/build_mlx_metallib.sh release
.build/release/clef-decide /path/to/Clef-flash-9B-MLX-4bit request.json
```

```json
{
  "state": "Please turn the kitchen lights on.",
  "questions": [
    {"id": "action", "type": "choice", "instructions": "Which action?", "choices": {"on": "Lights on", "off": "Lights off", "other": "Other"}},
    {"id": "urgent", "type": "noul", "instructions": "Is this urgent?"},
    {"id": "priority", "type": "score", "instructions": "Rate urgency", "levels": ["normal", "urgent"]}
  ]
}
```

The command emits JSON with probabilities, input token count, and `decision_seconds` (encoding through materialized probabilities, excluding model load). A single first inference includes warm-up and is not a steady-state latency benchmark. CLI input uses an ordered field array and a text state; it is not a drop-in SystemOne HTTP API.

## Validation

```sh
swift test -c release --no-parallel --filter ClefEncodingTests
swift test -c release --no-parallel --filter E2EClefHeadParityTests
CLEF_MODEL_DIR=/path/to/model swift test -c release --no-parallel --filter E2EClefTests
```

Run model tests and benchmarks sequentially. The head parity fixture uses the released PyTorch implementation with deterministic small tensors, independently checking logits and schema piece boundaries. Full model tests require local model files; they do not download weights implicitly.

## Memory and timing

On M5 Pro (48 GB), one loaded model handling a repeated 303-token request reached **5.11 GiB peak RSS** and **6.81 GiB peak macOS physical footprint**. The latter includes unified-memory costs that RSS misses. Twenty warmed runs averaged **291 ms**; model loading is excluded. Budget additional memory for longer inputs and the rest of your application. See [the measured report](../benchmarks/clef.md) for scope and methodology.

Timings vary with background activity: a later correctness run measured 462–897 ms while other applications were active.
