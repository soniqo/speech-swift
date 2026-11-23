# Inkling 2 Mini: native MLX Swift inference

`Inkling2Chat` in the `Qwen3Chat` library loads affine MLX exports of
[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini).
It supports streaming text, multi-turn history, and still-image input on
Apple Silicon. Export the source weights with
[export_inkling_mlx.py](../../scripts/export_inkling_mlx.py) first.
The original ModelOpt NVFP4/MXFP8 checkpoint is not an affine MLX bundle.

Published affine bundles:

| Bundle | Expert precision | Attention / vocabulary |
| --- | --- | --- |
| [aufklarer/Inkling-2-Mini-MLX-4bit](https://huggingface.co/aufklarer/Inkling-2-Mini-MLX-4bit) | INT4 | INT8 |
| [aufklarer/Inkling-2-Mini-MLX-5bit](https://huggingface.co/aufklarer/Inkling-2-Mini-MLX-5bit) | INT5 | INT8 |

Each model card includes installation steps, conversion provenance, a source
patch, and its fixed acceptance-test transcripts.
The same runtime supports both precisions and reads their settings from the
bundle configuration. Choose the model ID or local directory for the desired
variant.

The CLI defaults to MLX. For the Core ML INT8 exporter and `Inkling2CoreMLChat`,
see [Core ML export and inference](inkling2-mini-coreml.md). Select it with
`speech inkling --backend coreml --model-dir /path/to/Inkling-2-Mini-CoreML-INT8`.

See the [architecture diagram](../models/inkling2-mini.md) and
[validated local decision example](inkling2-mini-decisions.md) for text/image
action selection and its limits.

```bash
hf download aufklarer/Inkling-2-Mini-MLX-4bit --local-dir /path/to/Inkling-2-Mini-MLX-4bit
```

```swift
import Foundation
import Qwen3Chat

let model = try Inkling2Chat.fromDirectory(
    URL(fileURLWithPath: "/path/to/Inkling-2-Mini-MLX-4bit"))
let messages = [ChatMessage(role: .user, content: "What is 2 + 2?")]
for try await text in model.generateStream(messages: messages) {
    print(text, terminator: "")
}
```

`fromPretrained(modelId:)` downloads an explicitly selected affine export and
accepts `cacheDir`, `offlineMode`, and `DownloadOptions`. There is no implicit
hosted inference call. Weight loading checks every required decoder tensor's
shape and precision, and rejects missing scales or incomplete shard indexes.
The model shares weights across loop passes and creates independent cache
state for each logical layer visit.

```bash
speech inkling "What is 2 + 2?" \
  --model-dir /path/to/Inkling-2-Mini-MLX-4bit \
  --temperature 0 --max-tokens 512

speech inkling "Describe this image." \
  --model-dir /path/to/Inkling-2-Mini-MLX-4bit \
  --image photo.png --max-tokens 512
```

The CLI accepts `--system`, `--context-limit` (default 8,192), `--max-tokens`
(default 256), `--temperature` (default 1), and `--reasoning-effort` in
0…0.99 (default 0, requesting a direct answer). It uses top-p 1 and no
repetition penalty. The Swift interface accepts `ChatSamplingConfig`; its
shared package defaults differ from the CLI defaults. Structured response
formats are rejected because this backend does not implement constrained decoding.

Inkling's chat format uses native message/content markers and a thinking-effort
system turn. The model emits its own assistant header. The runtime streams only
text content, preserving partial UTF-8 sequences across token boundaries while
suppressing reasoning and structural markers. Native tool-call parsing is not
implemented; the low-level model and `nextTokenLogits(promptTokens:)` expose
scores for callers implementing custom generation.

The generation limit includes hidden reasoning and framing tokens. Even with
effort 0, the model may reason before answering; a very short token budget can
end before any visible text is emitted.

Still images attach to the last user turn:

```swift
let image = try Inkling2Image.load(URL(fileURLWithPath: "photo.png"))
let stream = model.generateStream(
    messages: [.init(role: .user, content: "Describe this image.")],
    reasoningEffort: 0,
    images: [image])
for try await text in stream { print(text, terminator: "") }
```

`Inkling2Image(width:height:rgb:)` also accepts packed RGB bytes. The processor
pads to 40-pixel boundaries and allows up to 1,024 merged patches per image by
default. Oversized images must be resized by the caller. Image tokens count
toward the request's context budget. The encoder validates required vision
tensors before inference. Video and audio input are not supported by this API.

Requests on one instance must be serialized; an overlapping request fails with
`Inkling2Error.busy`. Each request has fresh state and uses 128-token prefill
chunks. Terminating a stream requests cooperative cancellation. Model lifetimes
participate in the package's shared MLX buffer-cache coordinator. The default
context budget bounds request history; active weights and global KV history
still require memory beyond the exported file size. DSpark draft execution is
not implemented, even if an export includes `mtp.safetensors`.

The backend conforms to `Qwen35ChatBackend`, so an `Inkling2Chat` instance can
be passed to `Qwen35PipelineLLM(model:)` for a speech pipeline. The shared protocol
name does not change Inkling's native template or attention architecture.

Validation consists of CPU-only configuration, template, image-layout, argument,
and conversion tests; a small pinned-reference model exercising full and cached
logits with mixed INT5/INT8 weights; and opt-in full-weight generation tests.
Run model-loading tests sequentially, with a compiled MLX metallib.

```bash
swift test --filter Inkling2ConfigTests --disable-sandbox
swift test --filter E2EInkling2ParityTests --disable-sandbox
INKLING2_MODEL_DIR=/path/to/Inkling-2-Mini-MLX-4bit \
  swift test --filter E2EInkling2GenerationTests --disable-sandbox
SPEECH_CLI="$PWD/.build/debug/speech" \
  INKLING2_MODEL_DIR=/path/to/Inkling-2-Mini-MLX-4bit \
  swift test --filter E2EInklingCLITests --disable-sandbox
```

Regenerate the small fixture with `mlx-vlm==0.8.0`, `mlx`, `numpy`, and `pillow`:

```bash
python scripts/tests/generate_inkling_reference.py \
  --output Tests/Qwen3ChatTests/Fixtures/Inkling2
```

The generator fetches the reference Python implementation at an immutable
revision and performs local inference on deterministic small weights. It does
not download the full checkpoint or use hosted inference. The model's complete
architecture and source precision are documented in
[the model reference](../models/inkling2-mini.md).

The dated [INT4 quality report](../benchmarks/inkling2-mini.md#int4-acceptance) and
[INT5 quality report](../benchmarks/inkling2-mini.md#int5-acceptance) each record
25 selected text, language, image, and long-context checks. Run the acceptance
corpus with `INKLING2_MODEL_DIR` set and `swift test --filter E2EInkling2QualityTests`.
These checks do not establish general accuracy or multilingual parity.
