# Inkling 2 Mini: Core ML INT8 export and inference

The Core ML exporter and `Inkling2CoreMLChat` implement text generation and
still-image input through the `Qwen3Chat` library. Full-checkpoint conversion
has completed, with parity checks on all 27 partitions. All 25 fixed text/image acceptance
cases passed. All eight fixed decision cases also passed through
the release CLI. Download the complete
[Core ML INT8 bundle](https://huggingface.co/aufklarer/Inkling-2-Mini-CoreML-INT8)
from Hugging Face.
The published [MLX bundles](inkling2-mini.md) remain available.

INT8 describes weight storage. Core ML also supports floating-point weights;
INT8 is useful here because Inkling 2 Mini has 26.4 billion total parameters.
This exporter uses FP32 text activations and caches. Its FP16 decoder candidate
changed expert routing during the longer tiny-model regression test, so FP16
text execution is not the default.

## Partition layout and precision

The bundle contains 27 portable `.mlpackage` directories: an embedding table,
24 physical decoder layers, a vocabulary head, and the vision encoder. The
runtime reuses those decoder layers for 48 logical visits (`6 + 12 × 3 + 6`).
Each visit owns separate KV caches and four convolution histories. A request
starts with fresh state; model weights remain reusable across requests.

| Component | Storage and execution |
| --- | --- |
| Expert matrices | Symmetric INT8, groups of 64 input channels by default; FP16 scales; gather the selected experts before FP32 dequantization |
| Attention and vocabulary matrices | Core ML INT8 weight compression; FP32 execution |
| Routers, residual queries, controllers, convolution kernels | FP32 |
| Text activations, KV caches, convolution histories | FP32 |
| Vision matrices | FP16; normalization and attention accumulation use FP32; image features leave the encoder as FP32 |

Histories are explicit model inputs and outputs. This preserves FP32
convolution state without relying on native Core ML state tensors, which the
tested toolchain restricts to FP16. Single-token partition calls bound the
selected-expert workspace. Local KV histories retain the sliding window;
global histories grow up to the exported context limit.

## Download one representation

The release bundle contains portable packages and compiled partitions. Download
one representation to avoid storing both sets of weights. The compiled path is
the representation tested on macOS 26.6.2:

```bash
hf download aufklarer/Inkling-2-Mini-CoreML-INT8 \
  --local-dir /path/to/Inkling-2-Mini-CoreML-INT8 \
  --exclude '*.mlpackage/*'
```

For compilation on another Mac or OS, start with a fresh portable-only directory:

```bash
hf download aufklarer/Inkling-2-Mini-CoreML-INT8 \
  --local-dir /path/to/Inkling-2-Mini-CoreML-INT8-Portable \
  --exclude 'compiled/*'
```

Pass that directory to the same Swift API or CLI. The runtime compiles portable
packages into an owned temporary directory. Allow about 90 GiB for the portable
download, temporary compiled models and execution cache together. Other OS
versions and iOS remain unverified; local compilation does not establish support.

## Export locally

Tested conversion environment: Python 3.12, PyTorch 2.7.1, coremltools 9.0,
NumPy 2.2.6, and huggingface_hub 2.4.0 on Apple Silicon. Run conversion,
builds, and model-loading tests sequentially.

```bash
python3.12 -m venv /tmp/inkling-coreml-env
source /tmp/inkling-coreml-env/bin/activate
python -m pip install torch==2.7.1 coremltools==9.0 numpy==2.2.6 huggingface_hub==2.4.0
python scripts/prepare_inkling_coreml_bridge.py
python scripts/export_inkling_coreml.py \
  --output /path/to/Inkling-2-Mini-CoreML-INT8
```

Use an isolated Python 3.12 venv with CMake and Xcode installed. The bridge
helper builds Apple's [NumPy input lifetime fix](https://github.com/apple/coremltools/pull/2829)
at `a207b1a3452eafe02f2d870f486287834af2c82d` and installs it only in that venv.
The stable 9.0 Python bridge can otherwise crash asynchronously on recent
macOS versions while releasing input arrays. The converter remains at 9.0;
the bundle records the native bridge's source revision and binary hash.
Swift inference uses Apple's native framework directly.

The exporter streams tensor ranges from the immutable source revision
`thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7`.
It decodes the upstream NVFP4/MXFP8 weights directly, rounds decoded experts to
BF16 as in the MLX exporter, then quantizes them to INT8. It does not require an
MLX export on disk. Repacking upstream weights into INT8 does not recover
precision already removed by the source checkpoint's quantization.

Defaults are `--context-limit 8192`, `--prefill-size 1`,
`--image-patch-limit 1024`, and `--group-size 64`. Group sizes 32 and 128 are
also supported. `--source-dir` accepts a complete original source snapshot.
Use the same arguments with `--resume` after an interruption; the source
revision and exporter hashes must match completed partitions.

The conservative weight estimate is approximately 26.4 GiB. Conversion requires
that estimate plus 10 GiB of free space on a volume supporting copy-on-write.
When compilation needs separate physical copies, the initial requirement is
twice the estimate plus 10 GiB. Actual runtime memory includes decoded dense
matrices, selected experts, caches, and execution workspaces.

Core ML also creates a separate execution cache under the running process's
cache directory. This can occupy roughly another model's worth of storage;
allow about 65 GiB for a complete bundle and its execution cache. The initial
export check is a minimum staging-space check. Inactive compiler caches can be
cleared between interrupted/resumed conversion runs without removing completed
model partitions.

For an uninterrupted conversion, allow about 65 GiB free on APFS, or about
90 GiB when portable and compiled weights need separate physical copies.
The exporter stops below 4 GiB free and keeps completed partitions for resume.

The compiled partitions occupy 27.69 GB (25.79 GiB). On the tested 48 GB M5 Pro,
initialization took about 158 seconds and a short inventory question took
102 seconds to generate its final answer on CPU. This is an initial measurement;
full acceptance timings are recorded separately. Account for initialization
and hidden reasoning when measuring an application's response time.

Each partition is compiled and compared with its PyTorch module before being
recorded as complete. On APFS, identical compiled and portable weight files
share copy-on-write storage after their hashes match. `--no-compile` produces
portable packages without local compilation or parity checks. The manifest
records that distinction, source tensor hashes, tool versions, and file hashes.
Portable packages can be compiled on another Mac; compiled `.mlmodelc`
directories are optional local artifacts. The runtime owns temporary compiled
copies of portable packages and removes them when the model is released or
initialization fails. Those copies can need another model's worth of disk
space while the instance remains loaded. Use a fresh directory and download
only portable packages when testing on another OS; the supplied compiled
partitions have been tested on macOS 26.6.2.

## Swift and CLI

```swift
import Foundation
import Qwen3Chat

let model = try Inkling2CoreMLChat.fromDirectory(
    URL(fileURLWithPath: "/path/to/Inkling-2-Mini-CoreML-INT8"),
    contextLimit: 8192,
    computeUnits: .cpuOnly)
for try await text in model.generateStream(
    messages: [.init(role: .user, content: "What is 2 + 2?")]) {
    print(text, terminator: "")
}
```

The model conforms to `Qwen35ChatBackend`. It uses Inkling's native tokenizer,
message template, thinking-effort control, and visible-text filtering, together
with the shared sampler. Still images use `Inkling2Image` and the `images:`
argument, as in the [MLX examples](inkling2-mini.md). Concurrent generation on
one instance fails with `Inkling2Error.busy`. Stream termination requests
cooperative cancellation between partition calls.

```bash
speech inkling "What is 2 + 2?" \
  --backend coreml --coreml-compute-units cpu \
  --model-dir /path/to/Inkling-2-Mini-CoreML-INT8 \
  --temperature 0 --max-tokens 512
```

`--coreml-compute-units` accepts `cpu`, `gpu`, `ane`, or `all`; its default is
`cpu`. These options request eligible compute units. They do not establish
Neural Engine placement or a speed advantage. The shared Core ML loader also
honors `SPEECH_COREML_COMPUTE_UNITS`. The requested context must fit the exported
context limit. Image patches and hidden reasoning count toward token budgets.

The [decision example](inkling2-mini-decisions.md) accepts `--backend coreml`
and the same compute-unit choices. It validates a proposal against its action
allowlist before returning it.

## Validation

Before further INT8 quantization, the tiny PyTorch decoder matches four pinned
MLX logit references covering full prefill, continued decoding, sequential
decoding, and a 40-token grouped prompt. Maximum error is 0.00000175, and greedy
choices match. This checks the port's equations against an independent reference.

The deterministic tiny decoder covers every decoder component, all 16 logical
visits of its eight physical layers, cache growth, sliding eviction, fresh
requests, and separate call boundaries. Thirteen pipeline comparisons passed
with INT8 weights and FP32 activations: greedy choices matched, maximum logit
error was 0.0000195, and maximum state error was 0.000077. A separate small
complete vision encoder passed five image grids, including padding and changing
patch counts; maximum feature error was 0.000657. These structural checks do not
establish full-checkpoint answer quality.

```bash
# Fixture preparation requires MLX; it runs on the CPU without downloading weights.
python scripts/tests/prepare_inkling_coreml_fixture.py --output /tmp/inkling-coreml-fixture
python scripts/tests/test_inkling_coreml.py -v
python scripts/tests/e2e_inkling_coreml.py \
  --fixture-dir /tmp/inkling-coreml-fixture --output /tmp/inkling-coreml-tiny
python scripts/tests/e2e_inkling_coreml_vision.py --output /tmp/inkling-coreml-vision
INKLING2_COREML_TINY_DIR=/tmp/inkling-coreml-tiny \
  swift test --no-parallel --filter E2EInkling2CoreMLParityTests --disable-sandbox

# Run the same fixed 25-case corpus used for the published MLX variants.
INKLING2_QUALITY_BACKEND=coreml \
  INKLING2_MODEL_DIR=/path/to/Inkling-2-Mini-CoreML-INT8 \
  swift test --no-parallel --filter E2EInkling2QualityTests --disable-sandbox
```

All 25 fixed text/image cases passed on CPU across separate runs. A 2032-token
document returned `Oslo, 2028` in 3109.8 seconds (51.8 minutes), after model
initialization. The 24 short text/image cases took 71–309 seconds each, with a
median of 131.6 seconds. These debug measurements came from a shared 48 GB M5 Pro;
they are not a controlled throughput comparison. Two earlier document attempts
were interrupted by the disk-space guard. Their unfinished requests are excluded
from those timings; the original fixture completed after more space was reclaimed.

See [the validation report](../benchmarks/inkling2-mini.md#core-ml-int8) for method and scope.
The earlier [operator probe](../../scripts/probe_inkling_coreml.py)
is retained as a separate feasibility experiment; its compute-plan assignments
do not establish placement for this complete model.
