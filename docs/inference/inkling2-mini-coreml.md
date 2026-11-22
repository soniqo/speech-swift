# Inkling 2 Mini: Core ML feasibility

Status as of 2026-11-22: operator feasibility demonstrated; **no full-checkpoint
Core ML exporter or Swift Core ML backend is shipped**. The production runtime
and published bundle use MLX.

## What was converted and checked

`scripts/probe_inkling_coreml.py` constructs a synthetic width-64, eight-expert,
top-two MoE. It implements sigmoid routing with selection bias, dynamic expert
gather, softcapped SwiGLU, shared weights repeated three times, and three separate
convolution histories. It does not load Inkling weights or implement attention,
vision, the tokenizer, or complete decoder blocks.

On an Apple M5 Pro, with coremltools 9.0, PyTorch 2.7.1, and NumPy 2.2.6:

| Probe | Result |
| --- | --- |
| FP16 weights, three native FP16 states, four sequential calls and a fresh-state check | Passed; maximum output error versus eager PyTorch 0.0004883 |
| INT4 compressed expert weights, same state checks | Passed; outputs matched the same weights decompressed in Core ML exactly |
| CPU/GPU-allowed and CPU/Neural-Engine-allowed loads | Passed; compute plans selected CPU for this small graph |
| FP32 native convolution states | Conversion rejected: native state type only supports FP16 in this tested toolchain |
| FP32 convolution histories as ordinary inputs/outputs, FP16 weights, four sequential calls | Passed; output error at most 0.0002442, all three histories retained FP32 dtype |

The INT4 toy package is 106,422 bytes versus 250,547 bytes for FP16. It contains
three compressed-weight operations, reused across nine expert gathers, plus
three independent state reads/writes. This establishes compressed serialization
and shared weights, not full-model runtime memory savings. Quantizing the toy
weights changes its outputs: normalized RMS error versus the original FP16
network was approximately 4.8–5.1%. The INT4 parity reference uses the same
quantized weights decompressed, so that quantization error is not mistaken for a
converter error.

Compute-plan assignments describe placement for this toy graph. They neither
prove Neural Engine acceleration nor predict full-size placement or speed.
No performance comparison against the complete MLX model was run.

## Work required for a complete port

The full decoder has 128 experts, width 2,304, forty-eight logical layer visits,
learned relative-position attention, attention sinks, residual attention,
message controllers, and a vision encoder. It requires caches per logical visit
while sharing the physical weights. These paths need full conversion and
real-weight numerical/generation tests.

The MLX implementation retains FP32 convolution histories. A Core ML design can
keep those as explicit FP32 model inputs/outputs, as the probe demonstrates.
Changing them to FP16 native states would require a separate quality study.
The model also stores BF16-sensitive parameters; their Core ML representation
and compute precision need explicit decisions and parity checks.

A practical next step is one real decoder block with explicit FP32 convolution
history, bounded attention history, and compressed experts, followed by a
complete loop and full-model text/image acceptance tests. Keep the published
MLX runtime as the usable backend until those checks pass.

## Reproduce

Use a separate environment and run these sequentially, after other inference
jobs finish:

```bash
python -m pip install coremltools==9.0 torch==2.7.1 'numpy<2.3'
python scripts/probe_inkling_coreml.py --output /tmp/inkling-coreml-cpu --device cpu
python scripts/probe_inkling_coreml.py --output /tmp/inkling-coreml-gpu --device gpu
python scripts/probe_inkling_coreml.py --output /tmp/inkling-coreml-ane --device ane
python scripts/probe_inkling_coreml.py --output /tmp/inkling-coreml-explicit \
  --device cpu --state-dtype float32 --explicit-state
```

The unsupported native-state case is reproducible with `--state-dtype float32`
without `--explicit-state`; it exits with a conversion error and writes a report.

Reports: [CPU](https://huggingface.co/aufklarer/Inkling-2-Mini-MLX-4bit/blob/1033fa524242140fcc7168094839cb34fb3549b7/validation/coreml-probe-cpu.json),
[GPU allowed](https://huggingface.co/aufklarer/Inkling-2-Mini-MLX-4bit/blob/1033fa524242140fcc7168094839cb34fb3549b7/validation/coreml-probe-gpu.json),
[Neural Engine allowed](https://huggingface.co/aufklarer/Inkling-2-Mini-MLX-4bit/blob/1033fa524242140fcc7168094839cb34fb3549b7/validation/coreml-probe-ane.json),
[native FP32 failure](https://huggingface.co/aufklarer/Inkling-2-Mini-MLX-4bit/blob/1033fa524242140fcc7168094839cb34fb3549b7/validation/coreml-probe-fp32-state.json), and
[explicit FP32 success](https://huggingface.co/aufklarer/Inkling-2-Mini-MLX-4bit/blob/1033fa524242140fcc7168094839cb34fb3549b7/validation/coreml-probe-explicit-fp32.json).
Apple documents [stateful models](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)
and [weight compression](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html).
