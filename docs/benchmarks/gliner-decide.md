# GLiNER2.5-Decide benchmark

Apple M5 Pro, 48 GiB, macOS, idle machine (2026-09-27). Model: `fastino/GLiNER2.5-Decide` at revision `7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6`, published MLX conversions `aufklarer/GLiNER2.5-Decide-340M-MLX[-fp16|-8bit]`.

## Results

| Runtime | Routing median | Extraction median | Peak process memory | Max confidence drift vs upstream |
| --- | ---: | ---: | ---: | ---: |
| Swift, INT8 (default) | 7.64 ms | 8.87 ms | 0.85 GB | 0.0060 |
| Swift, FP16 | 8.75 ms | 9.95 ms | 1.58 GB | 0.0012 |
| Swift, FP32 | 11.10 ms | 12.57 ms | 2.55 GB | 0.0005 |
| Python gliner2-mlx 0.1.2, FP32 | 13.68 ms | 14.96 ms | | reference path |

- **Timing:** median full request including tokenization, model loaded once. 16 routing cases with six labels and 8 extraction cases with two labels, five warmups, then five timed calls per case. Swift p95 is within 0.5 ms of the median. Python ran in the same idle window with the same revision.
- **Memory:** peak physical footprint of the whole process, sampled every 20 ms, including the relative-position projections cached at load. It is not a minimum device requirement.
- **Fidelity:** every variant returns the same labels, spans and offsets as upstream PyTorch on 24 reference fixtures. Drift is the largest absolute confidence difference; the gates are 0.001 (FP32), 0.005 (FP16) and 0.02 (INT8).
- **Behavior:** all runtimes match 12/16 routing and 7/8 extraction expectations on the handwritten set. "Do not set a timer." routes to `set_timer`; "Call Alice, not Bob, at six PM" returns only Alice as a person. This set checks behavior, not broad accuracy.

## Memory options

A 64 MiB MLX cache limit lowers peak memory by 18% for FP16 and 7% for FP32 at unchanged speed, and barely changes INT8. Layerwise evaluation (`evaluateLayers: true`) lowers it by a further 4 to 8% but adds 5 to 6 ms per request. Neither is the default.

| Variant | Default | Cache limit 64 MiB | Cache limit + layerwise |
| --- | ---: | ---: | ---: |
| INT8 | 0.85 GB, 7.6 ms | 0.85 GB, 7.7 ms | 0.78 GB, 13.0 ms |
| FP16 | 1.58 GB, 8.8 ms | 1.30 GB, 8.9 ms | 1.24 GB, 13.9 ms |
| FP32 | 2.55 GB, 11.1 ms | 2.37 GB, 11.2 ms | 2.25 GB, 17.4 ms |

Times are routing medians.

## Reproduce

```sh
swift build -c release --product gliner-bench --disable-sandbox
scripts/build_mlx_metallib.sh release
python3 scripts/profile_gliner_memory.py \
  --fp32 <fp32-bundle> --fp16 <fp16-bundle> --int8 <int8-bundle> --output <scratch-dir>
```

The profiler runs one process per configuration, waits for other model jobs and compilers, discards contended runs, and checks each variant against `scripts/tests/fixtures/gliner/reference-fixtures.json`. Bundles are the `fromPretrained` cache directories under `~/Library/Caches/qwen3-speech/models/aufklarer/`.
