# Clef-flash local validation

Apple M5 Pro, 48 GB unified memory; native MLX Swift, release build with compiled Metal shaders. Checkpoint: `TrevorJS/clef-flash-mlx-4bit` at revision `6d4dc3ff7f43fba6065f1caef5c7a135315dfc8d`. Text-only, affine 4-bit groups of 64, float32 joint head. No remote decision API.

## Current result — 2026-10-03

The unchanged runtime averaged **about 276 ms** in each of two fresh runs of twenty warmed repeats, with a combined range of **274–284 ms**. This is one 303-token, three-field request on this Mac; model load is excluded. A separate earlier process measured **5.11 GiB peak RSS** and **6.81 GiB peak physical footprint**. The thirteen focused Clef tests, including the local full-model reference, passed; maximum reference probability difference was **0.0026641**. The existing Qwen3.5 chat path also matched `main` in the small prefill/decode comparison described below. These checks do not establish broad decision accuracy or full-checkpoint chat parity.

The following sections preserve the earlier measurements and the subsequent optimization and regression checks.

## Initial fidelity — 2026-10-02

Nine distinct focused tests passed: the eight-test suite including the locally loaded checkpoint (no skipped tests), followed by a two-test grouped/equal-head regression run after adding the equal-head case. The small PyTorch head fixture matches at 2e-5 tolerance on CPU and 5e-4 on Metal. The observed Metal discrepancy also occurs in Python MLX; CPU agrees within 2e-7.

The full-model fixture uses “Please turn the kitchen lights on.” with three fields: action (three choices), whether it is a factual question (noul), and urgency (two score levels). All **303 encoded token IDs match** the Python reference. Both implementations select lights on, not a factual question, and normal urgency. The largest absolute probability difference is approximately **0.00292**, within the test's 0.006 tolerance. This is a fidelity smoke test, not a quality benchmark.

## Initial timing — 2026-10-02

For the same three-field request, five sequential warmed native runs had a **0.918 s median**, approximately **0.921 s mean**, and a **0.916–0.930 s range**. Timing starts before schema encoding and ends after probabilities are materialized; it excludes model load. The first standalone run took 4.02 s. No percentile tail or cross-model performance conclusion is justified by five identical requests.

The implementation uses recurrent 64-token chunks. These measurements are local elapsed times, not the vendor's hosted inference figures. No memory benchmark or varied-input quality benchmark was performed.

## Reproduce

Build `clef-decide` in release mode, compile the Metal library, then run the request in `docs/inference/clef.md`. The exact three-field validation request and warm-run loop live in `Tests/ClefTests/ClefTests.swift`.

The focused suite was run through a temporary SwiftPM test harness depending on the local `Clef` product and linking the repository's test sources/resources. This avoids compiling unrelated model test targets; the full package test suite was not run. Set `CLEF_MODEL_DIR` to a local model directory to enable the real-checkpoint test. Run all model work sequentially.

## Fused recurrence update — 2026-10-03

Reusing MLXLMCommon's fused Metal gated-delta update and increasing the prefill chunk from 64 to 256 tokens reduces the same request's mean from **0.921 s to 0.325 s** (about **2.83× faster**, or **65% less elapsed time**). Weights, schema, input, and timing boundaries are unchanged. The earlier baseline used five repeats; the new run uses twenty, measured in separate runs on the same Mac, not interleaved.

| Measurement | Optimized runtime |
| --- | --- |
| Warm median | 0.324660 s |
| Warm mean | 0.324897 s |
| Warm range | 0.322045–0.330572 s |
| Repeats | 20, same 303-token request |
| Maximum probability difference from Python | 0.003041 (test tolerance 0.006) |

All three selected decisions remain unchanged. The intermediate fused implementation with 64-token chunks averaged approximately 0.394 s over five repeats. The old 4.02 s first-run number is historical and has not been remeasured for the optimized runtime.

Ten focused tests were executed; the full-model reference and twenty-repeat check passed. After strengthening the long-sequence regression to compare whole and split inference separately against the original implementation, all three backbone tests passed. The long synthetic sequence exposes an existing quantized-matmul batch-shape rounding difference; both optimized paths match their corresponding original paths within 1e-5. Small head parity tests and request encoding tests also passed. The full package suite was not run.

The CPU and unsupported head dimensions retain the original recurrence. The fused path keeps recurrent state in float32. No model weights were changed, and no broad quality conclusion follows from this one real-model request. Set `CLEF_BENCH_REPEATS=20` for the longer timing loop. Raw logs remain outside the repository in the local benchmark scratch directory.

## Native convolution and memory — 2026-10-03

The current runtime replaces per-token convolution windows with MLX grouped `conv1d` and uses 512-token chunks. The same 303-token request fits in one pass. All ten existing focused tests passed, including the full-model fixture; maximum probability difference from Python is **0.0026641**, with unchanged selected answers. A separate native process then measured twenty warmed requests:

| Measurement | Result |
| --- | --- |
| Warm mean | 0.290641 s |
| Warm median | 0.290458 s |
| Warm range | 0.280454–0.299195 s |
| RSS after loading | 4.904 GiB |
| Peak process RSS | 5.112 GiB (5,488,869,376 bytes) |
| Peak macOS physical footprint | 6.806 GiB (7,307,679,336 bytes) |
| MLX peak active allocation | 6.321 GiB |
| MLX allocation cache after warm runs | 1.433 GiB |

RSS and footprint were measured with Darwin task APIs and checked against `/usr/bin/time -l`. MLX allocation figures are separate views of the same memory, not numbers to add to RSS. macOS footprint better reflects unified-memory pressure than RSS alone. After clearing the MLX allocation cache, RSS was 4.926 GiB and footprint 6.419 GiB; peak figures are unchanged. Larger contexts and additional model instances can use more memory. This run used the default MLX allocation-cache policy and one loaded model, not a memory-constrained configuration.

Against the original 0.921-second mean this is about 3.17× faster (68% less elapsed time); against the previous 0.325-second mean it is about 11% less time. These were separate runs, not interleaved comparisons. A temporary phase profile placed roughly 266 ms in the backbone, 8 ms in the head, and 2 ms in encoding; profiling timings are not the primary benchmark. Raw measurements are in the local scratch files `rss-report.json` and `rss-time.log`.

### Integration verification and timing variability

The release `speech` binary built successfully. `speech clef decide` returned valid JSON with the expected three decisions, and executable checks rejected a missing request and out-of-range token limits. The final focused suite passed **12 tests**, including the shared JSON request decoder, token parity, head parity, and real-model checks. Full package tests were not run.

The final correctness run's five repeated timings were **0.462–0.897 s**, slower than the separate twenty-run memory probe. Immediately afterward, another local model process was active (about 31% CPU and 8.9% process memory), along with editor activity. This suggests contention but does not establish its exact cause; no controlled interference test was performed and no applications were stopped. Treat the 0.291-second mean as a measured run, not a guaranteed latency under background load. Runtime dependencies matched between the benchmark harness and main package: mlx-swift 0.31.6, mlx-swift-lm 3.31.4, swift-transformers 1.3.4.

Before opening the pull request, optimized DeltaNet execution was made an explicit initializer option. Clef enables it; existing Qwen3.5 chat callers retain their original execution defaults. The measured Clef execution path is unchanged.

## Regression audit and further latency trial — 2026-10-03

A temporary comparison harness loaded the Qwen3.5 implementation from `main` at `ca382ee` and this branch with identical weights. A small two-layer model with both linear and full attention was tested at 4-, 5-, and 8-bit quantization. Logits and recurrent states matched exactly for a five-token prefill followed by three decoding steps. This checks the existing chat execution path; it is not a full-checkpoint chat quality benchmark. A permanent test also verifies that model-level optimization requires explicit opt-in.

CI for the initial PR revision passed its debug build, Metal build, demo builds, and unit suite: 1,588 tests reported, 46 skipped, zero failures. The CLI's two new argument-parsing tests passed. Model-download E2E suites were excluded from that CI run.

After reverting the normalization trial, the thirteen focused Clef tests and the separate `main` comparison passed again. The full-model fixture retained all 303 reference token IDs, the same three selected answers, and a maximum probability difference of 0.0026641 (tolerance 0.006).

Two additional runs of the unchanged Clef runtime, each with twenty warmed repeats of the same 303-token request, averaged **276.03 ms** and **275.70 ms**. The combined observed range was **274.46–284.44 ms**. A trial using fused RMS normalization instead of the existing Q/K normalization averaged 396.08 ms and 283.80 ms in the corresponding runs. The order was current → trial → current → trial, in separate sequential processes. All selected answers matched. The trial did not demonstrate a speed improvement and was reverted; it is not part of the runtime. These timings exclude model load and do not replace the separate memory measurements above.
