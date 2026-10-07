# Qwen3-ASR Cache Regression Probe

Measured on 2026-10-03 to validate the memory policy in PR #498. This is a
focused repeated-request probe, not a dataset quality benchmark.

## Setup

- Apple M5 Pro, 48 GiB unified memory; macOS 26.6.2; Xcode 27.0.
- Release-compiled package modules, MLX Swift 0.31.6, swift-transformers 1.3.4,
  and compiled Metal shaders from the matching MLX checkout.
- Models: `aufklarer/Qwen3-ASR-0.6B-MLX-4bit` and
  `aufklarer/Qwen3-ASR-1.7B-MLX-8bit`, loaded from the local cache.
- Base: `ca382eec35c3675be9670081612e19ad9496f4c7`.
  Submitted PR: `c53f608fcbabed95f04af7284f90de314cff161d`.
  Fixed implementation: `5dd31eaa048f945ef21b3a9bbfe7bfb3e0dc6e0a`.
- Each revision/model runs alone in a fresh process. Only the Qwen3-ASR
  sources change between revisions; dependencies and other modules match.
- Input is `Tests/Qwen3ASRTests/Resources/test_audio.wav`, resampled to
  24 kHz. Trim to the first and last samples whose absolute amplitude
  exceeds 0.01, with 100 ms padding, then repeat that speech to form
  2.5, 5, 10, 20, 40, 7.5, 3.5, 15, 25, and 30 second clips.
- Two untimed 5 second warmups, then three cycles through those ten clips:
  30 measured requests per process. Use
  `transcribeCheckingCancellation(audio:sampleRate:options:)` with default
  `Qwen3DecodingOptions`, including automatic language detection and
  adaptive decoding. No HTTP or model-loading time is included.
- Measure elapsed time through transcription return. Synchronize the
  default MLX stream before idle memory samples. Physical footprint and
  its high-water mark come from `TASK_VM_INFO`; reusable cache comes from
  `MLX.Memory.snapshot()`. These are different memory measurements.

## First comparison

| Model | Revision | Mean request (ms) | Peak footprint (GiB) | Final footprint (GiB) | Final MLX cache (MiB) |
|---|---|---:|---:|---:|---:|
| 0.6B / 4-bit | Base | 109.2 | 7.05 | 7.04 | 6352.4 |
| 0.6B / 4-bit | Submitted PR | 121.5 | 2.74 | 1.79 | 3.0 |
| 0.6B / 4-bit | Fixed PR | 128.1 | 2.74 | 1.75 | 3.0 |
| 1.7B / 8-bit | Base | 267.7 | 7.57 | 6.57 | 4096.5 |
| 1.7B / 8-bit | Submitted PR | 269.3 | 4.94 | 4.50 | 4.5 |
| 1.7B / 8-bit | Fixed PR | 281.7 | 4.95 | 3.86 | 4.5 |

Every one of the 60 base-versus-fixed transcript comparisons matched
exactly, and all transcripts were nonempty. The caller's cache limit was
restored after unloading each model.

The fixed 0.6B run retained 1.75 GiB rather than 7.04 GiB after 30 requests.
Mean request time rose by 18.8 ms (17.2%). For 1.7B the final footprint fell
from 6.57 GiB to 3.86 GiB and mean request time rose by 14.0 ms (5.2%).
Clearing the buffer pool prevents reuse between requests; waiting for
submitted decoder work also ensures it cannot refill the pool after
cleanup. This memory policy has a latency cost relative to the base.

## Repeated comparison with the submitted PR

Two additional fresh-process runs per model and revision reversed the
submitted/fixed run order. Combined across three runs (90 requests each):

| Model | Submitted mean (ms) | Fixed mean (ms) | Submitted run means (ms) | Fixed run means (ms) |
|---|---:|---:|---|---|
| 0.6B / 4-bit | 129.9 | 132.6 | 121.5, 129.7, 138.5 | 128.1, 142.9, 126.9 |
| 1.7B / 8-bit | 290.0 | 289.0 | 269.3, 308.0, 292.8 | 281.7, 286.3, 299.0 |

The fixes are within observed run-to-run timing variation of the submitted
PR: +2.1% for 0.6B and -0.4% for 1.7B in the aggregate. All repeated
transcripts matched the corresponding base transcript. This does not
establish a speedup or prove equal latency on other workloads.

## Regression validation

- Full package build with tests passed.
- Local CI unit selection: 1,556 passed, 41 skipped, zero failures.
- Seven isolated Qwen3-ASR E2E classes: 21 passed, covering cache ownership,
  both decoder paths, cancellation, deterministic output, batch parity,
  and long-form transcription.
- Cache cleanup suite: ten additional fresh processes, all 50 executions
  passed, including cancellation with submitted GPU work and a scoped
  stream.
- Hosted `build-and-test` passed on the fixed implementation.

The repeated fixture exercises buffer reuse and varied sequence lengths.
It does not measure WER on diverse recordings, multi-model throughput, or
all memory configurations. Cache ceilings constrain reusable buffers,
not active weights or total process memory. Greedy batch transcription
uses the shared ceiling but does not flush its pool after each batch.

See [Qwen3-ASR inference](../inference/qwen3-asr-inference.md#mlx-cache-budget)
for the cache ownership and cleanup behavior. Raw per-run data remains
outside the repository.
