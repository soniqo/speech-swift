# Inworld Voice Isolator v1 benchmarks

The offline checkpoint has 64,047,112 parameters. The layouts use the
same Float32 weights and eight-second windows with a two-second crossfade.
The original tiled layout grouped independent rows in eight bands or
96 frames while keeping each row's full sequence. [Inworld model card](https://github.com/inworld-ai/voice-isolator/blob/c73b3fab81a870410cbac50d2f05f25edc4d10aa/MODEL_CARD.md)

## Speed and memory

Measured on an Apple M5 Pro with 48 GB RAM and macOS 26.6.2, using the Swift
inference sources built with `swiftc -O`. Each layout ran the same eleven-second
golden input three times in a separate process. RTF is processing time divided
by audio duration, with the median of the last two passes reported as warm.
The timer includes DSP, chunking, and inference. Peak memory is the process
physical footprint reported by `/usr/bin/time -l` across the three passes.

| Offline layout | Warm RTF | Peak physical memory | Peak RSS |
| --- | ---: | ---: | ---: |
| Original flexible stages | 0.414 | 6.66 GB | 3.92 GB |
| Fixed-shape full-window stages | 0.333 | 5.26 GB | 2.78 GB |
| Tiled with fixed-shape scan projections | 0.382 | 1.70 GB | 0.92 GB |

The tiled layout used 68% less peak physical memory than the fixed-shape
full-window layout. All three produced 528,000 samples and matched the
eleven-second golden within `3.6e-7` maximum absolute sample error. [Upstream golden fixtures](https://github.com/inworld-ai/voice-isolator/tree/c73b3fab81a870410cbac50d2f05f25edc4d10aa/tests/fixtures)

## Profiling

The compact bundle splits the eleven-second fixture into two eight-second
model windows. In the optimized Swift build, the mean time spent in each
phase across two warm runs was as follows.

| Phase | Seconds per recording | Share of wall time |
| --- | ---: | ---: |
| Core ML predictions | 3.06 | 75.7% |
| Core ML stage loading | 0.55 | 13.7% |
| Metal scans, including command submission and waits | 0.16 | 3.9% |
| Packing and restoring tile rows | 0.14 | 3.4% |
| Forward and inverse STFT | 0.03 | 0.7% |

Metal command-buffer timestamps attribute 0.089 seconds to GPU scan
execution. Task VM peak sampling attributed the largest physical memory peak
to the Core ML merge after a time-axis Mamba scan.

Specializing full attention and merge tiles reduced warm RTF from 0.367 to
0.316 and peak physical memory from 1.64 GB to 1.45 GB in the same profile.
The exported bundle retains flexible functions for smaller shapes. A separate
run without phase instrumentation measured 0.321 warm RTF, 0.420 first-pass
RTF, and 1.43 GB peak physical memory. Both offline golden fixtures passed
through the Swift CLI with a maximum absolute sample error of `3.6e-7`.

The specialized run still made 304 predictions and 44 stage loads. Mamba
projection stages, which include the depthwise convolutions, accounted for
29.5% of warm wall time. Mamba merge stages took 14.2%, attention stages 26.1%,
and loading 14.7%. The two eight-second windows overlap by five seconds on
this fixture, so the network processes sixteen seconds of input.
A single linear projection for both Mamba directions produced identical
output to the baseline bundle. Warm RTF was 0.314 versus 0.313 for the
baseline, and peak physical memory was 1.44 GB versus 1.43 GB.

### Offline window scheduling

Processing adjacent windows in pairs allows the Swift runtime to load each of
the 22 stages once for the pair. Both windows retain the same eight-second
context and are crossfaded at the original positions. In the prototype
comparison, the paired runner measured 0.265 warm RTF versus 0.313 for one
window at a time. Peak physical memory moved from 1.43 GB to 1.52 GB. One
earlier timed pass encountered a Core ML execution cache recompile. The
integrated Swift runtime produced all 528,000 samples identically to the
one-window runner. Both compact bundle generations passed the 3- and
11-second Swift CLI golden tests. Swift compilation was active during the
integrated runtime's timed run.

### Offline tile sizes

On the same eleven-second fixture, six time rows reduced the maximum scan
workspace and 89 band rows divided the 801 frames into nine full tiles.
The checkpoint's context, Float32 arithmetic, and waveform crossfade
were held constant. RTF is the median of the last two of three passes.

| Time rows | Band rows | Warm RTF | Peak physical memory |
| ---: | ---: | ---: | ---: |
| 8 | 96 | 0.260 | 1.52 GB |
| 12 | 116 | 0.263 | 2.04 GB |
| 6 | 89 | 0.260 | 1.24 GB |

The new tiled export reduced peak memory by 18%. Its maximum sample error
across both offline fixtures was `3.6e-7`.
The same comparison with the shared Swift scan used by both runtimes
measured 1.52 GB and 1.25 GB peak respectively. The smaller layout's warm
RTF was 0.268 in that run.

### Streaming attention

The streaming Core ML validation runner spent 76% of processing time in its
four attention stages on the eleven-second fixture. Passing fewer unused
key/value cache entries cut the total size of the cache tensors presented to
Core ML from 128 GB to 93 GB. In the same profile, RTF dropped from 1.99 to
1.75 and peak physical memory from 4.02 GB to 3.95 GB. A separate run of the
updated Python validator measured 1.77 RTF on the eleven-second fixture, with
both streaming fixtures within `2.8e-7` maximum absolute sample error.

A separate comparison used the same eleven-second fixture with 500 ms input
pushes. A maximum of sixteen frames per prediction reduced the four
attention stages from 364 to 112 calls each and reduced key/value tensor
inputs from 92.95 GB to 29.18 GB. In that comparison, RTF fell from 2.45 to
1.26 and peak physical memory rose from 3.94 GB to 5.37 GB. The maximum sample
error was `3.0e-7`.

At four frames per prediction, applying one shared softmax over separate
cached and new scores avoided concatenating the key and value activations.
Peak physical memory fell from 3.94–3.95 GB to 3.75–3.76 GB. RTF was 2.31
versus 2.45 in the first pair and 2.16 versus 2.15 with the order reversed.
Maximum sample error was `2.8e-7` on both streaming fixtures.

### Stateful streaming cache

The optional stateful Core ML export keeps the key/value tensors in Float16
state across calls. It updates the slots for new frames and retains Float32
attention, model weights, and scans. The Python runner supplies the cache
position, rotary tables, and mask. The normal four-frame prototype and the
explicit-cache export used the same eleven-second audio, 500 ms pushes, and
20 ms look-ahead on the M5 Pro.

| Four-frame streaming path | RTF | Peak physical memory |
| --- | ---: | ---: |
| Explicit cache inputs | 2.21 | 3.91 GB |
| Stateful cache | 1.12 | 2.13 GB |

The stateful run's maximum sample error was `1.10e-5` with a `1e-4` golden
tolerance. Both the four- and eight-frame compiled exports passed the
eleven-second golden with stagewise validation. The eight-frame export also
passed with 80 ms input pushes. These runs held one model stage at a time
for output validation.

### Streaming block size and precision

An isolated comparison of the two stateful bundles used 80 ms input pushes
on the eleven-second fixture, with Float32 attention and weights, Float16
cache storage, and 20 ms of look-ahead. RTF fell from 0.928 with four-frame
blocks to 0.633 with eight-frame blocks. Peak physical memory rose from 2.18 GB
to 2.42 GB. The 95th percentile push time fell from 120.5 ms to 85.4 ms, and
pushes exceeding their audio duration fell from 40 to 17 of 138. The two
renders differed by `2.04e-6` maximum absolute sample error.
The same block-size comparison on the six FLEURS pairs used 500 ms pushes.
Noisy-input SI-SDR changed by −0.00001 to +0.00003 dB, and clean-input SDR
changed by −0.00235 to +0.00029 dB. The largest sample difference was
`2.73e-5` across the twelve renders.

The precision comparison used the same eight-frame scheduler with 500 ms
input pushes. Each timed variant ran in its own process. Bundle sizes are
decimal megabytes. The golden limit was `1e-4` maximum absolute sample error.

| Linear layers | Compiled bundle | RTF | Peak physical memory | Max. error across both goldens |
| --- | ---: | ---: | ---: | ---: |
| Float32 baseline | 234 MB | 0.693 | 2.35 GB | `1.07e-5` |
| Float16 compute in the five middle stages | 179 MB | 0.698 | 2.30 GB | `2.57e-5` |
| Float16 weight storage in the three frequency stages | 215 MB | 0.807 | 2.26 GB | `1.08e-5` |
| Float16 compute for all large linears | 118 MB | — | — | `1.54e-4` |
| INT8 per-channel weights for all large linears | 61 MB | — | — | `3.84e-3` |

With INT8 applied only to the three frequency stages, the eleven-second
maximum error was `8.49e-4`. The Float16 weight-storage variant across all
stages encountered a Core ML runtime error. The mixed Float16 compute
variant kept the encoder and decoder stages in Float32. Across the six FLEURS
pairs described below, its noisy-input SI-SDR changed by −0.00192 to
+0.00001 dB, and clean-input SDR by −0.00847 to +0.00755 dB. Clean-input
SDR compares the output with the original waveform at its original gain.
Storing the three frequency stages' weights in Float16 changed noisy SI-SDR
by +0.00002 to +0.00030 dB and clean SDR by +0.00040 to +0.00820 dB.
With 80 ms pushes, the mixed Float16 compute variant measured 0.643 RTF and
2.37 GB peak physical memory, with 7 of 138 pushes exceeding their input
duration in that run. The same run series measured 0.635 RTF, 2.42 GB peak,
and 15 of 138 late pushes for the Float32 baseline.

### Native Core ML streaming

The Swift session loads the same eight-frame stateful export as the Python
validator. It reuses the rotary tables across stages and keeps the Mamba
recurrence in shared Metal buffers. The sample rate was 48 kHz and look-ahead
was 20 ms. Timings include feature extraction, Core ML, the scan, synthesis,
and flush. The p95 is the duration of a push call.

| Runtime on the 11-second fixture | Audio per push | RTF | Push p95 | Late pushes | Peak physical memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Python validator | 20 ms | 1.533 | 48.9 ms | 549 / 550 | 1.97 GB |
| Native Swift | 20 ms | 1.217 | 40.0 ms | 215 / 550 | 1.42 GB |
| Native Swift | 80 ms | 0.614 | 85.3 ms | 8 / 138 | 1.78 GB |

The native 20 ms run repeated at 1.227 RTF and 39.6 ms p95. Both reference
clips passed at 20 ms pushes with a maximum absolute sample error of
`1.12e-5`. The eleven-second clip exercises two-lane crossfades and exceeds
the 20 ms budget at p95.

On the six-language FLEURS sample set below, the native session used 80 ms
pushes and the Python validator used 500 ms pushes with the same eight-frame
block limit. Noisy-input SI-SDR changed by −0.00002 to +0.00004 dB, and
clean-input SDR changed by −0.00184 to +0.00588 dB. Maximum sample difference
across all twelve renders was `6.8e-5`.

Batched band projections, a frame-major state cache, and folding the Mamba
output projections preserved the golden tolerance. Their steady-state speed
was close to the existing graphs, so the benchmarked Swift path uses the
existing graph layouts.

### Batched streaming prototype

The second Swift harness uses fixed one- and two-frame Core ML functions.
During a crossfade it sends both lanes through one prediction per stage and
shares the initial spectrum projection. The time-axis recurrence runs
inside the Core ML stages and uses Float32 carried state. Metal updates
the cached attention slots and computes the time attention with Float32
accumulation. The key/value cache uses Float16 storage.

| Runtime on the 11-second fixture | RTF | p95 at 20 ms per push | Late pushes | Peak physical memory |
| --- | ---: | ---: | ---: | ---: |
| Existing native Swift path | 1.279 | 43.1 ms | 255 / 550 | 1.44 GB |
| Batched prototype, first run | 0.786 | 14.2 ms | 4 / 550 | 1.83 GB |
| Batched prototype, repeat | 0.660 | 14.5 ms | 8 / 550 | 1.82 GB |

A 33-second render repeating the test audio had 0.647 RTF, 14.8 ms p95,
and 4 late pushes out of 1,650. Its maximum sample difference from the
existing Swift stream was `5.9e-6`. The three- and eleven-second golden
renders passed with maximum absolute sample error at most `1.11e-5`. The
three-second run on first use measured 1.183 RTF and 12.1 ms p95.

Across the six-language FLEURS sample set below, noisy-input SI-SDR changed
by −0.00002 to +0.00000 dB and clean-input SDR changed by −0.00217 to
+0.00332 dB. The maximum sample difference across all twelve renders
was `5.22e-5`. The prototype used 20 ms input pushes and the earlier Python
validator renders used 500 ms input pushes with eight-frame blocks.

### Streaming memory

The paired harness originally reserved 800 key/value slots per lane. An
800-frame retiring lane and its 200-frame successor fit in a shared pool of
1,000 slots. Slot ownership changes when the old lane retires. Attention
traverses the same logical frame order using Float16 storage and Float32
accumulation. The pool reduced GPU cache allocation from 603 MB to 377 MB.

The lowest-memory run also used one two-lane, two-frame Core ML function per
stage. Inactive lanes and missing frames are masked in attention. A padded
packet advances the convolution history and recurrent state only for its
real input. This lets a single-frame tail use the already loaded function.

| Runtime on the 11-second fixture | RTF | p95 at 20 ms per push | Late pushes | Peak physical memory |
| --- | ---: | ---: | ---: | ---: |
| Paired with separate caches | 0.764 | 16.4 ms | 8 / 550 | 1.85 GB |
| Shared FP16 cache | 0.725 | 14.9 ms | 5 / 550 | 1.62 GB |
| Shared FP16 cache and one Core ML specialization | 0.734 | 16.1 ms | 5 / 550 | 0.94 GB |

The separate-cache run sampled 1.48 GB just before flush and 1.85 GB after
loading the single-frame tail function and flushing. The one-function run
sampled 0.94 GB during inference and through flush. The shared pool produced bit-identical output on
the 11-second and 33-second renders. Combining it with one specialization
changed the 11-second output by at most `2.46e-6` and passed the 3- and
11-second upstream fixtures. With 10 ms pushes on the three-second fixture,
the same path passed the waveform limit and measured 14.7 ms p95.

An Int8 key/value cache with Float32 scales reduced the cache allocation to
200 MB and measured 1.43 GB peak with the original set of Core ML functions.
Its upstream sample error was `4.42e-4` with a `1e-4` tolerance.

### Swift low-memory streaming

The following runs use the public Swift session, compiled Core ML stages,
and `.cpuAndGPU` on the M5 Pro. Each process uses 20 ms pushes and 20 ms of
look-ahead. RTF includes the flush and excludes model loading. Peak physical
memory covers the entire process.

| Export | Test audio | RTF | Push p95 | Late pushes | Peak physical memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stateful | 11 s | 1.549 | 58.6 ms | 406 / 550 | 1.43 GB |
| Low memory | 11 s | 0.685 | 14.4 ms | 4 / 550 | 0.93 GB |
| Low memory | 33 s | 0.750 | 20.7 ms | 173 / 1,650 | 0.93 GB |

The 11-second render differed from the upstream golden by at most `1.11e-5`.
The 33-second test repeats the same input three times to exercise cache
reuse. Its output differed from the earlier paired runner by at most
`2.64e-6`. A three-second golden run with 10 ms pushes reached `8.66e-6`
maximum error and took 17.6 ms at p95.

On the six FLEURS clips below, 20 ms pushes changed noisy-input SI-SDR by
−0.00001 to +0.00001 dB and clean-input SDR by −0.00216 to +0.00334 dB
relative to the eight-frame Float32 stateful export. The largest sample
difference was `5.71e-5` across clean and noisy inputs.

### Streaming RTF profile

An instrumented run of the 33-second input used 1,650 pushes of 20 ms each.
It took 25.32 seconds (0.767 RTF). The timings below separate the wall time
around each Metal command from its GPU execution time.

| Phase | Wall time | Share |
| --- | ---: | ---: |
| Core ML predictions | 20.76 s | 82.0% |
| Metal time attention | 2.75 s | 10.9% |
| Metal frequency scans | 1.04 s | 4.1% |
| Time recurrence state copies | 0.53 s | 2.1% |
| STFT analysis and synthesis | 0.07 s | 0.3% |

The three stages preparing the frequency scans and the final decoder stage
used 16.29 seconds, or 64.3% of total processing time. The attention and
frequency scan commands spent 1.56 and 0.19 seconds, respectively, on the
GPU. Crossfade pushes had a 27.1 ms p95 and missed 160 of 500 deadlines.
Single-window pushes had a 17.0 ms p95 and missed 20 of 1,149 deadlines.

The following runs used the same 33-second input and cadence. The Float16
trial converted linear operations with at least 262,144 weights in stages
1, 3, 5, and 7. Time recurrence and the attention cache used their existing
precision. The fixed one- and two-lane trial loaded a separate specialization
for each active lane count and reused the first projection.

| Trial in run order | RTF | Push p95 | Late pushes | Peak physical memory |
| --- | ---: | ---: | ---: | ---: |
| Current Float32, before | 0.697 | 14.9 ms | 3 / 1,650 | 0.931 GB |
| Selected Float16 linear operations | 0.722 | 16.1 ms | 18 / 1,650 | 0.839 GB |
| Current Float32, after | 0.883 | 26.2 ms | 412 / 1,650 | 0.932 GB |
| Fixed one- and two-lane models | 0.665 | 16.8 ms | 5 / 1,650 | 1.237 GB |

The Float16 trial used about 10% less peak memory. Its RTF fell within
the spread of the Float32 controls. The 3- and 11-second upstream goldens
passed with a maximum sample error of `3.49e-5`. On the 33-second input,
the Float16 and fixed-lane trials differed from the current Float32
render by at most `3.72e-5` and `2.64e-6`, respectively.

With 11 seconds of audio, the dynamic one- and two-lane graph used
0.882 GB and ran at 1.141 RTF. Core ML rejected direct output backings for
its dynamic tensors, so the runtime had to transfer them into the Metal
workspaces. The batched band projections and the revised time recurrence
passed the waveform check. Their timings fell within the Float32
variation on repeat.

### Packed streaming projections

One Float32 trial concatenated the forward and backward frequency Mamba
input weights. It also folded the adjacent time and frequency output
projections into single linear operations. A second trial grouped the
directional convolution and the smaller projections as well. Both kept the
same Core ML stage interfaces.

The 11-second screening used 20 ms pushes and the upstream waveform
fixture. Every render in the table passed the `1e-4` sample tolerance.

| Trial | RTF | Push p95 | Peak physical memory |
| --- | ---: | ---: | ---: |
| Current Float32 | 0.685 | 14.4 ms | 0.918 GB |
| Packed input and folded output projections | 0.676 | 13.9 ms | 0.941 GB |
| Packed directional convolution and projections | 1.067 | 26.2 ms | 0.936 GB |
| Time recurrence in Metal | 0.849 | 28.5 ms | 0.897 GB |
| Float16 transformer blocks with Float32 normalization | 0.762 | 14.6 ms | 0.880 GB |

In a second comparison, the current and packed-projection sessions
alternated call order on each 20 ms push. They measured 0.698 and 0.702
RTF, respectively, and each missed two of 550 deadlines. The outputs
differed by at most `5.91e-6`. The isolated screening measured memory.

Moving the time recurrence to Metal reduced recurrent state copies. This
layout added three Core ML stages and three Metal dispatches per push.
The added scans took about 0.66 ms per push on the host, including 0.15 ms
on the GPU.

The Float16 transformer trial kept the Mamba recurrence in Float32. Its
maximum sample error was `4.63e-5` on the 11-second fixture.

A fixed-shape conditional graph also compiled with one- and two-lane
branches. The three- and eleven-second renders produced nonfinite samples
and failed output validation.

### Fused attention and packed cache

One Metal dispatch writes both frames' cache entries and evaluates both
queries. The queries reuse the loaded keys and values in each head. The
packed trial stores each key or value component in 12 bits with a
Float32 scale per head and frame. Its four cache layers reserve 294.4 MB,
down from 376.8 MB with Float16.

| 11-second stream, 20 ms pushes | RTF | Push p95 | Peak physical memory |
| --- | ---: | ---: | ---: |
| Original Float16 cache, adjacent control | 0.693 | 14.6 ms | 0.919 GB |
| Fused attention, Float16 cache | 0.687 | 14.4 ms | 0.932 GB |
| Fused attention, 12-bit cache | 0.686 | 14.6 ms | 0.836 GB |
| 12-bit cache and selected Float16 linear layers | 0.732 | 17.7 ms | 0.751 GB |

The 12-bit cache passed the three- and eleven-second upstream fixtures with
a maximum sample error of `2.90e-5`. Its 33-second run measured 0.711 RTF
and 0.849 GB peak with at most `3.15e-5` difference from the Float16-cache
render. The attention calls took 1.29 ms per push on the 11-second stream,
down from 1.44 ms in the adjacent control.

### Fixed lane specialization

At a 20 ms cadence, a one-lane Core ML graph runs outside the crossfade
and a two-lane graph runs during it. Both graphs use the same spectrum
encoder and packed Metal cache. Converting selected large linear layers
to Float16 offsets the memory cost of retaining both graph shapes.

| 33-second stream, 20 ms pushes | RTF | Push p95 | Late pushes | Peak physical memory |
| --- | ---: | ---: | ---: | ---: |
| Original, before | 0.711 | 16.0 ms | 49 / 1,650 | 0.932 GB |
| Fast specialization | 0.567 | 14.1 ms | 10 / 1,650 | 0.971 GB |
| Original, after | 0.688 | 14.9 ms | 6 / 1,650 | 0.932 GB |

The 11-second fast trial measured 0.614 RTF, 14.2 ms p95, and 0.975 GB
peak. The three- and eleven-second upstream fixtures passed with at most
`7.41e-5` absolute sample error. The 33-second output differed from the
original by at most `7.59e-5`.

The original compiled `--fast-streaming` export reproduced the gain through
the public Swift API. The 11-second render measured 0.570 RTF, 13.7 ms
p95, and 0.965 GB peak. The 33-second render measured 0.569 RTF, 14.3 ms
p95, and 0.981 GB peak, with 37 of 1,650 pushes exceeding 20 ms.
The compiled low-memory export measured 0.686 RTF and 0.850 GB on
the eleven-second fixture.

For the compiled low-memory export, the six FLEURS clips changed
noisy-input SI-SDR by −0.00024 to +0.00027 dB. The largest difference
across the clean and noisy renders was `9.70e-5` per sample.

Across the six FLEURS clips, noisy-input SI-SDR changed by −0.00013 to
+0.00065 dB and clean-input SDR by −0.00164 to +0.03676 dB relative to
the original Swift output. The largest absolute sample difference was
`4.59e-4`.

### Fast streaming stage profile

The original compiled fast export ran the 33-second, 20 ms push test at 0.594 RTF
under instrumentation, compared with 0.569 RTF in the earlier uninstrumented
run. It used 0.990 GB peak physical memory and produced a byte-identical
WAV. The table uses wall times and includes the final flush.

| Phase | Time | Share |
| --- | ---: | ---: |
| Core ML predictions | 15.69 s | 80.0% |
| Metal time attention | 2.34 s | 12.0% |
| Metal frequency scans | 0.93 s | 4.7% |
| Recurrence state copies | 0.33 s | 1.7% |
| STFT analysis and synthesis | 0.07 s | 0.4% |

Stages 1, 3, and 5 prepare the frequency scans and together took 8.45 s.
Stage 7, the final attention and spectral decoder, was the largest
individual stage at 3.27 s. The four stages used 59.8% of total processing
time. The native attention and frequency scan commands spent 1.21 s and
0.14 s on the GPU, respectively, and used their output backings directly
on all 1,651 blocks.

The Core ML compute plan preferred the GPU for the mapped operations in
stages 1 and 7 in both lane shapes. A separate Metal hardware trace
confirmed GPU execution and showed seven GPU command buffers per decoder
prediction. Each scan preparation stage used four with one lane and five
with two lanes. The decoder graph contains 99 linear operations, 92 of
which belong to the 46 band-specific output heads.
[Apple Core ML compute plans](https://developer.apple.com/documentation/coreml/analyzing-a-core-ml-model-s-performance-in-xcode)

The 58.2M-parameter checkpoint applies its shared projections across
46 bands. The eight stages ran 13,208 predictions, about 400 per second
of audio. Counting dense linear operations from the compiled graph gives
3.95 billion multiply–accumulates per two-frame block with one lane and
7.82 billion with two lanes. The count covers linear operations only.

| Push type | Count | Mean | p95 | Pushes over 20 ms |
| --- | ---: | ---: | ---: | ---: |
| One active lane | 1,149 | 10.8 ms | 12.5 ms | 6 |
| Two active lanes during crossfade | 500 | 14.5 ms | 15.9 ms | 8 |

The first push buffered audio. The following push took 172 ms, including
158 ms in Core ML. The first crossfade push took 103 ms, including 99 ms
in Core ML. In a diagnostic run, one silent pass through each graph shape
before processing took 225 ms. It brought those two pushes to 15.4 and
17.0 ms and produced the same WAV after resetting streaming state.

### Metal decoder and folded projections

The two-frame fast export composes the frequency scan output projections during
conversion. The final Core ML stage now returns the features used by the
46 band heads. Metal runs the heads in one command. The cache encoding,
20 ms look-ahead, and 20 ms push cadence match the preceding fast export.

Each row is a separate process on the M5 Pro. The 33-second input repeats
the eleven-second fixture three times.

| Export | Audio | RTF | Push p95 | Pushes over 20 ms | Peak physical memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous fast, before | 33 s | 0.556 | 14.1 ms | 5 / 1,650 | 0.984 GB |
| Metal decoder and folded projections | 33 s | 0.503 | 13.4 ms | 13 / 1,650 | 0.895 GB |
| Previous fast, after | 33 s | 0.554 | 13.8 ms | 5 / 1,650 | 0.971 GB |
| Metal decoder and folded projections | 11 s | 0.492 | 13.1 ms | 4 / 550 | 0.881 GB |

An all-Core ML trial with batched band heads and the same folded projections
measured 0.511 RTF and 0.958 GB on the 33-second input.

The optimized export passed the three- and eleven-second upstream golden
fixtures through the Swift API. Maximum absolute sample error was
`8.37e-5` against a tolerance of `1e-4`. The 33-second waveform differed
from the preceding fast export by at most `6.68e-5` per sample.

On the six FLEURS clips below, noisy-input SI-SDR changed by
−0.00027 to +0.00004 dB and clean-input SDR by −0.02411 to +0.00276 dB
relative to the preceding fast export. The largest absolute sample
difference was `2.97e-4`.

### Four-frame streaming prototype

At 40 ms per push, a four-frame graph with a packed cache measured
0.577 RTF and 0.903 GB on the eleven-second fixture. The two-frame
controls at the same cadence measured 0.678–0.681 RTF and 0.918 GB.
The four-frame prototype ran the 33-second stream at 0.506 RTF and
0.899 GB, with 24.6 ms p95 against a 40 ms push interval. Both upstream
goldens passed within `3.09e-5` maximum sample error.

### 80 ms streaming blocks

The fast export defaults to eight STFT frames per prediction. At 48 kHz,
3,840 input samples per push fill an 80 ms block. All runs below use 20 ms
of model look-ahead, CPU and GPU compute, and the same Apple M5 Pro.
RTF excludes model loading and includes the flush.

| Model block | Input push | Audio | RTF | Push p95 | Late pushes | Peak physical memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 ms | 20 ms | 33 s | 0.512 | 13.8 ms | 3 / 1,650 | 0.895 GB |
| 20 ms, original runtime | 80 ms | 33 s | 0.530 | 65.2 ms | 2 / 413 | 0.887 GB |
| 80 ms | 80 ms | 33 s | 0.292 | 31.4 ms | 2 / 413 | 1.141 GB |
| 40 ms | 40 ms | 11 s | 0.441 | 22.3 ms | 2 / 275 | 0.961 GB |
| 80 ms | 80 ms | 11 s | 0.288 | 33.3 ms | 2 / 138 | 1.119 GB |

Earlier eight-frame runs of the 33-second input measured 0.284–0.290 RTF.
Two-frame runs with 80 ms pushes in the final runtime varied from 0.614 to
0.717 RTF. The 20 ms export at its matching push cadence produced a
byte-identical waveform to the previous build. The eight-frame render
differed from it by at most `8.02e-5` per sample on the 33-second input.

The eight-frame export passed the upstream goldens at both regular and
irregular push sizes with a maximum absolute sample error of `7.30e-5`.
On the six FLEURS clips, noisy-input SI-SDR changed by −0.00015 to
+0.00025 dB and clean-input SDR by −0.00459 to +0.00411 dB against the
two-frame fast export.

### Profile and memory at 80 ms

An instrumented run of the previous eight-frame runtime processed the
33-second input in 7.00 seconds. These times are measured around calls
and include submission and waits for Metal commands.

| Phase | Time | Share |
| --- | ---: | ---: |
| Core ML attention and Mamba preparation, stages 1, 3, and 5 | 3.65 s | 52% |
| Other Core ML stages | 1.51 s | 22% |
| Metal time attention | 1.10 s | 16% |
| Metal frequency scans | 0.31 s | 4% |
| Metal decoder | 0.20 s | 3% |

Physical footprint was 0.860 GB before the first crossfade and 1.128 GB
after its first two-lane prediction. With the one-lane graph and shared
encoder trial, it was 0.739 GB on both sides of that transition. The
first crossfade push took 96.8 ms in the previous profile and 42.3 ms
in the reuse trial. The packed key/value pool reserves 294.4 MB.

The updated runtime evaluates each crossfade lane through the same
one-lane Core ML functions. The spectrum projection is computed once
and retained while both lanes reuse the scan and decoder workspaces.
These matched runs use the same compiled bundle, 33-second input,
80 ms pushes, and 20 ms model look-ahead on the M5 Pro.

| Runtime | RTF | Push p95 | Pushes over 80 ms | Peak physical memory |
| --- | ---: | ---: | ---: | ---: |
| Previous, three runs | 0.218–0.228 | 25.2–27.6 ms | 2 / 413 each | 1.132–1.133 GB |
| Reused one-lane functions, two runs | 0.223–0.230 | 27.4–28.0 ms | 1 / 413 each | 0.749–0.758 GB |

The two optimized 33-second renders were byte-identical and differed
from the previous runtime by at most `3.96e-5` per sample. Both upstream
goldens passed through the Swift API, with a maximum sample error of
`8.59e-5` at 80 ms. On six FLEURS clips, changes in noisy-input SI-SDR
and clean-input SDR were within 0.00001 dB of the preceding 80 ms runtime.

A separate trial moved the time recurrence into Metal and measured
0.214 RTF and 1.122 GB. Its adjacent controls measured 0.212 and 0.217
RTF. Its maximum sample error on the 11-second golden was `1.01e-4`.
A four-query attention tile reduced attention wall time from 1.13 s to
1.04 s. Its end-to-end RTF was 0.227, with adjacent controls at 0.220
and 0.221.

### Compact keys at 80 ms

The default 80 ms fast export packs keys in ten bits with a Float32 scale
per 32 components. Values use twelve bits with one Float32 scale per head.
The four layers' full capacity is 276.7 MB, down from 294.4 MB with
twelve-bit keys.

These runs used the same Swift executable, checkpoint, 33-second audio,
80 ms pushes, and 20 ms model look-ahead on the M5 Pro.

| Attention cache | RTF | Push p95 | Peak physical memory |
| --- | ---: | ---: | ---: |
| 12-bit keys and values, three runs | 0.218–0.241 | 25.9–35.7 ms | 0.749–0.758 GB |
| 10-bit keys, 12-bit values, two runs | 0.218–0.219 | 26.3–26.4 ms | 0.732–0.740 GB |

Both upstream golden fixtures passed through the Swift API, including
irregular pushes, with a maximum sample error of `6.73e-5`.
Across six FLEURS clips, noisy-input SI-SDR changed by −0.00037 to
+0.00042 dB, and clean-input SDR changed by −0.01771 to +0.00432 dB.
The largest absolute sample difference was `3.05e-4`. A trial with
ten-bit keys and values reached a worst clean-input SDR change of
−0.08820 dB.

Batching the Float32 time recurrence in groups of two, four, or eight
produced byte-identical audio on the 11-second fixture. The eight-frame
variant combined eight exponential operations into one batched operation.
In the 33-second trial, two-frame recurrence batches measured 0.258 and
0.332 RTF, with control runs between 0.252 and 0.260 RTF.

### Streaming cache pages

The 80 ms runtime allocates attention storage in 125-frame pages as the
cache grows on GPUs with Tier 2 Metal argument buffers. Three seconds of
audio uses three pages. The longer test fills all eight pages across
successive crossfades.

The runs used the same checkpoint, 80 ms pushes, 20 ms model look-ahead,
and CPU and GPU compute on the M5 Pro. Each layout ran twice per input.

| Audio | Attention storage | RTF | Push p95 | Peak physical memory |
| --- | --- | ---: | ---: | ---: |
| 3 s | Contiguous | 0.211–0.274 | 23.9–28.5 ms | 0.714 GB |
| 3 s | On-demand pages | 0.219–0.224 | 22.8–31.0 ms | 0.541–0.542 GB |
| 33 s | Contiguous | 0.245–0.261 | 31.0–31.9 ms | 0.732 GB |
| 33 s | On-demand pages | 0.256–0.288 | 32.2–46.0 ms | 0.733 GB |

Both long-stream layouts exceeded the 80 ms deadline once per run.
The 3- and 33-second renders were byte-identical between layouts. The
Swift golden tests passed, including a stream extended past its second
crossfade. The clean and noisy renders for all six FLEURS clips below
were also byte-identical.

### Larger blocks and shorter attention context

A 160 ms prototype with the full attention context measured 0.203 RTF,
57.2–57.8 ms push p95, and 0.887–0.888 GB peak physical memory on the
33-second input. Adjacent 80 ms controls measured 0.219–0.222 RTF,
26.5–27.2 ms p95, and 0.732–0.741 GB. Both goldens passed for 160 ms
with at most `8.41e-5` absolute sample error.

At 80 ms, limiting the attention context to four or six seconds measured
0.623 GB and 0.679 GB peak physical memory on the 33-second input. The
eleven-second golden errors were `1.60e-3` and `4.90e-4`, above the
`1e-4` tolerance.

A scaled-prefix formulation of the time recurrence passed both goldens
and measured 0.375–0.379 RTF against 0.226–0.235 RTF in adjacent
33-second controls. The compute plan for the first attention and Mamba
stage assigned all operations with an estimated cost to the GPU,
including with Neural Engine compute enabled. [Core ML compute plans](https://developer.apple.com/documentation/coreml/analyzing-a-core-ml-model-s-performance-in-xcode)

### Live stream alignment

The offline release bundle ran the 11-second fixture at 0.258 warm RTF
with 1.24 GB peak physical memory. The warm number averages the second
and third runs in one process.

For streaming, `prepare()` moves the first prediction before capture.
`preferredPushSamples` starts the eight-frame bundle with 4,384 samples
and uses 3,840 samples for subsequent full blocks at 48 kHz. This puts
crossfade boundaries between predictions during aligned capture.

The table reports individual paced replays from an active M5 Pro
workstation. First output is measured from the start of capture and
excludes model loading and warm-up. Warm p99 counts full 80 ms pushes
after the first, from scheduled arrival until the call returns.

| Input and schedule | First output | Crossfade splits | Warm p99 | Late full pushes after the first |
| --- | ---: | ---: | ---: | ---: |
| 33 s, fixed 80 ms, cold first prediction | 199.4 ms | 10 | 132.4 ms | 35 / 411 |
| 33 s, warm-up and aligned blocks | 123.3 ms | 0 | 97.0 ms | 13 / 411 |
| 99 s, warm-up and aligned blocks | 124.9 ms | 0 | 485.5 ms | 57 / 1,236 |
| 66 s, same schedule on a user-initiated thread | 128.1 ms | 0 | 108.7 ms | 45 / 823 |

Warm-up took 101–114 ms in these runs. Late pushes still occurred
outside crossfade boundaries under desktop load. The longest late
push in the user-initiated run spent 124.5 ms in the inference call
and 51.7 ms waiting past its scheduled arrival.

Both compiled release bundles passed the Swift golden tests. The
maximum absolute sample errors were `3.58e-7` for the offline bundle
and `6.73e-5` for the realtime bundle. Warmed and unwarmed aligned
streams gave byte-identical 3- and 11-second renders. Fixed and aligned
pushes produced identical samples on the six clean and six noisy
FLEURS clips below.

## Denoising and clean speech

The bench used one FLEURS recording per language from the package's
`Tests/HibikiTranslateTests/Resources` and `Tests/OmnilingualASRTests/Resources`
fixtures. Each 16 kHz clip was sinc-resampled to 48 kHz and peak-normalized to
0.4. Synthesized noise used NumPy seed `20261123` with an amplitude spectrum
proportional to `1 / sqrt(max(frequency, 80 Hz))`. Its RMS matched the clean
clip's RMS. SI-SDR measures agreement with the clean waveform after adjusting
for overall gain. Higher is better. [FLEURS dataset](https://huggingface.co/datasets/google/fleurs)

| Language | SI-SDR gain with added noise | RMS level shift on clean input | Clean-input SDR |
| --- | ---: | ---: | ---: |
| Arabic | +13.04 dB | +0.05 dB | 23.35 dB |
| English | +12.20 dB | +0.26 dB | 4.05 dB |
| French | +10.18 dB | −0.13 dB | 16.55 dB |
| German | +9.03 dB | −0.83 dB | 9.43 dB |
| Hindi | +12.23 dB | −0.06 dB | 25.93 dB |
| Portuguese | +11.95 dB | +0.37 dB | 16.70 dB |

The median gain was +12.08 dB. Tiled and full-window output samples differed
by at most `8.95e-8` on the six noisy inputs. Clean waveform fidelity varied,
with the largest change on the English recording. The upstream PyTorch model
reproduced the English and German changes within `9.1e-5` per sample. These
measurements describe individual clips under the stated conditions. Inworld
reports multilingual training without an enumerated language support list.
