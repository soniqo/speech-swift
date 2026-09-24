# Nemotron 3 short-cadence Core ML probe (24 September 2026)

This is an experimental short-cadence and incremental-session profile, **not**
a package promotion result. The source was the five-file local
VoxConverse dev manifest (1,057.5 seconds). All engines used the same RTTM
references, 0.25-second collar and benchmark scoring. Core ML runs used the
INT8 final Nemotron 3 export. The short-cadence probe padded each fixed-shape
pre-encoder input while committing only the core and updating speaker cache;
`core6` means 6 × 80 ms core plus 7 × 80 ms right context. The baseline
`sortformer-session` used the package's streaming Sortformer configuration.

| Engine | Core / right context | DER | Count agreement | xRT | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| Nemotron 3 Core ML `.all` offline | 27.2 / 3.2 s | 1.99% | 80% | 1,595.8 | 427 MB |
| Nemotron 3 Core ML `.all` | 0.48 / 0.56 s | 2.41% | 80% | 20.5 | 332 MB |
| Nemotron 3 Core ML `.all` | 1.28 / 0.56 s | 2.33% | 80% | 49.6 | 428 MB |
| Nemotron 3 Core ML CPU+ANE | 0.48 / 0.56 s | 2.22% | 80% | 3.2 | 487 MB |
| Nemotron 3 Core ML `.all` incremental session | 0.48 / 0.56 s | 2.41% | 80% | 13.3 | 258 MB |
| Sortformer streaming session | 0.48 / 0.56 s | 3.13% | 60% | 34.1 | 142 MB |

The `.all` and CPU+ANE results are not bit-identical. One of two four-speaker
files has a Nemotron 3 count error even in the offline baseline; Sortformer
gets the count right on both. Conversely, Sortformer misses the count on the
one three-speaker file and one of the two two-speaker files. These five files
are a diagnostic, not a promotion gate. The very high offline xRT is from
amortizing a few fixed-size graph calls across long audio and is not live
latency.

With CPU+ANE, the 0.48-second cadence spent most sampled time in the Core ML
head; the stack was frequently in CPU BNNS. Allowing `.all` improved throughput
on this set from 3.2× to 20.5×. Wrapping pre-encoder and head predictions in
local autorelease pools prevented temporary Core ML objects from accumulating:
on the first 148-second file, peak RSS dropped from 1,568 MB to 403 MB for the
CPU+ANE `core6` run, while throughput increased from 3.4× to 4.5×. These are
process high-water marks and depend on Core ML compilation/cache state.

The incremental session confirms its first 480 ms of audio once 1.06 seconds
have arrived. It retains only the PCM needed by the next window and processes
new 10 ms probabilities incrementally. On 500 ms pushes, its per-file median
latency was 19–40 ms and p95 was 37–96 ms in the final run. Its aggregate DER
and count agreement exactly match the short-cadence offline replay. A real
Core ML E2E test also compares every segment ID and boundary on the 148-second
sample against that replay. These are processing timings, not a measured
end-to-end UI label delay; a sustained meeting/phone replay and recording-local
ID-stability gate remain open. The four-speaker miss occurs even with full
offline decoding, so it is not fixed by this cadence or by lowering the
binarization threshold. Do not replace a low-latency Sortformer consumer on
this report alone.
