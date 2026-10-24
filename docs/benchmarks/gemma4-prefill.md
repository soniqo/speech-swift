# Gemma 4 12B prompt prefill

The native 12B text runtime skips vocabulary projections for intermediate
prompt positions and retains one immutable system/tool prefix. Working request
caches use independent copies of that prefix. User messages, conversation
history, and tool results are excluded from retention.

## Local measurement

Measured on an Apple M5 Pro with 48 GiB unified memory, macOS 26.6.2, a release
build with compiled MLX Metal shaders, and the staged
`mlx-community/gemma-4-12B-it-4bit` checkpoint at revision
`73bcf09092aa277861d5a191b989b666f7f32e8f`.

The synthetic prompt has 8,499 tokens: a repeated scheduling reference in the
system turn, a declared `record_note` tool, and a request to save `launch review`.
Greedy generation produces 15 tokens. Timing includes prompt tokenization,
prefill, generation, and tool parsing; it excludes model loading. The allocator
cache is cleared between requests. The explicit prefix budget is 1,024 MiB.

| Request | Reused prompt tokens | Wall seconds |
| --- | ---: | ---: |
| Former prefill, before optimized pair | 0 | 10.180 |
| Optimized cold request | 0 | 9.352 |
| Optimized warm request | 8,448 | 0.778 |
| Former prefill, after optimized pair | 0 | 10.292 |

All four requests produced identical prompt tokens, generated tokens, raw
output, and tool arguments. The retained prefix contains 557,514,752 bytes
(about 532 MiB); the former 512 MiB budget was insufficient for this prompt.
The 12B default is now 1,024 MiB. E4B keeps its existing 512 MiB default.

This is one synthetic prompt on one machine, with one timed request per row.
It is not a task-accuracy result or a general latency guarantee. Cold requests
still pay the full prefill cost; gains depend on the repeated prefix and the
remaining conversation length.

## Verification and reproduction

`E2EGemma4UnifiedTests` compares cold and warm greedy tokens to the original
prefill path, then interleaves a second user request longer than the sliding
window and repeats the original request. Both users' responses must match
fresh-cache reference generations. The test also releases the prefix snapshot.

Build the tests in release mode with testing enabled and compile the release
metallib. Run only this model suite, with no other model process or build active:

```bash
GEMMA4_12B_MODEL_DIR=/models/gemma-4-12B-it-4bit \
SPEECH_SWIFT_GEMMA4_PREFIX_CACHE_MB=1024 \
swift test -c release -Xswiftc -enable-testing --skip-build --no-parallel \
  --filter E2EGemma4UnifiedTests/testLongPromptCacheOnlyPrefillMatchesReferenceGreedyTokens
```

Set `SPEECH_SWIFT_GEMMA4_PREFIX_CACHE_MB=0` to disable retention. Native HTTP
responses report `cachedPrefixTokens`; provenance records the effective budget.
Record cold/warm requests separately when reporting timing.
