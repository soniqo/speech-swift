# Inworld Voice Isolator v1 architecture

Voice Isolator v1 is a 64-million-parameter offline speech denoiser. It
processes a centered 2,048-point Hann STFT with a 480-sample hop. The band
encoder maps the 1,025 complex frequency bins to 46 tokens per frame with
width 512. Seven dual-path blocks alternate time and band attention with
bidirectional Mamba. Per-band heads predict a complex mask delta and a bounded
complex residual. [Inworld model card](https://github.com/inworld-ai/voice-isolator/blob/c73b3fab81a870410cbac50d2f05f25edc4d10aa/MODEL_CARD.md)

The export keeps the band projections, attention, Mamba input and output
projections, depthwise convolutions, and decoder in seven Core ML programs.
Metal evaluates each of the six selective scans in Float32, keeping one
channel's 16-state recurrence in registers. Each scan runs along time or
frequency in both directions. The Core ML stages accept the exact STFT frame
count, from 1 to 801. Accelerate handles the STFT and inverse overlap-add,
and Swift applies the complex correction to the original spectrum.
Inference loads one Core ML stage at a time to bound prediction memory. The
optional `full_window_frames` manifest field identifies bundles with a
`full_window` function in each stage. Its static output shape lets Core ML
write each scan input directly into a Metal-backed array. The default function
handles shorter inputs using the same weights.

The tiled bundle uses the format `speech-swift-inworld-voice-isolator-tiled-v1`.
It divides each attention and Mamba operation into groups of independent rows
along the batch dimension. Each row retains its complete time or band sequence.
The runtime reuses a scan workspace for the rows processed by the active
partition. The `full_tile` functions in the scan projections allow direct
writes into that workspace for their fixed shapes. With `static_tiles` in the
manifest, the runtime also selects fixed-shape functions for the attention
and scan merge stages. Flexible functions handle smaller inputs. These
bundles group up to two adjacent windows for inference. The windows share
loaded stages and retain independent activations. The waveform crossfade
uses the original window positions and output order.

The seven-stage export uses `speech-swift-inworld-voice-isolator-v1`. Stages
`stage_0` through `stage_6` exchange `scan_features`, `scan_input`, and
`scan_output` as Float32 tensors. `scan_coefficients.bin` stores the learned
decay rates and skip gains as little-endian Float32 in scan-stage order.

## Realtime checkpoint

The realtime checkpoint contains 58,178,056 Float32 parameters, occupying
232.7 MB in the checkpoint. It uses causal time attention and forward-only
time scans. Frequency scans remain bidirectional. The
`inworld-voice-isolator-realtime-coreml-stateful-v1` export uses Float16
key/value cache storage and Float32 attention, projections, and recurrence.
Core ML holds the key/value cache for each active lane. Swift carries the
short depthwise convolution history and runs the state-space recurrence
with Float32 state in Metal.

The stream uses 600-frame attention windows with a 200-frame crossfade.
During the crossfade it starts a new attention lane, copies the Mamba state
into it, and blends the two decoder outputs before synthesis. It reuses the
retired lane's cache and masks unwritten slots. Synthesis uses the selected
look-ahead and releases samples when their overlap-add is complete.

The low-memory export uses the `inworld-voice-isolator-realtime-coreml-padded-attention-v1`
format. Its eight Core ML stages hold the Float32 learned layers and time
recurrence. Metal runs time attention over a pool of 1,000 key/value
slots in each attention layer and runs the bidirectional frequency scans.
A two-frame graph batches the overlap lanes, shares the input embedding
across them, and saves the time recurrence state at the last valid frame.
The packed cache stores each component in 12 bits with a Float32 scale
per head and frame. Its four layers reserve 294.4 MB. A manifest without
an explicit cache encoding uses Float16 buffers, which reserve 376.8 MB.
The process footprint also includes Core ML execution workspaces,
intermediate tensors, and the Metal scan and state buffers.

The fast `inworld-voice-isolator-realtime-coreml-metal-decoder-v1` export
includes one- and two-lane functions. For the default eight-frame export,
Swift loads the one-lane stages and runs the two crossfade lanes in
sequence, sharing the spectrum encoder output and scan workspace. Two-
and four-frame exports use the two-lane functions during crossfades.
Attention masks frames beyond the valid count. Time recurrence and
convolution history advance through the valid frames.

Before the first audio push, `prepare()` can run one zero-input
prediction, release its attention slots, and reset its recurrence state.
`preferredPushSamples` includes the analysis padding and stops at the
next crossfade boundary or the end of a model block, whichever comes
first.

The eight-frame manifest specifies `key_cache_bits: 10` and
`kv_cache_bits: 12`. Each key head packs 64 ten-bit components and two
Float32 scales, one per 32 components. Each value head packs 64 twelve-bit
components and one Float32 scale. The four layers use 276.7 MB at
capacity. On devices with Tier 2 Metal argument buffers, the 80 ms
runtime allocates 125-frame pages as the cache grows and reuses their
slots across crossfades. Other devices allocate contiguous buffers.
A manifest with only `kv_cache_bits` applies that encoding to both buffers.

The three frequency Mamba output stages compose the
directional projections and output projection ahead of time. Selected
large linear layers use Float16. The Mamba recurrence uses Float32.

The final Core ML stage supplies features for the 46 band heads. A single
Metal command normalizes the features, runs the first projection with
Float16 weights and Float32 accumulation, and produces the Float32 mask
and residual. The previous `specialized-attention-v1` format runs its
band heads inside the final Core ML stage.

See [Bundle and inference](../inference/inworld-voice-isolator.md) for local
setup and usage.
