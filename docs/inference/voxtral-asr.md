# Voxtral ASR inference

Load a local export or a published bundle through `VoxtralModel.load`:

```swift
import VoxtralASR

let model = try await VoxtralModel.load(
    "/path/to/Voxtral-Mini-3B-2507-MLX-5bit",
    offlineMode: true
)

let text = model.transcribe(
    audio: samples,
    sampleRate: sourceSampleRate,
    language: "en"
)
```

The directory must contain `config.json`, one or more `.safetensors` files,
and `tekken.json`. `preprocessor_config.json` and
`speech_models_export.json` are retained for provenance and parity auditing.

## CLI

```bash
speech transcribe recording.wav --engine voxtral
speech transcribe recording.wav --engine voxtral --model int8 --language fr
speech transcribe recording.wav --engine voxtral --model /path/to/local/export
```

The default is INT5. `--model` also accepts `int8`, `fp16`, a Hugging Face
repository ID, or a local export directory. Voxtral is currently a
non-streaming engine.

## Variants

`VoxtralVariant` exposes `.fp16`, `.int5`, and `.int8`. INT5 passed the
cross-precision artifact gate and is the default candidate once the bundles
are published. MLX does not support INT7; INT8 is the supported high-quality
substitute.

The decoder projects only the final prompt state through the 131,072-token
language-model head. Earlier builds projected every prompt/audio state even
though greedy decoding consumes only the final row. Removing that work
preserved every transcript in the pinned acceptance run and materially lowers
quantized-model RTF.

The exporter also has a separately named `int5-audio-ffn` candidate. It is not
part of `VoxtralVariant`; load its directory explicitly when evaluating it.
On the pinned English FLEURS acceptance set it reduced standard-INT5 peak RSS
from 4,035 to 3,517 MiB and physical footprint from 6,012 to 5,671 MiB while
preserving observed quality, at 0.0757 mean RTF in the clean isolated run.
Keep it separate from the standard INT5 artifact until broader multilingual
and conversational validation is complete.

## Language handling

Language may be an ISO code or English language name for en, fr, de, es, it,
pt, nl, or hi. Passing `nil` omits the language instruction and leaves
selection to the model. Unsupported explicit names currently fall back to
English.

## Long inputs and memory

The frontend pads input to complete 30-second chunks and runs the audio tower
as a chunk batch. Audio embeddings are concatenated into one prompt, so prompt
and decoder-cache memory scale with the number of chunks. Use bounded input
segments for very long recordings until a streaming/chunk-merge path is added.

## Validation

Unit tests cover official config shapes, Tekken special tokens, exact language
prompt IDs, 30-second packing geometry, and quantization rejection.
`E2EVoxtralASRTests` runs against a local bundle when
`VOXTRAL_MLX_MODEL_PATH` is set. Use the isolated benchmark protocol for final
WER, RTF, throughput, RSS, and physical-footprint acceptance. Use physical
footprint for deployment sizing; RSS can undercount memory-mapped MLX weights.
