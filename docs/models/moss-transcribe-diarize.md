# MOSS Transcribe Diarize CoreML

`MossTranscribe` is the native speech-swift runtime for
MOSS-Transcribe-Diarize 0.9B. It produces transcription text together with
speaker labels and segment timestamps. Audio preprocessing, prompt assembly,
state management, decoding, and output parsing are implemented in Swift; the
audio encoder and decoder run directly through Core ML.

## Model layout

| Component | Shape / configuration | Purpose |
|---|---|---|
| Whisper frontend | 16 kHz, 80 mel bins, 400-point FFT, 160-sample hop | Audio to `[1, 80, 3000]` log-mel features |
| Audio encoder | FP16 input/output, 30 s fixed input | Mel features to 12.5 audio embeddings per second |
| Token embedding | Core ML multifunction entry point | Token IDs to 1024-dimensional embeddings |
| Stateful decoder | 28 layers, 1024 hidden, 16 attention heads, 8 KV heads | Prompt prefill and greedy generation |
| KV state | 1024-token context | Shared state across prefill and one-token decode calls |
| Vocabulary | 151,936 tokens | Qwen tokenizer plus MOSS audio special tokens |

The decoder bundle exposes `embedding` and `decoder` functions from the same
compiled Core ML model, so both entry points share the exported weight file.
The runtime uses `FastPrediction` specialization and `.all` compute units.

## Published variants

| Variant | Audio encoder | Decoder | Bundle size | Recommended use |
|---|---|---|---:|---|
| `int8` (default) | FP16 | Symmetric linear INT8, block size 32 | 1.2 GB | Best speed and memory |
| `fp16` | FP16 | FP16 | 1.7 GB | FP16 reference |

The INT8 profile quantizes the homogeneous decoder linear layers while keeping
the audio encoder and decoder I/O in FP16. On the matched native benchmark it
retains the same aggregate English WER as FP16, runs about 11% faster when the
two order-reversed runs are pooled, and lowers sampled peak RSS by about 36%.
See [MOSS CoreML benchmark](../benchmarks/moss-coreml.md).

## Prompt and timestamps

MOSS represents audio with `<|audio_start|>`, repeated `<|audio_pad|>`
positions, and `<|audio_end|>`. The runtime injects the audio encoder output at
the placeholder positions. It also inserts numeric time markers every five
seconds, matching the published processor.

The expected decoded form is:

```text
[5.00][S01] Can you guarantee that the replacement part will be shipped tomorrow?[8.40]
```

`MossTranscriptParser` returns both the plain text and structured
`MossTranscriptSegment` values. If the model emits malformed structure, the
raw decoded text remains visible instead of being silently discarded.

## Native runtime responsibilities

`Sources/MossTranscribe/` implements:

- exact Hugging Face Whisper log-mel preprocessing with Accelerate;
- model-bundle and processor-contract validation;
- time-marker and chat-prompt construction;
- audio-embedding injection;
- 128-token decoder prefill plus one-token stateful decode;
- bounded token-embedding caching;
- greedy argmax generation and end-token handling;
- timestamp/speaker parsing and per-stage timing metrics.

The frontend is numerically regression-tested against pinned Hugging Face
fixtures. The tokenizer prompt IDs are also pinned against the exporter
reference.

## Model weights

- [aufklarer/MOSS-Transcribe-Diarize-0.9B-CoreML-INT8](https://huggingface.co/aufklarer/MOSS-Transcribe-Diarize-0.9B-CoreML-INT8)
- [aufklarer/MOSS-Transcribe-Diarize-0.9B-CoreML-FP16](https://huggingface.co/aufklarer/MOSS-Transcribe-Diarize-0.9B-CoreML-FP16)
- Source model:
  [OpenMOSS-Team/MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)

## Platform and concurrency

- Minimum deployment: macOS 15 or iOS 18.
- A model instance serializes inference because its embedding cache and Core ML
  stateful decoder are not re-entrant.
- Create separate model instances when concurrent transcription is required.
- The fixed 1024-token decoder context limits the combined audio prompt and
  generated transcript. The runtime processes audio in 30-second encoder
  chunks and rejects prompts that exceed that context.
