# MOSS Transcribe Diarize

`MossTranscribe` runs MOSS-Transcribe-Diarize 0.9B natively on Apple
Silicon. It jointly generates transcript text, timestamps, and anonymous
speaker labels such as `[S01]`; it is not an ASR model followed by a separate
speaker clustering pipeline.

speech-swift provides two execution backends:

| Backend | Context | Weights | Intended use |
|---|---:|---|---|
| MLX | 131,072 dynamic tokens | FP16 audio/VQ + affine INT5 or INT8 decoder | Long-form, globally contextualized offline transcription |
| Core ML | 1,024 fixed tokens | FP16 audio + block-32 INT8 or FP16 decoder | Short recordings and Neural Engine execution |

The upstream model and both speech-swift backends are offline and
autoregressive. No streaming checkpoint is published for this MOSS model: the
complete recording must be available before transcript generation begins, and
no partial transcript is emitted while recording.

## Architecture

| Component | Shape / configuration | Purpose |
|---|---|---|
| Whisper frontend | 16 kHz, 80 mel bins, 400-point FFT, 160-sample hop | Audio to fixed 30-second log-mel chunks |
| Whisper encoder | 24 layers, 1,024 hidden, 16 heads | Each chunk to contextual audio features |
| VQ adaptor | merge size 4 | 12.5 decoder audio embeddings per second |
| Qwen3 decoder | 28 layers, 1,024 hidden, 16 query heads, 8 KV heads | Greedy speaker-attributed transcript generation |
| Vocabulary | 151,936 tokens | Qwen tokenizer plus MOSS audio tokens |
| MLX context | 131,072 combined prompt/output tokens | Dynamic long-context KV cache |

For MLX, the encoder processes non-overlapping 30-second chunks in bounded
batches. All adapted audio embeddings are then concatenated into one prompt.
The decoder therefore sees the recording as one global context rather than
independently transcribing and stitching windows.

At 12.5 audio tokens per second, 90 minutes uses about 67,500 audio tokens.
The remaining context must hold instructions, time markers, and generated
transcript tokens. The upstream 90-minute figure is therefore a model
capability, not a fixed host guarantee: available unified memory, KV-cache
precision, transcript length, and content determine the practical limit.

## MLX quantization

The reproducible exporter keeps all 312,463,360 Whisper/VQ parameters in FP16
and quantizes the 596,049,920 tied Qwen3 decoder parameters with MLX affine
group-64 quantization. No training or fine-tuning is performed.

| MLX variant | Decoder | Audio weights | Decoder weights | Total weights | Default |
|---|---|---:|---:|---:|---|
| `int5` | Affine INT5, group 64 | 596.0 MiB | 391.0 MiB | 987.0 MiB | Yes, for MLX |
| `int8` | Affine INT8, group 64 | 596.0 MiB | 604.1 MiB | 1,200.1 MiB | Quality reference |

Decoder weight precision and KV-cache precision are independent. Matched
INT5/INT8 quality comparisons use the same FP16 audio weights and FP16 KV
cache. Quantizing the cache to INT8 further lowers long-context memory but
introduces another quality variable.

The matched measurements show a small English ASR advantage for INT8 and
effectively equal aggregate diarization error on the five-file test slice.
INT5 is the recommended MLX default because it is smaller, uses less peak
memory, and was faster in both runs. See
[MOSS MLX benchmark](../benchmarks/moss-mlx.md).

## Prompt and output

MOSS represents audio with `<|audio_start|>`, repeated `<|audio_pad|>`
positions, and `<|audio_end|>`. Numeric time markers are inserted every five
seconds, matching the upstream processor.

The canonical decoded form is:

```text
[5.00][S01]Can you guarantee that the replacement part will be shipped tomorrow?[8.40]
```

`MossTranscriptParser` preserves the raw generated form and returns both plain
text and `MossTranscriptSegment` values. Speaker labels are anonymous within a
recording and are not persistent identities.

## Languages

The upstream model card claims transcription and diarization across 50+
languages but does not publish an exhaustive language list. Publicly named
evidence includes Mandarin Chinese plus English, French, German, Italian,
Portuguese, Spanish, Japanese, Korean, Russian, Thai, Vietnamese, Tagalog,
Urdu, and Turkish. Treat the 50+ figure as an upstream capability claim and
validate the target language, accents, speaker count, and acoustic domain.

The speech-swift quantization comparison currently measures English ASR and an
English VoxConverse diarization slice; it does not establish quality parity
for every upstream language.

## Export and model weights

Run the pinned exporter with Python MLX:

```bash
python3.11 scripts/export_moss_mlx.py \
  --bits 5 \
  --output /path/to/MOSS-Transcribe-Diarize-0.9B-MLX-5bit

python3.11 scripts/export_moss_mlx.py \
  --bits 8 \
  --output /path/to/MOSS-Transcribe-Diarize-0.9B-MLX-INT8
```

The exporter pins source revision
`e6d68cdfcddbdad1a7e8454f0cb859cad76e2502`, validates the expected tensor
counts and architecture, writes artifact checksums, verifies a dequantized
reference tensor, copies the exact tokenizer/processor assets, and includes
the Apache 2.0 license.

Published model repositories used by the runtime:

- [MLX INT5](https://huggingface.co/aufklarer/MOSS-Transcribe-Diarize-0.9B-MLX-5bit)
- [MLX INT8](https://huggingface.co/aufklarer/MOSS-Transcribe-Diarize-0.9B-MLX-INT8)
- [Core ML INT8](https://huggingface.co/aufklarer/MOSS-Transcribe-Diarize-0.9B-CoreML-INT8)
- [Core ML FP16](https://huggingface.co/aufklarer/MOSS-Transcribe-Diarize-0.9B-CoreML-FP16)
- [Upstream model](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)
- [Technical report](https://arxiv.org/abs/2601.01554)

## License and platform

MOSS-Transcribe-Diarize and the converted weight bundles use Apache License
2.0. The speech-swift runtime supports macOS 15+ and iOS 18+ on Apple Silicon.
A model instance serializes inference; use separate instances only when the
additional model and cache memory is acceptable.
