# Nemotron 3 Diarization

`SpeechVAD` supports the public final [NVIDIA Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) model through compiled Core ML INT8 and native MLX INT8 backends. Both exports use the final source revision `a435e9867d79e789e90053f9b6d6834053af564a` and predict activity for up to eight anonymous speakers every 10 ms.

## Architecture and state

The model stacks 128-band log-mel frames by eight, projects them to 512-dimensional embeddings, then applies a 31-layer, eight-head Transformer with rotary position embeddings. A 192-dimensional head upsamples each 80 ms encoder row into eight 10 ms speaker-activity rows.

A call packs three sequences into the fixed 684-row head input:

| Sequence | Maximum rows | Purpose |
| --- | ---: | --- |
| Speaker cache | 264 | Keeps long-range arrival-order speaker information |
| FIFO | 40 | Keeps recent encoder history |
| New chunk and right context | 380 | Up to 340 new rows and 40 look-ahead rows |

The recommended offline-style preset emits up to 27.2 seconds of newly confirmed speech per call. The host updates the speaker cache every 300 encoder rows and fills reserved silence rows with the checkpoint's learned silence embedding. Short recordings use only the valid rows in the fixed graph.

## Artifacts

- [Core ML INT8](https://huggingface.co/aufklarer/Nemotron-3-Diarization-100M-CoreML-INT8): `Nemotron3PreEncoder.mlmodelc/`, `Nemotron3Head.mlmodelc/`, `learnable_silence.f32`, `config.json`.
- [MLX INT8](https://huggingface.co/aufklarer/Nemotron-3-Diarization-100M-MLX-INT8): `model.safetensors`, `config.json`.

Core ML uses block-32 INT8 linear weights with FP16 convolution and activation work. MLX uses affine group-64 INT8 linear weights with FP16 convolution weights. The runtime validates the export geometry before loading either backend.

## Runtime

`Nemotron3Diarizer.fromCoreMLPretrained()` and `fromMLXPretrained()` download the public exports into the normal model cache. `fromCoreMLDirectory(_:)` and `fromMLXDirectory(_:)` load an existing local copy. Both backends share mel extraction, cache maintenance, 10 ms binarization, and segment construction. Speaker IDs are arrival-order labels within one recording; they are not cross-recording identities.

## Validation

The export check compared both quantized backends with the unquantized source on three speech excerpts. The Swift end-to-end suite loads both published bundle formats and checks output shape and agreement. The diarization benchmark compares full audio-to-segment results against the same RTTM references used for MOSS and Sortformer; see `docs/benchmarks/diarization.md` for the current run and limits.
