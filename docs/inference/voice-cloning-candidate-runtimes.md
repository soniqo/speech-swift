# Voice Cloning Candidate Runtimes - Inference

These modules are loadable runtime contracts, not complete synthesis engines.
They are useful for wiring downloads, cache validation, model-size reporting,
CLI surfacing, and native-port tests before the full inference graphs land.

```swift
import IndexTTS2TTS

let bundle = URL(fileURLWithPath: "/path/to/IndexTTS2-MLX-fp16")
let model = try await IndexTTS2TTSModel.fromBundle(bundle)

print(model.manifest.parameterCount ?? "unknown")
print(model.memoryFootprint)
print(IndexTTS2TTSModel.auxiliaryModels.map(\.repository))

let tokenIds = try model.tokenizer?.encode("Hello from IndexTTS2")

// Throws until the native graph port is implemented.
_ = try await model.generate(text: "Hello", language: "en")
```

The same shape is available for `HiggsAudioTTSModel` and `F5TTSModel`.

## CLI Surface

IndexTTS2 is exposed through `speech speak` as a bundle-validation engine:

```bash
speech speak "Hello from IndexTTS2" \
  --engine indextts2 \
  --voice-sample reference.wav \
  --indextts2-bundle-dir /path/to/IndexTTS2-MLX-fp16
```

or, after the expanded bundle is published:

```bash
speech speak "Hello from IndexTTS2" \
  --engine indextts2 \
  --voice-sample reference.wav \
  --indextts2-model-id aufklarer/IndexTTS2-MLX-fp16
```

The command downloads or loads the exported bundle, prints manifest metadata,
then throws the same explicit unsupported-runtime error as the Swift API. This
keeps CLI behavior honest while making the published artifact testable through
the normal cache path.

The expanded IndexTTS2 export layout contains base artifacts at the root plus
w2v-BERT, MaskGCT semantic codec, CAMPPlus, and BigVGAN under `aux/`. Those
paths are exposed through
`IndexTTS2TTSModel.auxiliaryModels` for the native port, but the graph execution
is not yet wired into `generate`.

| Flag | Scope | Notes |
|---|---|---|
| `--engine indextts2` | `speech speak` | Selects the IndexTTS2 exported-bundle loader. |
| `--voice-sample <wav>` | `indextts2` | Required because IndexTTS2 is a zero-shot voice-cloning model. |
| `--indextts2-model-id <repo>` | `indextts2` | Defaults to `aufklarer/IndexTTS2-MLX-fp16`. |
| `--indextts2-bundle-dir <path>` | `indextts2` | Loads a local exported bundle instead of Hugging Face. |
| `--indextts2-emotion-audio <wav>` | `indextts2` | Accepted for the future native port; not used until synthesis is implemented. |

## Download Defaults

| Module | Default model id |
|---|---|
| `IndexTTS2TTSModel` | `aufklarer/IndexTTS2-MLX-fp16` |
| `HiggsAudioTTSModel` | `aufklarer/Higgs-Audio-v3-TTS-4B-MLX-fp16` |
| `F5TTSModel` | `aufklarer/F5-TTS-v1-Base-MLX-fp16` |

The default IDs are the expected outputs of the `speech-models`
`voice-cloning-candidates` exporter. Uploading those bundles to Hugging Face is
a separate release step and is not performed by the runtime.

## Tests

Fast unit tests avoid model downloads:

```bash
swift test --filter VoiceCloneCandidateTTSTests
```

The IndexTTS2 expanded-bundle E2E test is opt-in because it validates a multi-GB
bundle:

```bash
INDEXTTS2_E2E_BUNDLE=/path/to/IndexTTS2-MLX-fp16 \
  swift test --filter E2EIndexTTS2BundleTests --disable-sandbox

INDEXTTS2_E2E_DOWNLOAD=1 \
  swift test --filter E2EIndexTTS2BundleTests --disable-sandbox
```

The E2E verifies bundle metadata, auxiliary model records, tokenizer loading,
and the current explicit native-synthesis-not-ported error.
