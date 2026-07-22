# Cache Directory & Offline Mode

All `fromPretrained()` methods accept optional `cacheDir` and `offlineMode` parameters for apps that need control over model storage or want to avoid network calls.

## Custom Cache Directory

By default, models are cached in `~/Library/Caches/qwen3-speech/models/<org>/<model>/`. Pass `cacheDir` to override:

```swift
let appModels = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
    .appendingPathComponent("MyApp/models")

let asr = try await ParakeetASRModel.fromPretrained(
    cacheDir: appModels.appendingPathComponent("parakeet"))

let tts = try await KokoroTTSModel.fromPretrained(
    cacheDir: appModels.appendingPathComponent("kokoro"))
```

This is useful for:
- **Sandboxed macOS apps** that can't write to `~/Library/Caches/`
- **iOS apps** using the app container
- **Custom storage** (external drive, shared group container)

### Diarization Pipeline

The diarization pipeline downloads 3 models (segmentation, speaker embedding, optional VAD). Use `cacheBaseDir` to set a shared base — each sub-model gets its own subdirectory automatically:

```swift
let pipeline = try await PyannoteDiarizationPipeline.fromPretrained(
    cacheBaseDir: appModels)
// Segmentation → appModels/models/aufklarer/Pyannote-Segmentation-MLX/
// Embedding    → appModels/models/aufklarer/WeSpeaker-ResNet34-LM-MLX/
// VAD (opt.)   → appModels/models/aufklarer/Silero-VAD-v6.2.1-MLX/
```

Community-1 is one self-contained bundle, so it uses the regular `cacheDir`
parameter:

```swift
let community1 = try await Community1DiarizationPipeline.fromPretrained(
    cacheDir: appModels.appendingPathComponent("community1"),
    offlineMode: true
)
```

## HuggingFace Mirror (`HF_ENDPOINT`)

Downloads default to `https://huggingface.co`. Users in regions where that host is slow or blocked — notably mainland China — can point the downloader at a mirror by setting the `HF_ENDPOINT` environment variable (the same name Python's `huggingface_hub` uses):

```bash
export HF_ENDPOINT=https://hf-mirror.com
.build/release/speech transcribe recording.wav   # weights now fetch from hf-mirror.com
```

Notes:
- The value must be a full `http(s)://host` URL. A blank or malformed value is ignored and the default endpoint is used.
- The cache is keyed by repo id, not by host — switching `HF_ENDPOINT` reuses any weights already on disk and never forces a re-download. You can fetch from the mirror once and keep using the cache offline.
- Applies to every model and CLI command, since all downloads share one downloader.

## Offline Mode

When `offlineMode: true`, the downloader skips network requests if weights already exist on disk:

```swift
let model = try await Qwen3ASRModel.fromPretrained(offlineMode: true)
```

Behavior:
- Weights exist → returns immediately (no HuggingFace API calls)
- Weights missing → falls through to normal download (will fail if truly offline)

This avoids unnecessary network latency on app launch when models are already cached.

### Combining Both

```swift
let model = try await ParakeetASRModel.fromPretrained(
    cacheDir: bundledModelsDir,
    offlineMode: true)
```

Ship pre-downloaded models in your app bundle, point `cacheDir` at them, and set `offlineMode: true` to guarantee zero network calls.

## Supported Models

All models support both parameters:

| Model | Parameter |
|-------|-----------|
| `Qwen3ASRModel` | `cacheDir`, `offlineMode` |
| `ParakeetASRModel` | `cacheDir`, `offlineMode` |
| `CoreMLASRModel` | `cacheDir`, `offlineMode` |
| `KokoroTTSModel` | `cacheDir`, `offlineMode` |
| `Qwen3TTSModel` | `cacheDir`, `offlineMode` |
| `Qwen3TTSCoreMLModel` | `cacheDir`, `offlineMode` |
| `CosyVoiceTTSModel` | `cacheDir`, `offlineMode` |
| `PersonaPlexModel` | `cacheDir`, `offlineMode` |
| `SileroVADModel` | `cacheDir`, `offlineMode` |
| `PyannoteVADModel` | `cacheDir`, `offlineMode` |
| `FireRedVADModel` | `cacheDir`, `offlineMode` |
| `WeSpeakerModel` | `cacheDir`, `offlineMode` |
| `ReDimNet2SpeakerModel` | `cacheDir`, `offlineMode` |
| `SpeechEnhancer` | `cacheDir`, `offlineMode` |
| `LocalVQEEchoCanceller` | `cacheDir`, `offlineMode` |
| `SortformerDiarizer` | `cacheDir`, `offlineMode` |
| `Community1DiarizationPipeline` | `cacheDir`, `offlineMode` |
| `PyannoteDiarizationPipeline` | `cacheBaseDir`, `offlineMode` |
| `Qwen35CoreMLChat` | `cacheDir`, `offlineMode` |
| `Qwen35MLXChat` | `cacheDir`, `offlineMode` |
