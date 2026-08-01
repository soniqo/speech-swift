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

## Download Behavior

Every model shares one downloader. A repository is resolved once against the
Hub tree API, which returns each file's size and — for LFS-backed files — its
SHA-256, in a single request. Transfers then behave the same way for every
model:

- **Large files transfer as concurrent byte ranges** and record completed
  chunks alongside the data, so an interrupted download resumes where it
  stopped instead of restarting the file. This survives process exit: a
  download interrupted by quitting the app continues on next launch.
- **Progress is weighted by real transferred bytes**, across the whole file
  set rather than per file.
- **LFS-backed weights are checksummed** after transfer. A file whose digest
  doesn't match is deleted rather than kept, so a corrupted download is
  refetched instead of being cached permanently and resurfacing later as an
  unreadable-tensor error. Already-cached files are not re-hashed on load.
- **Sharded bundles fetch only the shards their index names.** A repository
  that publishes both a sharded set and a consolidated `model.safetensors`
  holds the same tensors twice; `model.safetensors.index.json` decides which
  copy is used.

In-flight files stage under `<cacheDir>/.incomplete/` and are moved into place
once complete. The directory is removed when empty; anything left in it is a
resume point for an interrupted transfer and is safe to delete manually if you
want to force a clean re-download.

### When the Hub is unreachable

Resolution always goes to the network, so that a re-exported model is picked
up rather than silently pinned to whatever was cached first. If the Hub cannot
be reached and **everything the call asked for is already on disk**, the load
proceeds from cache instead of failing — a dropped connection shouldn't break a
load that needs no bytes.

That fallback is deliberately narrow:

- It applies only to transport failures (no route, DNS, refused connection,
  timeout, stall). An HTTP answer is a real answer: a 404 for a model that
  doesn't exist still fails, and so does a checksum mismatch.
- The cache must be *complete* — every requested asset present, and for a
  sharded bundle every shard the index names. A cache with weights but no
  tokenizer fails, rather than loading and breaking later somewhere less
  obvious.

Set `offlineMode: true` when you want to guarantee no network calls at all
rather than relying on this.

### Tuning

| Variable | Default | Effect |
|---|---|---|
| `HF_ENDPOINT` | `https://huggingface.co` | Mirror host (see above) |
| `HF_TOKEN` | — | Bearer token for gated repositories |
| `HF_DOWNLOAD_STALL_TIMEOUT` | `300` | Seconds without progress before an attempt is abandoned and retried |
| `HF_DOWNLOAD_RANGE_CONCURRENCY` | `16` | Concurrent range requests per file (capped at 16) |
| `HF_DOWNLOAD_RANGE_THRESHOLD` | `8388608` (8 MB) | File size at or above which ranged transfer is used |
| `HF_DOWNLOAD_RANGE_CHUNK` | `8388608` (8 MB) | Bytes per range request |

Each in-flight chunk is buffered whole, so chunk size × concurrency is the
transient memory a large download costs — 128 MB at the defaults. Lower
`HF_DOWNLOAD_RANGE_CONCURRENCY` on memory-constrained devices.

Failed downloads retry five times with 5/15/30/60 s backoff. The stall timeout
is deliberately patient because app users cannot set environment variables and
flaky networks routinely stall for a minute or two before recovering; CI pins
it lower to fail fast.

## Offline Mode

When `offlineMode: true`, the downloader never touches the network:

```swift
let model = try await Qwen3ASRModel.fromPretrained(offlineMode: true)
```

Behavior:
- Weights exist → returns immediately (no HuggingFace API calls)
- Weights missing → throws a cache-miss error naming the directory it checked

A sharded bundle counts as present only when every shard its index names is on
disk, so a partially downloaded model is reported missing rather than loaded
with some tensors absent.

This avoids unnecessary network latency on app launch when models are already
cached.

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
| `Qwen3TTSModel` | `cacheDir`, `tokenizerCacheDir`, `offlineMode`; or explicit `fromLocal` directories |
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
