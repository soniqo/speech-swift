# Agent Instructions

AI speech models for Apple Silicon (MLX Swift). ASR, TTS, speech-to-speech, VAD, diarization, speech enhancement.

## Communication Style

- Communicate at the conceptual level first: what changed, why it matters, risk, tests, and recommendation.
- Keep status updates short and understandable. Do not dump command logs, low-level mechanics, or long implementation detail unless directly asked.
- For PRs and fixes, default to this summary shape: what changed, architecture fit, regression risk, tests run or needed, and recommendation.
- If there are bugs, blockers, or regression risks, lead with them clearly and briefly.
- When the user asks a direct question, answer it directly before adding supporting detail.

## Workflow

- **Always work in a separate git worktree** so concurrent agents don't fight over the same working directory. Create one with `git worktree add ../speech-swift-<task> <branch>`, do all edits there, push from there. Multiple agents may be running against this repo at the same time — checking out branches in the shared working copy clobbers their state and silently loses WIP files. Delete the worktree (`git worktree remove`) when the task is done.
- **Never commit, push, or comment on GitHub without explicit user confirmation.** Draft first, ask to confirm, then execute.
- **Every README.md change must update all 13 translations** (`README_zh.md`, `README_ja.md`, `README_ko.md`, `README_es.md`, `README_de.md`, `README_fr.md`, `README_hi.md`, `README_pt.md`, `README_ru.md`, `README_ar.md`, `README_th.md`, `README_tr.md`, `README_vi.md`). No exceptions.
- **Keep docs and comments scoped to this package.** Model docs, code comments, and PR descriptions describe this package's models, APIs, and formats only — never downstream consumer apps or their integration rules.

## Git Conventions

- Never mention Claude, Codex, AI, or any AI tool in commit messages, PR titles, PR descriptions, comments, docs, or co-author tags. Do not add tool-name prefixes such as `[codex]`; use neutral project wording.
- No `Co-Authored-By` lines in commits
- **Never amend commits or force push** unless the user explicitly asks for it
- Always use branches and PRs — commit history must be preserved

## Build

```bash
# Release build (recommended)
make build

# Debug build
make debug

# Run tests (builds debug first)
make test

# Clean
make clean
```

Or manually:

```bash
swift build -c release --disable-sandbox
./scripts/build_mlx_metallib.sh release
```

The metallib step compiles MLX Metal shaders — without it, inference runs ~5x slower due to JIT shader compilation.

## Skills (Slash Commands)

Project skills in `.claude/skills/`:

The same skills are exposed to agents that scan `.codex/skills/` through relative symlinks; keep both paths in sync when adding or renaming a project skill.

| Command | Description |
|---------|-------------|
| `/build` or `/build release` | Release build with metallib |
| `/build debug` | Debug build |
| `/test` or `/test unit` | Run unit tests (skip E2E) |
| `/test e2e` | Full test suite with model downloads |
| `/test FilterName` | Run specific test filter |
| `/review-pr <PR>` | Review PR architecture fit, regression risk, tests, and merge readiness |
| `/benchmark asr` | Benchmark ASR speed |
| `/benchmark tts` | Benchmark TTS speed |
| `/benchmark vad` | VAD benchmark on VoxConverse |
| `/benchmark diarize` | DER benchmark on VoxConverse |

## Project Structure

**ASR**
- `Sources/Qwen3ASR/` — Speech-to-text (Qwen3-ASR, MLX)
- `Sources/ParakeetASR/` — Speech-to-text (Parakeet TDT, CoreML)
- `Sources/ParakeetStreamingASR/` — Streaming speech-to-text (Parakeet EOU 120M, CoreML)
- `Sources/NemotronStreamingASR/` — Streaming speech-to-text (Nemotron, CoreML)
- `Sources/OmnilingualASR/` — Speech-to-text (Meta wav2vec2 CTC, 1,672 languages, CoreML 300M + MLX 300M/1B/3B/7B)
- `Sources/WhisperASR/` — Speech-to-text (Whisper Large-v3 Turbo, native CoreML runtime)

**TTS**
- `Sources/Qwen3TTS/` — Text-to-speech (Qwen3-TTS, MLX)
- `Sources/Qwen3TTSCoreML/` — Text-to-speech (Qwen3-TTS 0.6B, CoreML, 6-model pipeline)
- `Sources/CosyVoiceTTS/` — Text-to-speech (CosyVoice3, streaming, MLX)
- `Sources/VoxCPM2TTS/` — Text-to-speech (VoxCPM2, MLX, 48 kHz, voice cloning + voice design)
- `Sources/IndexTTS2TTS/` — Text-to-speech (IndexTTS2, zero-shot voice cloning, MLX, speaker/emotion/pause controls)
- `Sources/F5TTS/` — Text-to-speech (F5-TTS v1 Base, zero-shot voice cloning, DiT flow matching + Vocos, MLX, 24 kHz; non-commercial license)
- `Sources/HiggsTTS/` — Text-to-speech (Higgs TTS 3 4B, conversational + zero-shot voice cloning, Qwen3 backbone + fused codebooks, MLX, 24 kHz; non-commercial license)
- `Sources/KokoroTTS/` — Text-to-speech (Kokoro-82M, CoreML, iOS-ready)
- `Sources/MagpieTTS/` — Multilingual TTS (MLX, 8 codebooks, NanoCodec)
- `Sources/MagpieTTSCoreML/` — Magpie TTS CoreML variant
- `Sources/ChatterboxTTS/` — Text-to-speech (MLX)
- `Sources/OmniVoiceTTS/` — Text-to-speech (MLX)
- `Sources/VibeVoiceTTS/` — Text-to-speech (MLX)
- `Sources/SupertonicTTS/` — Text-to-speech (CoreML)
- `Sources/FishAudioTTS/` — Text-to-speech (Fish Audio S2 Pro, MLX, 44.1 kHz, voice cloning; research/non-commercial license)
- `Sources/IndicMioTTS/` — Text-to-speech (Indic-Mio, Hindi/Indic with inline emotion markers, MLX, 24 kHz, voice cloning)

**Speech-to-Speech & Translation**
- `Sources/PersonaPlex/` — Speech-to-speech (PersonaPlex 7B, full-duplex, MLX)
- `Sources/HibikiTranslate/` — Speech-to-speech translation (depends on PersonaPlex + MLX)
- `Sources/MADLADTranslation/` — Text translation (MLX)

**Audio Processing**
- `Sources/SpeechVAD/` — VAD (Silero + Pyannote), speaker diarization, speaker embedding (WeSpeaker)
- `Sources/SpeechEnhancement/` — Noise suppression (DeepFilterNet3, CoreML)
- `Sources/SpeechRestoration/` — Audio restoration (CoreML)
- `Sources/SourceSeparation/` — Source separation (MLX)
- `Sources/SpeechWakeWord/` — Wake-word detection

**Music Generation**
- `Sources/MAGNeTMusicGen/` — Text-to-music (MAGNeT, 30 s, 32 kHz, MLX)
- `Sources/StableAudio3MusicGen/` — Music generation (Stable Audio 3, MLX)
- `Sources/FlashSR/` — Audio super-resolution (MLX)

**Avatar**
- `Sources/Audio2Face3D/` — Speech-driven avatar motion (NVIDIA Audio2Face-3D, timestamped coefficient frames, MLX; `speech avatar-motion` CLI command)

**On-Device LLM**
- `Sources/Qwen3Chat/` — On-device LLM chat (Qwen3.5-0.8B, MLX + CoreML, INT4/INT8)
- `Sources/FunctionGemma/` — On-device Gemma function-calling (swift-transformers)

**Infrastructure**
- `Sources/MLXCommon/` — Shared MLX utilities (weight loading, quantized layers, memory estimation, `SDPA` multi-head attention helper)
- `Sources/AudioCommon/` — Audio I/O, protocols, HuggingFace downloader, shared `SentencePieceModel` protobuf reader
- `Sources/SpeechCore/` — Swift wrapper around `CSpeechCore` XCFramework (pre-built binary from `soniqo/speech-core` — cannot be modified in this repo)
- `Sources/SpeechUI/` — SwiftUI components for speech apps
- `Sources/AudioServer/` — Hummingbird HTTP + WebSocket server (backs `speech-server` binary)
- `Sources/AudioCLILib/` — CLI command implementations
- `Sources/AudioCLI/` — CLI entry point (`speech` binary; `audio` is a deprecated alias)
- `Sources/AudioServerCLI/` — Server entry point (`speech-server` binary)
- `Sources/AsrBenchmark/` — Multi-engine WER/RTF benchmark tool, includes WhisperKit comparison (backs `asr-bench` binary)
- `Sources/BenchmarkSupport/` — Shared benchmark utilities (dataset handling, metrics) for the benchmark binaries
- `Sources/VadBenchmark/` — VAD benchmark (backs `vad-bench` binary)
- `Sources/DiarizationBenchmark/` — Diarization DER benchmark (backs `diarization-bench` binary)
- `Tests/` — Unit and integration tests
- `scripts/` — Model conversion (PyTorch → MLX/CoreML), benchmarking
- `Examples/` — Demo apps (PersonaPlexDemo, SpeechDemo, iOSEchoDemo)

## Key Conventions

- Swift 6, macOS 15+ / iOS 18+, Apple Silicon (M-series)
- MLX for GPU inference (Metal), CoreML for Neural Engine (DeepFilterNet3, Kokoro, Qwen3-TTS, Silero VAD optional)
- Models are downloaded from HuggingFace on first use, cached in `~/Library/Caches/qwen3-speech/`
- All audio processing uses Float32 PCM, resampled to model-specific rates internally
- `DiarizedSegment`, `SpeechSegment`, protocol types defined in `Sources/AudioCommon/Protocols.swift`
- Tests that use MLX arrays require the compiled metallib; config/logic-only tests work without it
- `CSpeechCore` is a pre-built XCFramework downloaded from `soniqo/speech-core` releases — `Sources/SpeechCore/` is only a thin Swift wrapper. Changes to core behavior require a new release of that separate repo.

## Testing

Curated local smoke/regression suite. This builds debug + metallib first and
runs a focused no-download filter, not the full CI matrix:
```bash
make test
```

CI builds tests, builds the debug metallib, then runs `swift test --skip-build`
with explicit skips for E2E and model-heavy suites. Treat
`.github/workflows/tests.yml` as the source of truth for the exact hosted
runner command.

Run a single test:
```bash
swift test --filter TestClassName/testMethodName --disable-sandbox
```

Full local suite including E2E. Build the debug metallib first; tests may
download models and require Apple Silicon GPU/ANE support:
```bash
make debug
swift test --disable-sandbox
```

### Testing requirements for new code

**Every new feature, model, or module MUST include tests:**

- **Unit tests**: Config parsing, data structures, weight loading, math/DSP logic — no GPU or model downloads needed
- **E2E tests**: Full pipeline with real model weights — verify correct output (e.g., ASR round-trip, correct transcription text)
- **Regression tests**: When fixing bugs, add a test that would have caught the bug

**Test organization**: Place tests in `Tests/<ModuleName>Tests/`. Follow existing patterns (e.g., `Qwen3ASRTests/`, `SpeechVADTests/`).

**E2E test naming**: Prefix E2E test classes with `E2E` (e.g., `E2ETranscriptionTests`, `E2EDiarizationTests`). CI uses `--skip E2E` regex to filter out all E2E tests that require model downloads — only unit tests run in the pipeline. E2E tests run locally with the full-suite command above, not `make test`. **CRITICAL**: Any test class that downloads models or requires GPU inference MUST be prefixed with `E2E`. Unit test classes must NOT contain `E2E` in their name.

**What to test per category:**
| Change | Required tests |
|--------|---------------|
| New model/module | Unit (config, weight loading) + E2E (inference produces correct output) |
| New CLI command | Unit (argument parsing) + E2E (end-to-end with real files) |
| Bug fix | Regression test reproducing the bug |
| New protocol/type | Unit test for conformance and behavior |
| DSP/audio processing | Unit test with known input/output pairs |

## CLI

The `speech` binary is the main entry point (`audio` is a deprecated alias that still works but prints a warning):

```bash
.build/release/speech transcribe recording.wav          # ASR
.build/release/speech speak "Hello" --output hi.wav     # TTS (Qwen3-TTS default)
.build/release/speech speak "Hello" --engine voxcpm2 --voxcpm2-variant int8 -o hi.wav  # VoxCPM2 (48 kHz)
.build/release/speech respond --input q.wav             # Speech-to-speech
.build/release/speech diarize meeting.wav               # Speaker diarization (pyannote)
.build/release/speech diarize meeting.wav --engine sortformer  # Sortformer (CoreML, end-to-end)
.build/release/speech diarize meeting.wav --rttm        # RTTM output
.build/release/speech vad audio.wav                     # Voice activity detection
.build/release/speech embed-speaker voice.wav           # Speaker embedding
.build/release/speech denoise noisy.wav                 # Speech enhancement
.build/release/speech compose "happy rock" -o music.wav # MAGNeT text-to-music (30s, 32 kHz)
.build/release/speech kokoro "Hello" --voice af_heart   # Kokoro TTS (iOS)
.build/release/speech qwen3-tts-coreml "Hello"          # Qwen3-TTS CoreML (6-model pipeline)
```

The `speech-server` binary exposes an HTTP + WebSocket API (Hummingbird):

```bash
.build/release/speech-server --port 8080
```

The `asr-bench` binary benchmarks multiple ASR engines (WER, RTF, peak RSS) including WhisperKit comparison:

```bash
.build/release/asr-bench --help
```

## Documentation

### Local docs (`docs/`)

Architecture and implementation docs live in this repo:

```
docs/
  models/                       Model architecture, weights, layers
    asr-model.md                Qwen3-ASR architecture
    tts-model.md                Qwen3-TTS architecture
    cosyvoice-tts.md            CosyVoice3 architecture
    voxcpm2-tts.md              VoxCPM2 architecture (48 kHz, voice cloning + voice design)
    magpie-tts.md               Magpie-TTS Multilingual architecture (4-bundle MLX, 8 codebooks, NanoCodec)
    magnet-music-gen.md         MAGNeT music generation (T5 + EnCodec, masked parallel decoding)
    kokoro-tts.md               Kokoro-82M architecture
    parakeet-asr.md             Parakeet TDT architecture
    personaplex.md              PersonaPlex architecture
    fireredvad.md               FireRedVAD (DFSMN) architecture
  inference/                    Pipelines, usage, configs
    qwen3-asr-inference.md      Qwen3-ASR inference pipeline
    parakeet-asr-inference.md   Parakeet TDT inference (CoreML)
    qwen3-tts-inference.md      TTS inference pipeline
    voxcpm2-inference.md        VoxCPM2 inference (48 kHz, --voxcpm2-variant bf16/int8)
    magpie-tts.md               Magpie-TTS Multilingual inference (CLI flags, languages, performance)
    magnet-music-gen.md         MAGNeT music generation CLI + tuning
    forced-aligner.md           Forced alignment pipeline
    silero-vad.md               Silero VAD streaming
    fireredvad.md               FireRedVAD inference + tuning results
    speaker-diarization.md      Speaker diarization pipeline
    speech-enhancement.md       DeepFilterNet3 pipeline
  audio/                        Audio I/O, playback, voice pipeline
    playback.md                 Streaming playback, pre-buffer, Apple audio architecture
    voice-pipeline.md           VoicePipeline state machine, events, config
  benchmarks/                   WER, DER, RTF results
  shared-protocols.md       Protocol reference (cross-cutting)
```

**Keep local docs in sync when making code changes.**

### Documentation site (soniqo-web)

The public documentation is hosted at **https://soniqo.audio** (Firebase Hosting) and lives in a separate private repository: **soniqo-web**.

**Whenever code changes are made in this repo, both local docs AND the soniqo-web site must be updated.**

### What requires a docs update

- New features or capabilities added
- CLI commands added, removed, or flags changed
- Public API changes (protocols, types, function signatures)
- New models or model variants added
- Performance characteristics changed
- Build requirements or installation steps changed
- New modules or source structure changes

### Documentation site structure

```
soniqo-web/public/
  index.html                Landing page (feature grid, performance stats)
  getting-started/          Installation, build instructions, quick start
  guides/
    transcribe/             Qwen3-ASR guide
    parakeet/               Parakeet TDT guide
    speak/                  Qwen3-TTS guide
    cosyvoice/              CosyVoice3 guide
    voice-cloning/          Voice cloning guide
    respond/                PersonaPlex guide
    vad/                    VAD guide (Pyannote + Silero)
    diarize/                Speaker diarization guide
    embed-speaker/          Speaker embeddings guide
    denoise/                Speech enhancement guide
    align/                  Forced alignment guide
    kokoro/                 Kokoro-82M guide (iOS TTS)
  cli/                      CLI command reference (all flags/options)
  api/                      Protocols and shared types
  architecture/             Module structure, backends, weight formats
```

### README translations

Translated READMEs live in the repo root: `README_zh.md`, `README_ja.md`, `README_ko.md`, `README_es.md`, `README_de.md`, `README_fr.md`, `README_hi.md`, `README_pt.md`, `README_ru.md`, `README_ar.md`, `README_th.md`, `README_tr.md`, `README_vi.md`. **Whenever README.md is updated, all translations must be updated to match.** Each translation links back to the main README and lists all available languages at the top.

### Mapping: code changes → docs pages

| Code change | Local docs | soniqo-web page(s) |
|---|---|---|
| CLI flag added/changed | Relevant inference doc | `/cli/index.html` + relevant guide |
| New model/module | New model + inference doc | Landing page + new guide + architecture |
| Protocol change | `shared-protocols.md` | `/api/index.html` |
| Performance improvement | `benchmarks/` | Landing page perf section + relevant guide |
| Build/install change | — | `/getting-started/index.html` |
| New CLI command | Relevant inference doc | `/cli/index.html` + new guide + landing page |
| Build/dependency change in demo | `Examples/<Demo>/README.md` | — |
| New demo app | `Examples/<Demo>/README.md` | Landing page + relevant guide |
