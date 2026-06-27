# Speech Swift

AI speech models for Apple Silicon, powered by MLX Swift and CoreML.

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

On-device speech recognition, synthesis, and understanding for Mac and iOS. Runs locally on Apple Silicon — no cloud, no API keys, no data leaves your device.

**[📚 Full Documentation →](https://soniqo.audio)** · **[🤗 HuggingFace Models](https://huggingface.co/aufklarer)** · **[📝 Blog](https://blog.ivan.digital)** · **[💬 Discord](https://discord.gg/TnCryqEMgu)**

<p align="center">
  <a href="https://trendshift.io/repositories/24196?utm_source=trendshift-badge&amp;utm_medium=badge&amp;utm_campaign=badge-trendshift-24196" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/trendshift/repositories/24196/daily?language=Swift" alt="soniqo%2Fspeech-swift | Trendshift" width="250" height="55"/></a>
</p>

<p align="center">
  <a href="https://youtu.be/x9zgcaW0gUk">
    <img src="https://img.youtube.com/vi/x9zgcaW0gUk/maxresdefault.jpg" width="640" alt="Local Speech AI on a MacBook — watch the 4-minute open-source library tour on YouTube">
  </a>
</p>
<p align="center"><em>Local Speech AI on a MacBook — watch the 4-minute open-source library tour on YouTube</em></p>

**Use cases:** [Voice Agents](https://soniqo.audio/voice-agents) · [Transcription](https://soniqo.audio/transcription) · [Speech Generation](https://soniqo.audio/speech-generation)

**Capability groups:** STT / ASR · Alignment · TTS · LLMs & translation · Speech-to-speech · Enhancement/restoration · Source separation · Music/audio generation · Wake word, VAD, diarization & speaker identity

**STT / ASR**

- **[Qwen3-ASR](https://soniqo.audio/guides/transcribe)** — Speech-to-text (automatic speech recognition, 52 languages, MLX + CoreML)
- **[Parakeet TDT](https://soniqo.audio/guides/parakeet)** — Speech-to-text via CoreML (Neural Engine, NVIDIA FastConformer + TDT decoder, 25 languages)
- **[Omnilingual ASR](https://soniqo.audio/guides/omnilingual)** — Speech-to-text (Meta wav2vec2 + CTC, **1,672 languages** across 32 scripts, CoreML 300M + MLX 300M/1B/3B/7B)
- **[Streaming Dictation](https://soniqo.audio/guides/dictate)** — Real-time dictation with partials and end-of-utterance detection (Parakeet-EOU-120M)
- **[Nemotron Streaming (Multilingual)](https://soniqo.audio/guides/nemotron)** — Low-latency streaming ASR with native punctuation and capitalization (NVIDIA Nemotron-3.5-ASR-Streaming-0.6B, CoreML + MLX, **40 language-locales**)
- **[Nemotron Streaming (English)](https://soniqo.audio/guides/nemotron)** — Low-latency streaming ASR with native punctuation and capitalization (NVIDIA Nemotron-Speech-Streaming-0.6B, CoreML, English-only, smaller and faster than the multilingual variant)

**Alignment**

- **[Qwen3-ForcedAligner](https://soniqo.audio/guides/align)** — Word-level timestamp alignment (audio + text → timestamps)

**TTS / Speech Generation**

- **[Qwen3-TTS](https://soniqo.audio/guides/speak)** — Text-to-speech (highest quality, streaming, custom speakers, 10 languages)
- **[CosyVoice TTS](https://soniqo.audio/guides/cosyvoice)** — Streaming TTS with voice cloning, multi-speaker dialogue, emotion tags (9 languages)
- **[VoxCPM2](https://soniqo.audio/speech-generation)** — 48 kHz studio-quality TTS with voice cloning + instruction-driven voice design (2B, MLX bf16/int8, 30 languages)
- **[Kokoro TTS](https://soniqo.audio/guides/kokoro)** — On-device TTS (82M, CoreML/Neural Engine, 54 voices, iOS-ready, 10 languages)
- **[VibeVoice TTS](https://soniqo.audio/guides/vibevoice)** — Long-form / multi-speaker TTS (Microsoft VibeVoice Realtime-0.5B + 1.5B, MLX, up to 90-min podcast/audiobook synthesis, EN/ZH)
- **[Magpie TTS](https://soniqo.audio/guides/magpie)** — Multilingual TTS (NVIDIA Magpie-TTS Multilingual 357M, MLX INT8 411 MB or CoreML INT8 342 MB, 9 languages, 5 baked speakers, streaming on MLX)
- **[Supertonic TTS](https://soniqo.audio/guides/supertonic)** — On-device flow-matching TTS (Supertone Supertonic-3 99M, CoreML/Neural Engine, 31 languages, 10 voices, G2P-free, 44.1 kHz)
- **[Chatterbox TTS](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16)** — Multilingual TTS with zero-shot voice cloning (Resemble AI Chatterbox Multilingual, MLX fp16 ~1.3 GB, 23 languages, MIT)
- **[OmniVoice TTS](https://huggingface.co/aufklarer/OmniVoice-MLX-int8)** — Non-autoregressive diffusion TTS with zero-shot voice cloning (k2-fsa OmniVoice, Qwen3 backbone, MLX int8 ~1 GB / fp16, 600+ languages, Apache-2.0)

**LLMs & Translation**

- **[Qwen3Chat](https://soniqo.audio/guides/chat)** — On-device LLM chat (Qwen3.5-0.8B MLX/CoreML plus dense Qwen3 4B and Gemma 4 E2B/E4B MLX backends, streaming tokens)
- **[FunctionGemma](https://soniqo.audio/guides/function-calls)** — On-device LLM for structured function / tool calls (Gemma 3 270M, CoreML 8-bit palettized, Neural Engine, ~252 tok/s)
- **[MADLAD-400](https://soniqo.audio/guides/translate)** — Many-to-many translation across 400+ languages (3B, MLX INT4 + INT8, T5 v1.1, Apache 2.0)

**Speech-to-Speech & Voice Agents**

- **[Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate)** — Streaming speech-to-speech translation (FR/ES/PT/DE → EN, MLX INT4 + INT8, Kyutai Moshi/Mimi stack, CC-BY-4.0)
- **[PersonaPlex](https://soniqo.audio/guides/respond)** — Full-duplex speech-to-speech (7B, audio in → audio out, 18 voice presets)

**Enhancement, Separation & Audio Generation**

- **[DeepFilterNet3](https://soniqo.audio/guides/denoise)** — Real-time noise suppression (2.1M params, 48 kHz). Long-form audio above the 60 s single-shot cap is auto-chunked with crossfade — see `enhanceChunked(...)`
- **[Source Separation](https://soniqo.audio/guides/separate)** — Music source separation via HTDemucs (Demucs v4) + Open-Unmix (UMX-HQ / UMX-L, 4 stems: vocals/drums/bass/other, 44.1 kHz stereo)
- **[MAGNeT](https://soniqo.audio/guides/compose)** — Text-to-music generation (Meta MAGNeT Small 300M / Medium 1.5B, MLX INT8, 30 s clips at 32 kHz mono, masked parallel decoding)
- **[FlashSR](https://soniqo.audio/guides/upsample)** — Audio super-resolution (FlashSR ICASSP 2025, MLX, 48 kHz mono, 1-step distilled diffusion, INT4 363 MB / INT8 720 MB)

**Turn Detection, Diarization & Speaker Identity**

- **[Wake-word](https://soniqo.audio/guides/wake-word)** — On-device keyword spotting (KWS Zipformer 3M, CoreML, 26× real-time, configurable keyword list)
- **[VAD](https://soniqo.audio/guides/vad)** — Voice activity detection (Silero streaming, Pyannote offline, FireRedVAD 100+ languages)
- **[Speaker Diarization](https://soniqo.audio/guides/diarize)** — Who spoke when (Pyannote pipeline, Sortformer end-to-end on Neural Engine)
- **[Speaker Embeddings](https://soniqo.audio/guides/embed-speaker)** — WeSpeaker ResNet34 (256-dim), CAM++ (192-dim)

Papers: [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba) · [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba) · [Omnilingual ASR](https://arxiv.org/abs/2511.09690) (Meta) · [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA) · [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba) · [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2) · [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA) · [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai) · [Hibiki](https://arxiv.org/abs/2502.03382) (Kyutai) · [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## News

- **19 Apr 2026** — [MLX vs CoreML on Apple Silicon — A Practical Guide to Picking the Right Backend](https://blog.ivan.digital/mlx-vs-coreml-on-apple-silicon-a-practical-guide-to-picking-the-right-backend-and-why-you-should-f77ddea7b27a)
- **20 Mar 2026** — [We Beat Whisper Large v3 with a 600M Model Running Entirely on Your Mac](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **26 Feb 2026** — [Speaker Diarization and Voice Activity Detection on Apple Silicon — Native Swift with MLX](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **23 Feb 2026** — [NVIDIA PersonaPlex 7B on Apple Silicon — Full-Duplex Speech-to-Speech in Native Swift with MLX](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **12 Feb 2026** — [Qwen3-ASR Swift: On-Device ASR + TTS for Apple Silicon — Architecture and Benchmarks](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## Quick start

Add the package to your `Package.swift`:

```swift
.package(url: "https://github.com/soniqo/speech-swift", branch: "main")
```

Import only the modules you need — every model is its own SPM library, so you don't pay for what you don't use:

```swift
.product(name: "ParakeetStreamingASR", package: "speech-swift"),
.product(name: "SpeechUI",             package: "speech-swift"),  // optional SwiftUI views
```

**Transcribe an audio buffer in 3 lines:**

```swift
import ParakeetStreamingASR

let model = try await ParakeetStreamingASRModel.fromPretrained()
let text = try model.transcribeAudio(audioSamples, sampleRate: 16000)
```

**Live streaming with partials:**

```swift
for await partial in model.transcribeStream(audio: samples, sampleRate: 16000) {
    print(partial.isFinal ? "FINAL: \(partial.text)" : "... \(partial.text)")
}
```

**SwiftUI dictation view in ~10 lines:**

```swift
import SwiftUI
import ParakeetStreamingASR
import SpeechUI

@MainActor
struct DictateView: View {
    @State private var store = TranscriptionStore()

    var body: some View {
        TranscriptionView(finals: store.finalLines, currentPartial: store.currentPartial)
            .task {
                let model = try? await ParakeetStreamingASRModel.fromPretrained()
                guard let model else { return }
                for await p in model.transcribeStream(audio: samples, sampleRate: 16000) {
                    store.apply(text: p.text, isFinal: p.isFinal)
                }
            }
    }
}
```

`SpeechUI` ships only `TranscriptionView` (finals + partials) and `TranscriptionStore` (streaming ASR adapter). Use AVFoundation for audio visualization and playback.

Available SPM products: `Qwen3ASR`, `Qwen3TTS`, `Qwen3TTSCoreML`, `ParakeetASR`, `ParakeetStreamingASR`, `NemotronStreamingASR`, `OmnilingualASR`, `KokoroTTS`, `SupertonicTTS`, `VibeVoiceTTS`, `CosyVoiceTTS`, `VoxCPM2TTS`, `ChatterboxTTS`, `OmniVoiceTTS`, `IndicMioTTS`, `FishAudioTTS`, `MagpieTTS`, `MagpieTTSCoreML`, `MAGNeTMusicGen`, `FlashSR`, `PersonaPlex`, `HibikiTranslate`, `SpeechVAD`, `SpeechEnhancement`, `SourceSeparation`, `Qwen3Chat`, `FunctionGemma`, `SpeechCore`, `SpeechUI`, `AudioCommon`.

## Models

Compact view below. **[Full model catalogue with sizes, quantisations, download URLs, and memory tables → soniqo.audio/architecture](https://soniqo.audio/architecture)**.

| Model | Task | Backends | Sizes | Languages |
|-------|------|----------|-------|-----------|
| [Qwen3-ASR](https://soniqo.audio/guides/transcribe) | Speech → Text | MLX, CoreML (hybrid) | 0.6B, 1.7B | 52 |
| [Parakeet TDT](https://soniqo.audio/guides/parakeet) | Speech → Text | CoreML (ANE) | 0.6B | 25 European |
| [Parakeet EOU](https://soniqo.audio/guides/dictate) | Speech → Text (streaming) | CoreML (ANE) | 120M | 25 European |
| [Nemotron Streaming (Multilingual)](https://soniqo.audio/guides/nemotron) | Speech → Text (streaming, punctuated) | CoreML (ANE), MLX | 0.6B | **40** |
| [Nemotron Streaming (English)](https://soniqo.audio/guides/nemotron) | Speech → Text (streaming, punctuated) | CoreML (ANE) | 0.6B | EN |
| [Omnilingual ASR](https://soniqo.audio/guides/omnilingual) | Speech → Text | CoreML (ANE), MLX | 300M / 1B / 3B / 7B | **[1,672](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)** |
| [Qwen3-ForcedAligner](https://soniqo.audio/guides/align) | Audio + Text → Timestamps | MLX, CoreML | 0.6B | Multi |
| [Qwen3-TTS](https://soniqo.audio/guides/speak) | Text → Speech | MLX, CoreML | 0.6B, 1.7B | 10 |
| [CosyVoice3](https://soniqo.audio/guides/cosyvoice) | Text → Speech | MLX | 0.5B | 9 |
| [VoxCPM2](https://soniqo.audio/speech-generation) | Text → Speech (48 kHz, voice design + cloning) | MLX | 2B (bf16/int8) | 30 |
| [Kokoro-82M](https://soniqo.audio/guides/kokoro) | Text → Speech | CoreML (ANE) | 82M | 10 |
| [Supertonic-3](https://soniqo.audio/guides/supertonic) | Text → Speech (44.1 kHz, flow-matching, G2P-free) | CoreML (ANE) | 99M | 31 |
| [VibeVoice Realtime-0.5B](https://soniqo.audio/guides/vibevoice) | Text → Speech (long-form, multi-speaker) | MLX | 0.5B | EN/ZH |
| [VibeVoice 1.5B](https://soniqo.audio/guides/vibevoice) | Text → Speech (up to 90-min podcast) | MLX | 1.5B | EN/ZH |
| [Magpie-TTS Multilingual](https://soniqo.audio/guides/magpie) | Text → Speech (5 baked speakers, streaming) | MLX / CoreML | 357M (MLX INT8, CoreML INT8) | 9 (CoreML excludes JA) |
| [Chatterbox Multilingual](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16) | Text → Speech (zero-shot cloning) | MLX | 0.8B (fp16) | 23 |
| [OmniVoice](https://huggingface.co/aufklarer/OmniVoice-MLX-int8) | Text → Speech (NAR diffusion, zero-shot cloning) | MLX | 0.8B (int8/fp16) | **600+** |
| [Fish Audio S2 Pro](docs/models/fish-audio-s2-pro.md) | Text → Speech (zero-shot cloning, explicit style markers) | MLX | 0.5B-class (fp16) | Multilingual |
| [Qwen3Chat](https://soniqo.audio/guides/chat) | Text → Text (LLM) | MLX, CoreML | 0.8B, 4B, E2B/E4B | Multi |
| [FunctionGemma](https://soniqo.audio/guides/function-calls) | Text → Tool calls (LLM) | CoreML | 270M | EN-tuned |
| [MADLAD-400](https://soniqo.audio/guides/translate) | Text → Text (Translation) | MLX | 3B | **400+** |
| [Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate) | Speech → Speech (Translation) | MLX | 3B | FR/ES/PT/DE → EN |
| [PersonaPlex](https://soniqo.audio/guides/respond) | Speech → Speech | MLX | 7B | EN |
| [Silero VAD](https://soniqo.audio/guides/vad) | Voice Activity Detection | MLX, CoreML | 309K | Agnostic |
| [Pyannote](https://soniqo.audio/guides/diarize) | VAD + Diarization | MLX | 1.5M | Agnostic |
| [Sortformer](https://soniqo.audio/guides/diarize) | Diarization (E2E) | CoreML (ANE) | — | Agnostic |
| [DeepFilterNet3](https://soniqo.audio/guides/denoise) | Speech Enhancement | CoreML | 2.1M | Agnostic |
| [Sidon](https://soniqo.audio/guides/restore) | Speech Restoration (denoise + dereverb, 48 kHz) | CoreML | w2v-BERT 2.0 + DAC (fp16/int8) | Agnostic |
| [HTDemucs (Demucs v4)](https://soniqo.audio/guides/separate) | Source Separation | MLX | 168M | Agnostic |
| [Open-Unmix](https://soniqo.audio/guides/separate) | Source Separation | MLX | 8.6M | Agnostic |
| [MAGNeT](https://soniqo.audio/guides/compose) | Text → Music (30s @ 32 kHz) | MLX | 300M / 1.5B (int4/int8) | EN prompts |
| [FlashSR](https://soniqo.audio/guides/upsample) | Audio super-resolution (48 kHz) | MLX | 363 MB / 720 MB (int4/int8) | Agnostic |
| [WeSpeaker](https://soniqo.audio/guides/embed-speaker) | Speaker Embedding | MLX, CoreML | 6.6M | Agnostic |

## Installation

### Homebrew

Requires native ARM Homebrew (`/opt/homebrew`). Rosetta/x86_64 Homebrew is not supported.

```bash
brew install speech
```

Then:

```bash
speech transcribe recording.wav
speech speak "Hello world"
speech translate "Hello, how are you?" --to es
speech respond --input question.wav --transcript
speech-server --port 8080            # local HTTP / WebSocket server (OpenAI-compatible /v1/realtime + /v1/audio/transcriptions)
```

**[Full CLI reference →](https://soniqo.audio/cli)**

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

Import only what you need — every model is its own SPM target:

```swift
import Qwen3ASR             // Speech recognition (MLX)
import ParakeetASR          // Speech recognition (CoreML, batch)
import ParakeetStreamingASR // Streaming dictation with partials + EOU
import NemotronStreamingASR // Multilingual streaming ASR with native punctuation (0.6B, 40 langs)
import OmnilingualASR       // 1,672 languages (CoreML + MLX)
import Qwen3TTS             // Text-to-speech
import CosyVoiceTTS         // Text-to-speech with voice cloning
import VoxCPM2TTS           // 48 kHz TTS with voice cloning + voice design (2B)
import KokoroTTS            // Text-to-speech (iOS-ready)
import VibeVoiceTTS         // Long-form / multi-speaker TTS (EN/ZH)
import MagpieTTS            // Multilingual TTS (NVIDIA Magpie 357M, MLX, 9 langs)
import MagpieTTSCoreML      // Magpie CoreML backend (hybrid CoreML + MLX, 8 langs)
import FishAudioTTS         // Experimental Fish Audio S2 Pro runtime with voice cloning
import Qwen3Chat            // On-device LLM chat
import FunctionGemma    // On-device tool-call LLM
import MADLADTranslation    // Many-to-many translation across 400+ languages
import HibikiTranslate      // Streaming speech-to-speech translation (FR/ES/PT/DE → EN)
import PersonaPlex          // Full-duplex speech-to-speech
import SpeechVAD            // VAD + speaker diarization + embeddings
import SpeechEnhancement    // Noise suppression
import SpeechRestoration    // Speech restoration — denoise + dereverb (Sidon, CoreML, 48 kHz)
import SourceSeparation     // Music source separation (Open-Unmix, 4 stems)
import SpeechUI             // SwiftUI components for streaming transcripts
import AudioCommon          // Shared protocols and utilities
```

### Requirements

- Swift 6+, Xcode 16+ (with Metal Toolchain)
- macOS 15+ (Sequoia) or iOS 18+, Apple Silicon (M1/M2/M3/M4)

The macOS 15 / iOS 18 minimum comes from [MLState](https://developer.apple.com/documentation/coreml/mlstate) — Apple's persistent ANE state API used by the CoreML pipelines (Qwen3-ASR, Qwen3-Chat, Qwen3-TTS) to keep KV caches resident on the Neural Engine across token steps.

### Build from source

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

`make build` compiles the Swift package **and** the MLX Metal shader library. The Metal library is required for GPU inference — without it you'll see `Failed to load the default metallib` at runtime. `make debug` for debug builds, `make test` for the test suite.

**[Full build and install guide →](https://soniqo.audio/getting-started)**

## Demo apps

- **[DictateDemo](Examples/DictateDemo/)** ([docs](https://soniqo.audio/guides/dictate)) — macOS menu-bar streaming dictation with live partials, VAD-driven end-of-utterance detection, and one-click copy. Runs as a background agent (Parakeet-EOU-120M + Silero VAD).
- **[iOSEchoDemo](Examples/iOSEchoDemo/)** — iOS echo demo (Parakeet ASR + Kokoro TTS). Device and simulator.
- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — Conversational voice assistant with mic input, VAD, and multi-turn context. macOS. RTF ~0.94 on M2 Max (faster than real-time).
- **[SpeechDemo](Examples/SpeechDemo/)** — Dictation and TTS synthesis in a tabbed interface. macOS.

Each demo's README has build instructions.

## Code examples

The snippets below show the minimal path for each domain. Every section links to a full guide on [soniqo.audio](https://soniqo.audio) with configuration options, multiple backends, streaming patterns, and CLI recipes.

### Speech-to-Text — [full guide →](https://soniqo.audio/guides/transcribe)

```swift
import Qwen3ASR

let model = try await Qwen3ASRModel.fromPretrained()
let text = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

Alternative backends: [Parakeet TDT](https://soniqo.audio/guides/parakeet) (CoreML, 32× realtime), [Omnilingual ASR](https://soniqo.audio/guides/omnilingual) (1,672 languages, CoreML or MLX), [Streaming dictation](https://soniqo.audio/guides/dictate) (live partials).

### Forced Alignment — [full guide →](https://soniqo.audio/guides/align)

```swift
import Qwen3ASR

let aligner = try await Qwen3ForcedAligner.fromPretrained()
let aligned = aligner.align(
    audio: audioSamples,
    text: "Can you guarantee that the replacement part will be shipped tomorrow?",
    sampleRate: 24000
)
for word in aligned {
    print("[\(word.startTime)s - \(word.endTime)s] \(word.text)")
}
```

### Text-to-Speech — [full guide →](https://soniqo.audio/guides/speak)

```swift
import Qwen3TTS
import AudioCommon

let model = try await Qwen3TTSModel.fromPretrained()
let audio = model.synthesize(text: "Hello world", language: "english")
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

Alternative TTS engines: [CosyVoice3](https://soniqo.audio/guides/cosyvoice) (streaming + voice cloning + emotion tags), [Kokoro-82M](https://soniqo.audio/guides/kokoro) (iOS-ready, 54 voices), [VibeVoice](https://soniqo.audio/guides/vibevoice) (long-form podcast / multi-speaker, EN/ZH), [Fish Audio S2 Pro](docs/inference/fish-audio-s2-pro.md) (experimental zero-shot cloning + bracket style markers), [Voice cloning](https://soniqo.audio/guides/voice-cloning).

### Speech-to-Speech — [full guide →](https://soniqo.audio/guides/respond)

```swift
import PersonaPlex

let model = try await PersonaPlexModel.fromPretrained()
let responseAudio = model.respond(userAudio: userSamples)
// 24 kHz mono Float32 output ready for playback
```

### LLM Chat — [full guide →](https://soniqo.audio/guides/chat)

```swift
import Qwen3Chat
import FunctionGemma

let chat = try await Qwen35MLXChat.fromPretrained()
chat.chat(messages: [(.user, "Explain MLX in one sentence")]) { token, isFinal in
    print(token, terminator: "")
}
```

### Translation — [full guide →](https://soniqo.audio/guides/translate)

```swift
import MADLADTranslation

let translator = try await MADLADTranslator.fromPretrained()
let es = try translator.translate("Hello, how are you?", to: "es")
// → "Hola, ¿cómo estás?"
```

### Speech Translation — [full guide →](https://soniqo.audio/guides/audio-translate)

```swift
import HibikiTranslate
import AudioCommon

let model = try await HibikiTranslateModel.fromPretrained()
let pcm = try AudioFileLoader.load(url: input, targetSampleRate: 24000)
let (englishAudio, textTokens) = model.translate(
    sourceAudio: pcm, sourceLanguage: .fr
)
// Hibiki Zero-3B — FR/ES/PT/DE → EN, on-device, streaming Mimi codec
```

### Voice Activity Detection — [full guide →](https://soniqo.audio/guides/vad)

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
for s in segments { print("\(s.startTime)s → \(s.endTime)s") }
```

### Speaker Diarization — [full guide →](https://soniqo.audio/guides/diarize)

```swift
import SpeechVAD

let diarizer = try await DiarizationPipeline.fromPretrained()
let segments = diarizer.diarize(audio: samples, sampleRate: 16000)
for s in segments { print("Speaker \(s.speakerId): \(s.startTime)s - \(s.endTime)s") }
```

### Speech Enhancement — [full guide →](https://soniqo.audio/guides/denoise)

```swift
import SpeechEnhancement

let denoiser = try await DeepFilterNet3Model.fromPretrained()
let clean = try denoiser.enhance(audio: noisySamples, sampleRate: 48000)
```

### Speech Restoration — [full guide →](https://soniqo.audio/guides/restore)

Joint denoise **and** dereverb with [Sidon](https://arxiv.org/abs/2509.17052) (w2v-BERT 2.0 predictor + DAC vocoder, Core ML). Unlike a generic noise suppressor, Sidon is trained to preserve speaker identity, so it is well suited to cleaning a noisy or reverberant voice-cloning reference before TTS. Input is 16 kHz; output is 48 kHz mono.

```swift
import SpeechRestoration

let restorer = try await SpeechRestorer.fromPretrained()          // .fp16 (default) or .int8
let clean = try restorer.restore(audio: noisySamples, sampleRate: 16000)  // → 48 kHz
```

From the CLI:

```bash
speech restore noisy.wav -o clean.wav            # denoise + dereverb, 48 kHz output
speech restore noisy.wav --variant int8          # smaller, lower peak RAM

# Clean a voice-cloning reference before TTS (opt-in; preserves speaker identity):
speech speak "Hello" --engine voxcpm2 --voice-sample ref.wav --clean-reference
```

### Voice Pipeline (ASR → LLM → TTS) — [full guide →](https://soniqo.audio/voice-agents)

```swift
import SpeechCore

let pipeline = VoicePipeline(
    stt: parakeetASR,
    tts: qwen3TTS,
    vad: sileroVAD,
    config: .init(mode: .voicePipeline),
    onEvent: { event in print(event) }
)
pipeline.start()
pipeline.pushAudio(micSamples)
```

`VoicePipeline` is the real-time voice-agent state machine (powered by [speech-core](https://github.com/soniqo/speech-core)) with VAD-driven turn detection, interruption handling, and eager STT. It connects any `SpeechRecognitionModel` + `SpeechGenerationModel` + `StreamingVADProvider`.

### HTTP API server

```bash
speech-server --port 8080
```

Exposes every model via HTTP REST + WebSocket endpoints, including OpenAI-compatible APIs: a Realtime WebSocket at `/v1/realtime` and a transcription REST endpoint at `/v1/audio/transcriptions`. See [`Sources/AudioServer/`](Sources/AudioServer/).

## Architecture

speech-swift is split into one SPM target per model so consumers only pay for what they import. Shared infrastructure lives in `AudioCommon` (protocols, audio I/O, HuggingFace downloader, `SentencePieceModel`) and `MLXCommon` (weight loading, `QuantizedLinear` helpers, `SDPA` multi-head attention helper).

**[Full architecture diagram with backends, memory tables, and module map → soniqo.audio/architecture](https://soniqo.audio/architecture)** · **[API reference → soniqo.audio/api](https://soniqo.audio/api)** · **[Benchmarks → soniqo.audio/benchmarks](https://soniqo.audio/benchmarks)**

Local docs (repo):
- **Models:** [Qwen3-ASR](docs/models/asr-model.md) · [Qwen3-TTS](docs/models/tts-model.md) · [CosyVoice](docs/models/cosyvoice-tts.md) · [Kokoro](docs/models/kokoro-tts.md) · [VibeVoice](docs/models/vibevoice.md) · [Fish Audio S2 Pro](docs/models/fish-audio-s2-pro.md) · [Parakeet TDT](docs/models/parakeet-asr.md) · [Parakeet Streaming](docs/models/parakeet-streaming-asr.md) · [Nemotron Streaming](docs/models/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/models/omnilingual-asr.md) · [PersonaPlex](docs/models/personaplex.md) · [Hibiki](docs/models/hibiki.md) · [FireRedVAD](docs/models/fireredvad.md) · [Source Separation](docs/models/source-separation.md) · [HTDemucs](docs/models/htdemucs.md) · [MAGNeT](docs/models/magnet-music-gen.md) · [FlashSR](docs/models/flashsr.md) · [Qwen3Chat](docs/models/qwen35-chat.md) · [Gemma 4 Chat](docs/models/gemma4-chat.md) · [Qwen3 Dense Chat](docs/models/qwen3-dense-chat.md)
- **Inference:** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md) · [Parakeet TDT](docs/inference/parakeet-asr-inference.md) · [Parakeet Streaming](docs/inference/parakeet-streaming-asr-inference.md) · [Nemotron Streaming](docs/inference/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/inference/omnilingual-asr-inference.md) · [TTS](docs/inference/qwen3-tts-inference.md) · [VibeVoice](docs/inference/vibevoice-inference.md) · [Fish Audio S2 Pro](docs/inference/fish-audio-s2-pro.md) · [Hibiki](docs/inference/hibiki-inference.md) · [MAGNeT](docs/inference/magnet-music-gen.md) · [FlashSR](docs/inference/flashsr.md) · [Forced Aligner](docs/inference/forced-aligner.md) · [Silero VAD](docs/inference/silero-vad.md) · [Speaker Diarization](docs/inference/speaker-diarization.md) · [Speech Enhancement](docs/inference/speech-enhancement.md)
- **Reference:** [Shared Protocols](docs/shared-protocols.md)

## Cache configuration

Model weights download from HuggingFace on first use and cache to `~/Library/Caches/qwen3-speech/`. Override with `QWEN3_CACHE_DIR` (CLI) or `cacheDir:` (Swift API). All `fromPretrained()` entry points also accept `offlineMode: true` to skip network when weights are already cached.

Users in mainland China (or anywhere `huggingface.co` is slow/blocked) can fetch from a mirror by setting `HF_ENDPOINT`, e.g. `export HF_ENDPOINT=https://hf-mirror.com`.

See [`docs/inference/cache-and-offline.md`](docs/inference/cache-and-offline.md) for full details including sandboxed iOS container paths.

## MLX Metal library

If you see `Failed to load the default metallib` at runtime, the Metal shader library is missing. Run `make build` or `./scripts/build_mlx_metallib.sh release` after a manual `swift build`. If the Metal Toolchain is missing, install it first:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Testing

```bash
make test                            # full suite (unit + E2E with model downloads)
swift test --skip E2E                # unit only (CI-safe, no downloads)
swift test --filter Qwen3ASRTests    # specific module
```

E2E test classes use the `E2E` prefix so CI can filter them out with `--skip E2E`. See [CLAUDE.md](CLAUDE.md#testing) for the full testing convention.

## Contributing

PRs welcome — bug fixes, new model integrations, documentation. Fork, create a feature branch, `make build && make test`, open a PR against `main`.

## License

Apache 2.0
