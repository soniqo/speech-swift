# Speech Swift

KI-Sprachmodelle für Apple Silicon, basierend auf MLX Swift und CoreML.

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

Spracherkennung, -synthese und -verständnis auf dem Gerät für Mac und iOS. Läuft vollständig lokal auf Apple Silicon — keine Cloud, keine API-Schlüssel, keine Daten verlassen das Gerät.

**[📚 Vollständige Dokumentation →](https://soniqo.audio/de)** · **[🤗 HuggingFace-Modelle](https://huggingface.co/aufklarer)** · **[📝 Blog](https://blog.ivan.digital)** · **[💬 Discord](https://discord.gg/TnCryqEMgu)**

<p align="center">
  <a href="https://www.producthunt.com/products/speech-swift?embed=true&amp;utm_source=badge-featured&amp;utm_medium=badge&amp;utm_campaign=badge-speech-swift" target="_blank" rel="noopener noreferrer"><img alt="speech-swift -  The whole speech stack, on your laptop. | Product Hunt" width="250" height="54" src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1151422&amp;theme=light&amp;t=1779261593657"></a>
</p>

<p align="center">
  <a href="https://youtu.be/x9zgcaW0gUk">
    <img src="https://img.youtube.com/vi/x9zgcaW0gUk/maxresdefault.jpg" width="640" alt="Lokale Sprach-KI auf einem MacBook — die vierminütige Tour durch die Open-Source-Bibliothek auf YouTube ansehen">
  </a>
</p>
<p align="center"><em>Lokale Sprach-KI auf einem MacBook — die vierminütige Tour durch die Open-Source-Bibliothek auf YouTube ansehen</em></p>

**Anwendungsfälle:** [Sprachagenten](https://soniqo.audio/de/voice-agents) · [Transkription](https://soniqo.audio/de/transcription) · [Sprachsynthese](https://soniqo.audio/de/speech-generation)

- **[Qwen3-ASR](https://soniqo.audio/de/guides/transcribe)** — Sprache-zu-Text (automatische Spracherkennung, 52 Sprachen, MLX + CoreML)
- **[Parakeet TDT](https://soniqo.audio/de/guides/parakeet)** — Sprache-zu-Text über CoreML (Neural Engine, NVIDIA FastConformer + TDT-Decoder, 25 Sprachen)
- **[Omnilingual ASR](https://soniqo.audio/de/guides/omnilingual)** — Sprache-zu-Text (Meta wav2vec2 + CTC, **1.672 Sprachen** in 32 Schriften, CoreML 300M + MLX 300M/1B/3B/7B)
- **[Streaming-Diktat](https://soniqo.audio/de/guides/dictate)** — Echtzeit-Diktat mit Teilergebnissen und Äußerungsende-Erkennung (Parakeet-EOU-120M)
- **[Nemotron Streaming (Mehrsprachig)](https://soniqo.audio/de/guides/nemotron)** — Streaming-ASR mit geringer Latenz, nativer Interpunktion und Großschreibung (NVIDIA Nemotron-3.5-ASR-Streaming-0.6B, CoreML + MLX, **40 Sprach-Lokalisierungen**)
- **[Nemotron Streaming (Englisch)](https://soniqo.audio/guides/nemotron)** — Streaming-ASR mit geringer Latenz, nativer Interpunktion und Großschreibung (NVIDIA Nemotron-Speech-Streaming-0.6B, CoreML, nur Englisch, kleiner und schneller als die mehrsprachige Variante)
- **[Qwen3-ForcedAligner](https://soniqo.audio/de/guides/align)** — Wortgenaue Zeitstempel-Zuordnung (Audio + Text → Zeitstempel)
- **[Qwen3-TTS](https://soniqo.audio/de/guides/speak)** — Sprachsynthese (höchste Qualität, Streaming, benutzerdefinierte Sprecher, 10 Sprachen)
- **[CosyVoice TTS](https://soniqo.audio/de/guides/cosyvoice)** — Streaming-TTS mit Stimmklonen, Mehrsprecherdialog, Emotions-Tags (9 Sprachen)
- **[VoxCPM2](https://soniqo.audio/de/speech-generation)** — 48-kHz-Studio-TTS mit Stimmklonen und sprachbeschreibungsgesteuertem Voice Design (2B, MLX bf16/int8, 30 Sprachen)
- **[Kokoro TTS](https://soniqo.audio/de/guides/kokoro)** — TTS auf dem Gerät (82M, CoreML/Neural Engine, 54 Stimmen, iOS-tauglich, 10 Sprachen)
- **[VibeVoice TTS](https://soniqo.audio/de/guides/vibevoice)** — Langform-/Multi-Speaker-TTS (Microsoft VibeVoice Realtime-0.5B + 1.5B, MLX, bis zu 90 Min. Podcast-/Hörbuch-Synthese, EN/ZH)
- **[Magpie TTS](https://soniqo.audio/de/guides/magpie)** — Mehrsprachiges TTS (NVIDIA Magpie-TTS Multilingual 357M, MLX INT8 411 MB oder CoreML INT8 342 MB, 9 Sprachen, 5 vordefinierte Sprecher, Streaming auf MLX)
- **[Supertonic TTS](https://soniqo.audio/guides/supertonic)** — Flow-Matching-TTS auf dem Gerät (Supertone Supertonic-3 99M, CoreML/Neural Engine, 31 Sprachen, 10 Stimmen, G2P-frei, 44,1 kHz)
- **[Chatterbox TTS](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16)** — Mehrsprachiges TTS mit Zero-Shot-Stimmklonen (Resemble AI Chatterbox Multilingual, MLX fp16 ~1,3 GB, 23 Sprachen inkl. Arabisch & Hindi, MIT)
- **[Qwen3.5-Chat](https://soniqo.audio/de/guides/chat)** — LLM-Chat auf dem Gerät (0.8B, MLX INT4 + CoreML INT8, DeltaNet-Hybrid, Token-Streaming)
- **[FunctionGemma](https://soniqo.audio/de/guides/function-calls)** — On-Device-LLM für strukturierte Funktions- / Tool-Aufrufe (Gemma 3 270M, CoreML 8-Bit-Palettierung, Neural Engine, ~252 tok/s)
- **[MADLAD-400](https://soniqo.audio/de/guides/translate)** — Mehrsprachige Übersetzung über 400+ Sprachen (3B, MLX INT4 + INT8, T5 v1.1, Apache 2.0)
- **[Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate)** — Streaming-Sprache-zu-Sprache-Übersetzung (FR/ES/PT/DE → EN, MLX INT4 + INT8, Kyutai Moshi/Mimi-Stack, CC-BY-4.0)
- **[PersonaPlex](https://soniqo.audio/de/guides/respond)** — Vollduplex-Sprache-zu-Sprache (7B, Audio rein → Audio raus, 18 Stimmvoreinstellungen)
- **[DeepFilterNet3](https://soniqo.audio/de/guides/denoise)** — Echtzeit-Rauschunterdrückung (2,1M Parameter, 48 kHz). Langformaudio oberhalb der 60 s Single-Shot-Grenze wird automatisch mit Crossfade in Chunks zerlegt — siehe `enhanceChunked(...)`
- **[Quelltrennung](https://soniqo.audio/de/guides/separate)** — Musikquelltrennung mit HTDemucs (Demucs v4) + Open-Unmix (UMX-HQ / UMX-L, 4 Stems: Gesang/Drums/Bass/Rest, 44,1 kHz Stereo)
- **[MAGNeT](https://soniqo.audio/de/guides/compose)** — Text-zu-Musik-Generierung (Meta MAGNeT Small 300M / Medium 1.5B, MLX INT8, 30-Sekunden-Clips, 32 kHz Mono, maskierte parallele Dekodierung)
- **[FlashSR](https://soniqo.audio/de/guides/upsample)** — Audio-Super-Resolution (FlashSR ICASSP 2025, MLX, 48 kHz Mono, 1-Schritt destillierte Diffusion, INT4 363 MB / INT8 720 MB)
- **[Wake-Word](https://soniqo.audio/de/guides/wake-word)** — Schlüsselworterkennung auf dem Gerät (KWS Zipformer 3M, CoreML, 26× Echtzeit, konfigurierbare Stichwortliste)
- **[VAD](https://soniqo.audio/de/guides/vad)** — Sprachaktivitätserkennung (Silero Streaming, Pyannote Offline, FireRedVAD 100+ Sprachen)
- **[Sprecherdiarisierung](https://soniqo.audio/de/guides/diarize)** — Wer hat wann gesprochen (Pyannote-Pipeline, durchgängiger Sortformer auf der Neural Engine)
- **[Sprechereinbettungen](https://soniqo.audio/de/guides/embed-speaker)** — WeSpeaker ResNet34 (256-dim), CAM++ (192-dim)

Paper: [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba) · [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba) · [Omnilingual ASR](https://arxiv.org/abs/2511.09690) (Meta) · [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA) · [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba) · [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2) · [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA) · [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai) · [Hibiki](https://arxiv.org/abs/2502.03382) (Kyutai) · [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## Neuigkeiten

- **19. Apr. 2026** — [MLX vs. CoreML auf Apple Silicon — ein praktischer Leitfaden zur Wahl des Backends](https://blog.ivan.digital/mlx-vs-coreml-on-apple-silicon-a-practical-guide-to-picking-the-right-backend-and-why-you-should-f77ddea7b27a)
- **20. März 2026** — [Wir schlagen Whisper Large v3 mit einem 600M-Modell, das vollständig auf deinem Mac läuft](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **26. Feb. 2026** — [Sprecherdiarisierung und Sprachaktivitätserkennung auf Apple Silicon — natives Swift mit MLX](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **23. Feb. 2026** — [NVIDIA PersonaPlex 7B auf Apple Silicon — Vollduplex-Sprache-zu-Sprache in nativem Swift mit MLX](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **12. Feb. 2026** — [Qwen3-ASR Swift: ASR + TTS auf dem Gerät für Apple Silicon — Architektur und Benchmarks](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## Schnellstart

Füge das Paket zu deiner `Package.swift` hinzu:

```swift
.package(url: "https://github.com/soniqo/speech-swift", branch: "main")
```

Importiere nur die Module, die du benötigst — jedes Modell ist eine eigene SPM-Bibliothek, du zahlst nicht für das, was du nicht nutzt:

```swift
.product(name: "ParakeetStreamingASR", package: "speech-swift"),
.product(name: "SpeechUI",             package: "speech-swift"),  // optionale SwiftUI-Views
```

**Audio-Puffer in 3 Zeilen transkribieren:**

```swift
import ParakeetStreamingASR

let model = try await ParakeetStreamingASRModel.fromPretrained()
let text = try model.transcribeAudio(audioSamples, sampleRate: 16000)
```

**Live-Streaming mit Teilergebnissen:**

```swift
for await partial in model.transcribeStream(audio: samples, sampleRate: 16000) {
    print(partial.isFinal ? "FINAL: \(partial.text)" : "... \(partial.text)")
}
```

**SwiftUI-Diktat-View in ~10 Zeilen:**

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

`SpeechUI` liefert nur `TranscriptionView` (finale + partielle Ergebnisse) und `TranscriptionStore` (Streaming-ASR-Adapter). Verwende AVFoundation für Audio-Visualisierung und Wiedergabe.

Verfügbare SPM-Produkte: `Qwen3ASR`, `Qwen3TTS`, `Qwen3TTSCoreML`, `ParakeetASR`, `ParakeetStreamingASR`, `NemotronStreamingASR`, `OmnilingualASR`, `KokoroTTS`, `SupertonicTTS`, `VibeVoiceTTS`, `CosyVoiceTTS`, `VoxCPM2TTS`, `ChatterboxTTS`, `MagpieTTS`, `MagpieTTSCoreML`, `MAGNeTMusicGen`, `FlashSR`, `PersonaPlex`, `HibikiTranslate`, `SpeechVAD`, `SpeechEnhancement`, `SourceSeparation`, `Qwen3Chat`, `SpeechCore`, `SpeechUI`, `AudioCommon`.

## Modelle

Kompakte Übersicht unten. **[Vollständiger Modellkatalog mit Größen, Quantisierungen, Download-URLs und Speichertabellen → soniqo.audio/architecture](https://soniqo.audio/de/architecture)**.

| Modell | Aufgabe | Backends | Größen | Sprachen |
|-------|------|----------|-------|-----------|
| [Qwen3-ASR](https://soniqo.audio/de/guides/transcribe) | Sprache → Text | MLX, CoreML (hybrid) | 0.6B, 1.7B | 52 |
| [Parakeet TDT](https://soniqo.audio/de/guides/parakeet) | Sprache → Text | CoreML (ANE) | 0.6B | 25 europäisch |
| [Parakeet EOU](https://soniqo.audio/de/guides/dictate) | Sprache → Text (Streaming) | CoreML (ANE) | 120M | 25 europäisch |
| [Nemotron Streaming (Mehrsprachig)](https://soniqo.audio/de/guides/nemotron) | Sprache → Text (Streaming, mit Interpunktion) | CoreML (ANE), MLX | 0.6B | **40** |
| [Nemotron Streaming (Englisch)](https://soniqo.audio/guides/nemotron) | Sprache → Text (Streaming, mit Interpunktion) | CoreML (ANE) | 0.6B | EN |
| [Omnilingual ASR](https://soniqo.audio/de/guides/omnilingual) | Sprache → Text | CoreML (ANE), MLX | 300M / 1B / 3B / 7B | **[1.672](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)** |
| [Qwen3-ForcedAligner](https://soniqo.audio/de/guides/align) | Audio + Text → Zeitstempel | MLX, CoreML | 0.6B | Multi |
| [Qwen3-TTS](https://soniqo.audio/de/guides/speak) | Text → Sprache | MLX, CoreML | 0.6B, 1.7B | 10 |
| [CosyVoice3](https://soniqo.audio/de/guides/cosyvoice) | Text → Sprache | MLX | 0.5B | 9 |
| [VoxCPM2](https://soniqo.audio/de/speech-generation) | Text → Sprache (48 kHz, Voice Design + Klonen) | MLX | 2B (bf16/int8) | 30 |
| [Chatterbox Multilingual](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16) | Text → Sprache (Zero-Shot-Klonen) | MLX | 0.8B (fp16) | 23 |
| [Kokoro-82M](https://soniqo.audio/de/guides/kokoro) | Text → Sprache | CoreML (ANE) | 82M | 10 |
| [Supertonic-3](https://soniqo.audio/guides/supertonic) | Text → Sprache (44,1 kHz, Flow-Matching, G2P-frei) | CoreML (ANE) | 99M | 31 |
| [VibeVoice Realtime-0.5B](https://soniqo.audio/de/guides/vibevoice) | Text → Sprache (Langform, Multi-Speaker) | MLX | 0.5B | EN/ZH |
| [VibeVoice 1.5B](https://soniqo.audio/de/guides/vibevoice) | Text → Sprache (bis zu 90 Min. Podcast) | MLX | 1.5B | EN/ZH |
| [Magpie-TTS Multilingual](https://soniqo.audio/de/guides/magpie) | Text → Sprache (5 vordefinierte Sprecher, Streaming) | MLX / CoreML | 357M (MLX INT8, CoreML INT8) | 9 (CoreML ohne JA) |
| [Qwen3.5-Chat](https://soniqo.audio/de/guides/chat) | Text → Text (LLM) | MLX, CoreML | 0.8B | Multi |
| [FunctionGemma](https://soniqo.audio/de/guides/function-calls) | Text → Tool-Aufrufe (LLM) | CoreML | 270M | EN |
| [MADLAD-400](https://soniqo.audio/de/guides/translate) | Text → Text (Übersetzung) | MLX | 3B | **400+** |
| [Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate) | Sprache → Sprache (Übersetzung) | MLX | 3B | FR/ES/PT/DE → EN |
| [PersonaPlex](https://soniqo.audio/de/guides/respond) | Sprache → Sprache | MLX | 7B | EN |
| [Silero VAD](https://soniqo.audio/de/guides/vad) | Sprachaktivitätserkennung | MLX, CoreML | 309K | Sprachunabhängig |
| [Pyannote](https://soniqo.audio/de/guides/diarize) | VAD + Diarisierung | MLX | 1.5M | Sprachunabhängig |
| [Sortformer](https://soniqo.audio/de/guides/diarize) | Diarisierung (E2E) | CoreML (ANE) | — | Sprachunabhängig |
| [DeepFilterNet3](https://soniqo.audio/de/guides/denoise) | Sprachverbesserung | CoreML | 2.1M | Sprachunabhängig |
| [Sidon](https://soniqo.audio/de/guides/restore) | Sprachwiederherstellung (Rauschunterdrückung + Enthallung, 48 kHz) | CoreML | w2v-BERT 2.0 + DAC (fp16/int8) | Sprachunabhängig |
| [HTDemucs (Demucs v4)](https://soniqo.audio/de/guides/separate) | Quelltrennung | MLX | 168M | Agnostic |
| [Open-Unmix](https://soniqo.audio/de/guides/separate) | Quelltrennung | MLX | 8.6M | Agnostic |
| [MAGNeT](https://soniqo.audio/de/guides/compose) | Text → Musik (30 s @ 32 kHz) | MLX | 300M / 1.5B (int4/int8) | EN-Prompts |
| [FlashSR](https://soniqo.audio/de/guides/upsample) | Audio-Super-Resolution (48 kHz) | MLX | 363 MB / 720 MB (int4/int8) | Agnostisch |
| [WeSpeaker](https://soniqo.audio/de/guides/embed-speaker) | Sprechereinbettung | MLX, CoreML | 6.6M | Sprachunabhängig |

## Installation

### Homebrew

Erfordert natives ARM-Homebrew (`/opt/homebrew`). Rosetta/x86_64-Homebrew wird nicht unterstützt.

```bash
brew install speech
```

Dann:

```bash
speech transcribe recording.wav
speech speak "Hello world"
speech translate "Hello, how are you?" --to es
speech respond --input question.wav --transcript
speech-server --port 8080            # lokaler HTTP/WebSocket-Server (OpenAI-kompatibles /v1/realtime + /v1/audio/transcriptions)
```

**[Vollständige CLI-Referenz →](https://soniqo.audio/de/cli)**

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

Importiere nur, was du brauchst — jedes Modell hat sein eigenes SPM-Target:

```swift
import Qwen3ASR             // Spracherkennung (MLX)
import ParakeetASR          // Spracherkennung (CoreML, Batch)
import ParakeetStreamingASR // Streaming-Diktat mit Teilergebnissen + EOU
import NemotronStreamingASR // Mehrsprachiges Streaming-ASR mit nativer Interpunktion (0.6B, 40 Sprachen)
import OmnilingualASR       // 1.672 Sprachen (CoreML + MLX)
import Qwen3TTS             // Sprachsynthese
import CosyVoiceTTS         // Sprachsynthese mit Stimmklonen
import VoxCPM2TTS           // 48-kHz-TTS, Stimmklonen + Voice Design (2B)
import KokoroTTS            // Sprachsynthese (iOS-tauglich)
import VibeVoiceTTS         // Langform-/Multi-Speaker-TTS (EN/ZH)
import MagpieTTS            // Mehrsprachiges TTS (NVIDIA Magpie 357M, MLX, 9 Sprachen)
import MagpieTTSCoreML      // Magpie CoreML-Backend (Hybrid CoreML + MLX, 8 Sprachen)
import Qwen3Chat            // LLM-Chat auf dem Gerät
import FunctionGemma    // On-Device-LLM für Tool-Aufrufe
import MADLADTranslation    // Mehrsprachige Übersetzung über 400+ Sprachen
import HibikiTranslate      // Streaming-Sprache-zu-Sprache-Übersetzung (FR/ES/PT/DE → EN)
import PersonaPlex          // Vollduplex-Sprache-zu-Sprache
import SpeechVAD            // VAD + Sprecherdiarisierung + Einbettungen
import SpeechEnhancement    // Rauschunterdrückung
import SpeechRestoration    // Sprachwiederherstellung — Rauschunterdrückung + Enthallung (Sidon, CoreML, 48 kHz)
import SourceSeparation     // Musikquelltrennung (Open-Unmix, 4 Stems)
import MAGNeTMusicGen      // Text-zu-Musik-Generierung (30 s, 32 kHz)
import FlashSR             // Audio-Super-Resolution (48 kHz, 1-Schritt-Diffusion)
import SpeechUI             // SwiftUI-Komponenten für Streaming-Transkripte
import AudioCommon          // Geteilte Protokolle und Utilities
```

### Voraussetzungen

- Swift 6+, Xcode 16+ (mit Metal Toolchain)
- macOS 15+ (Sequoia) oder iOS 18+, Apple Silicon (M1/M2/M3/M4)

Die Mindestanforderung macOS 15 / iOS 18 kommt von [MLState](https://developer.apple.com/documentation/coreml/mlstate) —— Apples persistenter ANE-State-API —— die die CoreML-Pipelines (Qwen3-ASR, Qwen3-Chat, Qwen3-TTS) nutzen, um KV-Caches zwischen Token-Schritten auf der Neural Engine zu halten.

### Aus dem Quellcode bauen

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

`make build` kompiliert das Swift-Paket **und** die MLX-Metal-Shader-Bibliothek. Die Metal-Bibliothek ist für GPU-Inferenz erforderlich — ohne sie siehst du zur Laufzeit `Failed to load the default metallib`. `make debug` für Debug-Builds, `make test` für die Test-Suite.

**[Vollständige Build- und Installationsanleitung →](https://soniqo.audio/de/getting-started)**

## Demo-Apps

- **[DictateDemo](Examples/DictateDemo/)** ([Docs](https://soniqo.audio/de/guides/dictate)) — macOS-Menüleisten-Streaming-Diktat mit Live-Teilergebnissen, VAD-basierter Äußerungsende-Erkennung und Ein-Klick-Kopieren. Läuft als Hintergrund-agent (Parakeet-EOU-120M + Silero VAD).
- **[iOSEchoDemo](Examples/iOSEchoDemo/)** — iOS-Echo-Demo (Parakeet ASR + Kokoro TTS). Gerät und Simulator.
- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — Konversationeller Sprachassistent mit Mikrofoneingang, VAD und Multi-Turn-Kontext. macOS. RTF ~0.94 auf M2 Max (schneller als Echtzeit).
- **[SpeechDemo](Examples/SpeechDemo/)** — Diktat und TTS-Synthese in einer Tab-Oberfläche. macOS.

Die README jedes Demos enthält Bauanleitungen.

## Codebeispiele

Die folgenden Snippets zeigen den minimalen Pfad für jede Domäne. Jeder Abschnitt verlinkt auf eine vollständige Anleitung auf [soniqo.audio](https://soniqo.audio/de) mit Konfigurationsoptionen, mehreren Backends, Streaming-Mustern und CLI-Rezepten.

### Sprache-zu-Text — [vollständige Anleitung →](https://soniqo.audio/de/guides/transcribe)

```swift
import Qwen3ASR

let model = try await Qwen3ASRModel.fromPretrained()
let text = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

Alternative Backends: [Parakeet TDT](https://soniqo.audio/de/guides/parakeet) (CoreML, 32× Echtzeit), [Omnilingual ASR](https://soniqo.audio/de/guides/omnilingual) (1.672 Sprachen, CoreML oder MLX), [Streaming-Diktat](https://soniqo.audio/de/guides/dictate) (Live-Teilergebnisse).

### Forced Alignment — [vollständige Anleitung →](https://soniqo.audio/de/guides/align)

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

### Text-zu-Sprache — [vollständige Anleitung →](https://soniqo.audio/de/guides/speak)

```swift
import Qwen3TTS
import AudioCommon

let model = try await Qwen3TTSModel.fromPretrained()
let audio = model.synthesize(text: "Hello world", language: "english")
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

Alternative TTS-Engines: [CosyVoice3](https://soniqo.audio/de/guides/cosyvoice) (Streaming + Stimmklonen + Emotions-Tags), [Kokoro-82M](https://soniqo.audio/de/guides/kokoro) (iOS-tauglich, 54 Stimmen), [VibeVoice](https://soniqo.audio/de/guides/vibevoice) (Langform-Podcast / Multi-Speaker, EN/ZH), [Stimmklonen](https://soniqo.audio/de/guides/voice-cloning).

### Sprache-zu-Sprache — [vollständige Anleitung →](https://soniqo.audio/de/guides/respond)

```swift
import PersonaPlex

let model = try await PersonaPlexModel.fromPretrained()
let responseAudio = model.respond(userAudio: userSamples)
// 24 kHz Mono Float32-Ausgabe, bereit zur Wiedergabe
```

### LLM-Chat — [vollständige Anleitung →](https://soniqo.audio/de/guides/chat)

```swift
import Qwen3Chat
import FunctionGemma

let chat = try await Qwen35MLXChat.fromPretrained()
chat.chat(messages: [(.user, "Explain MLX in one sentence")]) { token, isFinal in
    print(token, terminator: "")
}
```

### Übersetzung — [vollständige Anleitung →](https://soniqo.audio/de/guides/translate)

```swift
import MADLADTranslation

let translator = try await MADLADTranslator.fromPretrained()
let es = try translator.translate("Hello, how are you?", to: "es")
// → "Hola, ¿cómo estás?"
```

### Sprachübersetzung — [vollständige Anleitung →](https://soniqo.audio/guides/audio-translate)

```swift
import HibikiTranslate
import AudioCommon

let model = try await HibikiTranslateModel.fromPretrained()
let pcm = try AudioFileLoader.load(url: input, targetSampleRate: 24000)
let (englishAudio, textTokens) = model.translate(
    sourceAudio: pcm, sourceLanguage: .fr
)
// Hibiki Zero-3B — FR/ES/PT/DE → EN, auf dem Gerät, Streaming-Mimi-Codec
```

### Sprachaktivitätserkennung — [vollständige Anleitung →](https://soniqo.audio/de/guides/vad)

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
for s in segments { print("\(s.startTime)s → \(s.endTime)s") }
```

### Sprecherdiarisierung — [vollständige Anleitung →](https://soniqo.audio/de/guides/diarize)

```swift
import SpeechVAD

let diarizer = try await DiarizationPipeline.fromPretrained()
let segments = diarizer.diarize(audio: samples, sampleRate: 16000)
for s in segments { print("Speaker \(s.speakerId): \(s.startTime)s - \(s.endTime)s") }
```

### Sprachverbesserung — [vollständige Anleitung →](https://soniqo.audio/de/guides/denoise)

```swift
import SpeechEnhancement

let denoiser = try await DeepFilterNet3Model.fromPretrained()
let clean = try denoiser.enhance(audio: noisySamples, sampleRate: 48000)
```

### Sprachwiederherstellung — [vollständige Anleitung →](https://soniqo.audio/de/guides/restore)

Gemeinsame Rauschunterdrückung **und** Enthallung mit [Sidon](https://arxiv.org/abs/2509.17052) (w2v-BERT-2.0-Prädiktor + DAC-Vocoder, Core ML). Anders als ein generischer Rauschunterdrücker ist Sidon darauf trainiert, die Sprecheridentität zu bewahren, und eignet sich daher gut, um eine verrauschte oder verhallte Referenz fürs Stimmklonen vor der TTS zu säubern. Die Eingabe ist 16 kHz, die Ausgabe 48 kHz Mono.

```swift
import SpeechRestoration

let restorer = try await SpeechRestorer.fromPretrained()          // .fp16 (default) or .int8
let clean = try restorer.restore(audio: noisySamples, sampleRate: 16000)  // → 48 kHz
```

Über die CLI:

```bash
speech restore noisy.wav -o clean.wav            # denoise + dereverb, 48 kHz output
speech restore noisy.wav --variant int8          # smaller, lower peak RAM

# Clean a voice-cloning reference before TTS (opt-in; preserves speaker identity):
speech speak "Hello" --engine voxcpm2 --voice-sample ref.wav --clean-reference
```

### Voice Pipeline (ASR → LLM → TTS) — [vollständige Anleitung →](https://soniqo.audio/de/voice-agents)

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

`VoicePipeline` ist die Echtzeit-Voice-agent-Zustandsmaschine (angetrieben von [speech-core](https://github.com/soniqo/speech-core)) mit VAD-basierter Sprecherwechsel-Erkennung, Unterbrechungsbehandlung und eager STT. Sie verbindet beliebige `SpeechRecognitionModel` + `SpeechGenerationModel` + `StreamingVADProvider`.

### HTTP-API-Server

```bash
speech-server --port 8080
```

Stellt jedes Modell über HTTP-REST- + WebSocket-Endpunkte bereit, einschließlich OpenAI-kompatibler APIs: einem Realtime-WebSocket unter `/v1/realtime` und einem Transkriptions-REST-Endpunkt unter `/v1/audio/transcriptions`. Siehe [`Sources/AudioServer/`](Sources/AudioServer/).

## Architektur

speech-swift ist in ein SPM-Target pro Modell aufgeteilt, sodass Konsumenten nur für das bezahlen, was sie importieren. Geteilte Infrastruktur lebt in `AudioCommon` (Protokolle, Audio-I/O, HuggingFace-Downloader, `SentencePieceModel`) und `MLXCommon` (Gewichtsladen, `QuantizedLinear`-Helfer, `SDPA`-Multi-Head-Attention-Helfer).

**[Vollständiges Architekturdiagramm mit Backends, Speichertabellen und Modulkarte → soniqo.audio/architecture](https://soniqo.audio/de/architecture)** · **[API-Referenz → soniqo.audio/api](https://soniqo.audio/de/api)** · **[Benchmarks → soniqo.audio/benchmarks](https://soniqo.audio/de/benchmarks)**

Lokale Docs (Repo):
- **Modelle:** [Qwen3-ASR](docs/models/asr-model.md) · [Qwen3-TTS](docs/models/tts-model.md) · [CosyVoice](docs/models/cosyvoice-tts.md) · [Kokoro](docs/models/kokoro-tts.md) · [VibeVoice](docs/models/vibevoice.md) · [Parakeet TDT](docs/models/parakeet-asr.md) · [Parakeet Streaming](docs/models/parakeet-streaming-asr.md) · [Nemotron Streaming](docs/models/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/models/omnilingual-asr.md) · [PersonaPlex](docs/models/personaplex.md) · [Hibiki](docs/models/hibiki.md) · [FireRedVAD](docs/models/fireredvad.md) · [Source Separation](docs/models/source-separation.md) · [HTDemucs](docs/models/htdemucs.md) · [MAGNeT](docs/models/magnet-music-gen.md) · [FlashSR](docs/models/flashsr.md)
- **Inferenz:** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md) · [Parakeet TDT](docs/inference/parakeet-asr-inference.md) · [Parakeet Streaming](docs/inference/parakeet-streaming-asr-inference.md) · [Nemotron Streaming](docs/inference/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/inference/omnilingual-asr-inference.md) · [TTS](docs/inference/qwen3-tts-inference.md) · [VibeVoice](docs/inference/vibevoice-inference.md) · [Hibiki](docs/inference/hibiki-inference.md) · [Forced Aligner](docs/inference/forced-aligner.md) · [Silero VAD](docs/inference/silero-vad.md) · [Sprecherdiarisierung](docs/inference/speaker-diarization.md) · [Sprachverbesserung](docs/inference/speech-enhancement.md) · [MAGNeT](docs/inference/magnet-music-gen.md) · [FlashSR](docs/inference/flashsr.md)
- **Referenz:** [Geteilte Protokolle](docs/shared-protocols.md)

## Cache-Konfiguration

Modellgewichte werden beim ersten Gebrauch von HuggingFace heruntergeladen und in `~/Library/Caches/qwen3-speech/` zwischengespeichert. Überschreibe mit `QWEN3_CACHE_DIR` (CLI) oder `cacheDir:` (Swift-API). Alle `fromPretrained()`-Einstiegspunkte akzeptieren `offlineMode: true`, um das Netzwerk zu überspringen, wenn die Gewichte bereits im Cache sind.

Nutzer in Festlandchina (oder überall dort, wo `huggingface.co` langsam oder blockiert ist) können über einen Mirror laden, indem sie `HF_ENDPOINT` setzen, z. B. `export HF_ENDPOINT=https://hf-mirror.com`.

Siehe [`docs/inference/cache-and-offline.md`](docs/inference/cache-and-offline.md) für vollständige Details einschließlich sandboxed iOS-Container-Pfade.

## MLX-Metal-Bibliothek

Wenn du zur Laufzeit `Failed to load the default metallib` siehst, fehlt die Metal-Shader-Bibliothek. Führe nach einem manuellen `swift build` `make build` oder `./scripts/build_mlx_metallib.sh release` aus. Falls das Metal Toolchain fehlt, installiere es zuerst:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Tests

```bash
make test                            # Vollständige Suite (Unit + E2E mit Modell-Downloads)
swift test --skip E2E                # Nur Unit (CI-sicher, keine Downloads)
swift test --filter Qwen3ASRTests    # Bestimmtes Modul
```

E2E-Testklassen verwenden das Präfix `E2E`, damit CI sie mit `--skip E2E` ausfiltern kann. Siehe [CLAUDE.md](CLAUDE.md#testing) für die vollständige Testkonvention.

## Mitwirken

PRs willkommen — Bugfixes, neue Modellintegrationen, Dokumentation. Fork, Feature-Branch anlegen, `make build && make test`, PR gegen `main` eröffnen.

## Lizenz

Apache 2.0
