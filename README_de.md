# Speech Swift

KI-Sprachmodelle für Apple Silicon, basierend auf MLX Swift und CoreML.

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

Spracherkennung, -synthese und -verständnis auf dem Gerät für Mac und iOS. Läuft vollständig lokal auf Apple Silicon — keine Cloud, keine API-Schlüssel, keine Daten verlassen das Gerät.

**[📚 Vollständige Dokumentation →](https://soniqo.audio/de)** · **[🤗 HuggingFace-Modelle](https://huggingface.co/aufklarer)** · **[📝 Blog](https://blog.ivan.digital)** · **[💬 Discord](https://discord.gg/TnCryqEMgu)**

<p align="center">
  <a href="https://formulae.brew.sh/formula/speech"><img src="https://img.shields.io/homebrew/installs/dm/speech.svg?logo=homebrew&amp;label=Homebrew%20installs&amp;color=FBB040" alt="Homebrew installs"></a>
  <a href="https://github.com/soniqo/speech-swift#built-with-speech-swift"><img src="https://img.shields.io/badge/verified%20public%20repositories-29-2ea44f?logo=github" alt="Verified public repositories: 29"></a>
</p>

<p align="center">
  <a href="https://trendshift.io/repositories/24196?utm_source=trendshift-badge&amp;utm_medium=badge&amp;utm_campaign=badge-trendshift-24196" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/trendshift/repositories/24196/daily?language=Swift" alt="soniqo%2Fspeech-swift | Trendshift" width="250" height="55"/></a>
</p>

<p align="center">
  <a href="https://youtu.be/x9zgcaW0gUk">
    <img src="https://img.youtube.com/vi/x9zgcaW0gUk/maxresdefault.jpg" width="640" alt="Lokale Sprach-KI auf einem MacBook — die vierminütige Tour durch die Open-Source-Bibliothek auf YouTube ansehen">
  </a>
</p>
<p align="center"><em>Lokale Sprach-KI auf einem MacBook — die vierminütige Tour durch die Open-Source-Bibliothek auf YouTube ansehen</em></p>

**Anwendungsfälle:** [Sprachagenten](https://soniqo.audio/de/voice-agents) · [Transkription](https://soniqo.audio/de/transcription) · [Sprachsynthese](https://soniqo.audio/de/speech-generation)

## Mit Speech Swift entwickelt

29 öffentliche Repositories mit überprüfbaren Verweisen auf das Speech-Swift-Paket.

[Palmier Pro](https://github.com/palmier-io/palmier-pro) · [Anarlog](https://github.com/fastrepl/anarlog) · [ClawdHome](https://github.com/ThinkInAIXYZ/clawdhome) · [Jabber](https://github.com/rselbach/jabber) · [Ora](https://github.com/wuwangzhang1216/ora) · [VoxFlow](https://github.com/xingbofeng/VoxFlow) · [LokalBot](https://github.com/stevyhacker/lokalbot) · [Voicey](https://github.com/jonathanKingston/voicey) · [HushType](https://github.com/felixfu824/HushType) · [DexDictate macOS](https://github.com/westkitty/DexDictate_MacOS) · [Watchtower](https://github.com/aiwatchtowers/watchtower) · [Wishper App](https://github.com/irangareddy/wishper-app) · [FriSpeak](https://github.com/KSubedi/FriSpeak) · [Scribe](https://github.com/itchat/Scribe) · [VoicePen](https://github.com/dot-sk/VoicePen) · [Anything Voice](https://github.com/jakemaly/anything-voice) · [Conversational MLX](https://github.com/ottokafka/conversational_mlx) · [HachiSpeak](https://github.com/sarinali/hachispeak) · [JustTalk](https://github.com/d0zingcat/JustTalk) · [Kioku](https://github.com/matthewmorrone/Kioku) · [Luxicon](https://github.com/DavidsonCollege/luxicon) · [Mako](https://github.com/bn-l/mako) · [Meeting Emo Transcriber](https://github.com/kouko/meeting-emo-transcriber) · [MeetingSummary](https://github.com/a9650615/MeetingSummary) · [Stenograf](https://github.com/ivan-digital/stenograf) · [Toast](https://github.com/drbh/toast) · [TxtVoiceApp](https://github.com/2mauis/TxtVoiceApp) · [video_to_srt](https://github.com/dogacan/video_to_srt) · [Warmth iOS](https://github.com/molyleelatham/gtmhackathon)

**Fähigkeitsgruppen:** STT / ASR · Alignment · TTS · LLMs und Übersetzung · Speech-to-Speech · Verbesserung / Restaurierung · Quellentrennung · Musik- / Audiogenerierung · Wake Word, VAD, Diarisierung und Sprecheridentität

**STT / ASR**

- **[Qwen3-ASR](https://soniqo.audio/de/guides/transcribe)** — Sprache-zu-Text (automatische Spracherkennung, 52 Sprachen, MLX + CoreML)
- **[WhisperASR](docs/models/whisper-asr.md)** — Whisper Large-v3 Turbo speech-to-text via native CoreML runtime (ANE, multilingual)
- **[Parakeet TDT](https://soniqo.audio/de/guides/parakeet)** — Sprache-zu-Text über CoreML (Neural Engine, NVIDIA FastConformer + TDT-Decoder, 25 Sprachen)
- **[Omnilingual ASR](https://soniqo.audio/de/guides/omnilingual)** — Sprache-zu-Text (Meta wav2vec2 + CTC, **1.672 Sprachen** in 32 Schriften, CoreML 300M + MLX 300M/1B/3B/7B)
- **[Streaming-Diktat](https://soniqo.audio/de/guides/dictate)** — Echtzeit-Diktat mit Teilergebnissen und Äußerungsende-Erkennung (Parakeet-EOU-120M)
- **[Nemotron Streaming (Mehrsprachig)](https://soniqo.audio/de/guides/nemotron)** — Streaming-ASR mit geringer Latenz, nativer Interpunktion und Großschreibung (NVIDIA Nemotron-3.5-ASR-Streaming-0.6B, CoreML + MLX, **40 Sprach-Lokalisierungen**)
- **[Nemotron Streaming (Englisch)](https://soniqo.audio/guides/nemotron)** — Streaming-ASR mit geringer Latenz, nativer Interpunktion und Großschreibung (NVIDIA Nemotron-Speech-Streaming-0.6B, CoreML, nur Englisch, kleiner und schneller als die mehrsprachige Variante)

**Alignment**

- **[Qwen3-ForcedAligner](https://soniqo.audio/de/guides/align)** — Wortgenaue Zeitstempel-Zuordnung (Audio + Text → Zeitstempel)

**TTS / Sprachsynthese**

- **[Qwen3-TTS](https://soniqo.audio/de/guides/speak)** — Sprachsynthese (höchste Qualität, Streaming, benutzerdefinierte Sprecher, 10 Sprachen)
- **[CosyVoice TTS](https://soniqo.audio/de/guides/cosyvoice)** — Streaming-TTS mit Stimmklonen, Mehrsprecherdialog, Emotions-Tags (9 Sprachen)
- **[VoxCPM2](https://soniqo.audio/de/speech-generation)** — 48-kHz-Studio-TTS mit Stimmklonen und sprachbeschreibungsgesteuertem Voice Design (2B, MLX bf16/int8, 30 Sprachen)
- **[IndexTTS2](docs/models/indextts2.md)** — Native MLX voice cloning from a reference voice (IndexTeam IndexTTS-2, 1.5B-class fp16 bundle, speaker/emotion/pause controls)
- **[F5-TTS](docs/models/f5-tts.md)** — Zero-shot voice cloning from a short reference clip + transcript (SWivid F5-TTS v1 Base, DiT flow matching + Vocos, MLX fp16, 24 kHz, English + Mandarin; non-commercial license)
- **[Higgs TTS 3](docs/models/higgs-tts.md)** — Conversational TTS with zero-shot voice cloning and inline emotion/style/SFX/prosody tags (Boson Higgs TTS 3, Qwen3-4B backbone, MLX bf16, 24 kHz, 100+ languages; research/non-commercial license)
- **[Kokoro TTS](https://soniqo.audio/de/guides/kokoro)** — TTS auf dem Gerät (82M, CoreML/Neural Engine, 54 Stimmen, iOS-tauglich, 10 Sprachen)
- **[VibeVoice TTS](https://soniqo.audio/de/guides/vibevoice)** — Langform-/Multi-Speaker-TTS (Microsoft VibeVoice Realtime-0.5B + 1.5B, MLX, bis zu 90 Min. Podcast-/Hörbuch-Synthese, EN/ZH)
- **[Magpie TTS](https://soniqo.audio/de/guides/magpie)** — Mehrsprachiges TTS (NVIDIA Magpie-TTS Multilingual 357M, MLX INT8 411 MB oder CoreML INT8 342 MB, 9 Sprachen, 5 vordefinierte Sprecher, Streaming auf MLX)
- **[Supertonic TTS](https://soniqo.audio/guides/supertonic)** — Flow-Matching-TTS auf dem Gerät (Supertone Supertonic-3 99M, CoreML/Neural Engine, 31 Sprachen, 10 Stimmen, G2P-frei, 44,1 kHz)
- **[Chatterbox TTS](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16)** — Mehrsprachiges TTS mit Zero-Shot-Stimmklonen (Resemble AI Chatterbox Multilingual, MLX fp16 ~1,3 GB, 23 Laufzeitsprachen; Hebräisch erfordert Nikkud, MIT)
- **[OmniVoice TTS](https://huggingface.co/aufklarer/OmniVoice-MLX-fp16)** — Nicht-autoregressives Diffusions-TTS mit Zero-Shot-Stimmklonen (k2-fsa OmniVoice, Qwen3-Backbone, MLX fp16 Standard / int8 verfügbar, 600+ Sprachen, Apache-2.0)
- **[Indic-Mio](docs/models/indic-mio-tts.md)** — Hindi/Indic TTS with inline emotion markers and optional reference-voice cloning (MLX, 24 kHz)

**LLMs und Übersetzung**

- **[Qwen3Chat](https://soniqo.audio/de/guides/chat)** — LLM-Chat auf dem Gerät (Qwen3.5-0.8B mit MLX/CoreML plus dense Qwen3-4B- und Gemma-4-E2B/E4B-MLX-Backends, Token-Streaming)
- **[FunctionGemma](https://soniqo.audio/de/guides/function-calls)** — On-Device-LLM für strukturierte Funktions- / Tool-Aufrufe (Gemma 3 270M, CoreML 8-Bit-Palettierung, Neural Engine, ~252 tok/s)
- **[MADLAD-400](https://soniqo.audio/de/guides/translate)** — Mehrsprachige Übersetzung über 400+ Sprachen (3B, MLX INT4 + INT8, T5 v1.1, Apache 2.0)

**Speech-to-Speech und Sprachagenten**

- **[Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate)** — Streaming-Sprache-zu-Sprache-Übersetzung (FR/ES/PT/DE → EN, MLX INT4 + INT8, Kyutai Moshi/Mimi-Stack, CC-BY-4.0)
- **[PersonaPlex](https://soniqo.audio/de/guides/respond)** — Vollduplex-Sprache-zu-Sprache (7B, Audio rein → Audio raus, 18 Stimmvoreinstellungen)
- **[Audio2Face-3D](docs/models/audio2face3d.md)** — Sprachgesteuerte Gesichtsanimation für Avatare (NVIDIA Audio2Face-3D v2.3 Mark, 301 Gesichtskoeffizienten, MLX)

**Verbesserung, Trennung und Audiogenerierung**

- **[DeepFilterNet3](https://soniqo.audio/de/guides/denoise)** — Echtzeit-Rauschunterdrückung (2,1M Parameter, 48 kHz). Langformaudio oberhalb der 60 s Single-Shot-Grenze wird automatisch mit Crossfade in Chunks zerlegt — siehe `enhanceChunked(...)`
- **[Quelltrennung](https://soniqo.audio/de/guides/separate)** — Musikquelltrennung mit HTDemucs (Demucs v4) + Open-Unmix (UMX-HQ / UMX-L, 4 Stems: Gesang/Drums/Bass/Rest, 44,1 kHz Stereo)
- **[MAGNeT](https://soniqo.audio/de/guides/compose)** — Text-zu-Musik-Generierung (Meta MAGNeT Small 300M / Medium 1.5B, MLX INT8, 30-Sekunden-Clips, 32 kHz Mono, maskierte parallele Dekodierung)
- **[Stable Audio 3](docs/models/stable-audio-3.md)** — Text-to-audio/music generation (Stable Audio 3 Medium, MLX INT8/INT4, 44.1 kHz stereo, variable length)
- **[FlashSR](https://soniqo.audio/de/guides/upsample)** — Audio-Super-Resolution (FlashSR ICASSP 2025, MLX, 48 kHz Mono, 1-Schritt destillierte Diffusion, INT4 363 MB / INT8 720 MB)

**Turn-Erkennung, Diarisierung und Sprecheridentität**

- **[Wake-Word](https://soniqo.audio/de/guides/wake-word)** — Schlüsselworterkennung auf dem Gerät (KWS Zipformer 3M, CoreML, 26× Echtzeit, konfigurierbare Stichwortliste)
- **[VAD](https://soniqo.audio/de/guides/vad)** — Sprachaktivitätserkennung (Silero Streaming, Pyannote Offline, FireRedVAD 100+ Sprachen)
- **[Sprecherdiarisierung](https://soniqo.audio/de/guides/diarize)** — Wer hat wann gesprochen (Pyannote-Pipeline, durchgängiger Sortformer auf der Neural Engine) — jetzt mit inkrementeller Streaming-Session (stabile Sprecher-IDs, Updates alle 480 ms)
- **[Sprechereinbettungen](https://soniqo.audio/de/guides/embed-speaker)** — WeSpeaker ResNet34 (256-dim), ReDimNet2-B6 für benannte Stimmidentität (192-dim), CAM++ (192-dim)

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

Verfügbare SPM-Produkte: `Qwen3ASR`, `WhisperASR`, `Qwen3TTS`, `Qwen3TTSCoreML`, `ParakeetASR`, `ParakeetStreamingASR`, `NemotronStreamingASR`, `OmnilingualASR`, `KokoroTTS`, `SupertonicTTS`, `VibeVoiceTTS`, `CosyVoiceTTS`, `VoxCPM2TTS`, `IndexTTS2TTS`, `F5TTS`, `HiggsTTS`, `ChatterboxTTS`, `OmniVoiceTTS`, `IndicMioTTS`, `FishAudioTTS`, `MagpieTTS`, `MagpieTTSCoreML`, `MAGNeTMusicGen`, `StableAudio3MusicGen`, `FlashSR`, `PersonaPlex`, `Audio2Face3D`, `HibikiTranslate`, `MADLADTranslation`, `SpeechVAD`, `SpeechWakeWord`, `SpeechEnhancement`, `SpeechRestoration`, `SourceSeparation`, `Qwen3Chat`, `FunctionGemma`, `SpeechCore`, `SpeechUI`, `AudioCommon`.

## Modelle

Kompakte Übersicht unten. **[Vollständiger Modellkatalog mit Größen, Quantisierungen, Download-URLs und Speichertabellen → soniqo.audio/architecture](https://soniqo.audio/de/architecture)**.

| Modell | Aufgabe | Backends | Größen | Sprachen |
|-------|------|----------|-------|-----------|
| [Qwen3-ASR](https://soniqo.audio/de/guides/transcribe) | Sprache → Text | MLX, CoreML (hybrid) | 0.6B, 1.7B | 52 |
| [WhisperASR](docs/models/whisper-asr.md) | Speech → Text | CoreML (ANE) | Large-v3 Turbo | Multi |
| [Parakeet TDT](https://soniqo.audio/de/guides/parakeet) | Sprache → Text | CoreML (ANE) | 0.6B | 25 europäisch |
| [Parakeet EOU](https://soniqo.audio/de/guides/dictate) | Sprache → Text (Streaming) | CoreML (ANE) | 120M | 25 europäisch |
| [Nemotron Streaming (Mehrsprachig)](https://soniqo.audio/de/guides/nemotron) | Sprache → Text (Streaming, mit Interpunktion) | CoreML (ANE), MLX | 0.6B | **40** |
| [Nemotron Streaming (Englisch)](https://soniqo.audio/guides/nemotron) | Sprache → Text (Streaming, mit Interpunktion) | CoreML (ANE) | 0.6B | EN |
| [Omnilingual ASR](https://soniqo.audio/de/guides/omnilingual) | Sprache → Text | CoreML (ANE), MLX | 300M / 1B / 3B / 7B | **[1.672](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)** |
| [Qwen3-ForcedAligner](https://soniqo.audio/de/guides/align) | Audio + Text → Zeitstempel | MLX, CoreML | 0.6B | Multi |
| [Qwen3-TTS](https://soniqo.audio/de/guides/speak) | Text → Sprache | MLX, CoreML | 0.6B, 1.7B | 10 |
| [CosyVoice3](https://soniqo.audio/de/guides/cosyvoice) | Text → Sprache | MLX | 0.5B | 9 |
| [VoxCPM2](https://soniqo.audio/de/speech-generation) | Text → Sprache (48 kHz, Voice Design + Klonen) | MLX | 2B (bf16/int8) | 30 |
| [IndexTTS2](docs/models/indextts2.md) | Text → Speech (zero-shot voice cloning) | MLX | 1.5B-class (fp16) | EN/ZH |
| [F5-TTS](docs/models/f5-tts.md) | Text → Speech (zero-shot voice cloning) | MLX | 336M (fp16) | EN/ZH |
| [Higgs TTS 3](docs/models/higgs-tts.md) | Text → Speech (conversational, zero-shot voice cloning) | MLX | 4B (bf16) | 100+ |
| [Chatterbox Multilingual](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16) | Text → Sprache (Zero-Shot-Klonen) | MLX | 0.8B (fp16) | 23 (HE erfordert Nikkud) |
| [OmniVoice](https://huggingface.co/aufklarer/OmniVoice-MLX-fp16) | Text → Sprache (NAR-Diffusion, Zero-Shot-Klonen) | MLX | 0.8B (fp16 Standard / int8) | 600+ |
| [Indic-Mio](docs/models/indic-mio-tts.md) | Text → Speech (Hindi/Indic, emotion tags, voice cloning) | MLX | fp16 | Hindi / Indic |
| [Fish Audio S2 Pro](docs/models/fish-audio-s2-pro.md) | Text → Sprache (Zero-Shot-Klonen, explizite Stilmarker) | MLX | 0.5B-class (fp16) | Mehrsprachig |
| [Kokoro-82M](https://soniqo.audio/de/guides/kokoro) | Text → Sprache | CoreML (ANE) | 82M | 10 |
| [Supertonic-3](https://soniqo.audio/guides/supertonic) | Text → Sprache (44,1 kHz, Flow-Matching, G2P-frei) | CoreML (ANE) | 99M | 31 |
| [VibeVoice Realtime-0.5B](https://soniqo.audio/de/guides/vibevoice) | Text → Sprache (Langform, Multi-Speaker) | MLX | 0.5B | EN/ZH |
| [VibeVoice 1.5B](https://soniqo.audio/de/guides/vibevoice) | Text → Sprache (bis zu 90 Min. Podcast) | MLX | 1.5B | EN/ZH |
| [Magpie-TTS Multilingual](https://soniqo.audio/de/guides/magpie) | Text → Sprache (5 vordefinierte Sprecher, Streaming) | MLX / CoreML | 357M (MLX INT8, CoreML INT8) | 9 (CoreML ohne JA) |
| [Qwen3.5 Chat](docs/models/qwen35-chat.md) | Text → Text (LLM) | MLX, CoreML | 0.8B | Multi |
| [Qwen3 Dense Chat](docs/models/qwen3-dense-chat.md) | Text → Text (LLM) | MLX | 4B | Multi |
| [Gemma 4 Chat](docs/models/gemma4-chat.md) | Text → Text (LLM) | MLX | E2B / E4B (4-bit) | Multi |
| [FunctionGemma](https://soniqo.audio/de/guides/function-calls) | Text → Tool-Aufrufe (LLM) | CoreML | 270M | EN |
| [MADLAD-400](https://soniqo.audio/de/guides/translate) | Text → Text (Übersetzung) | MLX | 3B | **400+** |
| [Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate) | Sprache → Sprache (Übersetzung) | MLX | 3B | FR/ES/PT/DE → EN |
| [PersonaPlex](https://soniqo.audio/de/guides/respond) | Sprache → Sprache | MLX | 7B | EN |
| [Audio2Face-3D](docs/models/audio2face3d.md) | Sprache → Gesichtsanimation | MLX | v2.3 Mark | Sprachunabhängig |
| [Silero VAD](https://soniqo.audio/de/guides/vad) | Sprachaktivitätserkennung | MLX, CoreML | 309K | Sprachunabhängig |
| [KWS Zipformer](docs/models/kws-zipformer.md) | Audio → Wake word | CoreML (ANE) | 3M | EN/custom keywords |
| [Pyannote](https://soniqo.audio/de/guides/diarize) | VAD + Diarisierung | MLX | 1.5M | Sprachunabhängig |
| [Pyannote Community-1](https://huggingface.co/aufklarer/Pyannote-Community-1-CoreML) | Diarisierung + Sprechereinbettungen | CoreML (ANE) + Swift VBx | 8.35M | Sprachunabhängig |
| [Sortformer](https://huggingface.co/aufklarer/Sortformer-Diarization-CoreML) | [Diarisierung (E2E), inkrementelles Streaming](https://soniqo.audio/de/guides/diarize) | CoreML (ANE) | 117M | Sprachunabhängig |
| [DeepFilterNet3](https://soniqo.audio/de/guides/denoise) | Sprachverbesserung | CoreML | 2.1M | Sprachunabhängig |
| [Sidon](https://soniqo.audio/de/guides/restore) | Sprachwiederherstellung (Rauschunterdrückung + Enthallung, 48 kHz) | CoreML | w2v-BERT 2.0 + DAC (fp16/int8) | Sprachunabhängig |
| [HTDemucs (Demucs v4)](https://soniqo.audio/de/guides/separate) | Quelltrennung | MLX | 168M | Agnostic |
| [Open-Unmix](https://soniqo.audio/de/guides/separate) | Quelltrennung | MLX | 8.6M | Agnostic |
| [MAGNeT](https://soniqo.audio/de/guides/compose) | Text → Musik (30 s @ 32 kHz) | MLX | 300M / 1.5B (int4/int8) | EN-Prompts |
| [Stable Audio 3](docs/models/stable-audio-3.md) | Text → Music/audio (44.1 kHz stereo) | MLX | Medium 1.4B (int4/int8) | EN prompts |
| [FlashSR](https://soniqo.audio/de/guides/upsample) | Audio-Super-Resolution (48 kHz) | MLX | 363 MB / 720 MB (int4/int8) | Agnostisch |
| [WeSpeaker](https://soniqo.audio/de/guides/embed-speaker) | Sprechereinbettung | MLX, CoreML | 6.6M | Sprachunabhängig |
| [ReDimNet2-B6](https://huggingface.co/aufklarer/ReDimNet2-B6-CoreML) | Benannte Stimmidentität | CoreML | 12.3M | Sprachunabhängig |

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
import WhisperASR           // Whisper Large-v3 Turbo (CoreML)
import ParakeetASR          // Spracherkennung (CoreML, Batch)
import ParakeetStreamingASR // Streaming-Diktat mit Teilergebnissen + EOU
import NemotronStreamingASR // Mehrsprachiges Streaming-ASR mit nativer Interpunktion (0.6B, 40 Sprachen)
import OmnilingualASR       // 1.672 Sprachen (CoreML + MLX)
import Qwen3TTS             // Sprachsynthese
import CosyVoiceTTS         // Sprachsynthese mit Stimmklonen
import VoxCPM2TTS           // 48-kHz-TTS, Stimmklonen + Voice Design (2B)
import IndexTTS2TTS         // Native MLX voice cloning from reference audio
import F5TTS                // Zero-shot voice cloning (DiT flow matching + Vocos)
import HiggsTTS             // Conversational TTS + cloning (Qwen3 backbone, control tags)
import KokoroTTS            // Sprachsynthese (iOS-tauglich)
import VibeVoiceTTS         // Langform-/Multi-Speaker-TTS (EN/ZH)
import MagpieTTS            // Mehrsprachiges TTS (NVIDIA Magpie 357M, MLX, 9 Sprachen)
import MagpieTTSCoreML      // Magpie CoreML-Backend (Hybrid CoreML + MLX, 8 Sprachen)
import FishAudioTTS         // Experimentelle Fish Audio S2 Pro Runtime mit Stimmklonen
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

Alternative Backends: [WhisperASR](docs/inference/whisper-asr-inference.md) (Whisper Large-v3 Turbo, native CoreML), [Parakeet TDT](https://soniqo.audio/de/guides/parakeet) (CoreML, 32× Echtzeit), [Omnilingual ASR](https://soniqo.audio/de/guides/omnilingual) (1.672 Sprachen, CoreML oder MLX), [Streaming-Diktat](https://soniqo.audio/de/guides/dictate) (Live-Teilergebnisse).

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

Alternative TTS-Engines: [CosyVoice3](https://soniqo.audio/de/guides/cosyvoice) (Streaming + Stimmklonen + Emotions-Tags), [Kokoro-82M](https://soniqo.audio/de/guides/kokoro) (iOS-tauglich, 54 Stimmen), [VibeVoice](https://soniqo.audio/de/guides/vibevoice) (Langform-Podcast / Multi-Speaker, EN/ZH), [Fish Audio S2 Pro](docs/inference/fish-audio-s2-pro.md) (experimentelles Zero-Shot-Klonen + Stilmarker in eckigen Klammern), [Stimmklonen](https://soniqo.audio/de/guides/voice-cloning).

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
- **Modelle:** [Qwen3-ASR](docs/models/asr-model.md) · [WhisperASR](docs/models/whisper-asr.md) · [Qwen3-TTS](docs/models/tts-model.md) · [CosyVoice](docs/models/cosyvoice-tts.md) · [Kokoro](docs/models/kokoro-tts.md) · [VoxCPM2](docs/models/voxcpm2-tts.md) · [IndexTTS2](docs/models/indextts2.md) · [F5-TTS](docs/models/f5-tts.md) · [Higgs TTS 3](docs/models/higgs-tts.md) · [VibeVoice](docs/models/vibevoice.md) · [Supertonic](docs/models/supertonic-tts.md) · [Chatterbox](docs/models/chatterbox-tts.md) · [Indic-Mio](docs/models/indic-mio-tts.md) · [Fish Audio S2 Pro](docs/models/fish-audio-s2-pro.md) · [Magpie TTS](docs/models/magpie-tts.md) · [Parakeet TDT](docs/models/parakeet-asr.md) · [Parakeet Streaming](docs/models/parakeet-streaming-asr.md) · [Nemotron Streaming](docs/models/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/models/omnilingual-asr.md) · [PersonaPlex](docs/models/personaplex.md) · [Hibiki](docs/models/hibiki.md) · [MADLAD-400](docs/models/madlad-translation.md) · [FunctionGemma](docs/models/function-gemma.md) · [Qwen3.5 Chat](docs/models/qwen35-chat.md) · [Gemma 4 Chat](docs/models/gemma4-chat.md) · [Qwen3 Dense Chat](docs/models/qwen3-dense-chat.md) · [FireRedVAD](docs/models/fireredvad.md) · [KWS Zipformer](docs/models/kws-zipformer.md) · [Sidon](docs/models/sidon.md) · [Source Separation](docs/models/source-separation.md) · [HTDemucs](docs/models/htdemucs.md) · [MAGNeT](docs/models/magnet-music-gen.md) · [Stable Audio 3](docs/models/stable-audio-3.md) · [FlashSR](docs/models/flashsr.md) · [Audio2Face-3D](docs/models/audio2face3d.md)
- **Inferenz:** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md) · [WhisperASR](docs/inference/whisper-asr-inference.md) · [Parakeet TDT](docs/inference/parakeet-asr-inference.md) · [Parakeet Streaming](docs/inference/parakeet-streaming-asr-inference.md) · [Nemotron Streaming](docs/inference/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/inference/omnilingual-asr-inference.md) · [TTS](docs/inference/qwen3-tts-inference.md) · [VoxCPM2](docs/inference/voxcpm2-inference.md) · [IndexTTS2](docs/inference/indextts2.md) · [F5-TTS](docs/inference/f5-tts.md) · [Higgs TTS 3](docs/inference/higgs-tts.md) · [VibeVoice](docs/inference/vibevoice-inference.md) · [Fish Audio S2 Pro](docs/inference/fish-audio-s2-pro.md) · [Magpie TTS](docs/inference/magpie-tts.md) · [Hibiki](docs/inference/hibiki-inference.md) · [MADLAD-400](docs/inference/madlad-translation.md) · [MAGNeT](docs/inference/magnet-music-gen.md) · [Stable Audio 3](docs/inference/stable-audio-3.md) · [FlashSR](docs/inference/flashsr.md) · [Forced Aligner](docs/inference/forced-aligner.md) · [Silero VAD](docs/inference/silero-vad.md) · [FireRedVAD](docs/inference/fireredvad.md) · [Wake-word](docs/inference/wake-word.md) · [Speaker Diarization](docs/inference/speaker-diarization.md) · [Speech Enhancement](docs/inference/speech-enhancement.md) · [Sidon](docs/inference/sidon.md) · [Cache/offline](docs/inference/cache-and-offline.md)
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
