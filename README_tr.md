# Speech Swift

Apple Silicon için yapay zeka destekli konuşma modelleri; MLX Swift ve CoreML ile çalışır.

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

Mac ve iOS için cihaz üzerinde konuşma tanıma, sentezleme ve anlama. Apple Silicon üzerinde yerel olarak çalışır — bulut yok, API anahtarı yok, hiçbir veri cihazınızdan dışarı çıkmaz.

**[📚 Tam Dokümantasyon →](https://soniqo.audio)** · **[🤗 HuggingFace Modelleri](https://huggingface.co/aufklarer)** · **[📝 Blog](https://blog.ivan.digital)** · **[💬 Discord](https://discord.gg/TnCryqEMgu)**

<p align="center">
  <a href="https://www.producthunt.com/products/speech-swift?embed=true&amp;utm_source=badge-featured&amp;utm_medium=badge&amp;utm_campaign=badge-speech-swift" target="_blank" rel="noopener noreferrer"><img alt="speech-swift -  The whole speech stack, on your laptop. | Product Hunt" width="250" height="54" src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1151422&amp;theme=light&amp;t=1779261593657"></a>
</p>

<p align="center">
  <a href="https://youtu.be/x9zgcaW0gUk">
    <img src="https://img.youtube.com/vi/x9zgcaW0gUk/maxresdefault.jpg" width="640" alt="MacBook üzerinde yerel konuşma yapay zekası — açık kaynak kütüphanenin 4 dakikalık tanıtımını YouTube'da izleyin">
  </a>
</p>
<p align="center"><em>MacBook üzerinde yerel konuşma yapay zekası — açık kaynak kütüphanenin 4 dakikalık tanıtımını YouTube'da izleyin</em></p>

**Kullanım senaryoları:** [Sesli Ajanlar](https://soniqo.audio/voice-agents) · [Transkripsiyon](https://soniqo.audio/transcription) · [Konuşma Üretimi](https://soniqo.audio/speech-generation)

- **[Qwen3-ASR](https://soniqo.audio/guides/transcribe)** — Konuşmadan metne (otomatik konuşma tanıma, 52 dil, MLX + CoreML)
- **[Parakeet TDT](https://soniqo.audio/guides/parakeet)** — CoreML üzerinden konuşmadan metne (Neural Engine, NVIDIA FastConformer + TDT kod çözücü, 25 dil)
- **[Omnilingual ASR](https://soniqo.audio/guides/omnilingual)** — Konuşmadan metne (Meta wav2vec2 + CTC, 32 yazı sistemi üzerinde **1.672 dil**, CoreML 300M + MLX 300M/1B/3B/7B)
- **[Akış Dikte](https://soniqo.audio/guides/dictate)** — Kısmi sonuçlar ve söyleyiş sonu algılaması ile gerçek zamanlı dikte (Parakeet-EOU-120M)
- **[Nemotron Streaming](https://soniqo.audio/guides/nemotron)** — Yerel noktalama ve büyük harf desteğiyle düşük gecikmeli akış ASR (NVIDIA Nemotron-Speech-Streaming-0.6B, CoreML, İngilizce)
- **[Qwen3-ForcedAligner](https://soniqo.audio/guides/align)** — Kelime düzeyinde zaman damgası hizalama (ses + metin → zaman damgaları)
- **[Qwen3-TTS](https://soniqo.audio/guides/speak)** — Metinden konuşmaya (en yüksek kalite, akış, özel konuşmacılar, 10 dil)
- **[CosyVoice TTS](https://soniqo.audio/guides/cosyvoice)** — Ses klonlama, çok konuşmacılı diyalog, duygu etiketleri ile akış TTS (9 dil)
- **[VoxCPM2](https://soniqo.audio/speech-generation)** — Ses klonlama ve talimat odaklı ses tasarımı ile 48 kHz stüdyo kalitesinde TTS (2B, MLX bf16/int8/int4, 30 dil)
- **[Kokoro TTS](https://soniqo.audio/guides/kokoro)** — Cihaz üzerinde TTS (82M, CoreML/Neural Engine, 54 ses, iOS için hazır, 10 dil)
- **[VibeVoice TTS](https://soniqo.audio/guides/vibevoice)** — Uzun biçimli / çok konuşmacılı TTS (Microsoft VibeVoice Realtime-0.5B + 1.5B, MLX, 90 dakikaya kadar podcast/sesli kitap sentezi, EN/ZH)
- **[Magpie TTS](https://soniqo.audio/guides/magpie)** — Çok dilli TTS (NVIDIA Magpie-TTS Multilingual 357M, MLX INT4 247 MB / INT8 411 MB veya CoreML INT8 342 MB, 9 dil, 5 hazır konuşmacı, MLX'te akış)
- **[Qwen3.5-Chat](https://soniqo.audio/guides/chat)** — Cihaz üzerinde LLM sohbet (0.8B, MLX INT4 + CoreML INT8, DeltaNet hibrit, akış token'ları)
- **[MADLAD-400](https://soniqo.audio/guides/translate)** — 400+ dil arasında çoktan-çoğa çeviri (3B, MLX INT4 + INT8, T5 v1.1, Apache 2.0)
- **[PersonaPlex](https://soniqo.audio/guides/respond)** — Tam çift yönlü (full-duplex) konuşmadan konuşmaya (7B, ses girişi → ses çıkışı, 18 ses ön ayarı)
- **[DeepFilterNet3](https://soniqo.audio/guides/denoise)** — Gerçek zamanlı gürültü bastırma (2.1M parametre, 48 kHz)
- **[Kaynak Ayrıştırma](https://soniqo.audio/guides/separate)** — HTDemucs (Demucs v4) + Open-Unmix ile müzik kaynağı ayrıştırma (UMX-HQ / UMX-L, 4 katman: vokal/davul/bas/diğer, 44,1 kHz stereo)
- **[MAGNeT](https://soniqo.audio/guides/compose)** — Metinden müziğe üretim (Meta MAGNeT Small 300M / Medium 1.5B, MLX INT4/INT8, 32 kHz mono'da 30 sn klipler, maskelenmiş paralel kod çözme)
- **[FlashSR](https://soniqo.audio/guides/upsample)** — Ses süper çözünürlük (FlashSR ICASSP 2025, MLX, 48 kHz mono, 1 adımda damıtılmış difüzyon, INT4 363 MB / INT8 720 MB)
- **[Uyandırma kelimesi](https://soniqo.audio/guides/wake-word)** — Cihaz üzerinde anahtar kelime tespiti (KWS Zipformer 3M, CoreML, 26× gerçek zaman, yapılandırılabilir anahtar kelime listesi)
- **[VAD](https://soniqo.audio/guides/vad)** — Ses etkinlik algılama (Silero akış, Pyannote çevrimdışı, FireRedVAD 100+ dil)
- **[Konuşmacı Ayrımı](https://soniqo.audio/guides/diarize)** — Kim ne zaman konuştu (Pyannote pipeline, Neural Engine üzerinde uçtan uca Sortformer)
- **[Konuşmacı Gömmeleri](https://soniqo.audio/guides/embed-speaker)** — WeSpeaker ResNet34 (256-boyut), CAM++ (192-boyut)

Makaleler: [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba) · [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba) · [Omnilingual ASR](https://arxiv.org/abs/2511.09690) (Meta) · [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA) · [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba) · [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2) · [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA) · [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai) · [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## Haberler

- **19 Nis 2026** — [Apple Silicon'da MLX vs CoreML — Doğru Backend'i Seçmek İçin Pratik Bir Rehber](https://blog.ivan.digital/mlx-vs-coreml-on-apple-silicon-a-practical-guide-to-picking-the-right-backend-and-why-you-should-f77ddea7b27a)
- **20 Mar 2026** — [Tamamen Mac'inizde Çalışan 600M'lik Bir Modelle Whisper Large v3'ü Geçtik](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **26 Şub 2026** — [Apple Silicon'da Konuşmacı Ayrımı ve Ses Etkinlik Algılama — MLX ile Yerel Swift](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **23 Şub 2026** — [Apple Silicon'da NVIDIA PersonaPlex 7B — MLX ile Yerel Swift'te Tam Çift Yönlü Konuşmadan Konuşmaya](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **12 Şub 2026** — [Qwen3-ASR Swift: Apple Silicon İçin Cihaz Üzerinde ASR + TTS — Mimari ve Benchmark'lar](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## Hızlı başlangıç

Paketi `Package.swift` dosyanıza ekleyin:

```swift
.package(url: "https://github.com/soniqo/speech-swift", branch: "main")
```

Yalnızca ihtiyacınız olan modülleri içe aktarın — her model kendi SPM kütüphanesidir, böylece kullanmadığınız şey için bedel ödemezsiniz:

```swift
.product(name: "ParakeetStreamingASR", package: "speech-swift"),
.product(name: "SpeechUI",             package: "speech-swift"),  // isteğe bağlı SwiftUI görünümleri
```

**Bir ses tamponunu 3 satırda transkribe edin:**

```swift
import ParakeetStreamingASR

let model = try await ParakeetStreamingASRModel.fromPretrained()
let text = try model.transcribeAudio(audioSamples, sampleRate: 16000)
```

**Kısmi sonuçlarla canlı akış:**

```swift
for await partial in model.transcribeStream(audio: samples, sampleRate: 16000) {
    print(partial.isFinal ? "FINAL: \(partial.text)" : "... \(partial.text)")
}
```

**~10 satırda SwiftUI dikte görünümü:**

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

`SpeechUI` yalnızca `TranscriptionView` (kesin sonuçlar + kısmi sonuçlar) ve `TranscriptionStore` (akış ASR adaptörü) sunar. Ses görselleştirme ve oynatma için AVFoundation kullanın.

Mevcut SPM ürünleri: `Qwen3ASR`, `Qwen3TTS`, `Qwen3TTSCoreML`, `ParakeetASR`, `ParakeetStreamingASR`, `NemotronStreamingASR`, `OmnilingualASR`, `KokoroTTS`, `VibeVoiceTTS`, `CosyVoiceTTS`, `VoxCPM2TTS`, `MagpieTTS`, `MagpieTTSCoreML`, `MAGNeTMusicGen`, `FlashSR`, `PersonaPlex`, `SpeechVAD`, `SpeechEnhancement`, `SourceSeparation`, `Qwen3Chat`, `SpeechCore`, `SpeechUI`, `AudioCommon`.

## Modeller

Aşağıda kompakt bir görünüm. **[Boyutlar, kuantizasyonlar, indirme URL'leri ve bellek tablolarıyla tam model kataloğu → soniqo.audio/architecture](https://soniqo.audio/architecture)**.

| Model | Görev | Backend'ler | Boyutlar | Diller |
|-------|------|----------|-------|-----------|
| [Qwen3-ASR](https://soniqo.audio/guides/transcribe) | Konuşma → Metin | MLX, CoreML (hibrit) | 0.6B, 1.7B | 52 |
| [Parakeet TDT](https://soniqo.audio/guides/parakeet) | Konuşma → Metin | CoreML (ANE) | 0.6B | 25 Avrupa dili |
| [Parakeet EOU](https://soniqo.audio/guides/dictate) | Konuşma → Metin (akış) | CoreML (ANE) | 120M | 25 Avrupa dili |
| [Nemotron Streaming](https://soniqo.audio/guides/nemotron) | Konuşma → Metin (akış, noktalamalı) | CoreML (ANE) | 0.6B | EN |
| [Omnilingual ASR](https://soniqo.audio/guides/omnilingual) | Konuşma → Metin | CoreML (ANE), MLX | 300M / 1B / 3B / 7B | **[1.672](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)** |
| [Qwen3-ForcedAligner](https://soniqo.audio/guides/align) | Ses + Metin → Zaman damgaları | MLX, CoreML | 0.6B | Çoklu |
| [Qwen3-TTS](https://soniqo.audio/guides/speak) | Metin → Konuşma | MLX, CoreML | 0.6B, 1.7B | 10 |
| [CosyVoice3](https://soniqo.audio/guides/cosyvoice) | Metin → Konuşma | MLX | 0.5B | 9 |
| [VoxCPM2](https://soniqo.audio/speech-generation) | Metin → Konuşma (48 kHz, ses tasarımı + klonlama) | MLX | 2B (bf16/int8/int4) | 30 |
| [Kokoro-82M](https://soniqo.audio/guides/kokoro) | Metin → Konuşma | CoreML (ANE) | 82M | 10 |
| [VibeVoice Realtime-0.5B](https://soniqo.audio/guides/vibevoice) | Metin → Konuşma (uzun biçimli, çok konuşmacılı) | MLX | 0.5B | EN/ZH |
| [VibeVoice 1.5B](https://soniqo.audio/guides/vibevoice) | Metin → Konuşma (90 dakikaya kadar podcast) | MLX | 1.5B | EN/ZH |
| [Magpie-TTS Multilingual](https://soniqo.audio/guides/magpie) | Metin → Konuşma (5 hazır konuşmacı, akış) | MLX / CoreML | 357M (MLX INT4/INT8, CoreML INT8) | 9 (CoreML, JA hariç) |
| [Qwen3.5-Chat](https://soniqo.audio/guides/chat) | Metin → Metin (LLM) | MLX, CoreML | 0.8B | Çoklu |
| [MADLAD-400](https://soniqo.audio/guides/translate) | Metin → Metin (Çeviri) | MLX | 3B | **400+** |
| [PersonaPlex](https://soniqo.audio/guides/respond) | Konuşma → Konuşma | MLX | 7B | EN |
| [Silero VAD](https://soniqo.audio/guides/vad) | Ses Etkinlik Algılama | MLX, CoreML | 309K | Bağımsız |
| [Pyannote](https://soniqo.audio/guides/diarize) | VAD + Konuşmacı Ayrımı | MLX | 1.5M | Bağımsız |
| [Sortformer](https://soniqo.audio/guides/diarize) | Konuşmacı Ayrımı (E2E) | CoreML (ANE) | — | Bağımsız |
| [DeepFilterNet3](https://soniqo.audio/guides/denoise) | Konuşma İyileştirme | CoreML | 2.1M | Bağımsız |
| [HTDemucs (Demucs v4)](https://soniqo.audio/guides/separate) | Kaynak Ayrıştırma | MLX | 168M | Bağımsız |
| [Open-Unmix](https://soniqo.audio/guides/separate) | Kaynak Ayrıştırma | MLX | 8.6M | Bağımsız |
| [MAGNeT](https://soniqo.audio/guides/compose) | Metin → Müzik (30s @ 32 kHz) | MLX | 300M / 1.5B (int4/int8) | EN prompt'ları |
| [FlashSR](https://soniqo.audio/guides/upsample) | Ses süper çözünürlük (48 kHz) | MLX | 363 MB / 720 MB (int4/int8) | Bağımsız |
| [WeSpeaker](https://soniqo.audio/guides/embed-speaker) | Konuşmacı Gömme | MLX, CoreML | 6.6M | Bağımsız |

## Kurulum

### Homebrew

Yerel ARM Homebrew (`/opt/homebrew`) gerektirir. Rosetta/x86_64 Homebrew desteklenmez.

```bash
brew install soniqo/tap/speech
```

Ardından:

```bash
speech transcribe recording.wav
speech speak "Hello world"
speech translate "Hello, how are you?" --to es
speech respond --input question.wav --transcript
speech-server --port 8080            # yerel HTTP / WebSocket sunucusu (OpenAI-compatible /v1/realtime + /v1/audio/transcriptions)
```

**[Tam CLI referansı →](https://soniqo.audio/cli)**

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

Yalnızca ihtiyacınız olanı içe aktarın — her model kendi SPM hedefidir:

```swift
import Qwen3ASR             // Konuşma tanıma (MLX)
import ParakeetASR          // Konuşma tanıma (CoreML, batch)
import ParakeetStreamingASR // Kısmi sonuçlar + EOU ile akış dikte
import NemotronStreamingASR // Yerel noktalama ile İngilizce akış ASR (0.6B)
import OmnilingualASR       // 1.672 dil (CoreML + MLX)
import Qwen3TTS             // Metinden konuşmaya
import CosyVoiceTTS         // Ses klonlama ile metinden konuşmaya
import VoxCPM2TTS           // Ses klonlama + ses tasarımı ile 48 kHz TTS (2B)
import KokoroTTS            // Metinden konuşmaya (iOS için hazır)
import VibeVoiceTTS         // Uzun biçimli / çok konuşmacılı TTS (EN/ZH)
import MagpieTTS            // Çok dilli TTS (NVIDIA Magpie 357M, MLX, 9 dil)
import MagpieTTSCoreML      // Magpie CoreML backend'i (hibrit CoreML + MLX, 8 dil)
import Qwen3Chat            // Cihaz üzerinde LLM sohbet
import MADLADTranslation    // 400+ dil arasında çoktan-çoğa çeviri
import PersonaPlex          // Tam çift yönlü konuşmadan konuşmaya
import SpeechVAD            // VAD + konuşmacı ayrımı + gömmeler
import SpeechEnhancement    // Gürültü bastırma
import SourceSeparation     // Müzik kaynak ayrıştırma (Open-Unmix, 4 katman)
import SpeechUI             // Akış transkriptleri için SwiftUI bileşenleri
import AudioCommon          // Paylaşılan protokoller ve yardımcılar
```

### Gereksinimler

- Swift 6+, Xcode 16+ (Metal Toolchain ile)
- macOS 15+ (Sequoia) veya iOS 18+, Apple Silicon (M1/M2/M3/M4)

macOS 15 / iOS 18 minimum gereksinimi [MLState](https://developer.apple.com/documentation/coreml/mlstate)'ten gelir — Apple'ın CoreML pipeline'larının (Qwen3-ASR, Qwen3-Chat, Qwen3-TTS) KV önbelleklerini token adımları arasında Neural Engine'de tutmak için kullandığı kalıcı ANE durum API'sı.

### Kaynaktan derleme

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

`make build`, Swift paketini **ve** MLX Metal shader kütüphanesini derler. Metal kütüphanesi GPU çıkarımı için gereklidir — onsuz çalışma zamanında `Failed to load the default metallib` hatası görürsünüz. Debug build'leri için `make debug`, test paketi için `make test`.

**[Tam derleme ve kurulum rehberi →](https://soniqo.audio/getting-started)**

## Demo uygulamalar

- **[DictateDemo](Examples/DictateDemo/)** ([dokümantasyon](https://soniqo.audio/guides/dictate)) — Canlı kısmi sonuçlar, VAD tabanlı söyleyiş sonu algılaması ve tek tıklama kopyalama ile macOS menü çubuğu akış dikte. Arka plan ajanı olarak çalışır (Parakeet-EOU-120M + Silero VAD).
- **[iOSEchoDemo](Examples/iOSEchoDemo/)** — iOS yankı demosu (Parakeet ASR + Kokoro TTS). Cihaz ve simülatör.
- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — Mikrofon girişi, VAD ve çok turlu bağlam ile konuşmacı sesli asistan. macOS. M2 Max üzerinde RTF ~0.94 (gerçek zamandan hızlı).
- **[SpeechDemo](Examples/SpeechDemo/)** — Sekmeli arayüzde dikte ve TTS sentezi. macOS.

Her demonun README'sinde derleme talimatları yer alır.

## Kod örnekleri

Aşağıdaki kod parçacıkları her alan için minimum yolu gösterir. Her bölüm, yapılandırma seçenekleri, birden fazla backend, akış desenleri ve CLI tarifleriyle [soniqo.audio](https://soniqo.audio) üzerindeki tam bir rehbere bağlanır.

### Konuşmadan metne — [tam rehber →](https://soniqo.audio/guides/transcribe)

```swift
import Qwen3ASR

let model = try await Qwen3ASRModel.fromPretrained()
let text = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

Alternatif backend'ler: [Parakeet TDT](https://soniqo.audio/guides/parakeet) (CoreML, 32× gerçek zaman), [Omnilingual ASR](https://soniqo.audio/guides/omnilingual) (1.672 dil, CoreML veya MLX), [Akış dikte](https://soniqo.audio/guides/dictate) (canlı kısmi sonuçlar).

### Zorunlu Hizalama — [tam rehber →](https://soniqo.audio/guides/align)

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

### Metinden konuşmaya — [tam rehber →](https://soniqo.audio/guides/speak)

```swift
import Qwen3TTS
import AudioCommon

let model = try await Qwen3TTSModel.fromPretrained()
let audio = model.synthesize(text: "Hello world", language: "english")
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

Alternatif TTS motorları: [CosyVoice3](https://soniqo.audio/guides/cosyvoice) (akış + ses klonlama + duygu etiketleri), [Kokoro-82M](https://soniqo.audio/guides/kokoro) (iOS için hazır, 54 ses), [VibeVoice](https://soniqo.audio/guides/vibevoice) (uzun biçimli podcast / çok konuşmacılı, EN/ZH), [Ses klonlama](https://soniqo.audio/guides/voice-cloning).

### Konuşmadan konuşmaya — [tam rehber →](https://soniqo.audio/guides/respond)

```swift
import PersonaPlex

let model = try await PersonaPlexModel.fromPretrained()
let responseAudio = model.respond(userAudio: userSamples)
// Oynatmaya hazır 24 kHz mono Float32 çıkış
```

### LLM Sohbet — [tam rehber →](https://soniqo.audio/guides/chat)

```swift
import Qwen3Chat

let chat = try await Qwen35MLXChat.fromPretrained()
chat.chat(messages: [(.user, "Explain MLX in one sentence")]) { token, isFinal in
    print(token, terminator: "")
}
```

### Çeviri — [tam rehber →](https://soniqo.audio/guides/translate)

```swift
import MADLADTranslation

let translator = try await MADLADTranslator.fromPretrained()
let es = try translator.translate("Hello, how are you?", to: "es")
// → "Hola, ¿cómo estás?"
```

### Ses Etkinlik Algılama — [tam rehber →](https://soniqo.audio/guides/vad)

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
for s in segments { print("\(s.startTime)s → \(s.endTime)s") }
```

### Konuşmacı Ayrımı — [tam rehber →](https://soniqo.audio/guides/diarize)

```swift
import SpeechVAD

let diarizer = try await DiarizationPipeline.fromPretrained()
let segments = diarizer.diarize(audio: samples, sampleRate: 16000)
for s in segments { print("Speaker \(s.speakerId): \(s.startTime)s - \(s.endTime)s") }
```

### Konuşma İyileştirme — [tam rehber →](https://soniqo.audio/guides/denoise)

```swift
import SpeechEnhancement

let denoiser = try await DeepFilterNet3Model.fromPretrained()
let clean = try denoiser.enhance(audio: noisySamples, sampleRate: 48000)
```

### Ses Pipeline'ı (ASR → LLM → TTS) — [tam rehber →](https://soniqo.audio/voice-agents)

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

`VoicePipeline`, VAD tabanlı tur algılama, kesinti yönetimi ve eager STT ile gerçek zamanlı sesli ajan durum makinesidir ([speech-core](https://github.com/soniqo/speech-core) tarafından desteklenir). Herhangi bir `SpeechRecognitionModel` + `SpeechGenerationModel` + `StreamingVADProvider` bileşenini birbirine bağlar.

### HTTP API sunucusu

```bash
speech-server --port 8080
```

OpenAI uyumlu API'lar dahil olmak üzere her modeli HTTP REST + WebSocket uç noktaları üzerinden sunar: `/v1/realtime` üzerindeki Realtime WebSocket ve `/v1/audio/transcriptions` üzerindeki transkripsiyon REST uç noktası. Bkz. [`Sources/AudioServer/`](Sources/AudioServer/).

## Mimari

speech-swift, kullanıcıların yalnızca içe aktardıkları şey için bedel ödemesi adına model başına bir SPM hedefine ayrılmıştır. Paylaşılan altyapı `AudioCommon` (protokoller, ses G/Ç, HuggingFace indirici, `SentencePieceModel`) ve `MLXCommon` (ağırlık yükleme, `QuantizedLinear` yardımcıları, `SDPA` çok başlı dikkat yardımcısı) içinde yer alır.

**[Backend'ler, bellek tabloları ve modül haritasıyla tam mimari diyagramı → soniqo.audio/architecture](https://soniqo.audio/architecture)** · **[API referansı → soniqo.audio/api](https://soniqo.audio/api)** · **[Benchmark'lar → soniqo.audio/benchmarks](https://soniqo.audio/benchmarks)**

Yerel dokümantasyon (depo):
- **Modeller:** [Qwen3-ASR](docs/models/asr-model.md) · [Qwen3-TTS](docs/models/tts-model.md) · [CosyVoice](docs/models/cosyvoice-tts.md) · [Kokoro](docs/models/kokoro-tts.md) · [VibeVoice](docs/models/vibevoice.md) · [Parakeet TDT](docs/models/parakeet-asr.md) · [Parakeet Streaming](docs/models/parakeet-streaming-asr.md) · [Nemotron Streaming](docs/models/nemotron-streaming.md) · [Omnilingual ASR](docs/models/omnilingual-asr.md) · [PersonaPlex](docs/models/personaplex.md) · [FireRedVAD](docs/models/fireredvad.md) · [Source Separation](docs/models/source-separation.md) · [HTDemucs](docs/models/htdemucs.md) · [MAGNeT](docs/models/magnet-music-gen.md) · [FlashSR](docs/models/flashsr.md)
- **Çıkarım:** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md) · [Parakeet TDT](docs/inference/parakeet-asr-inference.md) · [Parakeet Streaming](docs/inference/parakeet-streaming-asr-inference.md) · [Nemotron Streaming](docs/inference/nemotron-streaming-inference.md) · [Omnilingual ASR](docs/inference/omnilingual-asr-inference.md) · [TTS](docs/inference/qwen3-tts-inference.md) · [VibeVoice](docs/inference/vibevoice-inference.md) · [MAGNeT](docs/inference/magnet-music-gen.md) · [FlashSR](docs/inference/flashsr.md) · [Forced Aligner](docs/inference/forced-aligner.md) · [Silero VAD](docs/inference/silero-vad.md) · [Konuşmacı Ayrımı](docs/inference/speaker-diarization.md) · [Konuşma İyileştirme](docs/inference/speech-enhancement.md)
- **Referans:** [Paylaşılan Protokoller](docs/shared-protocols.md)

## Önbellek yapılandırması

Model ağırlıkları ilk kullanımda HuggingFace'ten indirilir ve `~/Library/Caches/qwen3-speech/` dizinine önbelleğe alınır. `QWEN3_CACHE_DIR` (CLI) veya `cacheDir:` (Swift API) ile geçersiz kılabilirsiniz. Tüm `fromPretrained()` giriş noktaları, ağırlıklar zaten önbellekteyse ağı atlamak için `offlineMode: true` parametresini de kabul eder.

Sandbox'lı iOS konteyner yolları dahil tam ayrıntılar için [`docs/inference/cache-and-offline.md`](docs/inference/cache-and-offline.md) belgesine bakın.

## MLX Metal kütüphanesi

Çalışma zamanında `Failed to load the default metallib` görürseniz Metal shader kütüphanesi eksiktir. Manuel bir `swift build` sonrası `make build` veya `./scripts/build_mlx_metallib.sh release` komutunu çalıştırın. Metal Toolchain eksikse, önce onu kurun:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Test

```bash
make test                            # tam paket (birim + model indirmeleriyle E2E)
swift test --skip E2E                # yalnızca birim (CI için güvenli, indirme yok)
swift test --filter Qwen3ASRTests    # belirli modül
```

E2E test sınıfları, CI'nin `--skip E2E` ile bunları filtreleyebilmesi için `E2E` önekini kullanır. Tam test kuralı için [CLAUDE.md](CLAUDE.md#testing) belgesine bakın.

## Katkıda bulunma

PR'lar memnuniyetle karşılanır — hata düzeltmeleri, yeni model entegrasyonları, dokümantasyon. Fork'layın, bir özellik dalı oluşturun, `make build && make test` çalıştırın, `main`'e karşı bir PR açın.

## Lisans

Apache 2.0
