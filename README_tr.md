# Speech Swift

Apple Silicon için yapay zeka destekli konuşma modelleri; MLX Swift ve CoreML ile çalışır.

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

Mac ve iOS için cihaz üzerinde konuşma tanıma, sentezleme ve anlama. Apple Silicon üzerinde yerel olarak çalışır — bulut yok, API anahtarı yok, hiçbir veri cihazınızdan dışarı çıkmaz.

**[📚 Tam Dokümantasyon →](https://soniqo.audio)** · **[🤗 HuggingFace Modelleri](https://huggingface.co/aufklarer)** · **[📝 Blog](https://blog.ivan.digital)** · **[💬 Discord](https://discord.gg/TnCryqEMgu)**

<p align="center">
  <a href="https://trendshift.io/repositories/24196?utm_source=trendshift-badge&amp;utm_medium=badge&amp;utm_campaign=badge-trendshift-24196" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/trendshift/repositories/24196/daily?language=Swift" alt="soniqo%2Fspeech-swift | Trendshift" width="250" height="55"/></a>
</p>

<p align="center">
  <a href="https://youtu.be/x9zgcaW0gUk">
    <img src="https://img.youtube.com/vi/x9zgcaW0gUk/maxresdefault.jpg" width="640" alt="MacBook üzerinde yerel konuşma yapay zekası — açık kaynak kütüphanenin 4 dakikalık tanıtımını YouTube'da izleyin">
  </a>
</p>
<p align="center"><em>MacBook üzerinde yerel konuşma yapay zekası — açık kaynak kütüphanenin 4 dakikalık tanıtımını YouTube'da izleyin</em></p>

**Kullanım senaryoları:** [Sesli Ajanlar](https://soniqo.audio/voice-agents) · [Transkripsiyon](https://soniqo.audio/transcription) · [Konuşma Üretimi](https://soniqo.audio/speech-generation)

## Speech Swift ile geliştirildi

Doğrulanabilir Speech Swift entegrasyonlarına sahip açık projeler.

[Palmier Pro](https://github.com/palmier-io/palmier-pro) · [Anarlog](https://github.com/fastrepl/anarlog) · [VoxFlow](https://github.com/xingbofeng/VoxFlow) · [Ora](https://github.com/wuwangzhang1216/ora) · [Jabber](https://github.com/rselbach/jabber) · [ClawdHome](https://github.com/ThinkInAIXYZ/clawdhome)

**Yetenek grupları:** STT / ASR · Hizalama · TTS · LLM ve çeviri · Speech-to-Speech · İyileştirme / restorasyon · Kaynak ayırma · Müzik / ses üretimi · Wake word, VAD, konuşmacı ayrıştırma ve konuşmacı kimliği

**STT / ASR**

- **[Qwen3-ASR](https://soniqo.audio/guides/transcribe)** — Konuşmadan metne (otomatik konuşma tanıma, 52 dil, MLX + CoreML)
- **[WhisperASR](docs/models/whisper-asr.md)** — Whisper Large-v3 Turbo speech-to-text via native CoreML runtime (ANE, multilingual)
- **[Parakeet TDT](https://soniqo.audio/guides/parakeet)** — CoreML üzerinden konuşmadan metne (Neural Engine, NVIDIA FastConformer + TDT kod çözücü, 25 dil)
- **[Omnilingual ASR](https://soniqo.audio/guides/omnilingual)** — Konuşmadan metne (Meta wav2vec2 + CTC, 32 yazı sistemi üzerinde **1.672 dil**, CoreML 300M + MLX 300M/1B/3B/7B)
- **[Akış Dikte](https://soniqo.audio/guides/dictate)** — Kısmi sonuçlar ve söyleyiş sonu algılaması ile gerçek zamanlı dikte (Parakeet-EOU-120M)
- **[Nemotron Streaming (Çok dilli)](https://soniqo.audio/guides/nemotron)** — Yerel noktalama ve büyük harf desteğiyle düşük gecikmeli akış ASR (NVIDIA Nemotron-3.5-ASR-Streaming-0.6B, CoreML + MLX, **40 dil-yerel ayarı**)
- **[Nemotron Streaming (İngilizce)](https://soniqo.audio/guides/nemotron)** — Yerel noktalama ve büyük harf desteğiyle düşük gecikmeli akış ASR (NVIDIA Nemotron-Speech-Streaming-0.6B, CoreML, yalnızca İngilizce, çok dilli varyanttan daha küçük ve hızlı)

**Hizalama**

- **[Qwen3-ForcedAligner](https://soniqo.audio/guides/align)** — Kelime düzeyinde zaman damgası hizalama (ses + metin → zaman damgaları)

**TTS / Konuşma üretimi**

- **[Qwen3-TTS](https://soniqo.audio/guides/speak)** — Metinden konuşmaya (en yüksek kalite, akış, özel konuşmacılar, 10 dil)
- **[CosyVoice TTS](https://soniqo.audio/guides/cosyvoice)** — Ses klonlama, çok konuşmacılı diyalog, duygu etiketleri ile akış TTS (9 dil)
- **[VoxCPM2](https://soniqo.audio/speech-generation)** — Ses klonlama ve talimat odaklı ses tasarımı ile 48 kHz stüdyo kalitesinde TTS (2B, MLX bf16/int8, 30 dil)
- **[IndexTTS2](docs/models/indextts2.md)** — Native MLX voice cloning from a reference voice (IndexTeam IndexTTS-2, 1.5B-class fp16 bundle, speaker/emotion/pause controls)
- **[F5-TTS](docs/models/f5-tts.md)** — Zero-shot voice cloning from a short reference clip + transcript (SWivid F5-TTS v1 Base, DiT flow matching + Vocos, MLX fp16, 24 kHz, English + Mandarin; non-commercial license)
- **[Higgs TTS 3](docs/models/higgs-tts.md)** — Conversational TTS with zero-shot voice cloning and inline emotion/style/SFX/prosody tags (Boson Higgs TTS 3, Qwen3-4B backbone, MLX bf16, 24 kHz, 100+ languages; research/non-commercial license)
- **[Kokoro TTS](https://soniqo.audio/guides/kokoro)** — Cihaz üzerinde TTS (82M, CoreML/Neural Engine, 54 ses, iOS için hazır, 10 dil)
- **[VibeVoice TTS](https://soniqo.audio/guides/vibevoice)** — Uzun biçimli / çok konuşmacılı TTS (Microsoft VibeVoice Realtime-0.5B + 1.5B, MLX, 90 dakikaya kadar podcast/sesli kitap sentezi, EN/ZH)
- **[Magpie TTS](https://soniqo.audio/guides/magpie)** — Çok dilli TTS (NVIDIA Magpie-TTS Multilingual 357M, MLX INT8 411 MB veya CoreML INT8 342 MB, 9 dil, 5 hazır konuşmacı, MLX'te akış)
- **[Supertonic TTS](https://soniqo.audio/guides/supertonic)** — Cihaz üzerinde flow-matching TTS (Supertone Supertonic-3 99M, CoreML/Neural Engine, 31 dil, 10 ses, G2P-free, 44.1 kHz)
- **[Chatterbox TTS](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16)** — Zero-shot ses klonlama özellikli çok dilli TTS (Resemble AI Chatterbox Multilingual, MLX fp16 ~1,3 GB, 23 çalışma zamanı dili; İbranice için niqqud gerekir, MIT)
- **[OmniVoice TTS](https://huggingface.co/aufklarer/OmniVoice-MLX-fp16)** — Zero-shot ses klonlama ile otoregresif olmayan difüzyon TTS (k2-fsa OmniVoice, Qwen3 omurgası, MLX fp16 varsayılan / int8 kullanılabilir, 600+ dil, Apache-2.0)
- **[Indic-Mio](docs/models/indic-mio-tts.md)** — Hindi/Indic TTS with inline emotion markers and optional reference-voice cloning (MLX, 24 kHz)

**LLM ve çeviri**

- **[Qwen3Chat](https://soniqo.audio/guides/chat)** — Cihaz üzerinde LLM sohbet (Qwen3.5-0.8B MLX/CoreML artı dense Qwen3 4B ve Gemma 4 E2B/E4B MLX arka uçları, akış token'ları)
- **[FunctionGemma](https://soniqo.audio/guides/function-calls)** — Cihaz üzerinde yapılandırılmış fonksiyon / araç çağrıları için LLM (Gemma 3 270M, CoreML 8-bit paletleme, Neural Engine, ~252 tok/s)
- **[MADLAD-400](https://soniqo.audio/guides/translate)** — 400+ dil arasında çoktan-çoğa çeviri (3B, MLX INT4 + INT8, T5 v1.1, Apache 2.0)

**Speech-to-Speech ve sesli ajanlar**

- **[Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate)** — Akışlı konuşmadan konuşmaya çeviri (FR/ES/PT/DE → EN, MLX INT4 + INT8, Kyutai Moshi/Mimi yığını, CC-BY-4.0)
- **[PersonaPlex](https://soniqo.audio/guides/respond)** — Tam çift yönlü (full-duplex) konuşmadan konuşmaya (7B, ses girişi → ses çıkışı, 18 ses ön ayarı)
- **[Audio2Face-3D](docs/models/audio2face3d.md)** — Konuşmayla sürülen avatar yüz animasyonu (NVIDIA Audio2Face-3D v2.3 Mark, 301 yüz katsayısı, MLX)

**İyileştirme, ayırma ve ses üretimi**

- **[DeepFilterNet3](https://soniqo.audio/guides/denoise)** — Gerçek zamanlı gürültü bastırma (2.1M parametre, 48 kHz)
- **[Kaynak Ayrıştırma](https://soniqo.audio/guides/separate)** — HTDemucs (Demucs v4) + Open-Unmix ile müzik kaynağı ayrıştırma (UMX-HQ / UMX-L, 4 katman: vokal/davul/bas/diğer, 44,1 kHz stereo)
- **[MAGNeT](https://soniqo.audio/guides/compose)** — Metinden müziğe üretim (Meta MAGNeT Small 300M / Medium 1.5B, MLX INT8, 32 kHz mono'da 30 sn klipler, maskelenmiş paralel kod çözme)
- **[Stable Audio 3](docs/models/stable-audio-3.md)** — Text-to-audio/music generation (Stable Audio 3 Medium, MLX INT8/INT4, 44.1 kHz stereo, variable length)
- **[FlashSR](https://soniqo.audio/guides/upsample)** — Ses süper çözünürlük (FlashSR ICASSP 2025, MLX, 48 kHz mono, 1 adımda damıtılmış difüzyon, INT4 363 MB / INT8 720 MB)

**Sıra algılama, konuşmacı ayrıştırma ve konuşmacı kimliği**

- **[Uyandırma kelimesi](https://soniqo.audio/guides/wake-word)** — Cihaz üzerinde anahtar kelime tespiti (KWS Zipformer 3M, CoreML, 26× gerçek zaman, yapılandırılabilir anahtar kelime listesi)
- **[VAD](https://soniqo.audio/guides/vad)** — Ses etkinlik algılama (Silero akış, Pyannote çevrimdışı, FireRedVAD 100+ dil)
- **[Konuşmacı Ayrımı](https://soniqo.audio/guides/diarize)** — Kim ne zaman konuştu (Pyannote pipeline, Neural Engine üzerinde uçtan uca Sortformer) — artık artımlı akış oturumuyla (sabit konuşmacı kimlikleri, 480 ms'de bir güncelleme)
- **[Konuşmacı Gömmeleri](https://soniqo.audio/guides/embed-speaker)** — WeSpeaker ResNet34 (256-boyut), ReDimNet2-B6 ile adlandırılmış ses kimliği (192-boyut), CAM++ (192-boyut)

Makaleler: [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba) · [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba) · [Omnilingual ASR](https://arxiv.org/abs/2511.09690) (Meta) · [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA) · [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba) · [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2) · [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA) · [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai) · [Hibiki](https://arxiv.org/abs/2502.03382) (Kyutai) · [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

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

Mevcut SPM ürünleri: `Qwen3ASR`, `WhisperASR`, `Qwen3TTS`, `Qwen3TTSCoreML`, `ParakeetASR`, `ParakeetStreamingASR`, `NemotronStreamingASR`, `OmnilingualASR`, `KokoroTTS`, `SupertonicTTS`, `VibeVoiceTTS`, `CosyVoiceTTS`, `VoxCPM2TTS`, `IndexTTS2TTS`, `F5TTS`, `HiggsTTS`, `ChatterboxTTS`, `OmniVoiceTTS`, `IndicMioTTS`, `FishAudioTTS`, `MagpieTTS`, `MagpieTTSCoreML`, `MAGNeTMusicGen`, `StableAudio3MusicGen`, `FlashSR`, `PersonaPlex`, `Audio2Face3D`, `HibikiTranslate`, `MADLADTranslation`, `SpeechVAD`, `SpeechWakeWord`, `SpeechEnhancement`, `SpeechRestoration`, `SourceSeparation`, `Qwen3Chat`, `FunctionGemma`, `SpeechCore`, `SpeechUI`, `AudioCommon`.

## Modeller

Aşağıda kompakt bir görünüm. **[Boyutlar, kuantizasyonlar, indirme URL'leri ve bellek tablolarıyla tam model kataloğu → soniqo.audio/architecture](https://soniqo.audio/architecture)**.

| Model | Görev | Backend'ler | Boyutlar | Diller |
|-------|------|----------|-------|-----------|
| [Qwen3-ASR](https://soniqo.audio/guides/transcribe) | Konuşma → Metin | MLX, CoreML (hibrit) | 0.6B, 1.7B | 52 |
| [WhisperASR](docs/models/whisper-asr.md) | Speech → Text | CoreML (ANE) | Large-v3 Turbo | Multi |
| [Parakeet TDT](https://soniqo.audio/guides/parakeet) | Konuşma → Metin | CoreML (ANE) | 0.6B | 25 Avrupa dili |
| [Parakeet EOU](https://soniqo.audio/guides/dictate) | Konuşma → Metin (akış) | CoreML (ANE) | 120M | 25 Avrupa dili |
| [Nemotron Streaming (Çok dilli)](https://soniqo.audio/guides/nemotron) | Konuşma → Metin (akış, noktalamalı) | CoreML (ANE), MLX | 0.6B | **40** |
| [Nemotron Streaming (İngilizce)](https://soniqo.audio/guides/nemotron) | Konuşma → Metin (akış, noktalamalı) | CoreML (ANE) | 0.6B | EN |
| [Omnilingual ASR](https://soniqo.audio/guides/omnilingual) | Konuşma → Metin | CoreML (ANE), MLX | 300M / 1B / 3B / 7B | **[1.672](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)** |
| [Qwen3-ForcedAligner](https://soniqo.audio/guides/align) | Ses + Metin → Zaman damgaları | MLX, CoreML | 0.6B | Çoklu |
| [Qwen3-TTS](https://soniqo.audio/guides/speak) | Metin → Konuşma | MLX, CoreML | 0.6B, 1.7B | 10 |
| [CosyVoice3](https://soniqo.audio/guides/cosyvoice) | Metin → Konuşma | MLX | 0.5B | 9 |
| [VoxCPM2](https://soniqo.audio/speech-generation) | Metin → Konuşma (48 kHz, ses tasarımı + klonlama) | MLX | 2B (bf16/int8) | 30 |
| [IndexTTS2](docs/models/indextts2.md) | Text → Speech (zero-shot voice cloning) | MLX | 1.5B-class (fp16) | EN/ZH |
| [F5-TTS](docs/models/f5-tts.md) | Text → Speech (zero-shot voice cloning) | MLX | 336M (fp16) | EN/ZH |
| [Higgs TTS 3](docs/models/higgs-tts.md) | Text → Speech (conversational, zero-shot voice cloning) | MLX | 4B (bf16) | 100+ |
| [Kokoro-82M](https://soniqo.audio/guides/kokoro) | Metin → Konuşma | CoreML (ANE) | 82M | 10 |
| [Supertonic-3](https://soniqo.audio/guides/supertonic) | Metin → Konuşma (44.1 kHz, flow-matching, G2P-free) | CoreML (ANE) | 99M | 31 |
| [VibeVoice Realtime-0.5B](https://soniqo.audio/guides/vibevoice) | Metin → Konuşma (uzun biçimli, çok konuşmacılı) | MLX | 0.5B | EN/ZH |
| [VibeVoice 1.5B](https://soniqo.audio/guides/vibevoice) | Metin → Konuşma (90 dakikaya kadar podcast) | MLX | 1.5B | EN/ZH |
| [Magpie-TTS Multilingual](https://soniqo.audio/guides/magpie) | Metin → Konuşma (5 hazır konuşmacı, akış) | MLX / CoreML | 357M (MLX INT8, CoreML INT8) | 9 (CoreML, JA hariç) |
| [Chatterbox Multilingual](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16) | Metin → Konuşma (zero-shot klonlama) | MLX | 0.8B (fp16) | 23 (HE için niqqud gerekir) |
| [OmniVoice](https://huggingface.co/aufklarer/OmniVoice-MLX-fp16) | Metin → Konuşma (NAR difüzyon, zero-shot klonlama) | MLX | 0.8B (fp16 varsayılan / int8) | **600+** |
| [Indic-Mio](docs/models/indic-mio-tts.md) | Text → Speech (Hindi/Indic, emotion tags, voice cloning) | MLX | fp16 | Hindi / Indic |
| [Fish Audio S2 Pro](docs/models/fish-audio-s2-pro.md) | Metin → Konuşma (zero-shot klonlama, açık stil işaretleri) | MLX | 0.5B-class (fp16) | Çok dilli |
| [Qwen3.5 Chat](docs/models/qwen35-chat.md) | Text → Text (LLM) | MLX, CoreML | 0.8B | Multi |
| [Qwen3 Dense Chat](docs/models/qwen3-dense-chat.md) | Text → Text (LLM) | MLX | 4B | Multi |
| [Gemma 4 Chat](docs/models/gemma4-chat.md) | Text → Text (LLM) | MLX | E2B / E4B (4-bit) | Multi |
| [FunctionGemma](docs/models/function-gemma.md) | Metin → Araç çağrıları (LLM) | CoreML | 270M | EN |
| [MADLAD-400](https://soniqo.audio/guides/translate) | Metin → Metin (Çeviri) | MLX | 3B | **400+** |
| [Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate) | Konuşma → Konuşma (Çeviri) | MLX | 3B | FR/ES/PT/DE → EN |
| [PersonaPlex](https://soniqo.audio/guides/respond) | Konuşma → Konuşma | MLX | 7B | EN |
| [Audio2Face-3D](docs/models/audio2face3d.md) | Konuşma → Yüz animasyonu | MLX | v2.3 Mark | Bağımsız |
| [Silero VAD](https://soniqo.audio/guides/vad) | Ses Etkinlik Algılama | MLX, CoreML | 309K | Bağımsız |
| [KWS Zipformer](docs/models/kws-zipformer.md) | Audio → Wake word | CoreML (ANE) | 3M | EN/custom keywords |
| [Pyannote](https://soniqo.audio/guides/diarize) | VAD + Konuşmacı Ayrımı | MLX | 1.5M | Bağımsız |
| [Pyannote Community-1](https://huggingface.co/aufklarer/Pyannote-Community-1-CoreML) | Konuşmacı ayrıştırma + konuşmacı gömmeleri | CoreML (ANE) + Swift VBx | 8.35M | Bağımsız |
| [Sortformer](https://huggingface.co/aufklarer/Sortformer-Diarization-CoreML) | [Konuşmacı Ayrımı (E2E), artımlı akış](https://soniqo.audio/tr/guides/diarize) | CoreML (ANE) | 117M | Bağımsız |
| [DeepFilterNet3](https://soniqo.audio/guides/denoise) | Konuşma İyileştirme | CoreML | 2.1M | Bağımsız |
| [Sidon](https://soniqo.audio/guides/restore) | Konuşma Onarımı (gürültü bastırma + yankı giderme, 48 kHz) | CoreML | w2v-BERT 2.0 + DAC (fp16/int8) | Bağımsız |
| [HTDemucs (Demucs v4)](https://soniqo.audio/guides/separate) | Kaynak Ayrıştırma | MLX | 168M | Bağımsız |
| [Open-Unmix](https://soniqo.audio/guides/separate) | Kaynak Ayrıştırma | MLX | 8.6M | Bağımsız |
| [MAGNeT](https://soniqo.audio/guides/compose) | Metin → Müzik (30s @ 32 kHz) | MLX | 300M / 1.5B (int4/int8) | EN prompt'ları |
| [Stable Audio 3](docs/models/stable-audio-3.md) | Text → Music/audio (44.1 kHz stereo) | MLX | Medium 1.4B (int4/int8) | EN prompts |
| [FlashSR](https://soniqo.audio/guides/upsample) | Ses süper çözünürlük (48 kHz) | MLX | 363 MB / 720 MB (int4/int8) | Bağımsız |
| [WeSpeaker](https://soniqo.audio/guides/embed-speaker) | Konuşmacı Gömme | MLX, CoreML | 6.6M | Bağımsız |
| [ReDimNet2-B6](https://huggingface.co/aufklarer/ReDimNet2-B6-CoreML) | Adlandırılmış Ses Kimliği | CoreML | 12.3M | Bağımsız |

## Kurulum

### Homebrew

[![Homebrew installs](https://img.shields.io/homebrew/installs/dm/speech.svg?logo=homebrew&label=Homebrew%20installs&color=FBB040)](https://formulae.brew.sh/formula/speech)

Yerel ARM Homebrew (`/opt/homebrew`) gerektirir. Rosetta/x86_64 Homebrew desteklenmez.

```bash
brew install speech
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
import WhisperASR           // Whisper Large-v3 Turbo (CoreML)
import ParakeetASR          // Konuşma tanıma (CoreML, batch)
import ParakeetStreamingASR // Kısmi sonuçlar + EOU ile akış dikte
import NemotronStreamingASR // Yerel noktalama ile çok dilli akış ASR (0.6B, 40 dil)
import OmnilingualASR       // 1.672 dil (CoreML + MLX)
import Qwen3TTS             // Metinden konuşmaya
import CosyVoiceTTS         // Ses klonlama ile metinden konuşmaya
import VoxCPM2TTS           // Ses klonlama + ses tasarımı ile 48 kHz TTS (2B)
import IndexTTS2TTS         // Native MLX voice cloning from reference audio
import F5TTS                // Zero-shot voice cloning (DiT flow matching + Vocos)
import HiggsTTS             // Conversational TTS + cloning (Qwen3 backbone, control tags)
import KokoroTTS            // Metinden konuşmaya (iOS için hazır)
import VibeVoiceTTS         // Uzun biçimli / çok konuşmacılı TTS (EN/ZH)
import MagpieTTS            // Çok dilli TTS (NVIDIA Magpie 357M, MLX, 9 dil)
import MagpieTTSCoreML      // Magpie CoreML backend'i (hibrit CoreML + MLX, 8 dil)
import FishAudioTTS         // Ses klonlama özellikli deneysel Fish Audio S2 Pro runtime
import Qwen3Chat            // Cihaz üzerinde LLM sohbet
import FunctionGemma    // Cihaz üzerinde araç çağrı LLM'i
import MADLADTranslation    // 400+ dil arasında çoktan-çoğa çeviri
import HibikiTranslate      // Akışlı konuşmadan konuşmaya çeviri (FR/ES/PT/DE → EN)
import PersonaPlex          // Tam çift yönlü konuşmadan konuşmaya
import SpeechVAD            // VAD + konuşmacı ayrımı + gömmeler
import SpeechEnhancement    // Gürültü bastırma
import SpeechRestoration    // Konuşma onarımı — gürültü bastırma + yankı giderme (Sidon, CoreML, 48 kHz)
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

Alternatif backend'ler: [WhisperASR](docs/inference/whisper-asr-inference.md) (Whisper Large-v3 Turbo, native CoreML), [Parakeet TDT](https://soniqo.audio/guides/parakeet) (CoreML, 32× gerçek zaman), [Omnilingual ASR](https://soniqo.audio/guides/omnilingual) (1.672 dil, CoreML veya MLX), [Akış dikte](https://soniqo.audio/guides/dictate) (canlı kısmi sonuçlar).

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

Alternatif TTS motorları: [CosyVoice3](https://soniqo.audio/guides/cosyvoice) (akış + ses klonlama + duygu etiketleri), [Kokoro-82M](https://soniqo.audio/guides/kokoro) (iOS için hazır, 54 ses), [VibeVoice](https://soniqo.audio/guides/vibevoice) (uzun biçimli podcast / çok konuşmacılı, EN/ZH), [Fish Audio S2 Pro](docs/inference/fish-audio-s2-pro.md) (deneysel zero-shot klonlama + köşeli parantez stil işaretleri), [Ses klonlama](https://soniqo.audio/guides/voice-cloning).

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
import FunctionGemma

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

### Konuşma Çevirisi — [tam rehber →](https://soniqo.audio/guides/audio-translate)

```swift
import HibikiTranslate
import AudioCommon

let model = try await HibikiTranslateModel.fromPretrained()
let pcm = try AudioFileLoader.load(url: input, targetSampleRate: 24000)
let (englishAudio, textTokens) = model.translate(
    sourceAudio: pcm, sourceLanguage: .fr
)
// Hibiki Zero-3B — FR/ES/PT/DE → EN, cihaz üzerinde, akışlı Mimi codec
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

### Konuşma Onarımı — [tam rehber →](https://soniqo.audio/guides/restore)

[Sidon](https://arxiv.org/abs/2509.17052) ile birlikte gürültü bastırma **ve** yankı giderme (w2v-BERT 2.0 tahmincisi + DAC vokoderi, Core ML). Genel bir gürültü bastırıcının aksine, Sidon konuşmacı kimliğini koruyacak şekilde eğitilmiştir; bu nedenle TTS öncesinde gürültülü veya yankılı bir ses klonlama referansını temizlemek için çok uygundur. Giriş 16 kHz; çıkış 48 kHz mono'dur.

```swift
import SpeechRestoration

let restorer = try await SpeechRestorer.fromPretrained()          // .fp16 (default) or .int8
let clean = try restorer.restore(audio: noisySamples, sampleRate: 16000)  // → 48 kHz
```

CLI'dan:

```bash
speech restore noisy.wav -o clean.wav            # denoise + dereverb, 48 kHz output
speech restore noisy.wav --variant int8          # smaller, lower peak RAM

# Clean a voice-cloning reference before TTS (opt-in; preserves speaker identity):
speech speak "Hello" --engine voxcpm2 --voice-sample ref.wav --clean-reference
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
- **Modeller:** [Qwen3-ASR](docs/models/asr-model.md) · [WhisperASR](docs/models/whisper-asr.md) · [Qwen3-TTS](docs/models/tts-model.md) · [CosyVoice](docs/models/cosyvoice-tts.md) · [Kokoro](docs/models/kokoro-tts.md) · [VoxCPM2](docs/models/voxcpm2-tts.md) · [IndexTTS2](docs/models/indextts2.md) · [F5-TTS](docs/models/f5-tts.md) · [Higgs TTS 3](docs/models/higgs-tts.md) · [VibeVoice](docs/models/vibevoice.md) · [Supertonic](docs/models/supertonic-tts.md) · [Chatterbox](docs/models/chatterbox-tts.md) · [Indic-Mio](docs/models/indic-mio-tts.md) · [Fish Audio S2 Pro](docs/models/fish-audio-s2-pro.md) · [Magpie TTS](docs/models/magpie-tts.md) · [Parakeet TDT](docs/models/parakeet-asr.md) · [Parakeet Streaming](docs/models/parakeet-streaming-asr.md) · [Nemotron Streaming](docs/models/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/models/omnilingual-asr.md) · [PersonaPlex](docs/models/personaplex.md) · [Hibiki](docs/models/hibiki.md) · [MADLAD-400](docs/models/madlad-translation.md) · [FunctionGemma](docs/models/function-gemma.md) · [Qwen3.5 Chat](docs/models/qwen35-chat.md) · [Gemma 4 Chat](docs/models/gemma4-chat.md) · [Qwen3 Dense Chat](docs/models/qwen3-dense-chat.md) · [FireRedVAD](docs/models/fireredvad.md) · [KWS Zipformer](docs/models/kws-zipformer.md) · [Sidon](docs/models/sidon.md) · [Source Separation](docs/models/source-separation.md) · [HTDemucs](docs/models/htdemucs.md) · [MAGNeT](docs/models/magnet-music-gen.md) · [Stable Audio 3](docs/models/stable-audio-3.md) · [FlashSR](docs/models/flashsr.md) · [Audio2Face-3D](docs/models/audio2face3d.md)
- **Çıkarım:** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md) · [WhisperASR](docs/inference/whisper-asr-inference.md) · [Parakeet TDT](docs/inference/parakeet-asr-inference.md) · [Parakeet Streaming](docs/inference/parakeet-streaming-asr-inference.md) · [Nemotron Streaming](docs/inference/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/inference/omnilingual-asr-inference.md) · [TTS](docs/inference/qwen3-tts-inference.md) · [VoxCPM2](docs/inference/voxcpm2-inference.md) · [IndexTTS2](docs/inference/indextts2.md) · [F5-TTS](docs/inference/f5-tts.md) · [Higgs TTS 3](docs/inference/higgs-tts.md) · [VibeVoice](docs/inference/vibevoice-inference.md) · [Fish Audio S2 Pro](docs/inference/fish-audio-s2-pro.md) · [Magpie TTS](docs/inference/magpie-tts.md) · [Hibiki](docs/inference/hibiki-inference.md) · [MADLAD-400](docs/inference/madlad-translation.md) · [MAGNeT](docs/inference/magnet-music-gen.md) · [Stable Audio 3](docs/inference/stable-audio-3.md) · [FlashSR](docs/inference/flashsr.md) · [Forced Aligner](docs/inference/forced-aligner.md) · [Silero VAD](docs/inference/silero-vad.md) · [FireRedVAD](docs/inference/fireredvad.md) · [Wake-word](docs/inference/wake-word.md) · [Speaker Diarization](docs/inference/speaker-diarization.md) · [Speech Enhancement](docs/inference/speech-enhancement.md) · [Sidon](docs/inference/sidon.md) · [Cache/offline](docs/inference/cache-and-offline.md)
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
