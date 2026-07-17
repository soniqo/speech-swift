# Speech Swift

<div dir="rtl">

نماذج الذكاء الاصطناعي الصوتية لمعالجات Apple Silicon، مدعومة بـ MLX Swift و CoreML.

</div>

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

<div dir="rtl">

التعرف على الكلام وتوليده وفهمه على الجهاز لأنظمة Mac و iOS. يعمل محلياً على Apple Silicon — بدون سحابة، بدون مفاتيح API، ولا تغادر بياناتك جهازك.

**[📚 الوثائق الكاملة →](https://soniqo.audio/ar)** · **[🤗 نماذج HuggingFace](https://huggingface.co/aufklarer)** · **[📝 المدونة](https://blog.ivan.digital)** · **[💬 Discord](https://discord.gg/TnCryqEMgu)**

</div>

<p align="center">
  <a href="https://trendshift.io/repositories/24196?utm_source=trendshift-badge&amp;utm_medium=badge&amp;utm_campaign=badge-trendshift-24196" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/trendshift/repositories/24196/daily?language=Swift" alt="soniqo%2Fspeech-swift | Trendshift" width="250" height="55"/></a>
</p>

<p align="center">
  <a href="https://youtu.be/x9zgcaW0gUk">
    <img src="https://img.youtube.com/vi/x9zgcaW0gUk/maxresdefault.jpg" width="640" alt="الذكاء الاصطناعي الصوتي المحلي على MacBook — شاهد جولة الأربع دقائق في المكتبة مفتوحة المصدر على YouTube">
  </a>
</p>
<p align="center"><em>الذكاء الاصطناعي الصوتي المحلي على MacBook — شاهد جولة الأربع دقائق في المكتبة مفتوحة المصدر على YouTube</em></p>

<div dir="rtl">

**حالات الاستخدام:** [وكلاء الصوت](https://soniqo.audio/ar/voice-agents) · [النسخ النصي](https://soniqo.audio/ar/transcription) · [توليد الكلام](https://soniqo.audio/ar/speech-generation)

## مشاريع مبنية باستخدام Speech Swift

مشاريع عامة تتضمن تكاملات يمكن التحقق منها مع Speech Swift.

[Palmier Pro](https://github.com/palmier-io/palmier-pro) · [Anarlog](https://github.com/fastrepl/anarlog) · [VoxFlow](https://github.com/xingbofeng/VoxFlow) · [Ora](https://github.com/wuwangzhang1216/ora) · [Jabber](https://github.com/rselbach/jabber) · [ClawdHome](https://github.com/ThinkInAIXYZ/clawdhome)

**مجموعات القدرات:** STT / ASR · المحاذاة · TTS · نماذج LLM والترجمة · الكلام إلى كلام · التحسين / الاستعادة · فصل المصادر · توليد الموسيقى / الصوت · كلمة التنبيه، VAD، diarization وهوية المتحدث

**STT / ASR**

- **[Qwen3-ASR](https://soniqo.audio/ar/guides/transcribe)** — تحويل الكلام إلى نص (تعرف تلقائي على الكلام، 52 لغة، MLX + CoreML)
- **[WhisperASR](docs/models/whisper-asr.md)** — Whisper Large-v3 Turbo speech-to-text via native CoreML runtime (ANE, multilingual)
- **[Parakeet TDT](https://soniqo.audio/ar/guides/parakeet)** — تحويل الكلام إلى نص عبر CoreML (Neural Engine، NVIDIA FastConformer + مفكك ترميز TDT، 25 لغة)
- **[Omnilingual ASR](https://soniqo.audio/ar/guides/omnilingual)** — تحويل الكلام إلى نص (Meta wav2vec2 + CTC، **1,672 لغة** عبر 32 نظام كتابة، CoreML 300M + MLX 300M/1B/3B/7B)
- **[الإملاء التدفقي](https://soniqo.audio/ar/guides/dictate)** — إملاء فوري بنتائج جزئية واكتشاف نهاية النطق (Parakeet-EOU-120M)
- **[Nemotron Streaming (متعدد اللغات)](https://soniqo.audio/ar/guides/nemotron)** — تعرف تدفقي على الكلام بزمن استجابة منخفض مع علامات ترقيم وأحرف كبيرة أصلية (NVIDIA Nemotron-3.5-ASR-Streaming-0.6B، CoreML + MLX، **40 لغة-منطقة**)
- **[Nemotron Streaming (إنجليزي)](https://soniqo.audio/guides/nemotron)** — تعرف تدفقي على الكلام بزمن استجابة منخفض مع علامات ترقيم وأحرف كبيرة أصلية (NVIDIA Nemotron-Speech-Streaming-0.6B، CoreML، الإنجليزية فقط، أصغر وأسرع من المتغير متعدد اللغات)

**المحاذاة**

- **[Qwen3-ForcedAligner](https://soniqo.audio/ar/guides/align)** — محاذاة الطوابع الزمنية على مستوى الكلمة (صوت + نص → طوابع زمنية)

**TTS / توليد الكلام**

- **[Qwen3-TTS](https://soniqo.audio/ar/guides/speak)** — تحويل النص إلى كلام (أعلى جودة، تدفق، متحدثون مخصصون، 10 لغات)
- **[CosyVoice TTS](https://soniqo.audio/ar/guides/cosyvoice)** — تحويل تدفقي للنص إلى كلام مع استنساخ الصوت وحوار متعدد المتحدثين ووسوم المشاعر (9 لغات)
- **[VoxCPM2](https://soniqo.audio/ar/speech-generation)** — تحويل النص إلى كلام بجودة استوديو 48 كيلوهرتز مع استنساخ الصوت وتصميم الصوت بالأوامر (2B، MLX bf16/int8، 30 لغة)
- **[IndexTTS2](docs/models/indextts2.md)** — Native MLX voice cloning from a reference voice (IndexTeam IndexTTS-2, 1.5B-class fp16 bundle, speaker/emotion/pause controls)
- **[F5-TTS](docs/models/f5-tts.md)** — Zero-shot voice cloning from a short reference clip + transcript (SWivid F5-TTS v1 Base, DiT flow matching + Vocos, MLX fp16, 24 kHz, English + Mandarin; non-commercial license)
- **[Higgs TTS 3](docs/models/higgs-tts.md)** — Conversational TTS with zero-shot voice cloning and inline emotion/style/SFX/prosody tags (Boson Higgs TTS 3, Qwen3-4B backbone, MLX bf16, 24 kHz, 100+ languages; research/non-commercial license)
- **[Kokoro TTS](https://soniqo.audio/ar/guides/kokoro)** — تحويل النص إلى كلام على الجهاز (82M، CoreML/Neural Engine، 54 صوتاً، جاهز لـ iOS، 10 لغات)
- **[VibeVoice TTS](https://soniqo.audio/ar/guides/vibevoice)** — تحويل النص إلى كلام للنصوص الطويلة / متعدد المتحدثين (Microsoft VibeVoice Realtime-0.5B + 1.5B، MLX، توليد بودكاست/كتب صوتية حتى 90 دقيقة، EN/ZH)
- **[Magpie TTS](https://soniqo.audio/ar/guides/magpie)** — تحويل النص إلى كلام متعدد اللغات (NVIDIA Magpie-TTS Multilingual 357M، MLX INT8 411 ميغابايت أو CoreML INT8 342 ميغابايت، 9 لغات، 5 متحدثين جاهزين، تدفق على MLX)
- **[Supertonic TTS](https://soniqo.audio/guides/supertonic)** — تحويل النص إلى كلام على الجهاز بمطابقة التدفق (Supertone Supertonic-3 99M، CoreML/Neural Engine، 31 لغة، 10 أصوات، بدون G2P، 44.1 كيلوهرتز)
- **[Chatterbox TTS](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16)** — تحويل النص إلى كلام متعدد اللغات مع استنساخ الصوت بدون تدريب مسبق (Resemble AI Chatterbox Multilingual، MLX fp16 ~1.3 غيغابايت، 23 لغة وقت تشغيل؛ العبرية تتطلب النِّقُود، MIT)
- **[OmniVoice TTS](https://huggingface.co/aufklarer/OmniVoice-MLX-fp16)** — تحويل النص إلى كلام بالانتشار غير الانحداري التلقائي مع استنساخ الصوت بدون تدريب مسبق (k2-fsa OmniVoice، عمود Qwen3 الفقري، MLX fp16 افتراضي / int8 متاح، أكثر من 600 لغة، Apache-2.0)
- **[Indic-Mio](docs/models/indic-mio-tts.md)** — Hindi/Indic TTS with inline emotion markers and optional reference-voice cloning (MLX, 24 kHz)

**نماذج LLM والترجمة**

- **[Qwen3Chat](https://soniqo.audio/ar/guides/chat)** — محادثة LLM على الجهاز (Qwen3.5-0.8B عبر MLX/CoreML، إضافة إلى خلفيات MLX لـ Qwen3 dense بحجم 4B و Gemma 4 E2B/E4B، رموز تدفقية)
- **[FunctionGemma](https://soniqo.audio/ar/guides/function-calls)** — نموذج لغوي على الجهاز للاستدعاءات المنظمة للدوال / الأدوات (Gemma 3 270M، CoreML بترميز 8-بت، Neural Engine، حوالي 252 tok/s)
- **[MADLAD-400](https://soniqo.audio/ar/guides/translate)** — ترجمة متعددة الاتجاهات عبر أكثر من 400 لغة (3B، MLX INT4 + INT8، T5 v1.1، Apache 2.0)

**الكلام إلى كلام ووكلاء الصوت**

- **[Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate)** — ترجمة تدفقية من كلام إلى كلام (FR/ES/PT/DE → EN، MLX INT4 + INT8، حزمة Kyutai Moshi/Mimi، CC-BY-4.0)
- **[PersonaPlex](https://soniqo.audio/ar/guides/respond)** — تحويل صوت إلى صوت ثنائي الاتجاه الكامل (7B، صوت داخل → صوت خارج، 18 إعداداً صوتياً مسبقاً)
- **[Audio2Face-3D](docs/models/audio2face3d.md)** — تحريك وجه الأفاتار انطلاقًا من الكلام (NVIDIA Audio2Face-3D v2.3 Mark, 301 معامل وجه, MLX)

**التحسين والفصل وتوليد الصوت**

- **[DeepFilterNet3](https://soniqo.audio/ar/guides/denoise)** — قمع الضوضاء في الزمن الحقيقي (2.1M معامل، 48 كيلوهرتز) الصوت الطويل الذي يتجاوز حد اللقطة الواحدة البالغ 60 s يُقسَّم تلقائيًا إلى أجزاء مع crossfade — راجع `enhanceChunked(...)`
- **[فصل المصادر](https://soniqo.audio/ar/guides/separate)** — فصل المصادر الموسيقية عبر HTDemucs (Demucs v4) + Open-Unmix (UMX-HQ / UMX-L، 4 طبقات: غناء/طبول/باس/أخرى، 44.1 كيلوهرتز ستيريو)
- **[MAGNeT](https://soniqo.audio/ar/guides/compose)** — توليد الموسيقى من النص (Meta MAGNeT Small 300M / Medium 1.5B، MLX INT8، مقاطع 30 ثانية بجودة 32 كيلوهرتز مونو، فك ترميز متوازي مقنع)
- **[Stable Audio 3](docs/models/stable-audio-3.md)** — Text-to-audio/music generation (Stable Audio 3 Medium, MLX INT8/INT4, 44.1 kHz stereo, variable length)
- **[FlashSR](https://soniqo.audio/ar/guides/upsample)** — رفع دقة الصوت (FlashSR ICASSP 2025، MLX، 48 كيلوهرتز مونو، انتشار مقطر بخطوة واحدة، INT4 363 ميغابايت / INT8 720 ميغابايت)

**اكتشاف الدور، diarization وهوية المتحدث**

- **[كلمة التنبيه](https://soniqo.audio/ar/guides/wake-word)** — اكتشاف الكلمات المفتاحية على الجهاز (KWS Zipformer 3M، CoreML، 26× الزمن الحقيقي، قائمة كلمات مفتاحية قابلة للتهيئة)
- **[VAD](https://soniqo.audio/ar/guides/vad)** — اكتشاف النشاط الصوتي (Silero تدفقي، Pyannote دون اتصال، FireRedVAD أكثر من 100 لغة)
- **[تمييز المتحدثين](https://soniqo.audio/ar/guides/diarize)** — من تحدث متى (خط أنابيب Pyannote، Sortformer من طرف إلى طرف على Neural Engine) — الآن مع جلسة بث تزايدي (معرّفات متحدثين ثابتة وتحديث كل 480 مللي ثانية)
- **[تضمينات المتحدث](https://soniqo.audio/ar/guides/embed-speaker)** — WeSpeaker ResNet34 (256 بُعداً)، ReDimNet2-B6 للهوية الصوتية المسماة (192 بُعداً)، CAM++ (192 بُعداً)

</div>

الأوراق البحثية: [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba) · [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba) · [Omnilingual ASR](https://arxiv.org/abs/2511.09690) (Meta) · [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA) · [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba) · [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2) · [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA) · [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai) · [Hibiki](https://arxiv.org/abs/2502.03382) (Kyutai) · [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## الأخبار

<div dir="rtl">

- **19 أبريل 2026** — [MLX مقابل CoreML على Apple Silicon — دليل عملي لاختيار الواجهة الخلفية المناسبة](https://blog.ivan.digital/mlx-vs-coreml-on-apple-silicon-a-practical-guide-to-picking-the-right-backend-and-why-you-should-f77ddea7b27a)
- **20 مارس 2026** — [تفوقنا على Whisper Large v3 بنموذج 600M يعمل بالكامل على جهاز Mac الخاص بك](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **26 فبراير 2026** — [تمييز المتحدثين واكتشاف النشاط الصوتي على Apple Silicon — Swift أصلي مع MLX](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **23 فبراير 2026** — [NVIDIA PersonaPlex 7B على Apple Silicon — تحويل صوت إلى صوت ثنائي الاتجاه الكامل في Swift الأصلي مع MLX](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **12 فبراير 2026** — [Qwen3-ASR Swift: ASR + TTS على الجهاز لـ Apple Silicon — الهيكلة والاختبارات](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

</div>

## البدء السريع

<div dir="rtl">

أضف الحزمة إلى ملف `Package.swift` الخاص بك:

</div>

```swift
.package(url: "https://github.com/soniqo/speech-swift", branch: "main")
```

<div dir="rtl">

استورد فقط الوحدات التي تحتاجها — كل نموذج مكتبة SPM مستقلة، لذا لا تدفع مقابل ما لا تستخدمه:

</div>

```swift
.product(name: "ParakeetStreamingASR", package: "speech-swift"),
.product(name: "SpeechUI",             package: "speech-swift"),  // عروض SwiftUI اختيارية
```

<div dir="rtl">

**نسخ مخزن صوتي في 3 أسطر:**

</div>

```swift
import ParakeetStreamingASR

let model = try await ParakeetStreamingASRModel.fromPretrained()
let text = try model.transcribeAudio(audioSamples, sampleRate: 16000)
```

<div dir="rtl">

**تدفق مباشر مع نتائج جزئية:**

</div>

```swift
for await partial in model.transcribeStream(audio: samples, sampleRate: 16000) {
    print(partial.isFinal ? "FINAL: \(partial.text)" : "... \(partial.text)")
}
```

<div dir="rtl">

**عرض إملاء SwiftUI في ~10 أسطر:**

</div>

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

<div dir="rtl">

تشحن `SpeechUI` فقط `TranscriptionView` (النهائيات + الجزئيات) و `TranscriptionStore` (محول ASR تدفقي). استخدم AVFoundation لتصور الصوت وتشغيله.

منتجات SPM المتاحة: `Qwen3ASR`, `WhisperASR`, `Qwen3TTS`, `Qwen3TTSCoreML`, `ParakeetASR`, `ParakeetStreamingASR`, `NemotronStreamingASR`, `OmnilingualASR`, `KokoroTTS`, `SupertonicTTS`, `VibeVoiceTTS`, `CosyVoiceTTS`, `VoxCPM2TTS`, `IndexTTS2TTS`, `F5TTS`, `HiggsTTS`, `ChatterboxTTS`, `OmniVoiceTTS`, `IndicMioTTS`, `FishAudioTTS`, `MagpieTTS`, `MagpieTTSCoreML`, `MAGNeTMusicGen`, `StableAudio3MusicGen`, `FlashSR`, `PersonaPlex`, `Audio2Face3D`, `HibikiTranslate`, `MADLADTranslation`, `SpeechVAD`, `SpeechWakeWord`, `SpeechEnhancement`, `SpeechRestoration`, `SourceSeparation`, `Qwen3Chat`, `FunctionGemma`, `SpeechCore`, `SpeechUI`, `AudioCommon`.

</div>

## النماذج

<div dir="rtl">

عرض مختصر أدناه. **[الكتالوج الكامل للنماذج مع الأحجام والتكميمات وعناوين التنزيل وجداول الذاكرة → soniqo.audio/architecture](https://soniqo.audio/ar/architecture)**.

</div>

| النموذج | المهمة | الواجهات الخلفية | الأحجام | اللغات |
|-------|------|----------|-------|-----------|
| [Qwen3-ASR](https://soniqo.audio/ar/guides/transcribe) | كلام → نص | MLX, CoreML (هجين) | 0.6B, 1.7B | 52 |
| [WhisperASR](docs/models/whisper-asr.md) | Speech → Text | CoreML (ANE) | Large-v3 Turbo | Multi |
| [Parakeet TDT](https://soniqo.audio/ar/guides/parakeet) | كلام → نص | CoreML (ANE) | 0.6B | 25 أوروبية |
| [Parakeet EOU](https://soniqo.audio/ar/guides/dictate) | كلام → نص (تدفقي) | CoreML (ANE) | 120M | 25 أوروبية |
| [Nemotron Streaming (متعدد اللغات)](https://soniqo.audio/ar/guides/nemotron) | كلام → نص (تدفقي، مع علامات ترقيم) | CoreML (ANE), MLX | 0.6B | **40** |
| [Nemotron Streaming (إنجليزي)](https://soniqo.audio/guides/nemotron) | كلام → نص (تدفقي، مع علامات ترقيم) | CoreML (ANE) | 0.6B | EN |
| [Omnilingual ASR](https://soniqo.audio/ar/guides/omnilingual) | كلام → نص | CoreML (ANE), MLX | 300M / 1B / 3B / 7B | **[1,672](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)** |
| [Qwen3-ForcedAligner](https://soniqo.audio/ar/guides/align) | صوت + نص → طوابع زمنية | MLX, CoreML | 0.6B | متعدد |
| [Qwen3-TTS](https://soniqo.audio/ar/guides/speak) | نص → كلام | MLX, CoreML | 0.6B, 1.7B | 10 |
| [CosyVoice3](https://soniqo.audio/ar/guides/cosyvoice) | نص → كلام | MLX | 0.5B | 9 |
| [VoxCPM2](https://soniqo.audio/ar/speech-generation) | نص → كلام (48 كيلوهرتز، تصميم + استنساخ صوت) | MLX | 2B (bf16/int8) | 30 |
| [IndexTTS2](docs/models/indextts2.md) | Text → Speech (zero-shot voice cloning) | MLX | 1.5B-class (fp16) | EN/ZH |
| [F5-TTS](docs/models/f5-tts.md) | Text → Speech (zero-shot voice cloning) | MLX | 336M (fp16) | EN/ZH |
| [Higgs TTS 3](docs/models/higgs-tts.md) | Text → Speech (conversational, zero-shot voice cloning) | MLX | 4B (bf16) | 100+ |
| [Chatterbox Multilingual](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16) | نص → كلام (استنساخ صوت بدون تدريب) | MLX | 0.8B (fp16) | 23 (HE تتطلب النِّقُود) |
| [OmniVoice](https://huggingface.co/aufklarer/OmniVoice-MLX-fp16) | نص → كلام (انتشار NAR، استنساخ بدون تدريب) | MLX | 0.8B (fp16 افتراضي / int8) | 600+ |
| [Indic-Mio](docs/models/indic-mio-tts.md) | Text → Speech (Hindi/Indic, emotion tags, voice cloning) | MLX | fp16 | Hindi / Indic |
| [Fish Audio S2 Pro](docs/models/fish-audio-s2-pro.md) | نص → كلام (استنساخ صوت بدون تدريب، علامات نمط صريحة) | MLX | 0.5B-class (fp16) | متعدد اللغات |
| [Kokoro-82M](https://soniqo.audio/ar/guides/kokoro) | نص → كلام | CoreML (ANE) | 82M | 10 |
| [Supertonic-3](https://soniqo.audio/guides/supertonic) | نص → كلام (44.1 كيلوهرتز، مطابقة التدفق، بدون G2P) | CoreML (ANE) | 99M | 31 |
| [VibeVoice Realtime-0.5B](https://soniqo.audio/ar/guides/vibevoice) | نص → كلام (نص طويل، متعدد المتحدثين) | MLX | 0.5B | EN/ZH |
| [VibeVoice 1.5B](https://soniqo.audio/ar/guides/vibevoice) | نص → كلام (بودكاست حتى 90 دقيقة) | MLX | 1.5B | EN/ZH |
| [Magpie-TTS Multilingual](https://soniqo.audio/ar/guides/magpie) | نص → كلام (5 متحدثين جاهزين، تدفق) | MLX / CoreML | 357M (MLX INT8, CoreML INT8) | 9 (CoreML يستثني JA) |
| [Qwen3.5 Chat](docs/models/qwen35-chat.md) | Text → Text (LLM) | MLX, CoreML | 0.8B | Multi |
| [Qwen3 Dense Chat](docs/models/qwen3-dense-chat.md) | Text → Text (LLM) | MLX | 4B | Multi |
| [Gemma 4 Chat](docs/models/gemma4-chat.md) | Text → Text (LLM) | MLX | E2B / E4B (4-bit) | Multi |
| [FunctionGemma](https://soniqo.audio/ar/guides/function-calls) | نص → استدعاءات الأدوات (LLM) | CoreML | 270M | EN |
| [MADLAD-400](https://soniqo.audio/ar/guides/translate) | نص → نص (ترجمة) | MLX | 3B | **أكثر من 400** |
| [Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate) | كلام → كلام (ترجمة) | MLX | 3B | FR/ES/PT/DE → EN |
| [PersonaPlex](https://soniqo.audio/ar/guides/respond) | كلام → كلام | MLX | 7B | EN |
| [Audio2Face-3D](docs/models/audio2face3d.md) | كلام → تحريك الوجه | MLX | v2.3 Mark | محايد للغة |
| [Silero VAD](https://soniqo.audio/ar/guides/vad) | اكتشاف النشاط الصوتي | MLX, CoreML | 309K | محايد للغة |
| [KWS Zipformer](docs/models/kws-zipformer.md) | Audio → Wake word | CoreML (ANE) | 3M | EN/custom keywords |
| [Pyannote](https://soniqo.audio/ar/guides/diarize) | VAD + تمييز | MLX | 1.5M | محايد للغة |
| [Pyannote Community-1](https://huggingface.co/aufklarer/Pyannote-Community-1-CoreML) | فرز المتحدثين + تضمينات المتحدث | CoreML (ANE) + Swift VBx | 8.35M | محايد للغة |
| [Sortformer](https://huggingface.co/aufklarer/Sortformer-Diarization-CoreML) | [تمييز (E2E), بث تزايدي](https://soniqo.audio/ar/guides/diarize) | CoreML (ANE) | 117M | محايد للغة |
| [DeepFilterNet3](https://soniqo.audio/ar/guides/denoise) | تحسين الكلام | CoreML | 2.1M | محايد للغة |
| [Sidon](https://soniqo.audio/ar/guides/restore) | استعادة الكلام (إزالة الضوضاء + إزالة الصدى، 48 kHz) | CoreML | w2v-BERT 2.0 + DAC (fp16/int8) | محايد للغة |
| [HTDemucs (Demucs v4)](https://soniqo.audio/ar/guides/separate) | فصل المصادر | MLX | 168M | محايد للغة |
| [Open-Unmix](https://soniqo.audio/ar/guides/separate) | فصل المصادر | MLX | 8.6M | محايد للغة |
| [MAGNeT](https://soniqo.audio/ar/guides/compose) | نص → موسيقى (30 ث @ 32 كيلوهرتز) | MLX | 300M / 1.5B (int4/int8) | أوامر بالإنجليزية |
| [Stable Audio 3](docs/models/stable-audio-3.md) | Text → Music/audio (44.1 kHz stereo) | MLX | Medium 1.4B (int4/int8) | EN prompts |
| [FlashSR](https://soniqo.audio/ar/guides/upsample) | رفع دقة الصوت (48 كيلوهرتز) | MLX | 363 ميغابايت / 720 ميغابايت (int4/int8) | محايد للغة |
| [WeSpeaker](https://soniqo.audio/ar/guides/embed-speaker) | تضمين المتحدث | MLX, CoreML | 6.6M | محايد للغة |
| [ReDimNet2-B6](https://huggingface.co/aufklarer/ReDimNet2-B6-CoreML) | هوية صوتية مسماة | CoreML | 12.3M | محايد للغة |

## التثبيت

### Homebrew

[![Homebrew installs](https://img.shields.io/homebrew/installs/dm/speech.svg?logo=homebrew&label=Homebrew%20installs&color=FBB040)](https://formulae.brew.sh/formula/speech)

<div dir="rtl">

يتطلب Homebrew ARM الأصلي (`/opt/homebrew`). Homebrew Rosetta/x86_64 غير مدعوم.

</div>

```bash
brew install speech
```

<div dir="rtl">

ثم:

</div>

```bash
speech transcribe recording.wav
speech speak "Hello world"
speech translate "Hello, how are you?" --to es
speech respond --input question.wav --transcript
speech-server --port 8080            # خادم HTTP / WebSocket محلي (متوافق مع OpenAI /v1/realtime + /v1/audio/transcriptions)
```

<div dir="rtl">

**[المرجع الكامل لواجهة سطر الأوامر →](https://soniqo.audio/ar/cli)**

</div>

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

<div dir="rtl">

استورد فقط ما تحتاجه — كل نموذج له هدف SPM خاص به:

</div>

```swift
import Qwen3ASR             // التعرف على الكلام (MLX)
import WhisperASR           // Whisper Large-v3 Turbo (CoreML)
import ParakeetASR          // التعرف على الكلام (CoreML، دفعة)
import ParakeetStreamingASR // إملاء تدفقي مع جزئيات + EOU
import NemotronStreamingASR // ASR تدفقي متعدد اللغات مع علامات ترقيم أصلية (0.6B، 40 لغة)
import OmnilingualASR       // 1,672 لغة (CoreML + MLX)
import Qwen3TTS             // تحويل النص إلى كلام
import CosyVoiceTTS         // تحويل النص إلى كلام مع استنساخ الصوت
import VoxCPM2TTS           // TTS بـ 48 كيلوهرتز، استنساخ + تصميم صوت (2B)
import IndexTTS2TTS         // Native MLX voice cloning from reference audio
import F5TTS                // Zero-shot voice cloning (DiT flow matching + Vocos)
import HiggsTTS             // Conversational TTS + cloning (Qwen3 backbone, control tags)
import KokoroTTS            // تحويل النص إلى كلام (جاهز لـ iOS)
import VibeVoiceTTS         // TTS للنصوص الطويلة / متعدد المتحدثين (EN/ZH)
import MagpieTTS            // TTS متعدد اللغات (NVIDIA Magpie 357M، MLX، 9 لغات)
import MagpieTTSCoreML      // واجهة CoreML الخلفية لـ Magpie (هجين CoreML + MLX، 8 لغات)
import FishAudioTTS         // وقت تشغيل تجريبي لـ Fish Audio S2 Pro مع استنساخ الصوت
import Qwen3Chat            // محادثة LLM على الجهاز
import FunctionGemma    // نموذج لغوي على الجهاز لاستدعاءات الأدوات
import MADLADTranslation    // ترجمة متعددة الاتجاهات عبر أكثر من 400 لغة
import HibikiTranslate      // ترجمة تدفقية من كلام إلى كلام (FR/ES/PT/DE → EN)
import PersonaPlex          // تحويل صوت إلى صوت ثنائي الاتجاه
import SpeechVAD            // VAD + تمييز + تضمينات
import SpeechEnhancement    // قمع الضوضاء
import SpeechRestoration    // استعادة الكلام — إزالة الضوضاء + إزالة الصدى (Sidon، CoreML، 48 kHz)
import SourceSeparation     // فصل المصادر الموسيقية (Open-Unmix، 4 طبقات)
import SpeechUI             // مكونات SwiftUI للنسخ النصي التدفقي
import AudioCommon          // البروتوكولات والمرافق المشتركة
```

### المتطلبات

<div dir="rtl">

- Swift 6+, Xcode 16+ (مع Metal Toolchain)
- macOS 15+ (Sequoia) أو iOS 18+, Apple Silicon (M1/M2/M3/M4)

الحد الأدنى لنظامي macOS 15 / iOS 18 يأتي من [MLState](https://developer.apple.com/documentation/coreml/mlstate) — واجهة Apple لحالة ANE المستديمة التي تستخدمها خطوط أنابيب CoreML (Qwen3-ASR, Qwen3-Chat, Qwen3-TTS) للحفاظ على ذاكرات KV المؤقتة مقيمة على Neural Engine عبر خطوات الرموز.

</div>

### البناء من المصدر

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

<div dir="rtl">

يقوم `make build` بتصنيف حزمة Swift **و** مكتبة شيدر MLX Metal. مكتبة Metal مطلوبة لاستدلال GPU — بدونها سترى `Failed to load the default metallib` عند التشغيل. `make debug` لإصدارات التصحيح، `make test` لمجموعة الاختبارات.

**[دليل البناء والتثبيت الكامل →](https://soniqo.audio/ar/getting-started)**

</div>

## تطبيقات العرض التوضيحي

<div dir="rtl">

- **[DictateDemo](Examples/DictateDemo/)** ([الوثائق](https://soniqo.audio/ar/guides/dictate)) — إملاء تدفقي على شريط قائمة macOS مع جزئيات حية، اكتشاف نهاية النطق المعتمد على VAD، ونسخ بنقرة واحدة. يعمل كوكيل في الخلفية (Parakeet-EOU-120M + Silero VAD).
- **[iOSEchoDemo](Examples/iOSEchoDemo/)** — عرض صدى iOS (Parakeet ASR + Kokoro TTS). للجهاز والمحاكي.
- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — مساعد صوتي محادثاتي مع إدخال ميكروفون، VAD، وسياق متعدد الأدوار. macOS. RTF ~0.94 على M2 Max (أسرع من الزمن الحقيقي).
- **[SpeechDemo](Examples/SpeechDemo/)** — إملاء وتوليد TTS في واجهة بعلامات تبويب. macOS.

يحتوي README كل عرض على تعليمات البناء.

</div>

## أمثلة برمجية

<div dir="rtl">

تعرض المقتطفات أدناه الحد الأدنى من المسار لكل مجال. كل قسم يرتبط بدليل كامل على [soniqo.audio](https://soniqo.audio/ar) مع خيارات التكوين والواجهات الخلفية المتعددة وأنماط التدفق ووصفات CLI.

</div>

### تحويل الكلام إلى نص — [الدليل الكامل →](https://soniqo.audio/ar/guides/transcribe)

```swift
import Qwen3ASR

let model = try await Qwen3ASRModel.fromPretrained()
let text = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

<div dir="rtl">

واجهات خلفية بديلة: [WhisperASR](docs/inference/whisper-asr-inference.md) (Whisper Large-v3 Turbo, native CoreML)، [Parakeet TDT](https://soniqo.audio/ar/guides/parakeet) (CoreML، 32× الزمن الحقيقي)، [Omnilingual ASR](https://soniqo.audio/ar/guides/omnilingual) (1,672 لغة، CoreML أو MLX)، [الإملاء التدفقي](https://soniqo.audio/ar/guides/dictate) (جزئيات حية).

</div>

### المحاذاة القسرية — [الدليل الكامل →](https://soniqo.audio/ar/guides/align)

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

### تحويل النص إلى كلام — [الدليل الكامل →](https://soniqo.audio/ar/guides/speak)

```swift
import Qwen3TTS
import AudioCommon

let model = try await Qwen3TTSModel.fromPretrained()
let audio = model.synthesize(text: "Hello world", language: "english")
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

<div dir="rtl">

محركات TTS بديلة: [CosyVoice3](https://soniqo.audio/ar/guides/cosyvoice) (تدفق + استنساخ + وسوم مشاعر)، [Kokoro-82M](https://soniqo.audio/ar/guides/kokoro) (جاهز لـ iOS، 54 صوتاً)، [VibeVoice](https://soniqo.audio/ar/guides/vibevoice) (بودكاست/متعدد متحدثين، EN/ZH)، [Fish Audio S2 Pro](docs/inference/fish-audio-s2-pro.md) (استنساخ zero-shot تجريبي + علامات نمط بين أقواس)، [استنساخ الصوت](https://soniqo.audio/ar/guides/voice-cloning).

</div>

### تحويل صوت إلى صوت — [الدليل الكامل →](https://soniqo.audio/ar/guides/respond)

```swift
import PersonaPlex

let model = try await PersonaPlexModel.fromPretrained()
let responseAudio = model.respond(userAudio: userSamples)
// خرج Float32 مونو 24 كيلوهرتز جاهز للتشغيل
```

### محادثة LLM — [الدليل الكامل →](https://soniqo.audio/ar/guides/chat)

```swift
import Qwen3Chat
import FunctionGemma

let chat = try await Qwen35MLXChat.fromPretrained()
chat.chat(messages: [(.user, "Explain MLX in one sentence")]) { token, isFinal in
    print(token, terminator: "")
}
```

### الترجمة — [الدليل الكامل →](https://soniqo.audio/ar/guides/translate)

```swift
import MADLADTranslation

let translator = try await MADLADTranslator.fromPretrained()
let es = try translator.translate("Hello, how are you?", to: "es")
// → "Hola, ¿cómo estás?"
```

### ترجمة الكلام — [الدليل الكامل →](https://soniqo.audio/guides/audio-translate)

```swift
import HibikiTranslate
import AudioCommon

let model = try await HibikiTranslateModel.fromPretrained()
let pcm = try AudioFileLoader.load(url: input, targetSampleRate: 24000)
let (englishAudio, textTokens) = model.translate(
    sourceAudio: pcm, sourceLanguage: .fr
)
// Hibiki Zero-3B — FR/ES/PT/DE → EN، على الجهاز، ترميز Mimi تدفقي
```

### اكتشاف النشاط الصوتي — [الدليل الكامل →](https://soniqo.audio/ar/guides/vad)

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
for s in segments { print("\(s.startTime)s → \(s.endTime)s") }
```

### تمييز المتحدثين — [الدليل الكامل →](https://soniqo.audio/ar/guides/diarize)

```swift
import SpeechVAD

let diarizer = try await DiarizationPipeline.fromPretrained()
let segments = diarizer.diarize(audio: samples, sampleRate: 16000)
for s in segments { print("Speaker \(s.speakerId): \(s.startTime)s - \(s.endTime)s") }
```

### تحسين الكلام — [الدليل الكامل →](https://soniqo.audio/ar/guides/denoise)

```swift
import SpeechEnhancement

let denoiser = try await DeepFilterNet3Model.fromPretrained()
let clean = try denoiser.enhance(audio: noisySamples, sampleRate: 48000)
```

### استعادة الكلام — [الدليل الكامل →](https://soniqo.audio/ar/guides/restore)

إزالة الضوضاء **و** إزالة الصدى معاً باستخدام [Sidon](https://arxiv.org/abs/2509.17052) (متنبئ w2v-BERT 2.0 + مرمّز صوتي DAC، Core ML). على عكس مكبِّت الضوضاء العام، دُرِّب Sidon على الحفاظ على هوية المتحدث، لذا فهو مناسب تماماً لتنظيف عينة مرجعية صاخبة أو ذات صدى لاستنساخ الصوت قبل تحويل النص إلى كلام. المُدخل بتردد 16 كيلوهرتز؛ والمُخرج أحادي بتردد 48 كيلوهرتز.

```swift
import SpeechRestoration

let restorer = try await SpeechRestorer.fromPretrained()          // .fp16 (الافتراضي) أو .int8
let clean = try restorer.restore(audio: noisySamples, sampleRate: 16000)  // ← 48 كيلوهرتز
```

من سطر الأوامر:

```bash
speech restore noisy.wav -o clean.wav            # إزالة الضوضاء + إزالة الصدى، مُخرج بتردد 48 كيلوهرتز
speech restore noisy.wav --variant int8          # أصغر، وذاكرة ذروة أقل

# تنظيف عينة مرجعية لاستنساخ الصوت قبل تحويل النص إلى كلام (اختياري؛ يحافظ على هوية المتحدث):
speech speak "Hello" --engine voxcpm2 --voice-sample ref.wav --clean-reference
```

### خط أنابيب الصوت (ASR → LLM → TTS) — [الدليل الكامل →](https://soniqo.audio/ar/voice-agents)

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

<div dir="rtl">

`VoicePipeline` هي آلة حالة وكيل الصوت في الزمن الحقيقي (مدعومة بـ [speech-core](https://github.com/soniqo/speech-core)) مع اكتشاف الأدوار المعتمد على VAD، ومعالجة المقاطعات، و STT حريص. تربط بين أي `SpeechRecognitionModel` + `SpeechGenerationModel` + `StreamingVADProvider`.

</div>

### خادم HTTP API

```bash
speech-server --port 8080
```

<div dir="rtl">

يعرض كل نموذج عبر نقاط نهاية HTTP REST + WebSocket، بما في ذلك واجهات برمجة متوافقة مع OpenAI: WebSocket Realtime على `/v1/realtime` ونقطة نهاية REST للنسخ الصوتي على `/v1/audio/transcriptions`. انظر [`Sources/AudioServer/`](Sources/AudioServer/).

</div>

## الهيكلة

<div dir="rtl">

speech-swift مقسم إلى هدف SPM واحد لكل نموذج بحيث يدفع المستهلكون فقط مقابل ما يستوردونه. توجد البنية التحتية المشتركة في `AudioCommon` (البروتوكولات، إدخال/إخراج الصوت، منزّل HuggingFace، `SentencePieceModel`) و `MLXCommon` (تحميل الأوزان، مساعدات `QuantizedLinear`، مساعد انتباه متعدد الرؤوس `SDPA`).

**[مخطط الهيكلة الكامل مع الواجهات الخلفية وجداول الذاكرة وخريطة الوحدات → soniqo.audio/architecture](https://soniqo.audio/ar/architecture)** · **[مرجع API → soniqo.audio/api](https://soniqo.audio/ar/api)** · **[الاختبارات → soniqo.audio/benchmarks](https://soniqo.audio/ar/benchmarks)**

الوثائق المحلية (المستودع):
- **النماذج:** [Qwen3-ASR](docs/models/asr-model.md) · [WhisperASR](docs/models/whisper-asr.md) · [Qwen3-TTS](docs/models/tts-model.md) · [CosyVoice](docs/models/cosyvoice-tts.md) · [Kokoro](docs/models/kokoro-tts.md) · [VoxCPM2](docs/models/voxcpm2-tts.md) · [IndexTTS2](docs/models/indextts2.md) · [F5-TTS](docs/models/f5-tts.md) · [Higgs TTS 3](docs/models/higgs-tts.md) · [VibeVoice](docs/models/vibevoice.md) · [Supertonic](docs/models/supertonic-tts.md) · [Chatterbox](docs/models/chatterbox-tts.md) · [Indic-Mio](docs/models/indic-mio-tts.md) · [Fish Audio S2 Pro](docs/models/fish-audio-s2-pro.md) · [Magpie TTS](docs/models/magpie-tts.md) · [Parakeet TDT](docs/models/parakeet-asr.md) · [Parakeet Streaming](docs/models/parakeet-streaming-asr.md) · [Nemotron Streaming](docs/models/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/models/omnilingual-asr.md) · [PersonaPlex](docs/models/personaplex.md) · [Hibiki](docs/models/hibiki.md) · [MADLAD-400](docs/models/madlad-translation.md) · [FunctionGemma](docs/models/function-gemma.md) · [Qwen3.5 Chat](docs/models/qwen35-chat.md) · [Gemma 4 Chat](docs/models/gemma4-chat.md) · [Qwen3 Dense Chat](docs/models/qwen3-dense-chat.md) · [FireRedVAD](docs/models/fireredvad.md) · [KWS Zipformer](docs/models/kws-zipformer.md) · [Sidon](docs/models/sidon.md) · [Source Separation](docs/models/source-separation.md) · [HTDemucs](docs/models/htdemucs.md) · [MAGNeT](docs/models/magnet-music-gen.md) · [Stable Audio 3](docs/models/stable-audio-3.md) · [FlashSR](docs/models/flashsr.md) · [Audio2Face-3D](docs/models/audio2face3d.md)
- **الاستدلال:** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md) · [WhisperASR](docs/inference/whisper-asr-inference.md) · [Parakeet TDT](docs/inference/parakeet-asr-inference.md) · [Parakeet Streaming](docs/inference/parakeet-streaming-asr-inference.md) · [Nemotron Streaming](docs/inference/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/inference/omnilingual-asr-inference.md) · [TTS](docs/inference/qwen3-tts-inference.md) · [VoxCPM2](docs/inference/voxcpm2-inference.md) · [IndexTTS2](docs/inference/indextts2.md) · [F5-TTS](docs/inference/f5-tts.md) · [Higgs TTS 3](docs/inference/higgs-tts.md) · [VibeVoice](docs/inference/vibevoice-inference.md) · [Fish Audio S2 Pro](docs/inference/fish-audio-s2-pro.md) · [Magpie TTS](docs/inference/magpie-tts.md) · [Hibiki](docs/inference/hibiki-inference.md) · [MADLAD-400](docs/inference/madlad-translation.md) · [MAGNeT](docs/inference/magnet-music-gen.md) · [Stable Audio 3](docs/inference/stable-audio-3.md) · [FlashSR](docs/inference/flashsr.md) · [Forced Aligner](docs/inference/forced-aligner.md) · [Silero VAD](docs/inference/silero-vad.md) · [FireRedVAD](docs/inference/fireredvad.md) · [Wake-word](docs/inference/wake-word.md) · [Speaker Diarization](docs/inference/speaker-diarization.md) · [Speech Enhancement](docs/inference/speech-enhancement.md) · [Sidon](docs/inference/sidon.md) · [Cache/offline](docs/inference/cache-and-offline.md)
- **المرجع:** [البروتوكولات المشتركة](docs/shared-protocols.md)

</div>

## تكوين الذاكرة المؤقتة

<div dir="rtl">

تُنزَّل أوزان النموذج من HuggingFace عند الاستخدام الأول وتُخزَّن في `~/Library/Caches/qwen3-speech/`. يمكن استبدالها بـ `QWEN3_CACHE_DIR` (CLI) أو `cacheDir:` (Swift API). تقبل جميع نقاط دخول `fromPretrained()` أيضاً `offlineMode: true` لتخطي الشبكة عندما تكون الأوزان مخزنة مؤقتاً بالفعل.

يمكن للمستخدمين في الصين القارية (أو في أي مكان يكون فيه `huggingface.co` بطيئاً أو محجوباً) التنزيل من مرآة عبر ضبط `HF_ENDPOINT`، مثل `export HF_ENDPOINT=https://hf-mirror.com`.

انظر [`docs/inference/cache-and-offline.md`](docs/inference/cache-and-offline.md) للتفاصيل الكاملة بما في ذلك مسارات حاوية iOS المعزولة.

</div>

## مكتبة MLX Metal

<div dir="rtl">

إذا ظهر لك `Failed to load the default metallib` عند التشغيل، فإن مكتبة شيدر Metal مفقودة. شغّل `make build` أو `./scripts/build_mlx_metallib.sh release` بعد `swift build` يدوي. إذا كان Metal Toolchain مفقوداً، ثبّته أولاً:

</div>

```bash
xcodebuild -downloadComponent MetalToolchain
```

## الاختبارات

```bash
make test                            # المجموعة الكاملة (وحدة + E2E مع تنزيلات النماذج)
swift test --skip E2E                # وحدة فقط (آمن لـ CI، بدون تنزيلات)
swift test --filter Qwen3ASRTests    # وحدة محددة
```

<div dir="rtl">

تستخدم فئات اختبار E2E بادئة `E2E` بحيث يمكن لـ CI تصفيتها باستخدام `--skip E2E`. راجع [CLAUDE.md](CLAUDE.md#testing) لاتفاقية الاختبار الكاملة.

</div>

## المساهمة

<div dir="rtl">

طلبات السحب مرحب بها — إصلاح الأخطاء، دمج نماذج جديدة، التوثيق. اعمل fork، أنشئ فرع ميزة، `make build && make test`، افتح PR ضد `main`.

</div>

## الرخصة

Apache 2.0
