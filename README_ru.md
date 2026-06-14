# Speech Swift

Модели ИИ для обработки речи на Apple Silicon, на базе MLX Swift и CoreML.

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

Распознавание, синтез и понимание речи на устройстве для Mac и iOS. Работает полностью локально на Apple Silicon — без облака, без API-ключей, данные не покидают устройство.

**[📚 Полная документация →](https://soniqo.audio/ru)** · **[🤗 Модели на HuggingFace](https://huggingface.co/aufklarer)** · **[📝 Блог](https://blog.ivan.digital)**

<p align="center">
  <a href="https://www.producthunt.com/products/speech-swift?embed=true&amp;utm_source=badge-featured&amp;utm_medium=badge&amp;utm_campaign=badge-speech-swift" target="_blank" rel="noopener noreferrer"><img alt="speech-swift -  The whole speech stack, on your laptop. | Product Hunt" width="250" height="54" src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1151422&amp;theme=light&amp;t=1779261593657"></a>
</p>

<p align="center">
  <a href="https://youtu.be/x9zgcaW0gUk">
    <img src="https://img.youtube.com/vi/x9zgcaW0gUk/maxresdefault.jpg" width="640" alt="Локальный речевой ИИ на MacBook — четырёхминутный обзор open source библиотеки на YouTube">
  </a>
</p>
<p align="center"><em>Локальный речевой ИИ на MacBook — четырёхминутный обзор open source библиотеки на YouTube</em></p>

**Сценарии:** [Голосовые агенты](https://soniqo.audio/ru/voice-agents) · [Транскрипция](https://soniqo.audio/ru/transcription) · [Синтез речи](https://soniqo.audio/ru/speech-generation)

- **[Qwen3-ASR](https://soniqo.audio/ru/guides/transcribe)** — Распознавание речи (автоматическое распознавание речи, 52 языка, MLX + CoreML)
- **[Parakeet TDT](https://soniqo.audio/ru/guides/parakeet)** — Распознавание речи через CoreML (Neural Engine, NVIDIA FastConformer + TDT-декодер, 25 языков)
- **[Omnilingual ASR](https://soniqo.audio/ru/guides/omnilingual)** — Распознавание речи (Meta wav2vec2 + CTC, **1 672 языка** в 32 письменностях, CoreML 300M + MLX 300M/1B/3B/7B)
- **[Streaming Dictation](https://soniqo.audio/ru/guides/dictate)** — Диктовка в реальном времени с частичными результатами и детекцией окончания реплики (Parakeet-EOU-120M)
- **[Nemotron Streaming (Многоязычный)](https://soniqo.audio/ru/guides/nemotron)** — Потоковое ASR с низкой задержкой, нативной пунктуацией и капитализацией (NVIDIA Nemotron-3.5-ASR-Streaming-0.6B, CoreML + MLX, **40 языковых локалей**)
- **[Nemotron Streaming (Английский)](https://soniqo.audio/guides/nemotron)** — Потоковое ASR с низкой задержкой, нативной пунктуацией и капитализацией (NVIDIA Nemotron-Speech-Streaming-0.6B, CoreML, только английский, меньше и быстрее многоязычного варианта)
- **[Qwen3-ForcedAligner](https://soniqo.audio/ru/guides/align)** — Выравнивание временных меток на уровне слов (аудио + текст → временные метки)
- **[Qwen3-TTS](https://soniqo.audio/ru/guides/speak)** — Синтез речи (наивысшее качество, потоковый режим, пользовательские голоса, 10 языков)
- **[CosyVoice TTS](https://soniqo.audio/ru/guides/cosyvoice)** — Потоковый синтез речи с клонированием голоса, многоголосым диалогом, тегами эмоций (9 языков)
- **[VoxCPM2](https://soniqo.audio/ru/speech-generation)** — TTS студийного качества 48 кГц с клонированием голоса и описанием голоса инструкцией (2B, MLX bf16/int8/int4, 30 языков)
- **[Kokoro TTS](https://soniqo.audio/ru/guides/kokoro)** — Синтез речи на устройстве (82M, CoreML/Neural Engine, 54 голоса, готов для iOS, 10 языков)
- **[VibeVoice TTS](https://soniqo.audio/ru/guides/vibevoice)** — Длинный формат / многоголосый TTS (Microsoft VibeVoice Realtime-0.5B + 1.5B, MLX, синтез подкастов/аудиокниг до 90 минут, EN/ZH)
- **[Magpie TTS](https://soniqo.audio/ru/guides/magpie)** — Многоязычный TTS (NVIDIA Magpie-TTS Multilingual 357M, MLX INT4 247 МБ / INT8 411 МБ или CoreML INT8 342 МБ, 9 языков, 5 встроенных голосов, стриминг на MLX)
- **[Qwen3.5-Chat](https://soniqo.audio/ru/guides/chat)** — Локальный чат на базе LLM (0.8B, MLX INT4 + CoreML INT8, гибрид DeltaNet, потоковая генерация токенов)
- **[MADLAD-400](https://soniqo.audio/ru/guides/translate)** — Многоязычный перевод между 400+ языками (3B, MLX INT4 + INT8, T5 v1.1, Apache 2.0)
- **[Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate)** — Потоковый перевод речи в речь (FR/ES/PT/DE → EN, MLX INT4 + INT8, стек Kyutai Moshi/Mimi, CC-BY-4.0)
- **[PersonaPlex](https://soniqo.audio/ru/guides/respond)** — Полнодуплексная генерация речи из речи (7B, аудио на входе → аудио на выходе, 18 голосовых пресетов)
- **[DeepFilterNet3](https://soniqo.audio/ru/guides/denoise)** — Подавление шума в реальном времени (2.1M параметров, 48 кГц). Длинное аудио, превышающее лимит одиночного прохода в 60 s, автоматически разбивается на чанки с crossfade — см. `enhanceChunked(...)`
- **[Разделение источников](https://soniqo.audio/ru/guides/separate)** — Разделение музыкальных источников через HTDemucs (Demucs v4) + Open-Unmix (UMX-HQ / UMX-L, 4 стема: вокал/ударные/бас/остальное, 44,1 кГц стерео)
- **[MAGNeT](https://soniqo.audio/ru/guides/compose)** — Генерация музыки по тексту (Meta MAGNeT Small 300M / Medium 1.5B, MLX INT4/INT8, 30-секундные клипы 32 кГц моно, маскированное параллельное декодирование)
- **[FlashSR](https://soniqo.audio/ru/guides/upsample)** — Аудио супер-разрешение (FlashSR ICASSP 2025, MLX, 48 кГц моно, дистиллированная диффузия за 1 шаг, INT4 363 МБ / INT8 720 МБ)
- **[Активационное слово](https://soniqo.audio/ru/guides/wake-word)** — Локальное распознавание ключевых слов (KWS Zipformer 3M, CoreML, 26× реального времени, настраиваемый список ключевых слов)
- **[VAD](https://soniqo.audio/ru/guides/vad)** — Обнаружение голосовой активности (Silero потоковый, Pyannote офлайн, FireRedVAD 100+ языков)
- **[Speaker Diarization](https://soniqo.audio/ru/guides/diarize)** — Кто говорил и когда (Pyannote-пайплайн, сквозной Sortformer на Neural Engine)
- **[Speaker Embeddings](https://soniqo.audio/ru/guides/embed-speaker)** — WeSpeaker ResNet34 (256-мерные векторы), CAM++ (192-мерные)

Статьи: [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba) · [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba) · [Omnilingual ASR](https://arxiv.org/abs/2511.09690) (Meta) · [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA) · [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba) · [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2) · [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA) · [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai) · [Hibiki](https://arxiv.org/abs/2502.03382) (Kyutai) · [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## Новости

- **19 апреля 2026** — [MLX vs CoreML on Apple Silicon — A Practical Guide to Picking the Right Backend](https://blog.ivan.digital/mlx-vs-coreml-on-apple-silicon-a-practical-guide-to-picking-the-right-backend-and-why-you-should-f77ddea7b27a)
- **20 марта 2026** — [We Beat Whisper Large v3 with a 600M Model Running Entirely on Your Mac](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **26 февраля 2026** — [Speaker Diarization and Voice Activity Detection on Apple Silicon — Native Swift with MLX](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **23 февраля 2026** — [NVIDIA PersonaPlex 7B on Apple Silicon — Full-Duplex Speech-to-Speech in Native Swift with MLX](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **12 февраля 2026** — [Qwen3-ASR Swift: On-Device ASR + TTS for Apple Silicon — Architecture and Benchmarks](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## Быстрый старт

Добавьте пакет в `Package.swift`:

```swift
.package(url: "https://github.com/soniqo/speech-swift", branch: "main")
```

Импортируйте только те модули, которые вам нужны — каждая модель это отдельная SPM-библиотека, поэтому вы не платите за то, чем не пользуетесь:

```swift
.product(name: "ParakeetStreamingASR", package: "speech-swift"),
.product(name: "SpeechUI",             package: "speech-swift"),  // опциональные SwiftUI-компоненты
```

**Транскрибировать аудиобуфер в 3 строки:**

```swift
import ParakeetStreamingASR

let model = try await ParakeetStreamingASRModel.fromPretrained()
let text = try model.transcribeAudio(audioSamples, sampleRate: 16000)
```

**Потоковый режим с частичными результатами:**

```swift
for await partial in model.transcribeStream(audio: samples, sampleRate: 16000) {
    print(partial.isFinal ? "FINAL: \(partial.text)" : "... \(partial.text)")
}
```

**SwiftUI-вид для диктовки примерно в 10 строк:**

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

`SpeechUI` предоставляет только `TranscriptionView` (финальные + частичные результаты) и `TranscriptionStore` (адаптер для потокового ASR). Для визуализации и воспроизведения аудио используйте AVFoundation.

Доступные SPM-продукты: `Qwen3ASR`, `Qwen3TTS`, `Qwen3TTSCoreML`, `ParakeetASR`, `ParakeetStreamingASR`, `NemotronStreamingASR`, `OmnilingualASR`, `KokoroTTS`, `VibeVoiceTTS`, `CosyVoiceTTS`, `VoxCPM2TTS`, `MagpieTTS`, `MagpieTTSCoreML`, `MAGNeTMusicGen`, `FlashSR`, `PersonaPlex`, `HibikiTranslate`, `SpeechVAD`, `SpeechEnhancement`, `SourceSeparation`, `Qwen3Chat`, `SpeechCore`, `SpeechUI`, `AudioCommon`.

## Модели

Краткий обзор ниже. **[Полный каталог моделей со всеми размерами, квантизациями, ссылками на скачивание и таблицами памяти → soniqo.audio/architecture](https://soniqo.audio/ru/architecture)**.

| Модель | Задача | Бэкенды | Размеры | Языки |
|--------|--------|---------|---------|-------|
| [Qwen3-ASR](https://soniqo.audio/ru/guides/transcribe) | Речь → Текст | MLX, CoreML (гибрид) | 0.6B, 1.7B | 52 |
| [Parakeet TDT](https://soniqo.audio/ru/guides/parakeet) | Речь → Текст | CoreML (ANE) | 0.6B | 25 европейских |
| [Parakeet EOU](https://soniqo.audio/ru/guides/dictate) | Речь → Текст (потоковый) | CoreML (ANE) | 120M | 25 европейских |
| [Nemotron Streaming (Многоязычный)](https://soniqo.audio/ru/guides/nemotron) | Речь → Текст (потоковый, с пунктуацией) | CoreML (ANE), MLX | 0.6B | **40** |
| [Nemotron Streaming (Английский)](https://soniqo.audio/guides/nemotron) | Речь → Текст (потоковый, с пунктуацией) | CoreML (ANE) | 0.6B | EN |
| [Omnilingual ASR](https://soniqo.audio/ru/guides/omnilingual) | Речь → Текст | CoreML (ANE), MLX | 300M / 1B / 3B / 7B | **[1 672](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)** |
| [Qwen3-ForcedAligner](https://soniqo.audio/ru/guides/align) | Аудио + Текст → Метки | MLX, CoreML | 0.6B | Многоязычный |
| [Qwen3-TTS](https://soniqo.audio/ru/guides/speak) | Текст → Речь | MLX, CoreML | 0.6B, 1.7B | 10 |
| [CosyVoice3](https://soniqo.audio/ru/guides/cosyvoice) | Текст → Речь | MLX | 0.5B | 9 |
| [VoxCPM2](https://soniqo.audio/ru/speech-generation) | Текст → Речь (48 кГц, описание голоса + клонирование) | MLX | 2B (bf16/int8/int4) | 30 |
| [Kokoro-82M](https://soniqo.audio/ru/guides/kokoro) | Текст → Речь | CoreML (ANE) | 82M | 10 |
| [VibeVoice Realtime-0.5B](https://soniqo.audio/ru/guides/vibevoice) | Текст → Речь (длинный формат, многоголосый) | MLX | 0.5B | EN/ZH |
| [VibeVoice 1.5B](https://soniqo.audio/ru/guides/vibevoice) | Текст → Речь (подкаст до 90 минут) | MLX | 1.5B | EN/ZH |
| [Magpie-TTS Multilingual](https://soniqo.audio/ru/guides/magpie) | Текст → Речь (5 встроенных голосов, стриминг) | MLX / CoreML | 357M (MLX INT4/INT8, CoreML INT8) | 9 (CoreML без JA) |
| [Qwen3.5-Chat](https://soniqo.audio/ru/guides/chat) | Текст → Текст (LLM) | MLX, CoreML | 0.8B | Многоязычный |
| [MADLAD-400](https://soniqo.audio/ru/guides/translate) | Текст → Текст (Перевод) | MLX | 3B | **400+** |
| [Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate) | Речь → Речь (Перевод) | MLX | 3B | FR/ES/PT/DE → EN |
| [PersonaPlex](https://soniqo.audio/ru/guides/respond) | Речь → Речь | MLX | 7B | EN |
| [Silero VAD](https://soniqo.audio/ru/guides/vad) | Детектор речи | MLX, CoreML | 309K | Универсальный |
| [Pyannote](https://soniqo.audio/ru/guides/diarize) | VAD + Диаризация | MLX | 1.5M | Универсальный |
| [Sortformer](https://soniqo.audio/ru/guides/diarize) | Диаризация (E2E) | CoreML (ANE) | — | Универсальный |
| [DeepFilterNet3](https://soniqo.audio/ru/guides/denoise) | Улучшение речи | CoreML | 2.1M | Универсальный |
| [Sidon](https://soniqo.audio/ru/guides/restore) | Восстановление речи (подавление шума + дереверберация, 48 кГц) | CoreML | w2v-BERT 2.0 + DAC (fp16/int8) | Универсальный |
| [HTDemucs (Demucs v4)](https://soniqo.audio/ru/guides/separate) | Разделение источников | MLX | 168M | Agnostic |
| [Open-Unmix](https://soniqo.audio/ru/guides/separate) | Разделение источников | MLX | 8.6M | Agnostic |
| [MAGNeT](https://soniqo.audio/ru/guides/compose) | Текст → Музыка (30 с @ 32 кГц) | MLX | 300M / 1.5B (int4/int8) | EN-промпты |
| [FlashSR](https://soniqo.audio/ru/guides/upsample) | Аудио супер-разрешение (48 кГц) | MLX | 363 МБ / 720 МБ (int4/int8) | Языконезависимое |
| [WeSpeaker](https://soniqo.audio/ru/guides/embed-speaker) | Эмбеддинги спикеров | MLX, CoreML | 6.6M | Универсальный |

## Установка

### Homebrew

Требуется нативный ARM Homebrew (`/opt/homebrew`). Rosetta/x86_64-версия Homebrew не поддерживается.

```bash
brew install speech
```

Затем:

```bash
speech transcribe recording.wav
speech speak "Hello world"
speech translate "Hello, how are you?" --to es
speech respond --input question.wav --transcript
speech-server --port 8080            # локальный HTTP / WebSocket сервер (OpenAI-совместимый /v1/realtime + /v1/audio/transcriptions)
```

**[Полный справочник CLI →](https://soniqo.audio/ru/cli)**

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

Импортируйте только то, что вам нужно — каждая модель это отдельный SPM-таргет:

```swift
import Qwen3ASR             // Распознавание речи (MLX)
import ParakeetASR          // Распознавание речи (CoreML, пакетный режим)
import ParakeetStreamingASR // Потоковая диктовка с частичными результатами + EOU
import NemotronStreamingASR // Многоязычное потоковое ASR с нативной пунктуацией (0.6B, 40 языков)
import OmnilingualASR       // 1 672 языка (CoreML + MLX)
import Qwen3TTS             // Синтез речи
import CosyVoiceTTS         // Синтез речи с клонированием голоса
import VoxCPM2TTS           // TTS 48 кГц, клонирование голоса + описание голоса (2B)
import KokoroTTS            // Синтез речи (готов для iOS)
import VibeVoiceTTS         // Длинный формат / многоголосый TTS (EN/ZH)
import MagpieTTS            // Многоязычный TTS (NVIDIA Magpie 357M, MLX, 9 языков)
import MagpieTTSCoreML      // CoreML-бэкенд Magpie (гибрид CoreML + MLX, 8 языков)
import Qwen3Chat            // Локальный чат на базе LLM
import MADLADTranslation    // Многоязычный перевод между 400+ языками
import HibikiTranslate      // Потоковый перевод речи в речь (FR/ES/PT/DE → EN)
import PersonaPlex          // Полнодуплексная речь-в-речь
import SpeechVAD            // VAD + диаризация спикеров + эмбеддинги
import SpeechEnhancement    // Подавление шума
import SpeechRestoration    // Восстановление речи — подавление шума + дереверберация (Sidon, CoreML, 48 кГц)
import SourceSeparation     // Разделение музыкальных источников (Open-Unmix, 4 стема)
import MAGNeTMusicGen      // Генерация музыки по тексту (30 с, 32 кГц)
import FlashSR             // Аудио супер-разрешение (48 кГц, 1-шаговая диффузия)
import SpeechUI             // SwiftUI-компоненты для потоковой транскрипции
import AudioCommon          // Общие протоколы и утилиты
```

### Требования

- Swift 6+, Xcode 16+ (с Metal Toolchain)
- macOS 15+ (Sequoia) или iOS 18+, Apple Silicon (M1/M2/M3/M4)

Минимум macOS 15 / iOS 18 следует из [MLState](https://developer.apple.com/documentation/coreml/mlstate) —— API Apple для персистентного состояния на ANE —— которое CoreML-пайплайны (Qwen3-ASR, Qwen3-Chat, Qwen3-TTS) используют, чтобы держать KV-кэши на Neural Engine между шагами токенов.

### Сборка из исходников

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

`make build` компилирует Swift-пакет **и** библиотеку Metal-шейдеров MLX. Metal-библиотека необходима для GPU-инференса — без неё при запуске будет ошибка `Failed to load the default metallib`. `make debug` для отладочных сборок, `make test` для тестового набора.

**[Полное руководство по сборке и установке →](https://soniqo.audio/ru/getting-started)**

## Демо-приложения

- **[DictateDemo](Examples/DictateDemo/)** ([документация](https://soniqo.audio/ru/guides/dictate)) — macOS-приложение в строке меню с потоковой диктовкой, живыми частичными результатами, детекцией окончания реплики по VAD и копированием в один клик. Работает как фоновый агент (Parakeet-EOU-120M + Silero VAD).
- **[iOSEchoDemo](Examples/iOSEchoDemo/)** — iOS-эхо-демо (Parakeet ASR + Kokoro TTS). Устройство и симулятор.
- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — Диалоговый голосовой ассистент с микрофонным вводом, VAD и многоходовым контекстом. macOS. RTF ~0.94 на M2 Max (быстрее реального времени).
- **[SpeechDemo](Examples/SpeechDemo/)** — Диктовка и синтез TTS в интерфейсе с вкладками. macOS.

В README каждого демо есть инструкции по сборке.

## Примеры кода

Сниппеты ниже показывают минимальный путь для каждой области. Каждый раздел ссылается на полное руководство на [soniqo.audio](https://soniqo.audio/ru) с опциями конфигурации, множественными бэкендами, шаблонами потоковой обработки и примерами CLI.

### Распознавание речи — [полное руководство →](https://soniqo.audio/ru/guides/transcribe)

```swift
import Qwen3ASR

let model = try await Qwen3ASRModel.fromPretrained()
let text = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

Альтернативные бэкенды: [Parakeet TDT](https://soniqo.audio/ru/guides/parakeet) (CoreML, 32× быстрее реального времени), [Omnilingual ASR](https://soniqo.audio/ru/guides/omnilingual) (1 672 языка, CoreML или MLX), [Потоковая диктовка](https://soniqo.audio/ru/guides/dictate) (живые частичные результаты).

### Выравнивание с форсированием — [полное руководство →](https://soniqo.audio/ru/guides/align)

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

### Синтез речи — [полное руководство →](https://soniqo.audio/ru/guides/speak)

```swift
import Qwen3TTS
import AudioCommon

let model = try await Qwen3TTSModel.fromPretrained()
let audio = model.synthesize(text: "Hello world", language: "english")
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

Альтернативные TTS-движки: [CosyVoice3](https://soniqo.audio/ru/guides/cosyvoice) (потоковый + клонирование голоса + теги эмоций), [Kokoro-82M](https://soniqo.audio/ru/guides/kokoro) (готов для iOS, 54 голоса), [VibeVoice](https://soniqo.audio/ru/guides/vibevoice) (длинный подкаст / многоголосый, EN/ZH), [Клонирование голоса](https://soniqo.audio/ru/guides/voice-cloning).

### Речь-в-речь — [полное руководство →](https://soniqo.audio/ru/guides/respond)

```swift
import PersonaPlex

let model = try await PersonaPlexModel.fromPretrained()
let responseAudio = model.respond(userAudio: userSamples)
// 24 кГц моно Float32 — готово к воспроизведению
```

### LLM-чат — [полное руководство →](https://soniqo.audio/ru/guides/chat)

```swift
import Qwen3Chat

let chat = try await Qwen35MLXChat.fromPretrained()
chat.chat(messages: [(.user, "Explain MLX in one sentence")]) { token, isFinal in
    print(token, terminator: "")
}
```

### Перевод — [полное руководство →](https://soniqo.audio/ru/guides/translate)

```swift
import MADLADTranslation

let translator = try await MADLADTranslator.fromPretrained()
let es = try translator.translate("Hello, how are you?", to: "es")
// → "Hola, ¿cómo estás?"
```

### Перевод речи — [полное руководство →](https://soniqo.audio/guides/audio-translate)

```swift
import HibikiTranslate
import AudioCommon

let model = try await HibikiTranslateModel.fromPretrained()
let pcm = try AudioFileLoader.load(url: input, targetSampleRate: 24000)
let (englishAudio, textTokens) = model.translate(
    sourceAudio: pcm, sourceLanguage: .fr
)
// Hibiki Zero-3B — FR/ES/PT/DE → EN, на устройстве, потоковый кодек Mimi
```

### Детектор голосовой активности — [полное руководство →](https://soniqo.audio/ru/guides/vad)

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
for s in segments { print("\(s.startTime)s → \(s.endTime)s") }
```

### Диаризация спикеров — [полное руководство →](https://soniqo.audio/ru/guides/diarize)

```swift
import SpeechVAD

let diarizer = try await DiarizationPipeline.fromPretrained()
let segments = diarizer.diarize(audio: samples, sampleRate: 16000)
for s in segments { print("Speaker \(s.speakerId): \(s.startTime)s - \(s.endTime)s") }
```

### Улучшение речи — [полное руководство →](https://soniqo.audio/ru/guides/denoise)

```swift
import SpeechEnhancement

let denoiser = try await DeepFilterNet3Model.fromPretrained()
let clean = try denoiser.enhance(audio: noisySamples, sampleRate: 48000)
```

### Восстановление речи — [полное руководство →](https://soniqo.audio/ru/guides/restore)

Совместное подавление шума **и** дереверберация с помощью [Sidon](https://arxiv.org/abs/2509.17052) (предиктор w2v-BERT 2.0 + вокодер DAC, Core ML). В отличие от обычного шумоподавителя, Sidon обучен сохранять идентичность говорящего, поэтому он хорошо подходит для очистки зашумлённого или реверберирующего референса для клонирования голоса перед TTS. Вход — 16 кГц; выход — 48 кГц моно.

```swift
import SpeechRestoration

let restorer = try await SpeechRestorer.fromPretrained()          // .fp16 (по умолчанию) или .int8
let clean = try restorer.restore(audio: noisySamples, sampleRate: 16000)  // → 48 кГц
```

Из CLI:

```bash
speech restore noisy.wav -o clean.wav            # подавление шума + дереверберация, выход 48 кГц
speech restore noisy.wav --variant int8          # компактнее, ниже пиковое потребление RAM

# Очистка референса для клонирования голоса перед TTS (по выбору; сохраняет идентичность говорящего):
speech speak "Hello" --engine voxcpm2 --voice-sample ref.wav --clean-reference
```

### Голосовой пайплайн (ASR → LLM → TTS) — [полное руководство →](https://soniqo.audio/ru/voice-agents)

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

`VoicePipeline` — это конечный автомат голосового агента реального времени (на базе [speech-core](https://github.com/soniqo/speech-core)) с детекцией переключения реплик по VAD, обработкой прерываний и eager STT. Он соединяет любой `SpeechRecognitionModel` + `SpeechGenerationModel` + `StreamingVADProvider`.

### HTTP API-сервер

```bash
speech-server --port 8080
```

Предоставляет все модели через HTTP REST + WebSocket-эндпоинты, включая совместимые с OpenAI API: Realtime WebSocket по адресу `/v1/realtime` и REST-эндпоинт транскрипции по адресу `/v1/audio/transcriptions`. См. [`Sources/AudioServer/`](Sources/AudioServer/).

## Архитектура

speech-swift разделён на отдельные SPM-таргеты для каждой модели, чтобы потребители платили только за то, что импортируют. Общая инфраструктура находится в `AudioCommon` (протоколы, аудио ввод/вывод, загрузчик HuggingFace, `SentencePieceModel`) и `MLXCommon` (загрузка весов, хелперы `QuantizedLinear`, хелпер `SDPA` для multi-head attention).

**[Полная архитектурная диаграмма с бэкендами, таблицами памяти и картой модулей → soniqo.audio/architecture](https://soniqo.audio/ru/architecture)** · **[Справочник API → soniqo.audio/api](https://soniqo.audio/ru/api)** · **[Бенчмарки → soniqo.audio/benchmarks](https://soniqo.audio/ru/benchmarks)**

Локальная документация (в репозитории):
- **Модели:** [Qwen3-ASR](docs/models/asr-model.md) · [Qwen3-TTS](docs/models/tts-model.md) · [CosyVoice](docs/models/cosyvoice-tts.md) · [Kokoro](docs/models/kokoro-tts.md) · [VibeVoice](docs/models/vibevoice.md) · [Parakeet TDT](docs/models/parakeet-asr.md) · [Parakeet Streaming](docs/models/parakeet-streaming-asr.md) · [Nemotron Streaming](docs/models/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/models/omnilingual-asr.md) · [PersonaPlex](docs/models/personaplex.md) · [Hibiki](docs/models/hibiki.md) · [FireRedVAD](docs/models/fireredvad.md) · [Source Separation](docs/models/source-separation.md) · [HTDemucs](docs/models/htdemucs.md) · [MAGNeT](docs/models/magnet-music-gen.md) · [FlashSR](docs/models/flashsr.md)
- **Инференс:** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md) · [Parakeet TDT](docs/inference/parakeet-asr-inference.md) · [Parakeet Streaming](docs/inference/parakeet-streaming-asr-inference.md) · [Nemotron Streaming](docs/inference/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/inference/omnilingual-asr-inference.md) · [TTS](docs/inference/qwen3-tts-inference.md) · [VibeVoice](docs/inference/vibevoice-inference.md) · [Hibiki](docs/inference/hibiki-inference.md) · [Forced Aligner](docs/inference/forced-aligner.md) · [Silero VAD](docs/inference/silero-vad.md) · [Диаризация спикеров](docs/inference/speaker-diarization.md) · [Улучшение речи](docs/inference/speech-enhancement.md) · [MAGNeT](docs/inference/magnet-music-gen.md) · [FlashSR](docs/inference/flashsr.md)
- **Справочник:** [Общие протоколы](docs/shared-protocols.md)

## Настройка кэша

Веса моделей скачиваются с HuggingFace при первом использовании и кэшируются в `~/Library/Caches/qwen3-speech/`. Переопределите с помощью `QWEN3_CACHE_DIR` (CLI) или `cacheDir:` (Swift API). Все точки входа `fromPretrained()` также принимают `offlineMode: true` для пропуска сети, когда веса уже закэшированы.

Подробности, включая пути в песочницах iOS-контейнеров, см. в [`docs/inference/cache-and-offline.md`](docs/inference/cache-and-offline.md).

## Metal-библиотека MLX

Если при запуске вы видите `Failed to load the default metallib`, это значит что библиотека Metal-шейдеров отсутствует. Запустите `make build` или `./scripts/build_mlx_metallib.sh release` после ручного `swift build`. Если Metal Toolchain отсутствует, сначала установите его:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Тестирование

```bash
make test                            # полный набор (юнит + E2E с загрузкой моделей)
swift test --skip E2E                # только юнит-тесты (безопасно для CI, без загрузок)
swift test --filter Qwen3ASRTests    # конкретный модуль
```

Классы E2E-тестов используют префикс `E2E`, чтобы CI мог отфильтровать их с помощью `--skip E2E`. Полное соглашение о тестировании см. в [CLAUDE.md](CLAUDE.md#testing).

## Участие в разработке

Приветствуются PR — исправления багов, интеграции новых моделей, документация. Сделайте форк, создайте feature-ветку, `make build && make test`, откройте PR в `main`.

## Лицензия

Apache 2.0
