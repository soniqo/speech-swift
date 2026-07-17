# Speech Swift

Modelos de IA de voz para Apple Silicon, impulsados por MLX Swift y CoreML.

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

Reconocimiento, síntesis y comprensión de voz en el dispositivo para Mac e iOS. Se ejecuta localmente en Apple Silicon — sin nube, sin claves de API, ningún dato sale del dispositivo.

**[📚 Documentación completa →](https://soniqo.audio/es)** · **[🤗 Modelos en HuggingFace](https://huggingface.co/aufklarer)** · **[📝 Blog](https://blog.ivan.digital)** · **[💬 Discord](https://discord.gg/TnCryqEMgu)**

<p align="center">
  <a href="https://trendshift.io/repositories/24196?utm_source=trendshift-badge&amp;utm_medium=badge&amp;utm_campaign=badge-trendshift-24196" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/trendshift/repositories/24196/daily?language=Swift" alt="soniqo%2Fspeech-swift | Trendshift" width="250" height="55"/></a>
</p>

<p align="center">
  <a href="https://youtu.be/x9zgcaW0gUk">
    <img src="https://img.youtube.com/vi/x9zgcaW0gUk/maxresdefault.jpg" width="640" alt="IA de voz local en un MacBook — mira en YouTube el recorrido de cuatro minutos por la biblioteca open source">
  </a>
</p>
<p align="center"><em>IA de voz local en un MacBook — mira en YouTube el recorrido de cuatro minutos por la biblioteca open source</em></p>

**Casos de uso:** [Agentes de voz](https://soniqo.audio/es/voice-agents) · [Transcripción](https://soniqo.audio/es/transcription) · [Síntesis de voz](https://soniqo.audio/es/speech-generation)

## Proyectos creados con Speech Swift

Proyectos públicos con integraciones verificables de Speech Swift.

[Palmier Pro](https://github.com/palmier-io/palmier-pro) · [Anarlog](https://github.com/fastrepl/anarlog) · [VoxFlow](https://github.com/xingbofeng/VoxFlow) · [Ora](https://github.com/wuwangzhang1216/ora) · [Jabber](https://github.com/rselbach/jabber) · [ClawdHome](https://github.com/ThinkInAIXYZ/clawdhome)

**Grupos de capacidades:** STT / ASR · Alineación · TTS · LLM y traducción · Voz a voz · Mejora / restauración · Separación de fuentes · Generación de música / audio · Wake word, VAD, diarización e identidad de hablante

**STT / ASR**

- **[Qwen3-ASR](https://soniqo.audio/es/guides/transcribe)** — Voz a texto (reconocimiento automático del habla, 52 idiomas, MLX + CoreML)
- **[WhisperASR](docs/models/whisper-asr.md)** — Whisper Large-v3 Turbo speech-to-text via native CoreML runtime (ANE, multilingual)
- **[Parakeet TDT](https://soniqo.audio/es/guides/parakeet)** — Voz a texto vía CoreML (Neural Engine, NVIDIA FastConformer + decodificador TDT, 25 idiomas)
- **[Omnilingual ASR](https://soniqo.audio/es/guides/omnilingual)** — Voz a texto (Meta wav2vec2 + CTC, **1.672 idiomas** en 32 escrituras, CoreML 300M + MLX 300M/1B/3B/7B)
- **[Dictado en streaming](https://soniqo.audio/es/guides/dictate)** — Dictado en tiempo real con resultados parciales y detección de fin de enunciado (Parakeet-EOU-120M)
- **[Nemotron Streaming (Multilingüe)](https://soniqo.audio/es/guides/nemotron)** — ASR en streaming de baja latencia con puntuación y mayúsculas nativas (NVIDIA Nemotron-3.5-ASR-Streaming-0.6B, CoreML + MLX, **40 configuraciones regionales**)
- **[Nemotron Streaming (Inglés)](https://soniqo.audio/guides/nemotron)** — ASR en streaming de baja latencia con puntuación y mayúsculas nativas (NVIDIA Nemotron-Speech-Streaming-0.6B, CoreML, solo inglés, más pequeño y rápido que la variante multilingüe)

**Alineación**

- **[Qwen3-ForcedAligner](https://soniqo.audio/es/guides/align)** — Alineación de marcas temporales a nivel de palabra (audio + texto → marcas temporales)

**TTS / Síntesis de voz**

- **[Qwen3-TTS](https://soniqo.audio/es/guides/speak)** — Síntesis de texto a voz (máxima calidad, streaming, hablantes personalizados, 10 idiomas)
- **[CosyVoice TTS](https://soniqo.audio/es/guides/cosyvoice)** — TTS con streaming, clonación de voz, diálogo multi-hablante y etiquetas de emoción (9 idiomas)
- **[VoxCPM2](https://soniqo.audio/es/speech-generation)** — TTS de calidad de estudio a 48 kHz con clonación de voz y diseño de voz por instrucciones (2B, MLX bf16/int8, 30 idiomas)
- **[IndexTTS2](docs/models/indextts2.md)** — Native MLX voice cloning from a reference voice (IndexTeam IndexTTS-2, 1.5B-class fp16 bundle, speaker/emotion/pause controls)
- **[F5-TTS](docs/models/f5-tts.md)** — Zero-shot voice cloning from a short reference clip + transcript (SWivid F5-TTS v1 Base, DiT flow matching + Vocos, MLX fp16, 24 kHz, English + Mandarin; non-commercial license)
- **[Higgs TTS 3](docs/models/higgs-tts.md)** — Conversational TTS with zero-shot voice cloning and inline emotion/style/SFX/prosody tags (Boson Higgs TTS 3, Qwen3-4B backbone, MLX bf16, 24 kHz, 100+ languages; research/non-commercial license)
- **[Kokoro TTS](https://soniqo.audio/es/guides/kokoro)** — TTS en el dispositivo (82M, CoreML/Neural Engine, 54 voces, listo para iOS, 10 idiomas)
- **[VibeVoice TTS](https://soniqo.audio/es/guides/vibevoice)** — TTS de formato largo / múltiples hablantes (Microsoft VibeVoice Realtime-0.5B + 1.5B, MLX, síntesis de podcast/audiolibro de hasta 90 min, EN/ZH)
- **[Magpie TTS](https://soniqo.audio/es/guides/magpie)** — TTS multilingüe (NVIDIA Magpie-TTS Multilingual 357M, MLX INT8 411 MB o CoreML INT8 342 MB, 9 idiomas, 5 hablantes preconfigurados, streaming en MLX)
- **[Supertonic TTS](https://soniqo.audio/guides/supertonic)** — TTS de flow-matching en el dispositivo (Supertone Supertonic-3 99M, CoreML/Neural Engine, 31 idiomas, 10 voces, sin G2P, 44.1 kHz)
- **[Chatterbox TTS](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16)** — TTS multilingüe con clonación de voz zero-shot (Resemble AI Chatterbox Multilingual, MLX fp16 ~1,3 GB, 23 idiomas en runtime; hebreo requiere niqqud, MIT)
- **[OmniVoice TTS](https://huggingface.co/aufklarer/OmniVoice-MLX-fp16)** — TTS de difusión no autorregresivo con clonación de voz zero-shot (k2-fsa OmniVoice, backbone Qwen3, MLX fp16 por defecto / int8 disponible, 600+ idiomas, Apache-2.0)
- **[Indic-Mio](docs/models/indic-mio-tts.md)** — Hindi/Indic TTS with inline emotion markers and optional reference-voice cloning (MLX, 24 kHz)

**LLM y traducción**

- **[Qwen3Chat](https://soniqo.audio/es/guides/chat)** — Chat LLM en el dispositivo (Qwen3.5-0.8B con MLX/CoreML, más backends MLX dense Qwen3 4B y Gemma 4 E2B/E4B, tokens en streaming)
- **[FunctionGemma](https://soniqo.audio/es/guides/function-calls)** — LLM en el dispositivo para llamadas estructuradas a funciones / herramientas (Gemma 3 270M, CoreML paletizado de 8 bits, Neural Engine, ~252 tok/s)
- **[MADLAD-400](https://soniqo.audio/es/guides/translate)** — Traducción multidireccional entre 400+ idiomas (3B, MLX INT4 + INT8, T5 v1.1, Apache 2.0)

**Voz a voz y agentes de voz**

- **[Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate)** — Traducción voz a voz en streaming (FR/ES/PT/DE → EN, MLX INT4 + INT8, pila Kyutai Moshi/Mimi, CC-BY-4.0)
- **[PersonaPlex](https://soniqo.audio/es/guides/respond)** — Voz a voz full-duplex (7B, audio de entrada → audio de salida, 18 presets de voz)
- **[Audio2Face-3D](docs/models/audio2face3d.md)** — Animación facial de avatares impulsada por voz (NVIDIA Audio2Face-3D v2.3 Mark, 301 coeficientes faciales, MLX)

**Mejora, separación y generación de audio**

- **[DeepFilterNet3](https://soniqo.audio/es/guides/denoise)** — Supresión de ruido en tiempo real (2.1M parámetros, 48 kHz). El audio de larga duración por encima del límite de 60 s en una sola pasada se divide automáticamente en fragmentos con crossfade — ver `enhanceChunked(...)`
- **[Separación de fuentes](https://soniqo.audio/es/guides/separate)** — Separación de fuentes musicales con HTDemucs (Demucs v4) + Open-Unmix (UMX-HQ / UMX-L, 4 stems: voces/batería/bajo/otros, 44,1 kHz estéreo)
- **[MAGNeT](https://soniqo.audio/es/guides/compose)** — Generación de música a partir de texto (Meta MAGNeT Small 300M / Medium 1.5B, MLX INT8, clips de 30 s a 32 kHz mono, decodificación enmascarada en paralelo)
- **[Stable Audio 3](docs/models/stable-audio-3.md)** — Text-to-audio/music generation (Stable Audio 3 Medium, MLX INT8/INT4, 44.1 kHz stereo, variable length)
- **[FlashSR](https://soniqo.audio/es/guides/upsample)** — Super-resolución de audio (FlashSR ICASSP 2025, MLX, 48 kHz mono, difusión destilada en 1 paso, INT4 363 MB / INT8 720 MB)

**Detección de turnos, diarización e identidad de hablante**

- **[Palabra de activación](https://soniqo.audio/es/guides/wake-word)** — Detección de palabras clave en el dispositivo (KWS Zipformer 3M, CoreML, 26× tiempo real, lista de palabras clave configurable)
- **[VAD](https://soniqo.audio/es/guides/vad)** — Detección de actividad vocal (Silero streaming, Pyannote offline, FireRedVAD 100+ idiomas)
- **[Diarización de hablantes](https://soniqo.audio/es/guides/diarize)** — Quién habló cuándo (pipeline Pyannote, Sortformer de extremo a extremo en Neural Engine) — ahora con una sesión de streaming incremental (IDs de hablante estables, actualizaciones cada 480 ms)
- **[Embeddings de hablante](https://soniqo.audio/es/guides/embed-speaker)** — WeSpeaker ResNet34 (256 dim), ReDimNet2-B6 para identidad de voz con nombre (192 dim), CAM++ (192 dim)

Papers: [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba) · [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba) · [Omnilingual ASR](https://arxiv.org/abs/2511.09690) (Meta) · [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA) · [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba) · [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2) · [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA) · [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai) · [Hibiki](https://arxiv.org/abs/2502.03382) (Kyutai) · [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## Novedades

- **19 abr 2026** — [MLX frente a CoreML en Apple Silicon — guía práctica para elegir el backend correcto](https://blog.ivan.digital/mlx-vs-coreml-on-apple-silicon-a-practical-guide-to-picking-the-right-backend-and-why-you-should-f77ddea7b27a)
- **20 mar 2026** — [Superamos a Whisper Large v3 con un modelo de 600M ejecutándose completamente en tu Mac](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **26 feb 2026** — [Diarización de hablantes y detección de actividad vocal en Apple Silicon — Swift nativo con MLX](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **23 feb 2026** — [NVIDIA PersonaPlex 7B en Apple Silicon — Voz a voz full-duplex en Swift nativo con MLX](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **12 feb 2026** — [Qwen3-ASR Swift: ASR + TTS en el dispositivo para Apple Silicon — Arquitectura y benchmarks](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## Inicio rápido

Añade el paquete a tu `Package.swift`:

```swift
.package(url: "https://github.com/soniqo/speech-swift", branch: "main")
```

Importa solo los módulos que necesites — cada modelo es una librería SPM independiente, así no pagas por lo que no uses:

```swift
.product(name: "ParakeetStreamingASR", package: "speech-swift"),
.product(name: "SpeechUI",             package: "speech-swift"),  // vistas SwiftUI opcionales
```

**Transcribe un buffer de audio en 3 líneas:**

```swift
import ParakeetStreamingASR

let model = try await ParakeetStreamingASRModel.fromPretrained()
let text = try model.transcribeAudio(audioSamples, sampleRate: 16000)
```

**Streaming en vivo con resultados parciales:**

```swift
for await partial in model.transcribeStream(audio: samples, sampleRate: 16000) {
    print(partial.isFinal ? "FINAL: \(partial.text)" : "... \(partial.text)")
}
```

**Vista de dictado SwiftUI en ~10 líneas:**

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

`SpeechUI` solo incluye `TranscriptionView` (finales + parciales) y `TranscriptionStore` (adaptador de ASR en streaming). Usa AVFoundation para la visualización y reproducción de audio.

Productos SPM disponibles: `Qwen3ASR`, `WhisperASR`, `Qwen3TTS`, `Qwen3TTSCoreML`, `ParakeetASR`, `ParakeetStreamingASR`, `NemotronStreamingASR`, `OmnilingualASR`, `KokoroTTS`, `SupertonicTTS`, `VibeVoiceTTS`, `CosyVoiceTTS`, `VoxCPM2TTS`, `IndexTTS2TTS`, `F5TTS`, `HiggsTTS`, `ChatterboxTTS`, `OmniVoiceTTS`, `IndicMioTTS`, `FishAudioTTS`, `MagpieTTS`, `MagpieTTSCoreML`, `MAGNeTMusicGen`, `StableAudio3MusicGen`, `FlashSR`, `PersonaPlex`, `Audio2Face3D`, `HibikiTranslate`, `MADLADTranslation`, `SpeechVAD`, `SpeechWakeWord`, `SpeechEnhancement`, `SpeechRestoration`, `SourceSeparation`, `Qwen3Chat`, `FunctionGemma`, `SpeechCore`, `SpeechUI`, `AudioCommon`.

## Modelos

Vista compacta a continuación. **[Catálogo completo de modelos con tamaños, cuantizaciones, URLs de descarga y tablas de memoria → soniqo.audio/architecture](https://soniqo.audio/es/architecture)**.

| Modelo | Tarea | Backends | Tamaños | Idiomas |
|-------|------|----------|-------|-----------|
| [Qwen3-ASR](https://soniqo.audio/es/guides/transcribe) | Voz → Texto | MLX, CoreML (híbrido) | 0.6B, 1.7B | 52 |
| [WhisperASR](docs/models/whisper-asr.md) | Speech → Text | CoreML (ANE) | Large-v3 Turbo | Multi |
| [Parakeet TDT](https://soniqo.audio/es/guides/parakeet) | Voz → Texto | CoreML (ANE) | 0.6B | 25 europeos |
| [Parakeet EOU](https://soniqo.audio/es/guides/dictate) | Voz → Texto (streaming) | CoreML (ANE) | 120M | 25 europeos |
| [Nemotron Streaming (Multilingüe)](https://soniqo.audio/es/guides/nemotron) | Voz → Texto (streaming, con puntuación) | CoreML (ANE), MLX | 0.6B | **40** |
| [Nemotron Streaming (Inglés)](https://soniqo.audio/guides/nemotron) | Voz → Texto (streaming, con puntuación) | CoreML (ANE) | 0.6B | EN |
| [Omnilingual ASR](https://soniqo.audio/es/guides/omnilingual) | Voz → Texto | CoreML (ANE), MLX | 300M / 1B / 3B / 7B | **[1.672](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)** |
| [Qwen3-ForcedAligner](https://soniqo.audio/es/guides/align) | Audio + Texto → Marcas temp. | MLX, CoreML | 0.6B | Multi |
| [Qwen3-TTS](https://soniqo.audio/es/guides/speak) | Texto → Voz | MLX, CoreML | 0.6B, 1.7B | 10 |
| [CosyVoice3](https://soniqo.audio/es/guides/cosyvoice) | Texto → Voz | MLX | 0.5B | 9 |
| [VoxCPM2](https://soniqo.audio/es/speech-generation) | Texto → Voz (48 kHz, diseño de voz + clonación) | MLX | 2B (bf16/int8) | 30 |
| [IndexTTS2](docs/models/indextts2.md) | Text → Speech (zero-shot voice cloning) | MLX | 1.5B-class (fp16) | EN/ZH |
| [F5-TTS](docs/models/f5-tts.md) | Text → Speech (zero-shot voice cloning) | MLX | 336M (fp16) | EN/ZH |
| [Higgs TTS 3](docs/models/higgs-tts.md) | Text → Speech (conversational, zero-shot voice cloning) | MLX | 4B (bf16) | 100+ |
| [Chatterbox Multilingual](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16) | Texto → Voz (clonación zero-shot) | MLX | 0.8B (fp16) | 23 (HE requiere niqqud) |
| [OmniVoice](https://huggingface.co/aufklarer/OmniVoice-MLX-fp16) | Texto → Voz (difusión NAR, clonación zero-shot) | MLX | 0.8B (fp16 por defecto / int8) | 600+ |
| [Indic-Mio](docs/models/indic-mio-tts.md) | Text → Speech (Hindi/Indic, emotion tags, voice cloning) | MLX | fp16 | Hindi / Indic |
| [Fish Audio S2 Pro](docs/models/fish-audio-s2-pro.md) | Texto → Voz (clonación zero-shot, marcadores de estilo explícitos) | MLX | 0.5B-class (fp16) | Multilingüe |
| [Kokoro-82M](https://soniqo.audio/es/guides/kokoro) | Texto → Voz | CoreML (ANE) | 82M | 10 |
| [Supertonic-3](https://soniqo.audio/guides/supertonic) | Texto → Voz (44.1 kHz, flow-matching, sin G2P) | CoreML (ANE) | 99M | 31 |
| [VibeVoice Realtime-0.5B](https://soniqo.audio/es/guides/vibevoice) | Texto → Voz (formato largo, múltiples hablantes) | MLX | 0.5B | EN/ZH |
| [VibeVoice 1.5B](https://soniqo.audio/es/guides/vibevoice) | Texto → Voz (podcast de hasta 90 min) | MLX | 1.5B | EN/ZH |
| [Magpie-TTS Multilingual](https://soniqo.audio/es/guides/magpie) | Texto → Voz (5 hablantes preconfigurados, streaming) | MLX / CoreML | 357M (MLX INT8, CoreML INT8) | 9 (CoreML excluye JA) |
| [Qwen3.5 Chat](docs/models/qwen35-chat.md) | Text → Text (LLM) | MLX, CoreML | 0.8B | Multi |
| [Qwen3 Dense Chat](docs/models/qwen3-dense-chat.md) | Text → Text (LLM) | MLX | 4B | Multi |
| [Gemma 4 Chat](docs/models/gemma4-chat.md) | Text → Text (LLM) | MLX | E2B / E4B (4-bit) | Multi |
| [FunctionGemma](https://soniqo.audio/es/guides/function-calls) | Texto → Llamadas a herramientas (LLM) | CoreML | 270M | EN |
| [MADLAD-400](https://soniqo.audio/es/guides/translate) | Texto → Texto (Traducción) | MLX | 3B | **400+** |
| [Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate) | Voz → Voz (Traducción) | MLX | 3B | FR/ES/PT/DE → EN |
| [PersonaPlex](https://soniqo.audio/es/guides/respond) | Voz → Voz | MLX | 7B | EN |
| [Audio2Face-3D](docs/models/audio2face3d.md) | Voz → Animación facial | MLX | v2.3 Mark | Agnóstico |
| [Silero VAD](https://soniqo.audio/es/guides/vad) | Detección de actividad vocal | MLX, CoreML | 309K | Agnóstico |
| [KWS Zipformer](docs/models/kws-zipformer.md) | Audio → Wake word | CoreML (ANE) | 3M | EN/custom keywords |
| [Pyannote](https://soniqo.audio/es/guides/diarize) | VAD + Diarización | MLX | 1.5M | Agnóstico |
| [Pyannote Community-1](https://huggingface.co/aufklarer/Pyannote-Community-1-CoreML) | Diarización + embeddings de hablante | CoreML (ANE) + Swift VBx | 8.35M | Agnóstico |
| [Sortformer](https://huggingface.co/aufklarer/Sortformer-Diarization-CoreML) | [Diarización (E2E), streaming incremental](https://soniqo.audio/es/guides/diarize) | CoreML (ANE) | 117M | Agnóstico |
| [DeepFilterNet3](https://soniqo.audio/es/guides/denoise) | Mejora de voz | CoreML | 2.1M | Agnóstico |
| [Sidon](https://soniqo.audio/es/guides/restore) | Restauración de voz (supresión de ruido + dereverberación, 48 kHz) | CoreML | w2v-BERT 2.0 + DAC (fp16/int8) | Agnóstico |
| [HTDemucs (Demucs v4)](https://soniqo.audio/es/guides/separate) | Separación de fuentes | MLX | 168M | Agnostic |
| [Open-Unmix](https://soniqo.audio/es/guides/separate) | Separación de fuentes | MLX | 8.6M | Agnostic |
| [MAGNeT](https://soniqo.audio/es/guides/compose) | Texto → Música (30 s @ 32 kHz) | MLX | 300M / 1.5B (int4/int8) | Prompts en EN |
| [Stable Audio 3](docs/models/stable-audio-3.md) | Text → Music/audio (44.1 kHz stereo) | MLX | Medium 1.4B (int4/int8) | EN prompts |
| [FlashSR](https://soniqo.audio/es/guides/upsample) | Super-resolución de audio (48 kHz) | MLX | 363 MB / 720 MB (int4/int8) | Agnóstico |
| [WeSpeaker](https://soniqo.audio/es/guides/embed-speaker) | Embedding de hablante | MLX, CoreML | 6.6M | Agnóstico |
| [ReDimNet2-B6](https://huggingface.co/aufklarer/ReDimNet2-B6-CoreML) | Identidad de voz con nombre | CoreML | 12.3M | Agnóstico |

## Instalación

### Homebrew

[![Homebrew installs](https://img.shields.io/homebrew/installs/dm/speech.svg?logo=homebrew&label=Homebrew%20installs&color=FBB040)](https://formulae.brew.sh/formula/speech)

Requiere Homebrew ARM nativo (`/opt/homebrew`). Homebrew Rosetta/x86_64 no está soportado.

```bash
brew install speech
```

Luego:

```bash
speech transcribe recording.wav
speech speak "Hello world"
speech translate "Hello, how are you?" --to es
speech respond --input question.wav --transcript
speech-server --port 8080            # servidor HTTP / WebSocket local (OpenAI-compatible /v1/realtime + /v1/audio/transcriptions)
```

**[Referencia completa de la CLI →](https://soniqo.audio/es/cli)**

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

Importa solo lo que necesites — cada modelo es su propio target SPM:

```swift
import Qwen3ASR             // Reconocimiento de voz (MLX)
import WhisperASR           // Whisper Large-v3 Turbo (CoreML)
import ParakeetASR          // Reconocimiento de voz (CoreML, batch)
import ParakeetStreamingASR // Dictado en streaming con parciales + EOU
import NemotronStreamingASR // ASR streaming multilingüe con puntuación nativa (0.6B, 40 idiomas)
import OmnilingualASR       // 1.672 idiomas (CoreML + MLX)
import Qwen3TTS             // Síntesis de voz
import CosyVoiceTTS         // Síntesis de voz con clonación
import VoxCPM2TTS           // TTS a 48 kHz, clonación de voz + diseño de voz (2B)
import IndexTTS2TTS         // Native MLX voice cloning from reference audio
import F5TTS                // Zero-shot voice cloning (DiT flow matching + Vocos)
import HiggsTTS             // Conversational TTS + cloning (Qwen3 backbone, control tags)
import KokoroTTS            // Síntesis de voz (listo para iOS)
import VibeVoiceTTS         // TTS de formato largo / múltiples hablantes (EN/ZH)
import MagpieTTS            // TTS multilingüe (NVIDIA Magpie 357M, MLX, 9 idiomas)
import MagpieTTSCoreML      // Backend CoreML de Magpie (híbrido CoreML + MLX, 8 idiomas)
import FishAudioTTS         // Runtime experimental Fish Audio S2 Pro con clonación de voz
import Qwen3Chat            // Chat LLM en el dispositivo
import FunctionGemma    // LLM de llamada a herramientas en el dispositivo
import MADLADTranslation    // Traducción multidireccional entre 400+ idiomas
import HibikiTranslate      // Traducción voz a voz en streaming (FR/ES/PT/DE → EN)
import PersonaPlex          // Voz a voz full-duplex
import SpeechVAD            // VAD + diarización + embeddings
import SpeechEnhancement    // Supresión de ruido
import SpeechRestoration    // Restauración de voz — supresión de ruido + dereverberación (Sidon, CoreML, 48 kHz)
import SourceSeparation     // Separación de fuentes musicales (Open-Unmix, 4 stems)
import MAGNeTMusicGen      // Generación de música desde texto (30 s, 32 kHz)
import FlashSR             // Super-resolución de audio (48 kHz, difusión en 1 paso)
import SpeechUI             // Componentes SwiftUI para transcripciones en streaming
import AudioCommon          // Protocolos y utilidades compartidas
```

### Requisitos

- Swift 6+, Xcode 16+ (con Metal Toolchain)
- macOS 15+ (Sequoia) o iOS 18+, Apple Silicon (M1/M2/M3/M4)

El mínimo de macOS 15 / iOS 18 proviene de [MLState](https://developer.apple.com/documentation/coreml/mlstate) —— la API de estado persistente de ANE de Apple —— que los pipelines CoreML (Qwen3-ASR, Qwen3-Chat, Qwen3-TTS) usan para mantener las cachés KV residentes en el Neural Engine entre pasos de token.

### Compilar desde el código fuente

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

`make build` compila el paquete Swift **y** la librería de shaders MLX Metal. La librería Metal es necesaria para la inferencia en GPU — sin ella verás `Failed to load the default metallib` en tiempo de ejecución. `make debug` para builds de depuración, `make test` para la suite de pruebas.

**[Guía completa de compilación e instalación →](https://soniqo.audio/es/getting-started)**

## Aplicaciones de demostración

- **[DictateDemo](Examples/DictateDemo/)** ([documentación](https://soniqo.audio/es/guides/dictate)) — Dictado en streaming en la barra de menús de macOS con parciales en vivo, detección de fin de enunciado basada en VAD y copia con un clic. Se ejecuta como agent en segundo plano (Parakeet-EOU-120M + Silero VAD).
- **[iOSEchoDemo](Examples/iOSEchoDemo/)** — Demo de eco iOS (Parakeet ASR + Kokoro TTS). Dispositivo y simulador.
- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — Asistente de voz conversacional con entrada de micrófono, VAD y contexto multi-turno. macOS. RTF ~0.94 en M2 Max (más rápido que tiempo real).
- **[SpeechDemo](Examples/SpeechDemo/)** — Dictado y síntesis TTS en una interfaz de pestañas. macOS.

El README de cada demo tiene instrucciones de compilación.

## Ejemplos de código

Los fragmentos siguientes muestran el camino mínimo para cada dominio. Cada sección enlaza a una guía completa en [soniqo.audio](https://soniqo.audio/es) con opciones de configuración, múltiples backends, patrones de streaming y recetas de CLI.

### Voz a texto — [guía completa →](https://soniqo.audio/es/guides/transcribe)

```swift
import Qwen3ASR

let model = try await Qwen3ASRModel.fromPretrained()
let text = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

Backends alternativos: [WhisperASR](docs/inference/whisper-asr-inference.md) (Whisper Large-v3 Turbo, native CoreML), [Parakeet TDT](https://soniqo.audio/es/guides/parakeet) (CoreML, 32× tiempo real), [Omnilingual ASR](https://soniqo.audio/es/guides/omnilingual) (1.672 idiomas, CoreML o MLX), [Dictado en streaming](https://soniqo.audio/es/guides/dictate) (parciales en vivo).

### Alineación forzada — [guía completa →](https://soniqo.audio/es/guides/align)

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

### Texto a voz — [guía completa →](https://soniqo.audio/es/guides/speak)

```swift
import Qwen3TTS
import AudioCommon

let model = try await Qwen3TTSModel.fromPretrained()
let audio = model.synthesize(text: "Hello world", language: "english")
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

Motores TTS alternativos: [CosyVoice3](https://soniqo.audio/es/guides/cosyvoice) (streaming + clonación + etiquetas de emoción), [Kokoro-82M](https://soniqo.audio/es/guides/kokoro) (listo para iOS, 54 voces), [VibeVoice](https://soniqo.audio/es/guides/vibevoice) (podcast de formato largo / múltiples hablantes, EN/ZH), [Fish Audio S2 Pro](docs/inference/fish-audio-s2-pro.md) (clonación zero-shot experimental + marcadores de estilo entre corchetes), [Clonación de voz](https://soniqo.audio/es/guides/voice-cloning).

### Voz a voz — [guía completa →](https://soniqo.audio/es/guides/respond)

```swift
import PersonaPlex

let model = try await PersonaPlexModel.fromPretrained()
let responseAudio = model.respond(userAudio: userSamples)
// Salida Float32 mono 24 kHz lista para reproducir
```

### Chat LLM — [guía completa →](https://soniqo.audio/es/guides/chat)

```swift
import Qwen3Chat
import FunctionGemma

let chat = try await Qwen35MLXChat.fromPretrained()
chat.chat(messages: [(.user, "Explain MLX in one sentence")]) { token, isFinal in
    print(token, terminator: "")
}
```

### Traducción — [guía completa →](https://soniqo.audio/es/guides/translate)

```swift
import MADLADTranslation

let translator = try await MADLADTranslator.fromPretrained()
let es = try translator.translate("Hello, how are you?", to: "es")
// → "Hola, ¿cómo estás?"
```

### Traducción de voz — [guía completa →](https://soniqo.audio/guides/audio-translate)

```swift
import HibikiTranslate
import AudioCommon

let model = try await HibikiTranslateModel.fromPretrained()
let pcm = try AudioFileLoader.load(url: input, targetSampleRate: 24000)
let (englishAudio, textTokens) = model.translate(
    sourceAudio: pcm, sourceLanguage: .fr
)
// Hibiki Zero-3B — FR/ES/PT/DE → EN, en el dispositivo, códec Mimi en streaming
```

### Detección de actividad vocal — [guía completa →](https://soniqo.audio/es/guides/vad)

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
for s in segments { print("\(s.startTime)s → \(s.endTime)s") }
```

### Diarización de hablantes — [guía completa →](https://soniqo.audio/es/guides/diarize)

```swift
import SpeechVAD

let diarizer = try await DiarizationPipeline.fromPretrained()
let segments = diarizer.diarize(audio: samples, sampleRate: 16000)
for s in segments { print("Speaker \(s.speakerId): \(s.startTime)s - \(s.endTime)s") }
```

### Mejora de voz — [guía completa →](https://soniqo.audio/es/guides/denoise)

```swift
import SpeechEnhancement

let denoiser = try await DeepFilterNet3Model.fromPretrained()
let clean = try denoiser.enhance(audio: noisySamples, sampleRate: 48000)
```

### Restauración de voz — [guía completa →](https://soniqo.audio/es/guides/restore)

Supresión de ruido **y** dereverberación conjuntas con [Sidon](https://arxiv.org/abs/2509.17052) (predictor w2v-BERT 2.0 + vocoder DAC, Core ML). A diferencia de un supresor de ruido genérico, Sidon está entrenado para preservar la identidad del hablante, por lo que es ideal para limpiar una referencia de clonación de voz ruidosa o con reverberación antes del TTS. La entrada es de 16 kHz; la salida es mono a 48 kHz.

```swift
import SpeechRestoration

let restorer = try await SpeechRestorer.fromPretrained()          // .fp16 (default) or .int8
let clean = try restorer.restore(audio: noisySamples, sampleRate: 16000)  // → 48 kHz
```

Desde la CLI:

```bash
speech restore noisy.wav -o clean.wav            # denoise + dereverb, 48 kHz output
speech restore noisy.wav --variant int8          # smaller, lower peak RAM

# Clean a voice-cloning reference before TTS (opt-in; preserves speaker identity):
speech speak "Hello" --engine voxcpm2 --voice-sample ref.wav --clean-reference
```

### Pipeline de voz (ASR → LLM → TTS) — [guía completa →](https://soniqo.audio/es/voice-agents)

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

`VoicePipeline` es la máquina de estados de agent de voz en tiempo real (impulsada por [speech-core](https://github.com/soniqo/speech-core)) con detección de turnos basada en VAD, manejo de interrupciones y STT eager. Conecta cualquier `SpeechRecognitionModel` + `SpeechGenerationModel` + `StreamingVADProvider`.

### Servidor API HTTP

```bash
speech-server --port 8080
```

Expone cada modelo a través de endpoints HTTP REST + WebSocket, incluyendo APIs compatibles con OpenAI: un WebSocket Realtime en `/v1/realtime` y un endpoint REST de transcripción en `/v1/audio/transcriptions`. Ver [`Sources/AudioServer/`](Sources/AudioServer/).

## Arquitectura

speech-swift está dividido en un target SPM por modelo para que los consumidores solo paguen por lo que importan. La infraestructura compartida vive en `AudioCommon` (protocolos, E/S de audio, descargador de HuggingFace, `SentencePieceModel`) y `MLXCommon` (carga de pesos, helpers `QuantizedLinear`, helper de atención multi-head `SDPA`).

**[Diagrama completo de arquitectura con backends, tablas de memoria y mapa de módulos → soniqo.audio/architecture](https://soniqo.audio/es/architecture)** · **[Referencia de API → soniqo.audio/api](https://soniqo.audio/es/api)** · **[Benchmarks → soniqo.audio/benchmarks](https://soniqo.audio/es/benchmarks)**

Docs locales (repositorio):
- **Modelos:** [Qwen3-ASR](docs/models/asr-model.md) · [WhisperASR](docs/models/whisper-asr.md) · [Qwen3-TTS](docs/models/tts-model.md) · [CosyVoice](docs/models/cosyvoice-tts.md) · [Kokoro](docs/models/kokoro-tts.md) · [VoxCPM2](docs/models/voxcpm2-tts.md) · [IndexTTS2](docs/models/indextts2.md) · [F5-TTS](docs/models/f5-tts.md) · [Higgs TTS 3](docs/models/higgs-tts.md) · [VibeVoice](docs/models/vibevoice.md) · [Supertonic](docs/models/supertonic-tts.md) · [Chatterbox](docs/models/chatterbox-tts.md) · [Indic-Mio](docs/models/indic-mio-tts.md) · [Fish Audio S2 Pro](docs/models/fish-audio-s2-pro.md) · [Magpie TTS](docs/models/magpie-tts.md) · [Parakeet TDT](docs/models/parakeet-asr.md) · [Parakeet Streaming](docs/models/parakeet-streaming-asr.md) · [Nemotron Streaming](docs/models/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/models/omnilingual-asr.md) · [PersonaPlex](docs/models/personaplex.md) · [Hibiki](docs/models/hibiki.md) · [MADLAD-400](docs/models/madlad-translation.md) · [FunctionGemma](docs/models/function-gemma.md) · [Qwen3.5 Chat](docs/models/qwen35-chat.md) · [Gemma 4 Chat](docs/models/gemma4-chat.md) · [Qwen3 Dense Chat](docs/models/qwen3-dense-chat.md) · [FireRedVAD](docs/models/fireredvad.md) · [KWS Zipformer](docs/models/kws-zipformer.md) · [Sidon](docs/models/sidon.md) · [Source Separation](docs/models/source-separation.md) · [HTDemucs](docs/models/htdemucs.md) · [MAGNeT](docs/models/magnet-music-gen.md) · [Stable Audio 3](docs/models/stable-audio-3.md) · [FlashSR](docs/models/flashsr.md) · [Audio2Face-3D](docs/models/audio2face3d.md)
- **Inferencia:** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md) · [WhisperASR](docs/inference/whisper-asr-inference.md) · [Parakeet TDT](docs/inference/parakeet-asr-inference.md) · [Parakeet Streaming](docs/inference/parakeet-streaming-asr-inference.md) · [Nemotron Streaming](docs/inference/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/inference/omnilingual-asr-inference.md) · [TTS](docs/inference/qwen3-tts-inference.md) · [VoxCPM2](docs/inference/voxcpm2-inference.md) · [IndexTTS2](docs/inference/indextts2.md) · [F5-TTS](docs/inference/f5-tts.md) · [Higgs TTS 3](docs/inference/higgs-tts.md) · [VibeVoice](docs/inference/vibevoice-inference.md) · [Fish Audio S2 Pro](docs/inference/fish-audio-s2-pro.md) · [Magpie TTS](docs/inference/magpie-tts.md) · [Hibiki](docs/inference/hibiki-inference.md) · [MADLAD-400](docs/inference/madlad-translation.md) · [MAGNeT](docs/inference/magnet-music-gen.md) · [Stable Audio 3](docs/inference/stable-audio-3.md) · [FlashSR](docs/inference/flashsr.md) · [Forced Aligner](docs/inference/forced-aligner.md) · [Silero VAD](docs/inference/silero-vad.md) · [FireRedVAD](docs/inference/fireredvad.md) · [Wake-word](docs/inference/wake-word.md) · [Speaker Diarization](docs/inference/speaker-diarization.md) · [Speech Enhancement](docs/inference/speech-enhancement.md) · [Sidon](docs/inference/sidon.md) · [Cache/offline](docs/inference/cache-and-offline.md)
- **Referencia:** [Protocolos compartidos](docs/shared-protocols.md)

## Configuración de caché

Los pesos del modelo se descargan desde HuggingFace en el primer uso y se almacenan en `~/Library/Caches/qwen3-speech/`. Puedes sobrescribir con `QWEN3_CACHE_DIR` (CLI) o `cacheDir:` (API Swift). Todos los puntos de entrada `fromPretrained()` aceptan `offlineMode: true` para omitir la red cuando los pesos ya están en caché.

Los usuarios en China continental (o donde `huggingface.co` sea lento o esté bloqueado) pueden descargar desde un espejo configurando `HF_ENDPOINT`, p. ej. `export HF_ENDPOINT=https://hf-mirror.com`.

Consulta [`docs/inference/cache-and-offline.md`](docs/inference/cache-and-offline.md) para los detalles completos, incluyendo rutas de contenedor iOS sandboxed.

## Librería MLX Metal

Si ves `Failed to load the default metallib` en tiempo de ejecución, falta la librería de shaders Metal. Ejecuta `make build` o `./scripts/build_mlx_metallib.sh release` después de un `swift build` manual. Si falta el Metal Toolchain, instálalo primero:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Pruebas

```bash
make test                            # suite completa (unidad + E2E con descargas de modelos)
swift test --skip E2E                # solo unidad (seguro para CI, sin descargas)
swift test --filter Qwen3ASRTests    # módulo específico
```

Las clases de test E2E usan el prefijo `E2E` para que CI pueda filtrarlas con `--skip E2E`. Consulta [CLAUDE.md](CLAUDE.md#testing) para la convención completa de pruebas.

## Contribuir

PRs bienvenidos — correcciones de bugs, integraciones de nuevos modelos, documentación. Fork, crea una rama de feature, `make build && make test`, abre un PR contra `main`.

## Licencia

Apache 2.0
