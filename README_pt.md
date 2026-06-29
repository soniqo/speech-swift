# Speech Swift

Modelos de IA para fala em Apple Silicon, com tecnologia MLX Swift e CoreML.

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

Reconhecimento, sintese e compreensao de fala no dispositivo para Mac e iOS. Executa localmente no Apple Silicon — sem nuvem, sem chaves de API, nenhum dado sai do dispositivo.

**[📚 Documentacao completa →](https://soniqo.audio/pt)** · **[🤗 Modelos no HuggingFace](https://huggingface.co/aufklarer)** · **[📝 Blog](https://blog.ivan.digital)** · **[💬 Discord](https://discord.gg/TnCryqEMgu)**

<p align="center">
  <a href="https://trendshift.io/repositories/24196?utm_source=trendshift-badge&amp;utm_medium=badge&amp;utm_campaign=badge-trendshift-24196" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/trendshift/repositories/24196/daily?language=Swift" alt="soniqo%2Fspeech-swift | Trendshift" width="250" height="55"/></a>
</p>

<p align="center">
  <a href="https://youtu.be/x9zgcaW0gUk">
    <img src="https://img.youtube.com/vi/x9zgcaW0gUk/maxresdefault.jpg" width="640" alt="IA de voz local em um MacBook — assista no YouTube ao tour de quatro minutos pela biblioteca open source">
  </a>
</p>
<p align="center"><em>IA de voz local em um MacBook — assista no YouTube ao tour de quatro minutos pela biblioteca open source</em></p>

**Casos de uso:** [Agentes de voz](https://soniqo.audio/pt/voice-agents) · [Transcricao](https://soniqo.audio/pt/transcription) · [Sintese de voz](https://soniqo.audio/pt/speech-generation)

**Grupos de capacidades:** STT / ASR · Alinhamento · TTS · LLMs e tradução · Fala-para-fala · Aprimoramento / restauração · Separação de fontes · Geração de música / áudio · Wake word, VAD, diarização e identidade de falante

**STT / ASR**

- **[Qwen3-ASR](https://soniqo.audio/pt/guides/transcribe)** — Fala para texto (reconhecimento automatico de fala, 52 idiomas, MLX + CoreML)
- **[Parakeet TDT](https://soniqo.audio/pt/guides/parakeet)** — Fala para texto via CoreML (Neural Engine, NVIDIA FastConformer + decodificador TDT, 25 idiomas)
- **[Omnilingual ASR](https://soniqo.audio/pt/guides/omnilingual)** — Fala para texto (Meta wav2vec2 + CTC, **1.672 idiomas** em 32 escritas, CoreML 300M + MLX 300M/1B/3B/7B)
- **[Ditado em streaming](https://soniqo.audio/pt/guides/dictate)** — Ditado em tempo real com resultados parciais e deteccao de fim de enunciado (Parakeet-EOU-120M)
- **[Nemotron Streaming (Multilíngue)](https://soniqo.audio/pt/guides/nemotron)** — ASR de streaming de baixa latência com pontuação e capitalização nativas (NVIDIA Nemotron-3.5-ASR-Streaming-0.6B, CoreML + MLX, **40 idiomas-localidades**)
- **[Nemotron Streaming (Inglês)](https://soniqo.audio/guides/nemotron)** — ASR de streaming de baixa latência com pontuação e capitalização nativas (NVIDIA Nemotron-Speech-Streaming-0.6B, CoreML, apenas inglês, menor e mais rápido que a variante multilíngue)

**Alinhamento**

- **[Qwen3-ForcedAligner](https://soniqo.audio/pt/guides/align)** — Alinhamento de timestamps por palavra (audio + texto → timestamps)

**TTS / Síntese de voz**

- **[Qwen3-TTS](https://soniqo.audio/pt/guides/speak)** — Sintese de texto para fala (mais alta qualidade, streaming, locutores personalizados, 10 idiomas)
- **[CosyVoice TTS](https://soniqo.audio/pt/guides/cosyvoice)** — TTS em streaming com clonagem de voz, dialogo multi-locutor, tags de emocao (9 idiomas)
- **[VoxCPM2](https://soniqo.audio/pt/speech-generation)** — TTS de qualidade de estudio a 48 kHz com clonagem de voz e design de voz baseado em instrucoes (2B, MLX bf16/int8, 30 idiomas)
- **[Kokoro TTS](https://soniqo.audio/pt/guides/kokoro)** — TTS no dispositivo (82M, CoreML/Neural Engine, 54 vozes, pronto para iOS, 10 idiomas)
- **[VibeVoice TTS](https://soniqo.audio/pt/guides/vibevoice)** — TTS de formato longo / multi-alto-falante (Microsoft VibeVoice Realtime-0.5B + 1.5B, MLX, sintese de podcast/audiolivro de ate 90 min, EN/ZH)
- **[Magpie TTS](https://soniqo.audio/pt/guides/magpie)** — TTS multilíngue (NVIDIA Magpie-TTS Multilingual 357M, MLX INT8 411 MB ou CoreML INT8 342 MB, 9 idiomas, 5 oradores predefinidos, streaming em MLX)
- **[Supertonic TTS](https://soniqo.audio/guides/supertonic)** — TTS de flow-matching no dispositivo (Supertone Supertonic-3 99M, CoreML/Neural Engine, 31 idiomas, 10 vozes, sem G2P, 44.1 kHz)
- **[Chatterbox TTS](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16)** — TTS multilíngue com clonagem de voz zero-shot (Resemble AI Chatterbox Multilingual, MLX fp16 ~1,3 GB, 23 idiomas, MIT)
- **[OmniVoice TTS](https://huggingface.co/aufklarer/OmniVoice-MLX-int8)** — TTS de difusao nao-autorregressiva com clonagem de voz zero-shot (k2-fsa OmniVoice, backbone Qwen3, MLX int8 ~1 GB / fp16, 600+ idiomas, Apache-2.0)
- **[Indic-Mio](docs/models/indic-mio-tts.md)** — Hindi/Indic TTS with inline emotion markers and optional reference-voice cloning (MLX, 24 kHz)

**LLMs e tradução**

- **[Qwen3Chat](https://soniqo.audio/pt/guides/chat)** — Chat LLM no dispositivo (Qwen3.5-0.8B MLX/CoreML mais backends MLX dense Qwen3 4B e Gemma 4 E2B/E4B, tokens em streaming)
- **[FunctionGemma](https://soniqo.audio/pt/guides/function-calls)** — LLM no dispositivo para chamadas estruturadas de funcoes / ferramentas (Gemma 3 270M, CoreML paletizado de 8 bits, Neural Engine, ~252 tok/s)
- **[MADLAD-400](https://soniqo.audio/pt/guides/translate)** — Tradução multidirecional entre 400+ idiomas (3B, MLX INT4 + INT8, T5 v1.1, Apache 2.0)

**Fala-para-fala e agentes de voz**

- **[Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate)** — Tradução de fala para fala em streaming (FR/ES/PT/DE → EN, MLX INT4 + INT8, stack Kyutai Moshi/Mimi, CC-BY-4.0)
- **[PersonaPlex](https://soniqo.audio/pt/guides/respond)** — Fala-a-fala full-duplex (7B, audio de entrada → audio de saida, 18 presets de voz)

**Aprimoramento, separação e geração de áudio**

- **[DeepFilterNet3](https://soniqo.audio/pt/guides/denoise)** — Supressao de ruido em tempo real (2.1M parametros, 48 kHz). Audio longo acima do limite de 60 s em uma unica passagem e dividido automaticamente em blocos com crossfade — veja `enhanceChunked(...)`
- **[Separação de fontes](https://soniqo.audio/pt/guides/separate)** — Separação de fontes musicais com HTDemucs (Demucs v4) + Open-Unmix (UMX-HQ / UMX-L, 4 stems: vocais/bateria/baixo/outros, 44,1 kHz estéreo)
- **[MAGNeT](https://soniqo.audio/pt/guides/compose)** — Geração de música a partir de texto (Meta MAGNeT Small 300M / Medium 1.5B, MLX INT8, clipes de 30 s a 32 kHz mono, decodificação mascarada paralela)
- **[Stable Audio 3](docs/models/stable-audio-3.md)** — Text-to-audio/music generation (Stable Audio 3 Medium, MLX INT8/INT4, 44.1 kHz stereo, variable length)
- **[FlashSR](https://soniqo.audio/pt/guides/upsample)** — Super-resolução de áudio (FlashSR ICASSP 2025, MLX, 48 kHz mono, difusão destilada em 1 passo, INT4 363 MB / INT8 720 MB)

**Detecção de turnos, diarização e identidade de falante**

- **[Palavra de ativacao](https://soniqo.audio/pt/guides/wake-word)** — Deteccao de palavras-chave no dispositivo (KWS Zipformer 3M, CoreML, 26x tempo real, lista de palavras-chave configuravel)
- **[VAD](https://soniqo.audio/pt/guides/vad)** — Deteccao de atividade de voz (Silero streaming, Pyannote offline, FireRedVAD 100+ idiomas)
- **[Diarizacao de falantes](https://soniqo.audio/pt/guides/diarize)** — Quem falou quando (pipeline Pyannote, Sortformer ponta-a-ponta no Neural Engine)
- **[Embeddings de falante](https://soniqo.audio/pt/guides/embed-speaker)** — WeSpeaker ResNet34 (256 dim), CAM++ (192 dim)

Papers: [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba) · [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba) · [Omnilingual ASR](https://arxiv.org/abs/2511.09690) (Meta) · [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA) · [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba) · [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2) · [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA) · [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai) · [Hibiki](https://arxiv.org/abs/2502.03382) (Kyutai) · [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## Novidades

- **19 Abr 2026** — [MLX vs CoreML no Apple Silicon — guia prático para escolher o backend certo](https://blog.ivan.digital/mlx-vs-coreml-on-apple-silicon-a-practical-guide-to-picking-the-right-backend-and-why-you-should-f77ddea7b27a)
- **20 Mar 2026** — [Superamos o Whisper Large v3 com um modelo de 600M rodando inteiramente no seu Mac](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **26 Fev 2026** — [Diarizacao de falantes e deteccao de atividade de voz em Apple Silicon — Swift nativo com MLX](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **23 Fev 2026** — [NVIDIA PersonaPlex 7B em Apple Silicon — fala-a-fala full-duplex em Swift nativo com MLX](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **12 Fev 2026** — [Qwen3-ASR Swift: ASR + TTS no dispositivo para Apple Silicon — arquitetura e benchmarks](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## Inicio rapido

Adicione o pacote ao seu `Package.swift`:

```swift
.package(url: "https://github.com/soniqo/speech-swift", branch: "main")
```

Importe apenas os modulos que voce precisa — cada modelo e uma biblioteca SPM independente, entao voce nao paga pelo que nao usa:

```swift
.product(name: "ParakeetStreamingASR", package: "speech-swift"),
.product(name: "SpeechUI",             package: "speech-swift"),  // views SwiftUI opcionais
```

**Transcrever um buffer de audio em 3 linhas:**

```swift
import ParakeetStreamingASR

let model = try await ParakeetStreamingASRModel.fromPretrained()
let text = try model.transcribeAudio(audioSamples, sampleRate: 16000)
```

**Streaming ao vivo com resultados parciais:**

```swift
for await partial in model.transcribeStream(audio: samples, sampleRate: 16000) {
    print(partial.isFinal ? "FINAL: \(partial.text)" : "... \(partial.text)")
}
```

**View de ditado SwiftUI em ~10 linhas:**

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

`SpeechUI` inclui apenas `TranscriptionView` (finais + parciais) e `TranscriptionStore` (adaptador de ASR em streaming). Use AVFoundation para visualizacao e reproducao de audio.

Produtos SPM disponiveis: `Qwen3ASR`, `Qwen3TTS`, `Qwen3TTSCoreML`, `ParakeetASR`, `ParakeetStreamingASR`, `NemotronStreamingASR`, `OmnilingualASR`, `KokoroTTS`, `SupertonicTTS`, `VibeVoiceTTS`, `CosyVoiceTTS`, `VoxCPM2TTS`, `ChatterboxTTS`, `OmniVoiceTTS`, `IndicMioTTS`, `FishAudioTTS`, `MagpieTTS`, `MagpieTTSCoreML`, `MAGNeTMusicGen`, `StableAudio3MusicGen`, `FlashSR`, `PersonaPlex`, `HibikiTranslate`, `MADLADTranslation`, `SpeechVAD`, `SpeechWakeWord`, `SpeechEnhancement`, `SpeechRestoration`, `SourceSeparation`, `Qwen3Chat`, `FunctionGemma`, `SpeechCore`, `SpeechUI`, `AudioCommon`.

## Modelos

Vista compacta abaixo. **[Catalogo completo de modelos com tamanhos, quantizacoes, URLs de download e tabelas de memoria → soniqo.audio/architecture](https://soniqo.audio/pt/architecture)**.

| Modelo | Tarefa | Backends | Tamanhos | Idiomas |
|-------|------|----------|-------|-----------|
| [Qwen3-ASR](https://soniqo.audio/pt/guides/transcribe) | Fala → Texto | MLX, CoreML (hibrido) | 0.6B, 1.7B | 52 |
| [Parakeet TDT](https://soniqo.audio/pt/guides/parakeet) | Fala → Texto | CoreML (ANE) | 0.6B | 25 europeus |
| [Parakeet EOU](https://soniqo.audio/pt/guides/dictate) | Fala → Texto (streaming) | CoreML (ANE) | 120M | 25 europeus |
| [Nemotron Streaming (Multilíngue)](https://soniqo.audio/pt/guides/nemotron) | Fala → Texto (streaming, com pontuação) | CoreML (ANE), MLX | 0.6B | **40** |
| [Nemotron Streaming (Inglês)](https://soniqo.audio/guides/nemotron) | Fala → Texto (streaming, com pontuação) | CoreML (ANE) | 0.6B | EN |
| [Omnilingual ASR](https://soniqo.audio/pt/guides/omnilingual) | Fala → Texto | CoreML (ANE), MLX | 300M / 1B / 3B / 7B | **[1.672](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)** |
| [Qwen3-ForcedAligner](https://soniqo.audio/pt/guides/align) | Audio + Texto → Timestamps | MLX, CoreML | 0.6B | Multi |
| [Qwen3-TTS](https://soniqo.audio/pt/guides/speak) | Texto → Fala | MLX, CoreML | 0.6B, 1.7B | 10 |
| [CosyVoice3](https://soniqo.audio/pt/guides/cosyvoice) | Texto → Fala | MLX | 0.5B | 9 |
| [VoxCPM2](https://soniqo.audio/pt/speech-generation) | Texto → Fala (48 kHz, design de voz + clonagem) | MLX | 2B (bf16/int8) | 30 |
| [Kokoro-82M](https://soniqo.audio/pt/guides/kokoro) | Texto → Fala | CoreML (ANE) | 82M | 10 |
| [Supertonic-3](https://soniqo.audio/guides/supertonic) | Texto → Fala (44.1 kHz, flow-matching, sem G2P) | CoreML (ANE) | 99M | 31 |
| [VibeVoice Realtime-0.5B](https://soniqo.audio/pt/guides/vibevoice) | Texto → Fala (formato longo, multi-alto-falante) | MLX | 0.5B | EN/ZH |
| [VibeVoice 1.5B](https://soniqo.audio/pt/guides/vibevoice) | Texto → Fala (podcast de ate 90 min) | MLX | 1.5B | EN/ZH |
| [Magpie-TTS Multilingual](https://soniqo.audio/pt/guides/magpie) | Texto → Fala (5 oradores predefinidos, streaming) | MLX / CoreML | 357M (MLX INT8, CoreML INT8) | 9 (CoreML exclui JA) |
| [OmniVoice](https://huggingface.co/aufklarer/OmniVoice-MLX-int8) | Texto → Fala (difusao NAR, clonagem zero-shot) | MLX | 0.8B (int8/fp16) | 600+ |
| [Indic-Mio](docs/models/indic-mio-tts.md) | Text → Speech (Hindi/Indic, emotion tags, voice cloning) | MLX | fp16 | Hindi / Indic |
| [Fish Audio S2 Pro](docs/models/fish-audio-s2-pro.md) | Texto → Fala (clonagem zero-shot, marcadores de estilo explicitos) | MLX | 0.5B-class (fp16) | Multilingue |
| [Qwen3.5 Chat](docs/models/qwen35-chat.md) | Text → Text (LLM) | MLX, CoreML | 0.8B | Multi |
| [Qwen3 Dense Chat](docs/models/qwen3-dense-chat.md) | Text → Text (LLM) | MLX | 4B | Multi |
| [Gemma 4 Chat](docs/models/gemma4-chat.md) | Text → Text (LLM) | MLX | E2B / E4B (4-bit) | Multi |
| [FunctionGemma](https://soniqo.audio/pt/guides/function-calls) | Texto → Chamadas de ferramentas (LLM) | CoreML | 270M | EN |
| [MADLAD-400](https://soniqo.audio/pt/guides/translate) | Texto → Texto (Tradução) | MLX | 3B | **400+** |
| [Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate) | Fala → Fala (Tradução) | MLX | 3B | FR/ES/PT/DE → EN |
| [PersonaPlex](https://soniqo.audio/pt/guides/respond) | Fala → Fala | MLX | 7B | EN |
| [Silero VAD](https://soniqo.audio/pt/guides/vad) | Deteccao de atividade de voz | MLX, CoreML | 309K | Agnostico |
| [KWS Zipformer](docs/models/kws-zipformer.md) | Audio → Wake word | CoreML (ANE) | 3M | EN/custom keywords |
| [Pyannote](https://soniqo.audio/pt/guides/diarize) | VAD + Diarizacao | MLX | 1.5M | Agnostico |
| [Sortformer](https://soniqo.audio/pt/guides/diarize) | Diarizacao (E2E) | CoreML (ANE) | — | Agnostico |
| [DeepFilterNet3](https://soniqo.audio/pt/guides/denoise) | Aprimoramento de fala | CoreML | 2.1M | Agnostico |
| [Sidon](https://soniqo.audio/pt/guides/restore) | Restauracao de fala (denoise + dereverb, 48 kHz) | CoreML | w2v-BERT 2.0 + DAC (fp16/int8) | Agnostico |
| [HTDemucs (Demucs v4)](https://soniqo.audio/pt/guides/separate) | Separação de fontes | MLX | 168M | Agnostic |
| [Open-Unmix](https://soniqo.audio/pt/guides/separate) | Separação de fontes | MLX | 8.6M | Agnostic |
| [MAGNeT](https://soniqo.audio/pt/guides/compose) | Texto → Música (30 s @ 32 kHz) | MLX | 300M / 1.5B (int4/int8) | Prompts em EN |
| [Stable Audio 3](docs/models/stable-audio-3.md) | Text → Music/audio (44.1 kHz stereo) | MLX | Medium 1.4B (int4/int8) | EN prompts |
| [FlashSR](https://soniqo.audio/pt/guides/upsample) | Super-resolução de áudio (48 kHz) | MLX | 363 MB / 720 MB (int4/int8) | Agnóstico |
| [WeSpeaker](https://soniqo.audio/pt/guides/embed-speaker) | Embedding de falante | MLX, CoreML | 6.6M | Agnostico |

## Instalacao

### Homebrew

Requer Homebrew ARM nativo (`/opt/homebrew`). Homebrew Rosetta/x86_64 nao e suportado.

```bash
brew install speech
```

Depois:

```bash
speech transcribe recording.wav
speech speak "Hello world"
speech translate "Hello, how are you?" --to es
speech respond --input question.wav --transcript
speech-server --port 8080            # servidor HTTP / WebSocket local (OpenAI-compatible /v1/realtime + /v1/audio/transcriptions)
```

**[Referencia completa do CLI →](https://soniqo.audio/pt/cli)**

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

Importe apenas o que voce precisa — cada modelo e o seu proprio target SPM:

```swift
import Qwen3ASR             // Reconhecimento de fala (MLX)
import ParakeetASR          // Reconhecimento de fala (CoreML, batch)
import ParakeetStreamingASR // Ditado em streaming com parciais + EOU
import NemotronStreamingASR // ASR streaming multilíngue com pontuação nativa (0.6B, 40 idiomas)
import OmnilingualASR       // 1.672 idiomas (CoreML + MLX)
import Qwen3TTS             // Sintese de fala
import CosyVoiceTTS         // Sintese de fala com clonagem
import VoxCPM2TTS           // TTS de 48 kHz, clonagem de voz + design de voz (2B)
import KokoroTTS            // Sintese de fala (pronto para iOS)
import VibeVoiceTTS         // TTS de formato longo / multi-alto-falante (EN/ZH)
import MagpieTTS            // TTS multilíngue (NVIDIA Magpie 357M, MLX, 9 idiomas)
import MagpieTTSCoreML      // Backend CoreML do Magpie (híbrido CoreML + MLX, 8 idiomas)
import FishAudioTTS         // Runtime experimental Fish Audio S2 Pro com clonagem de voz
import Qwen3Chat            // Chat LLM no dispositivo
import FunctionGemma    // LLM no dispositivo para chamadas de ferramentas
import MADLADTranslation    // Tradução multidirecional entre 400+ idiomas
import HibikiTranslate      // Tradução de fala para fala em streaming (FR/ES/PT/DE → EN)
import PersonaPlex          // Fala-a-fala full-duplex
import SpeechVAD            // VAD + diarizacao + embeddings
import SpeechEnhancement    // Supressao de ruido
import SpeechRestoration    // Restauracao de fala — denoise + dereverb (Sidon, CoreML, 48 kHz)
import SourceSeparation     // Separação de fontes musicais (Open-Unmix, 4 stems)
import MAGNeTMusicGen      // Geração de música a partir de texto (30 s, 32 kHz)
import FlashSR             // Super-resolução de áudio (48 kHz, difusão em 1 passo)
import SpeechUI             // Componentes SwiftUI para transcricoes em streaming
import AudioCommon          // Protocolos e utilitarios compartilhados
```

### Requisitos

- Swift 6+, Xcode 16+ (com Metal Toolchain)
- macOS 15+ (Sequoia) ou iOS 18+, Apple Silicon (M1/M2/M3/M4)

O mínimo de macOS 15 / iOS 18 vem do [MLState](https://developer.apple.com/documentation/coreml/mlstate) —— a API de estado persistente do ANE da Apple —— que os pipelines CoreML (Qwen3-ASR, Qwen3-Chat, Qwen3-TTS) usam para manter caches KV residentes no Neural Engine entre passos de token.

### Compilar a partir do codigo-fonte

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

`make build` compila o pacote Swift **e** a biblioteca de shaders MLX Metal. A biblioteca Metal e necessaria para inferencia em GPU — sem ela voce vera `Failed to load the default metallib` em tempo de execucao. `make debug` para builds de debug, `make test` para a suite de testes.

**[Guia completo de build e instalacao →](https://soniqo.audio/pt/getting-started)**

## Aplicativos de demonstracao

- **[DictateDemo](Examples/DictateDemo/)** ([docs](https://soniqo.audio/pt/guides/dictate)) — Ditado em streaming na barra de menus do macOS com parciais ao vivo, deteccao de fim de enunciado baseada em VAD e copia com um clique. Roda como agent em segundo plano (Parakeet-EOU-120M + Silero VAD).
- **[iOSEchoDemo](Examples/iOSEchoDemo/)** — Demo de eco iOS (Parakeet ASR + Kokoro TTS). Dispositivo e simulador.
- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — Assistente de voz conversacional com entrada de microfone, VAD e contexto multi-turno. macOS. RTF ~0.94 em M2 Max (mais rapido que tempo real).
- **[SpeechDemo](Examples/SpeechDemo/)** — Ditado e sintese TTS em uma interface com abas. macOS.

O README de cada demo tem instrucoes de build.

## Exemplos de codigo

Os snippets abaixo mostram o caminho minimo para cada dominio. Cada secao tem link para um guia completo em [soniqo.audio](https://soniqo.audio/pt) com opcoes de configuracao, multiplos backends, padroes de streaming e receitas de CLI.

### Fala para texto — [guia completo →](https://soniqo.audio/pt/guides/transcribe)

```swift
import Qwen3ASR

let model = try await Qwen3ASRModel.fromPretrained()
let text = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

Backends alternativos: [Parakeet TDT](https://soniqo.audio/pt/guides/parakeet) (CoreML, 32× tempo real), [Omnilingual ASR](https://soniqo.audio/pt/guides/omnilingual) (1.672 idiomas, CoreML ou MLX), [Ditado em streaming](https://soniqo.audio/pt/guides/dictate) (parciais ao vivo).

### Alinhamento forcado — [guia completo →](https://soniqo.audio/pt/guides/align)

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

### Texto para fala — [guia completo →](https://soniqo.audio/pt/guides/speak)

```swift
import Qwen3TTS
import AudioCommon

let model = try await Qwen3TTSModel.fromPretrained()
let audio = model.synthesize(text: "Hello world", language: "english")
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

Engines TTS alternativas: [CosyVoice3](https://soniqo.audio/pt/guides/cosyvoice) (streaming + clonagem + tags de emocao), [Kokoro-82M](https://soniqo.audio/pt/guides/kokoro) (pronto para iOS, 54 vozes), [VibeVoice](https://soniqo.audio/pt/guides/vibevoice) (podcast de formato longo / multi-alto-falante, EN/ZH), [Fish Audio S2 Pro](docs/inference/fish-audio-s2-pro.md) (clonagem zero-shot experimental + marcadores de estilo entre colchetes), [Clonagem de voz](https://soniqo.audio/pt/guides/voice-cloning).

### Fala para fala — [guia completo →](https://soniqo.audio/pt/guides/respond)

```swift
import PersonaPlex

let model = try await PersonaPlexModel.fromPretrained()
let responseAudio = model.respond(userAudio: userSamples)
// Saida mono Float32 a 24 kHz pronta para reproducao
```

### Chat LLM — [guia completo →](https://soniqo.audio/pt/guides/chat)

```swift
import Qwen3Chat
import FunctionGemma

let chat = try await Qwen35MLXChat.fromPretrained()
chat.chat(messages: [(.user, "Explain MLX in one sentence")]) { token, isFinal in
    print(token, terminator: "")
}
```

### Tradução — [guia completo →](https://soniqo.audio/pt/guides/translate)

```swift
import MADLADTranslation

let translator = try await MADLADTranslator.fromPretrained()
let es = try translator.translate("Hello, how are you?", to: "es")
// → "Hola, ¿cómo estás?"
```

### Tradução de fala — [guia completo →](https://soniqo.audio/guides/audio-translate)

```swift
import HibikiTranslate
import AudioCommon

let model = try await HibikiTranslateModel.fromPretrained()
let pcm = try AudioFileLoader.load(url: input, targetSampleRate: 24000)
let (englishAudio, textTokens) = model.translate(
    sourceAudio: pcm, sourceLanguage: .fr
)
// Hibiki Zero-3B — FR/ES/PT/DE → EN, no dispositivo, codec Mimi em streaming
```

### Deteccao de atividade de voz — [guia completo →](https://soniqo.audio/pt/guides/vad)

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
for s in segments { print("\(s.startTime)s → \(s.endTime)s") }
```

### Diarizacao de falantes — [guia completo →](https://soniqo.audio/pt/guides/diarize)

```swift
import SpeechVAD

let diarizer = try await DiarizationPipeline.fromPretrained()
let segments = diarizer.diarize(audio: samples, sampleRate: 16000)
for s in segments { print("Speaker \(s.speakerId): \(s.startTime)s - \(s.endTime)s") }
```

### Aprimoramento de fala — [guia completo →](https://soniqo.audio/pt/guides/denoise)

```swift
import SpeechEnhancement

let denoiser = try await DeepFilterNet3Model.fromPretrained()
let clean = try denoiser.enhance(audio: noisySamples, sampleRate: 48000)
```

### Restauracao de fala — [guia completo →](https://soniqo.audio/pt/guides/restore)

Denoise **e** dereverb conjuntos com [Sidon](https://arxiv.org/abs/2509.17052) (preditor w2v-BERT 2.0 + vocoder DAC, Core ML). Ao contrario de um supressor de ruido generico, o Sidon e treinado para preservar a identidade do locutor, o que o torna ideal para limpar uma referencia de clonagem de voz ruidosa ou reverberante antes do TTS. A entrada e a 16 kHz; a saida e mono a 48 kHz.

```swift
import SpeechRestoration

let restorer = try await SpeechRestorer.fromPretrained()          // .fp16 (default) or .int8
let clean = try restorer.restore(audio: noisySamples, sampleRate: 16000)  // → 48 kHz
```

A partir do CLI:

```bash
speech restore noisy.wav -o clean.wav            # denoise + dereverb, 48 kHz output
speech restore noisy.wav --variant int8          # smaller, lower peak RAM

# Clean a voice-cloning reference before TTS (opt-in; preserves speaker identity):
speech speak "Hello" --engine voxcpm2 --voice-sample ref.wav --clean-reference
```

### Voice Pipeline (ASR → LLM → TTS) — [guia completo →](https://soniqo.audio/pt/voice-agents)

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

`VoicePipeline` e a maquina de estados de agent de voz em tempo real (movida por [speech-core](https://github.com/soniqo/speech-core)) com deteccao de turnos baseada em VAD, tratamento de interrupcoes e STT eager. Conecta qualquer `SpeechRecognitionModel` + `SpeechGenerationModel` + `StreamingVADProvider`.

### Servidor HTTP API

```bash
speech-server --port 8080
```

Expoe cada modelo via endpoints HTTP REST + WebSocket, incluindo APIs compativeis com OpenAI: um WebSocket Realtime em `/v1/realtime` e um endpoint REST de transcricao em `/v1/audio/transcriptions`. Veja [`Sources/AudioServer/`](Sources/AudioServer/).

## Arquitetura

speech-swift e dividido em um target SPM por modelo para que os consumidores paguem apenas pelo que importarem. A infraestrutura compartilhada fica em `AudioCommon` (protocolos, I/O de audio, downloader do HuggingFace, `SentencePieceModel`) e `MLXCommon` (carregamento de pesos, helpers `QuantizedLinear`, helper de atencao multi-head `SDPA`).

**[Diagrama completo de arquitetura com backends, tabelas de memoria e mapa de modulos → soniqo.audio/architecture](https://soniqo.audio/pt/architecture)** · **[Referencia de API → soniqo.audio/api](https://soniqo.audio/pt/api)** · **[Benchmarks → soniqo.audio/benchmarks](https://soniqo.audio/pt/benchmarks)**

Docs locais (repositorio):
- **Modelos:** [Qwen3-ASR](docs/models/asr-model.md) · [Qwen3-TTS](docs/models/tts-model.md) · [CosyVoice](docs/models/cosyvoice-tts.md) · [Kokoro](docs/models/kokoro-tts.md) · [VoxCPM2](docs/models/voxcpm2-tts.md) · [VibeVoice](docs/models/vibevoice.md) · [Supertonic](docs/models/supertonic-tts.md) · [Chatterbox](docs/models/chatterbox-tts.md) · [Indic-Mio](docs/models/indic-mio-tts.md) · [Fish Audio S2 Pro](docs/models/fish-audio-s2-pro.md) · [Magpie TTS](docs/models/magpie-tts.md) · [Parakeet TDT](docs/models/parakeet-asr.md) · [Parakeet Streaming](docs/models/parakeet-streaming-asr.md) · [Nemotron Streaming](docs/models/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/models/omnilingual-asr.md) · [PersonaPlex](docs/models/personaplex.md) · [Hibiki](docs/models/hibiki.md) · [MADLAD-400](docs/models/madlad-translation.md) · [FunctionGemma](docs/models/function-gemma.md) · [Qwen3.5 Chat](docs/models/qwen35-chat.md) · [Gemma 4 Chat](docs/models/gemma4-chat.md) · [Qwen3 Dense Chat](docs/models/qwen3-dense-chat.md) · [FireRedVAD](docs/models/fireredvad.md) · [KWS Zipformer](docs/models/kws-zipformer.md) · [Sidon](docs/models/sidon.md) · [Source Separation](docs/models/source-separation.md) · [HTDemucs](docs/models/htdemucs.md) · [MAGNeT](docs/models/magnet-music-gen.md) · [Stable Audio 3](docs/models/stable-audio-3.md) · [FlashSR](docs/models/flashsr.md)
- **Inferencia:** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md) · [Parakeet TDT](docs/inference/parakeet-asr-inference.md) · [Parakeet Streaming](docs/inference/parakeet-streaming-asr-inference.md) · [Nemotron Streaming](docs/inference/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/inference/omnilingual-asr-inference.md) · [TTS](docs/inference/qwen3-tts-inference.md) · [VoxCPM2](docs/inference/voxcpm2-inference.md) · [VibeVoice](docs/inference/vibevoice-inference.md) · [Fish Audio S2 Pro](docs/inference/fish-audio-s2-pro.md) · [Magpie TTS](docs/inference/magpie-tts.md) · [Hibiki](docs/inference/hibiki-inference.md) · [MADLAD-400](docs/inference/madlad-translation.md) · [MAGNeT](docs/inference/magnet-music-gen.md) · [Stable Audio 3](docs/inference/stable-audio-3.md) · [FlashSR](docs/inference/flashsr.md) · [Forced Aligner](docs/inference/forced-aligner.md) · [Silero VAD](docs/inference/silero-vad.md) · [FireRedVAD](docs/inference/fireredvad.md) · [Wake-word](docs/inference/wake-word.md) · [Speaker Diarization](docs/inference/speaker-diarization.md) · [Speech Enhancement](docs/inference/speech-enhancement.md) · [Sidon](docs/inference/sidon.md) · [Cache/offline](docs/inference/cache-and-offline.md)
- **Referencia:** [Protocolos compartilhados](docs/shared-protocols.md)

## Configuracao de cache

Os pesos dos modelos sao baixados do HuggingFace no primeiro uso e armazenados em cache em `~/Library/Caches/qwen3-speech/`. Sobrescreva com `QWEN3_CACHE_DIR` (CLI) ou `cacheDir:` (API Swift). Todos os pontos de entrada `fromPretrained()` aceitam `offlineMode: true` para pular a rede quando os pesos ja estao em cache.

Usuarios na China continental (ou onde `huggingface.co` for lento ou bloqueado) podem baixar de um espelho definindo `HF_ENDPOINT`, p. ex. `export HF_ENDPOINT=https://hf-mirror.com`.

Veja [`docs/inference/cache-and-offline.md`](docs/inference/cache-and-offline.md) para detalhes completos, incluindo caminhos de container iOS sandboxed.

## Biblioteca MLX Metal

Se voce ver `Failed to load the default metallib` em tempo de execucao, a biblioteca de shaders Metal esta faltando. Execute `make build` ou `./scripts/build_mlx_metallib.sh release` apos um `swift build` manual. Se o Metal Toolchain estiver faltando, instale-o primeiro:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Testes

```bash
make test                            # suite completa (unidade + E2E com downloads de modelos)
swift test --skip E2E                # somente unidade (seguro para CI, sem downloads)
swift test --filter Qwen3ASRTests    # modulo especifico
```

Classes de teste E2E usam o prefixo `E2E` para que a CI possa filtra-las com `--skip E2E`. Veja [CLAUDE.md](CLAUDE.md#testing) para a convencao completa de testes.

## Contribuindo

PRs bem-vindos — correcoes de bugs, integracoes de novos modelos, documentacao. Fork, crie uma branch de feature, `make build && make test`, abra um PR contra `main`.

## Licenca

Apache 2.0
