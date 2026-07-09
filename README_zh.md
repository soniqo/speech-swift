# Speech Swift

面向 Apple Silicon 的 AI 语音模型，基于 MLX Swift 和 CoreML。

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

端侧语音识别、合成与理解，适用于 Mac 与 iOS。完全在 Apple Silicon 上本地运行——无需云端、无需 API 密钥、数据不出设备。

**[📚 完整文档 →](https://soniqo.audio/zh)** · **[🤗 HuggingFace 模型](https://huggingface.co/aufklarer)** · **[📝 博客](https://blog.ivan.digital)**

<p align="center">
  <a href="https://trendshift.io/repositories/24196?utm_source=trendshift-badge&amp;utm_medium=badge&amp;utm_campaign=badge-trendshift-24196" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/trendshift/repositories/24196/daily?language=Swift" alt="soniqo%2Fspeech-swift | Trendshift" width="250" height="55"/></a>
</p>

<p align="center">
  <a href="https://youtu.be/x9zgcaW0gUk">
    <img src="https://img.youtube.com/vi/x9zgcaW0gUk/maxresdefault.jpg" width="640" alt="在 MacBook 上运行的本地语音 AI —— 在 YouTube 上观看四分钟开源库导览">
  </a>
</p>
<p align="center"><em>在 MacBook 上运行的本地语音 AI —— 在 YouTube 上观看四分钟开源库导览</em></p>

**使用场景：** [语音代理](https://soniqo.audio/zh/voice-agents) · [转录](https://soniqo.audio/zh/transcription) · [语音合成](https://soniqo.audio/zh/speech-generation)

**能力分组：** STT / ASR · 对齐 · TTS · LLM 与翻译 · 语音到语音 · 增强 / 修复 · 源分离 · 音乐 / 音频生成 · 唤醒词、VAD、说话人分离与说话人身份

**STT / ASR**

- **[Qwen3-ASR](https://soniqo.audio/zh/guides/transcribe)** — 语音转文字（自动语音识别，52 种语言，MLX + CoreML）
- **[WhisperASR](docs/models/whisper-asr.md)** — Whisper Large-v3 Turbo speech-to-text via native CoreML runtime (ANE, multilingual)
- **[Parakeet TDT](https://soniqo.audio/zh/guides/parakeet)** — 通过 CoreML 进行语音转文字（神经引擎，NVIDIA FastConformer + TDT 解码器，25 种语言）
- **[Omnilingual ASR](https://soniqo.audio/zh/guides/omnilingual)** — 语音转文字（Meta wav2vec2 + CTC，**1,672 种语言**，覆盖 32 种文字系统，CoreML 300M + MLX 300M/1B/3B/7B）
- **[流式听写](https://soniqo.audio/zh/guides/dictate)** — 带部分结果和句末检测的实时听写（Parakeet-EOU-120M）
- **[Nemotron 流式 (多语言)](https://soniqo.audio/zh/guides/nemotron)** — 具有原生标点和大小写的低延迟流式 ASR（NVIDIA Nemotron-3.5-ASR-Streaming-0.6B，CoreML + MLX，**40 种语言-区域设置**）
- **[Nemotron 流式 (英语)](https://soniqo.audio/guides/nemotron)** — 具有原生标点和大小写的低延迟流式 ASR （NVIDIA Nemotron-Speech-Streaming-0.6B，CoreML，仅英语，比多语言版本更小、更快）

**对齐**

- **[Qwen3-ForcedAligner](https://soniqo.audio/zh/guides/align)** — 词级时间戳对齐（音频 + 文本 → 时间戳）

**TTS / 语音生成**

- **[Qwen3-TTS](https://soniqo.audio/zh/guides/speak)** — 文本转语音（最高质量、流式输出、自定义说话人，10 种语言）
- **[CosyVoice TTS](https://soniqo.audio/zh/guides/cosyvoice)** — 流式 TTS，支持声音克隆、多说话人对话、情感标签（9 种语言）
- **[VoxCPM2](https://soniqo.audio/zh/speech-generation)** — 48 kHz 录音棚级 TTS，支持声音克隆与基于指令的声音设计（2B，MLX bf16/int8，30 种语言）
- **[IndexTTS2](docs/models/voice-cloning-candidate-runtimes.md)** — Native MLX voice cloning from a reference voice (IndexTeam IndexTTS-2, 1.5B-class fp16 bundle, speaker/emotion/pause controls)
- **[Kokoro TTS](https://soniqo.audio/zh/guides/kokoro)** — 端侧 TTS（82M，CoreML/神经引擎，54 种音色，iOS 就绪，10 种语言）
- **[VibeVoice TTS](https://soniqo.audio/zh/guides/vibevoice)** — 长篇 / 多说话人 TTS（Microsoft VibeVoice Realtime-0.5B + 1.5B，MLX，可合成最长 90 分钟的播客 / 有声书，英语 / 中文）
- **[Magpie TTS](https://soniqo.audio/zh/guides/magpie)** — 多语言 TTS（NVIDIA Magpie-TTS Multilingual 357M，MLX INT8 411 MB 或 CoreML INT8 342 MB，9 种语言，5 位预设说话人，MLX 端流式）
- **[Supertonic TTS](https://soniqo.audio/guides/supertonic)** — 端侧流匹配 TTS（Supertone Supertonic-3 99M，CoreML/神经引擎，31 种语言，10 种音色，G2P-free，44.1 kHz）
- **[Chatterbox TTS](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16)** — 支持零样本声音克隆的多语言 TTS（Resemble AI Chatterbox Multilingual，MLX fp16 ~1.3 GB，23 种运行时语言；希伯来语需要 niqqud，MIT）
- **[OmniVoice TTS](https://huggingface.co/aufklarer/OmniVoice-MLX-fp16)** — 支持零样本语音克隆的非自回归扩散 TTS（k2-fsa OmniVoice，Qwen3 骨干，MLX fp16 默认 / 可用 int8，600+ 种语言，Apache-2.0）
- **[Indic-Mio](docs/models/indic-mio-tts.md)** — Hindi/Indic TTS with inline emotion markers and optional reference-voice cloning (MLX, 24 kHz)

**LLM 与翻译**

- **[Qwen3Chat](https://soniqo.audio/zh/guides/chat)** — 端侧 LLM 对话（Qwen3.5-0.8B MLX/CoreML，加 dense Qwen3 4B 与 Gemma 4 E2B/E4B 的 MLX 后端，流式 token）
- **[FunctionGemma](https://soniqo.audio/zh/guides/function-calls)** — 端侧结构化函数 / 工具调用 LLM（Gemma 3 270M，CoreML 8 位调色板量化，Neural Engine，约 252 tok/s）
- **[MADLAD-400](https://soniqo.audio/zh/guides/translate)** — 400+ 语言间的多对多翻译（3B，MLX INT4 + INT8，T5 v1.1，Apache 2.0）

**语音到语音与语音代理**

- **[Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate)** — 流式语音到语音翻译（FR/ES/PT/DE → EN，MLX INT4 + INT8，Kyutai Moshi/Mimi 技术栈，CC-BY-4.0）
- **[PersonaPlex](https://soniqo.audio/zh/guides/respond)** — 全双工语音到语音（7B，音频输入 → 音频输出，18 种预设音色）
- **[Audio2Face-3D](docs/models/audio2face3d.md)** — 语音驱动的虚拟形象面部动画（NVIDIA Audio2Face-3D v2.3 Mark，301 个面部系数，MLX）

**增强、分离与音频生成**

- **[DeepFilterNet3](https://soniqo.audio/zh/guides/denoise)** — 实时噪声抑制（2.1M 参数，48 kHz）。超过 60 s 单次处理上限的长音频会自动分块并使用 crossfade 拼接 — 参见 `enhanceChunked(...)`
- **[音源分离](https://soniqo.audio/zh/guides/separate)** — 通过 HTDemucs (Demucs v4) + Open-Unmix 进行音乐源分离（UMX-HQ / UMX-L，4 声轨：人声/鼓/贝斯/其他，44.1 kHz 立体声）
- **[MAGNeT](https://soniqo.audio/zh/guides/compose)** — 文本到音乐生成（Meta MAGNeT Small 300M / Medium 1.5B，MLX INT8，30 秒片段 32 kHz 单声道，掩码并行解码）
- **[Stable Audio 3](docs/models/stable-audio-3.md)** — Text-to-audio/music generation (Stable Audio 3 Medium, MLX INT8/INT4, 44.1 kHz stereo, variable length)
- **[FlashSR](https://soniqo.audio/zh/guides/upsample)** — 音频超分辨率(FlashSR ICASSP 2025,MLX,48 kHz 单声道,1 步蒸馏扩散,INT4 363 MB / INT8 720 MB)

**轮次检测、说话人分离与说话人身份**

- **[唤醒词](https://soniqo.audio/zh/guides/wake-word)** — 设备端关键词识别（KWS Zipformer 3M，CoreML，26× 实时，可配置关键词列表）
- **[VAD](https://soniqo.audio/zh/guides/vad)** — 语音活动检测（Silero 流式、Pyannote 离线、FireRedVAD 100+ 种语言）
- **[说话人分离](https://soniqo.audio/zh/guides/diarize)** — 谁在什么时间说话（Pyannote 流水线，神经引擎上的端到端 Sortformer）
- **[说话人嵌入向量](https://soniqo.audio/zh/guides/embed-speaker)** — WeSpeaker ResNet34（256 维）、CAM++（192 维）

论文：[Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba) · [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba) · [Omnilingual ASR](https://arxiv.org/abs/2511.09690) (Meta) · [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA) · [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba) · [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2) · [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA) · [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai) · [Hibiki](https://arxiv.org/abs/2502.03382) (Kyutai) · [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## 动态

- **2026 年 4 月 19 日** — [Apple Silicon 上的 MLX 与 CoreML — 如何选择合适的推理后端](https://blog.ivan.digital/mlx-vs-coreml-on-apple-silicon-a-practical-guide-to-picking-the-right-backend-and-why-you-should-f77ddea7b27a)
- **2026 年 3 月 20 日** — [我们用一个 600M 模型在 Mac 上击败了 Whisper Large v3](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **2026 年 2 月 26 日** — [Apple Silicon 上的说话人分离与语音活动检测——基于 MLX 的原生 Swift 实现](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **2026 年 2 月 23 日** — [NVIDIA PersonaPlex 7B 在 Apple Silicon 上运行——基于 MLX 的原生 Swift 全双工语音到语音](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **2026 年 2 月 12 日** — [Qwen3-ASR Swift：面向 Apple Silicon 的端侧 ASR + TTS——架构与基准测试](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## 快速开始

将依赖添加到你的 `Package.swift`：

```swift
.package(url: "https://github.com/soniqo/speech-swift", branch: "main")
```

只引入你需要的模块——每个模型都是独立的 SPM 库，不用为你不使用的东西买单：

```swift
.product(name: "ParakeetStreamingASR", package: "speech-swift"),
.product(name: "SpeechUI",             package: "speech-swift"),  // 可选的 SwiftUI 视图
```

**3 行代码转写音频缓冲区：**

```swift
import ParakeetStreamingASR

let model = try await ParakeetStreamingASRModel.fromPretrained()
let text = try model.transcribeAudio(audioSamples, sampleRate: 16000)
```

**带部分结果的实时流式：**

```swift
for await partial in model.transcribeStream(audio: samples, sampleRate: 16000) {
    print(partial.isFinal ? "FINAL: \(partial.text)" : "... \(partial.text)")
}
```

**约 10 行写出 SwiftUI 听写视图：**

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

`SpeechUI` 只提供 `TranscriptionView`（最终结果 + 部分结果）与 `TranscriptionStore`（流式 ASR 适配器）。音频可视化和播放请使用 AVFoundation。

可用的 SPM products：`Qwen3ASR`, `WhisperASR`, `Qwen3TTS`, `Qwen3TTSCoreML`, `ParakeetASR`, `ParakeetStreamingASR`, `NemotronStreamingASR`, `OmnilingualASR`, `KokoroTTS`, `SupertonicTTS`, `VibeVoiceTTS`, `CosyVoiceTTS`, `VoxCPM2TTS`, `IndexTTS2TTS`, `ChatterboxTTS`, `OmniVoiceTTS`, `IndicMioTTS`, `FishAudioTTS`, `MagpieTTS`, `MagpieTTSCoreML`, `MAGNeTMusicGen`, `StableAudio3MusicGen`, `FlashSR`, `PersonaPlex`, `Audio2Face3D`, `HibikiTranslate`, `MADLADTranslation`, `SpeechVAD`, `SpeechWakeWord`, `SpeechEnhancement`, `SpeechRestoration`, `SourceSeparation`, `Qwen3Chat`, `FunctionGemma`, `SpeechCore`, `SpeechUI`, `AudioCommon`。

## 模型

下方是精简视图。**[完整模型目录（含大小、量化方式、下载地址、内存表）→ soniqo.audio/architecture](https://soniqo.audio/zh/architecture)**。

| 模型 | 任务 | 后端 | 大小 | 语言 |
|-------|------|----------|-------|-----------|
| [Qwen3-ASR](https://soniqo.audio/zh/guides/transcribe) | 语音 → 文字 | MLX、CoreML（混合） | 0.6B、1.7B | 52 |
| [WhisperASR](docs/models/whisper-asr.md) | Speech → Text | CoreML (ANE) | Large-v3 Turbo | Multi |
| [Parakeet TDT](https://soniqo.audio/zh/guides/parakeet) | 语音 → 文字 | CoreML (ANE) | 0.6B | 25 种欧洲语言 |
| [Parakeet EOU](https://soniqo.audio/zh/guides/dictate) | 语音 → 文字（流式） | CoreML (ANE) | 120M | 25 种欧洲语言 |
| [Nemotron Streaming (多语言)](https://soniqo.audio/zh/guides/nemotron) | 语音 → 文本（流式、带标点） | CoreML (ANE), MLX | 0.6B | **40** |
| [Nemotron Streaming (英语)](https://soniqo.audio/guides/nemotron) | 语音 → 文本（流式、带标点） | CoreML (ANE) | 0.6B | EN |
| [Omnilingual ASR](https://soniqo.audio/zh/guides/omnilingual) | 语音 → 文字 | CoreML (ANE)、MLX | 300M / 1B / 3B / 7B | **[1,672](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)** |
| [Qwen3-ForcedAligner](https://soniqo.audio/zh/guides/align) | 音频 + 文本 → 时间戳 | MLX、CoreML | 0.6B | 多语言 |
| [Qwen3-TTS](https://soniqo.audio/zh/guides/speak) | 文本 → 语音 | MLX、CoreML | 0.6B、1.7B | 10 |
| [CosyVoice3](https://soniqo.audio/zh/guides/cosyvoice) | 文本 → 语音 | MLX | 0.5B | 9 |
| [VoxCPM2](https://soniqo.audio/zh/speech-generation) | 文本 → 语音 (48 kHz, 声音设计 + 克隆) | MLX | 2B (bf16/int8) | 30 |
| [IndexTTS2](docs/models/voice-cloning-candidate-runtimes.md) | Text → Speech (zero-shot voice cloning) | MLX | 1.5B-class (fp16) | EN/ZH |
| [Kokoro-82M](https://soniqo.audio/zh/guides/kokoro) | 文本 → 语音 | CoreML (ANE) | 82M | 10 |
| [Supertonic-3](https://soniqo.audio/guides/supertonic) | 文本 → 语音（44.1 kHz，流匹配，G2P-free） | CoreML (ANE) | 99M | 31 |
| [VibeVoice Realtime-0.5B](https://soniqo.audio/zh/guides/vibevoice) | 文本 → 语音（长篇、多说话人） | MLX | 0.5B | EN/ZH |
| [VibeVoice 1.5B](https://soniqo.audio/zh/guides/vibevoice) | 文本 → 语音（最长 90 分钟播客） | MLX | 1.5B | EN/ZH |
| [Magpie-TTS Multilingual](https://soniqo.audio/zh/guides/magpie) | 文本 → 语音（5 位预设说话人，流式） | MLX / CoreML | 357M (MLX INT8, CoreML INT8) | 9（CoreML 不含日语） |
| [Chatterbox Multilingual](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16) | 文本 → 语音（零样本克隆） | MLX | 0.8B (fp16) | 23（HE 需要 niqqud） |
| [OmniVoice](https://huggingface.co/aufklarer/OmniVoice-MLX-fp16) | 文本 → 语音（NAR 扩散，零样本克隆） | MLX | 0.8B (fp16 默认 / int8) | **600+** |
| [Indic-Mio](docs/models/indic-mio-tts.md) | Text → Speech (Hindi/Indic, emotion tags, voice cloning) | MLX | fp16 | Hindi / Indic |
| [Fish Audio S2 Pro](docs/models/fish-audio-s2-pro.md) | 文本 → 语音（零样本克隆，显式风格标记） | MLX | 0.5B-class (fp16) | 多语言 |
| [Qwen3.5 Chat](docs/models/qwen35-chat.md) | Text → Text (LLM) | MLX, CoreML | 0.8B | Multi |
| [Qwen3 Dense Chat](docs/models/qwen3-dense-chat.md) | Text → Text (LLM) | MLX | 4B | Multi |
| [Gemma 4 Chat](docs/models/gemma4-chat.md) | Text → Text (LLM) | MLX | E2B / E4B (4-bit) | Multi |
| [FunctionGemma](https://soniqo.audio/zh/guides/function-calls) | 文本 → 工具调用（LLM） | CoreML | 270M | 主英语 |
| [MADLAD-400](https://soniqo.audio/zh/guides/translate) | 文本 → 文本（翻译） | MLX | 3B | **400+** |
| [Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate) | 语音 → 语音（翻译） | MLX | 3B | FR/ES/PT/DE → EN |
| [PersonaPlex](https://soniqo.audio/zh/guides/respond) | 语音 → 语音 | MLX | 7B | EN |
| [Audio2Face-3D](docs/models/audio2face3d.md) | 语音 → 面部动画 | MLX | v2.3 Mark | 语言无关 |
| [Silero VAD](https://soniqo.audio/zh/guides/vad) | 语音活动检测 | MLX、CoreML | 309K | 语言无关 |
| [KWS Zipformer](docs/models/kws-zipformer.md) | Audio → Wake word | CoreML (ANE) | 3M | EN/custom keywords |
| [Pyannote](https://soniqo.audio/zh/guides/diarize) | VAD + 说话人分离 | MLX | 1.5M | 语言无关 |
| [Sortformer](https://soniqo.audio/zh/guides/diarize) | 说话人分离（端到端） | CoreML (ANE) | — | 语言无关 |
| [DeepFilterNet3](https://soniqo.audio/zh/guides/denoise) | 语音增强 | CoreML | 2.1M | 语言无关 |
| [Sidon](https://soniqo.audio/zh/guides/restore) | 语音修复（降噪 + 去混响，48 kHz） | CoreML | w2v-BERT 2.0 + DAC (fp16/int8) | 语言无关 |
| [HTDemucs (Demucs v4)](https://soniqo.audio/zh/guides/separate) | 音源分离 | MLX | 168M | Agnostic |
| [Open-Unmix](https://soniqo.audio/zh/guides/separate) | 音源分离 | MLX | 8.6M | Agnostic |
| [MAGNeT](https://soniqo.audio/zh/guides/compose) | 文本 → 音乐 (30 秒 @ 32 kHz) | MLX | 300M / 1.5B (int4/int8) | 英文提示 |
| [Stable Audio 3](docs/models/stable-audio-3.md) | Text → Music/audio (44.1 kHz stereo) | MLX | Medium 1.4B (int4/int8) | EN prompts |
| [FlashSR](https://soniqo.audio/zh/guides/upsample) | 音频超分辨率 (48 kHz) | MLX | 363 MB / 720 MB (int4/int8) | 通用 |
| [WeSpeaker](https://soniqo.audio/zh/guides/embed-speaker) | 说话人嵌入向量 | MLX、CoreML | 6.6M | 语言无关 |

## 安装

### Homebrew

需要原生 ARM Homebrew（`/opt/homebrew`），不支持 Rosetta/x86_64 Homebrew。

```bash
brew install speech
```

然后：

```bash
speech transcribe recording.wav
speech speak "Hello world"
speech translate "Hello, how are you?" --to es
speech respond --input question.wav --transcript
speech-server --port 8080            # 本地 HTTP / WebSocket 服务器（OpenAI 兼容 /v1/realtime + /v1/audio/transcriptions）
```

**[完整 CLI 参考 →](https://soniqo.audio/zh/cli)**

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

只引入你需要的模块——每个模型都是独立的 SPM target：

```swift
import Qwen3ASR             // 语音识别 (MLX)
import WhisperASR           // Whisper Large-v3 Turbo (CoreML)
import ParakeetASR          // 语音识别 (CoreML，批量)
import ParakeetStreamingASR // 带部分结果和 EOU 的流式听写
import NemotronStreamingASR // 多语言流式 ASR，原生标点（0.6B，40 种语言）
import OmnilingualASR       // 1,672 种语言 (CoreML + MLX)
import Qwen3TTS             // 文本转语音
import CosyVoiceTTS         // 带声音克隆的文本转语音
import VoxCPM2TTS           // 48 kHz TTS，声音克隆 + 声音设计 (2B)
import IndexTTS2TTS         // Native MLX voice cloning from reference audio
import KokoroTTS            // 文本转语音 (iOS 就绪)
import VibeVoiceTTS         // 长篇 / 多说话人 TTS（英语 / 中文）
import MagpieTTS            // 多语言 TTS（NVIDIA Magpie 357M，MLX，9 种语言）
import MagpieTTSCoreML      // Magpie CoreML 后端（CoreML + MLX 混合，8 种语言）
import FishAudioTTS         // 实验性 Fish Audio S2 Pro 运行时，支持声音克隆
import Qwen3Chat            // 端侧 LLM 对话
import FunctionGemma    // 端侧函数 / 工具调用 LLM
import MADLADTranslation    // 400+ 语言间的多对多翻译
import HibikiTranslate      // 流式语音到语音翻译（FR/ES/PT/DE → EN）
import PersonaPlex          // 全双工语音到语音
import SpeechVAD            // VAD + 说话人分离 + 嵌入向量
import SpeechEnhancement    // 噪声抑制
import SpeechRestoration    // 语音修复 — 降噪 + 去混响（Sidon，CoreML，48 kHz）
import SourceSeparation     // 音乐源分离（Open-Unmix，4 声轨）
import MAGNeTMusicGen      // 文本到音乐生成（30 秒，32 kHz）
import FlashSR             // 音频超分辨率(48 kHz,1 步扩散)
import SpeechUI             // 流式转写的 SwiftUI 组件
import AudioCommon          // 共享协议与工具
```

### 环境要求

- Swift 6+、Xcode 16+（含 Metal Toolchain）
- macOS 15+（Sequoia）或 iOS 18+、Apple Silicon（M1/M2/M3/M4）

macOS 15 / iOS 18 的最低要求来自 [MLState](https://developer.apple.com/documentation/coreml/mlstate) —— Apple 的持久化 ANE 状态 API，CoreML 管线（Qwen3-ASR、Qwen3-Chat、Qwen3-TTS）使用它让 KV 缓存常驻 Neural Engine，跨 token 步骤复用。

### 从源代码构建

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

`make build` 会同时编译 Swift 包**和** MLX Metal 着色器库。Metal 库是 GPU 推理所必需的——若缺失，运行时会出现 `Failed to load the default metallib`。`make debug` 用于调试构建，`make test` 运行测试套件。

**[完整构建与安装指南 →](https://soniqo.audio/zh/getting-started)**

## 演示应用

- **[DictateDemo](Examples/DictateDemo/)**（[文档](https://soniqo.audio/zh/guides/dictate)）— macOS 菜单栏流式听写，带实时部分结果、VAD 驱动的句末检测、一键复制。以后台 agent 方式运行（Parakeet-EOU-120M + Silero VAD）。
- **[iOSEchoDemo](Examples/iOSEchoDemo/)** — iOS 回声 demo（Parakeet ASR + Kokoro TTS）。支持设备和模拟器。
- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — 带麦克风输入、VAD 和多轮上下文的对话式语音助手。macOS。M2 Max 上 RTF 约 0.94（快于实时）。
- **[SpeechDemo](Examples/SpeechDemo/)** — 标签式界面中的听写与 TTS 合成。macOS。

每个 demo 的 README 都包含构建说明。

## 代码示例

下方代码片段展示每个领域的最小使用路径。每一节都链接到 [soniqo.audio](https://soniqo.audio/zh) 上的完整指南，涵盖配置选项、多种后端、流式模式和 CLI 示例。

### 语音转文字 — [完整指南 →](https://soniqo.audio/zh/guides/transcribe)

```swift
import Qwen3ASR

let model = try await Qwen3ASRModel.fromPretrained()
let text = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

其他后端：[WhisperASR](docs/inference/whisper-asr-inference.md)（Whisper Large-v3 Turbo，native CoreML）、[Parakeet TDT](https://soniqo.audio/zh/guides/parakeet)（CoreML，32× 实时）、[Omnilingual ASR](https://soniqo.audio/zh/guides/omnilingual)（1,672 种语言，CoreML 或 MLX）、[流式听写](https://soniqo.audio/zh/guides/dictate)（实时部分结果）。

### 强制对齐 — [完整指南 →](https://soniqo.audio/zh/guides/align)

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

### 文本转语音 — [完整指南 →](https://soniqo.audio/zh/guides/speak)

```swift
import Qwen3TTS
import AudioCommon

let model = try await Qwen3TTSModel.fromPretrained()
let audio = model.synthesize(text: "Hello world", language: "english")
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

其他 TTS 引擎：[CosyVoice3](https://soniqo.audio/zh/guides/cosyvoice)（流式 + 声音克隆 + 情感标签）、[Kokoro-82M](https://soniqo.audio/zh/guides/kokoro)（iOS 就绪，54 种音色）、[VibeVoice](https://soniqo.audio/zh/guides/vibevoice)（长篇播客 / 多说话人，英语 / 中文）、[Fish Audio S2 Pro](docs/inference/fish-audio-s2-pro.md)（实验性零样本克隆 + 方括号风格标记）、[声音克隆](https://soniqo.audio/zh/guides/voice-cloning)。

### 语音到语音 — [完整指南 →](https://soniqo.audio/zh/guides/respond)

```swift
import PersonaPlex

let model = try await PersonaPlexModel.fromPretrained()
let responseAudio = model.respond(userAudio: userSamples)
// 24 kHz 单声道 Float32 输出，可直接播放
```

### LLM 对话 — [完整指南 →](https://soniqo.audio/zh/guides/chat)

```swift
import Qwen3Chat
import FunctionGemma

let chat = try await Qwen35MLXChat.fromPretrained()
chat.chat(messages: [(.user, "Explain MLX in one sentence")]) { token, isFinal in
    print(token, terminator: "")
}
```

### 翻译 — [完整指南 →](https://soniqo.audio/zh/guides/translate)

```swift
import MADLADTranslation

let translator = try await MADLADTranslator.fromPretrained()
let es = try translator.translate("Hello, how are you?", to: "es")
// → "Hola, ¿cómo estás?"
```

### 语音翻译 — [完整指南 →](https://soniqo.audio/guides/audio-translate)

```swift
import HibikiTranslate
import AudioCommon

let model = try await HibikiTranslateModel.fromPretrained()
let pcm = try AudioFileLoader.load(url: input, targetSampleRate: 24000)
let (englishAudio, textTokens) = model.translate(
    sourceAudio: pcm, sourceLanguage: .fr
)
// Hibiki Zero-3B —— FR/ES/PT/DE → EN，端侧运行，流式 Mimi 编解码器
```

### 语音活动检测 — [完整指南 →](https://soniqo.audio/zh/guides/vad)

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
for s in segments { print("\(s.startTime)s → \(s.endTime)s") }
```

### 说话人分离 — [完整指南 →](https://soniqo.audio/zh/guides/diarize)

```swift
import SpeechVAD

let diarizer = try await DiarizationPipeline.fromPretrained()
let segments = diarizer.diarize(audio: samples, sampleRate: 16000)
for s in segments { print("Speaker \(s.speakerId): \(s.startTime)s - \(s.endTime)s") }
```

### 语音增强 — [完整指南 →](https://soniqo.audio/zh/guides/denoise)

```swift
import SpeechEnhancement

let denoiser = try await DeepFilterNet3Model.fromPretrained()
let clean = try denoiser.enhance(audio: noisySamples, sampleRate: 48000)
```

### 语音修复 — [完整指南 →](https://soniqo.audio/zh/guides/restore)

使用 [Sidon](https://arxiv.org/abs/2509.17052)（w2v-BERT 2.0 预测器 + DAC 声码器，Core ML）同时进行降噪**和**去混响。与通用降噪器不同，Sidon 经过训练以保留说话人身份，因此非常适合在 TTS 之前清理嘈杂或带混响的声音克隆参考音频。输入为 16 kHz；输出为 48 kHz 单声道。

```swift
import SpeechRestoration

let restorer = try await SpeechRestorer.fromPretrained()          // .fp16（默认）或 .int8
let clean = try restorer.restore(audio: noisySamples, sampleRate: 16000)  // → 48 kHz
```

通过 CLI：

```bash
speech restore noisy.wav -o clean.wav            # 降噪 + 去混响，48 kHz 输出
speech restore noisy.wav --variant int8          # 更小，更低的峰值内存占用

# 在 TTS 之前清理声音克隆参考音频（可选；保留说话人身份）：
speech speak "Hello" --engine voxcpm2 --voice-sample ref.wav --clean-reference
```

### 语音流水线（ASR → LLM → TTS） — [完整指南 →](https://soniqo.audio/zh/voice-agents)

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

`VoicePipeline` 是实时语音 agent 的状态机（由 [speech-core](https://github.com/soniqo/speech-core) 提供），支持 VAD 驱动的轮次检测、打断处理和积极式 STT。它可以连接任意 `SpeechRecognitionModel` + `SpeechGenerationModel` + `StreamingVADProvider`。

### HTTP API 服务器

```bash
speech-server --port 8080
```

通过 HTTP REST + WebSocket 接口暴露每个模型，包括兼容 OpenAI 的 API：`/v1/realtime` 上的 Realtime WebSocket 和 `/v1/audio/transcriptions` 上的转录 REST 接口。详见 [`Sources/AudioServer/`](Sources/AudioServer/)。

## 架构

speech-swift 把每个模型拆成独立的 SPM target，因此使用者只为 import 的模块付费。共享基础设施在 `AudioCommon`（协议、音频 I/O、HuggingFace 下载器、`SentencePieceModel`）和 `MLXCommon`（权重加载、`QuantizedLinear` 辅助工具、`SDPA` 多头注意力辅助工具）。

**[完整架构图（含后端、内存表、模块映射）→ soniqo.audio/architecture](https://soniqo.audio/zh/architecture)** · **[API 参考 → soniqo.audio/api](https://soniqo.audio/zh/api)** · **[基准测试 → soniqo.audio/benchmarks](https://soniqo.audio/zh/benchmarks)**

本地文档（仓库内）：
- **模型：** [Qwen3-ASR](docs/models/asr-model.md) · [WhisperASR](docs/models/whisper-asr.md) · [Qwen3-TTS](docs/models/tts-model.md) · [CosyVoice](docs/models/cosyvoice-tts.md) · [Kokoro](docs/models/kokoro-tts.md) · [VoxCPM2](docs/models/voxcpm2-tts.md) · [IndexTTS2 / voice-cloning candidates](docs/models/voice-cloning-candidate-runtimes.md) · [VibeVoice](docs/models/vibevoice.md) · [Supertonic](docs/models/supertonic-tts.md) · [Chatterbox](docs/models/chatterbox-tts.md) · [Indic-Mio](docs/models/indic-mio-tts.md) · [Fish Audio S2 Pro](docs/models/fish-audio-s2-pro.md) · [Magpie TTS](docs/models/magpie-tts.md) · [Parakeet TDT](docs/models/parakeet-asr.md) · [Parakeet Streaming](docs/models/parakeet-streaming-asr.md) · [Nemotron Streaming](docs/models/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/models/omnilingual-asr.md) · [PersonaPlex](docs/models/personaplex.md) · [Hibiki](docs/models/hibiki.md) · [MADLAD-400](docs/models/madlad-translation.md) · [FunctionGemma](docs/models/function-gemma.md) · [Qwen3.5 Chat](docs/models/qwen35-chat.md) · [Gemma 4 Chat](docs/models/gemma4-chat.md) · [Qwen3 Dense Chat](docs/models/qwen3-dense-chat.md) · [FireRedVAD](docs/models/fireredvad.md) · [KWS Zipformer](docs/models/kws-zipformer.md) · [Sidon](docs/models/sidon.md) · [Source Separation](docs/models/source-separation.md) · [HTDemucs](docs/models/htdemucs.md) · [MAGNeT](docs/models/magnet-music-gen.md) · [Stable Audio 3](docs/models/stable-audio-3.md) · [FlashSR](docs/models/flashsr.md) · [Audio2Face-3D](docs/models/audio2face3d.md)
- **推理：** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md) · [WhisperASR](docs/inference/whisper-asr-inference.md) · [Parakeet TDT](docs/inference/parakeet-asr-inference.md) · [Parakeet Streaming](docs/inference/parakeet-streaming-asr-inference.md) · [Nemotron Streaming](docs/inference/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/inference/omnilingual-asr-inference.md) · [TTS](docs/inference/qwen3-tts-inference.md) · [VoxCPM2](docs/inference/voxcpm2-inference.md) · [IndexTTS2 / voice-cloning candidates](docs/inference/voice-cloning-candidate-runtimes.md) · [VibeVoice](docs/inference/vibevoice-inference.md) · [Fish Audio S2 Pro](docs/inference/fish-audio-s2-pro.md) · [Magpie TTS](docs/inference/magpie-tts.md) · [Hibiki](docs/inference/hibiki-inference.md) · [MADLAD-400](docs/inference/madlad-translation.md) · [MAGNeT](docs/inference/magnet-music-gen.md) · [Stable Audio 3](docs/inference/stable-audio-3.md) · [FlashSR](docs/inference/flashsr.md) · [Forced Aligner](docs/inference/forced-aligner.md) · [Silero VAD](docs/inference/silero-vad.md) · [FireRedVAD](docs/inference/fireredvad.md) · [Wake-word](docs/inference/wake-word.md) · [Speaker Diarization](docs/inference/speaker-diarization.md) · [Speech Enhancement](docs/inference/speech-enhancement.md) · [Sidon](docs/inference/sidon.md) · [Cache/offline](docs/inference/cache-and-offline.md)
- **参考：** [共享协议](docs/shared-protocols.md)

## 缓存配置

模型权重在首次使用时从 HuggingFace 下载并缓存到 `~/Library/Caches/qwen3-speech/`。可通过 `QWEN3_CACHE_DIR`（CLI）或 `cacheDir:`（Swift API）覆盖。所有 `fromPretrained()` 入口都接受 `offlineMode: true`，在权重已缓存时跳过网络。

中国大陆用户（或任何 `huggingface.co` 访问缓慢/受限的地区）可通过设置 `HF_ENDPOINT` 从镜像下载，例如 `export HF_ENDPOINT=https://hf-mirror.com`。

详见 [`docs/inference/cache-and-offline.md`](docs/inference/cache-and-offline.md)，包含 iOS 沙盒容器路径等完整说明。

## MLX Metal 库

如果运行时出现 `Failed to load the default metallib`，说明缺少 Metal 着色器库。在 `swift build` 之后运行 `make build` 或 `./scripts/build_mlx_metallib.sh release`。如果缺少 Metal Toolchain，先安装：

```bash
xcodebuild -downloadComponent MetalToolchain
```

## 测试

```bash
make test                            # 完整套件（单元 + 需要下载模型的 E2E）
swift test --skip E2E                # 仅单元测试（CI 安全，无下载）
swift test --filter Qwen3ASRTests    # 指定模块
```

E2E 测试类使用 `E2E` 前缀，这样 CI 就可以用 `--skip E2E` 过滤掉它们。完整测试规范见 [CLAUDE.md](CLAUDE.md#testing)。

## 贡献

欢迎 PR——bug 修复、新模型集成、文档改进。fork、创建功能分支、`make build && make test`，然后向 `main` 提交 PR。

## 许可证

Apache 2.0
