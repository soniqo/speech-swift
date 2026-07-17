# Speech Swift

Mô hình AI giọng nói cho Apple Silicon, được hỗ trợ bởi MLX Swift và CoreML.

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

Nhận dạng, tổng hợp và hiểu giọng nói trên thiết bị cho Mac và iOS. Chạy cục bộ trên Apple Silicon — không cần đám mây, không cần khóa API, không có dữ liệu nào rời khỏi thiết bị của bạn.

**[📚 Tài liệu đầy đủ →](https://soniqo.audio)** · **[🤗 Mô hình HuggingFace](https://huggingface.co/aufklarer)** · **[📝 Blog](https://blog.ivan.digital)** · **[💬 Discord](https://discord.gg/TnCryqEMgu)**

<p align="center">
  <a href="https://formulae.brew.sh/formula/speech"><img src="https://img.shields.io/homebrew/installs/dm/speech.svg?logo=homebrew&amp;label=Homebrew%20installs&amp;color=FBB040" alt="Homebrew installs"></a>
  <a href="https://github.com/soniqo/speech-swift#built-with-speech-swift"><img src="https://img.shields.io/badge/verified%20public%20repositories-15-2ea44f?logo=github" alt="Verified public repositories: 15"></a>
</p>

<p align="center">
  <a href="https://trendshift.io/repositories/24196?utm_source=trendshift-badge&amp;utm_medium=badge&amp;utm_campaign=badge-trendshift-24196" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/trendshift/repositories/24196/daily?language=Swift" alt="soniqo%2Fspeech-swift | Trendshift" width="250" height="55"/></a>
</p>

<p align="center">
  <a href="https://youtu.be/x9zgcaW0gUk">
    <img src="https://img.youtube.com/vi/x9zgcaW0gUk/maxresdefault.jpg" width="640" alt="AI giọng nói cục bộ trên MacBook — xem trên YouTube video 4 phút giới thiệu thư viện mã nguồn mở">
  </a>
</p>
<p align="center"><em>AI giọng nói cục bộ trên MacBook — xem trên YouTube video 4 phút giới thiệu thư viện mã nguồn mở</em></p>

**Trường hợp sử dụng:** [Voice Agents](https://soniqo.audio/voice-agents) · [Phiên âm](https://soniqo.audio/transcription) · [Tổng hợp giọng nói](https://soniqo.audio/speech-generation)

## Được xây dựng bằng Speech Swift

15 kho mã nguồn công khai có tham chiếu xác minh được đến gói Speech Swift.

[Palmier Pro](https://github.com/palmier-io/palmier-pro) · [Anarlog](https://github.com/fastrepl/anarlog) · [ClawdHome](https://github.com/ThinkInAIXYZ/clawdhome) · [Jabber](https://github.com/rselbach/jabber) · [Ora](https://github.com/wuwangzhang1216/ora) · [VoxFlow](https://github.com/xingbofeng/VoxFlow) · [LokalBot](https://github.com/stevyhacker/lokalbot) · [Voicey](https://github.com/jonathanKingston/voicey) · [HushType](https://github.com/felixfu824/HushType) · [DexDictate macOS](https://github.com/westkitty/DexDictate_MacOS) · [Watchtower](https://github.com/aiwatchtowers/watchtower) · [Wishper App](https://github.com/irangareddy/wishper-app) · [FriSpeak](https://github.com/KSubedi/FriSpeak) · [Scribe](https://github.com/itchat/Scribe) · [VoicePen](https://github.com/dot-sk/VoicePen)

**Nhóm năng lực:** STT / ASR · Căn chỉnh · TTS · LLM và dịch thuật · Speech-to-Speech · Nâng cấp / phục hồi · Tách nguồn · Tạo nhạc / âm thanh · Wake word, VAD, diarization và định danh người nói

**STT / ASR**

- **[Qwen3-ASR](https://soniqo.audio/guides/transcribe)** — Chuyển giọng nói thành văn bản (nhận dạng giọng nói tự động, 52 ngôn ngữ, MLX + CoreML)
- **[WhisperASR](docs/models/whisper-asr.md)** — Whisper Large-v3 Turbo speech-to-text via native CoreML runtime (ANE, multilingual)
- **[Parakeet TDT](https://soniqo.audio/guides/parakeet)** — Chuyển giọng nói thành văn bản qua CoreML (Neural Engine, NVIDIA FastConformer + bộ giải mã TDT, 25 ngôn ngữ)
- **[Omnilingual ASR](https://soniqo.audio/guides/omnilingual)** — Chuyển giọng nói thành văn bản (Meta wav2vec2 + CTC, **1.672 ngôn ngữ** trên 32 hệ chữ viết, CoreML 300M + MLX 300M/1B/3B/7B)
- **[Đọc chính tả streaming](https://soniqo.audio/guides/dictate)** — Đọc chính tả thời gian thực với kết quả tạm thời và phát hiện kết thúc phát ngôn (Parakeet-EOU-120M)
- **[Nemotron Streaming (Đa ngôn ngữ)](https://soniqo.audio/guides/nemotron)** — ASR streaming độ trễ thấp với dấu câu và viết hoa tự nhiên (NVIDIA Nemotron-3.5-ASR-Streaming-0.6B, CoreML + MLX, **40 ngôn ngữ-khu vực**)
- **[Nemotron Streaming (Tiếng Anh)](https://soniqo.audio/guides/nemotron)** — ASR streaming độ trễ thấp với dấu câu và viết hoa tự nhiên (NVIDIA Nemotron-Speech-Streaming-0.6B, CoreML, chỉ tiếng Anh, nhỏ gọn và nhanh hơn biến thể đa ngôn ngữ)

**Căn chỉnh**

- **[Qwen3-ForcedAligner](https://soniqo.audio/guides/align)** — Căn chỉnh dấu thời gian theo cấp độ từ (audio + văn bản → dấu thời gian)

**TTS / Tổng hợp giọng nói**

- **[Qwen3-TTS](https://soniqo.audio/guides/speak)** — Tổng hợp giọng nói (chất lượng cao nhất, streaming, người nói tùy chỉnh, 10 ngôn ngữ)
- **[CosyVoice TTS](https://soniqo.audio/guides/cosyvoice)** — TTS streaming với nhân bản giọng nói, hội thoại nhiều người nói, thẻ cảm xúc (9 ngôn ngữ)
- **[VoxCPM2](https://soniqo.audio/speech-generation)** — TTS chất lượng phòng thu 48 kHz với nhân bản giọng nói + thiết kế giọng nói theo chỉ dẫn (2B, MLX bf16/int8, 30 ngôn ngữ)
- **[IndexTTS2](docs/models/indextts2.md)** — Native MLX voice cloning from a reference voice (IndexTeam IndexTTS-2, 1.5B-class fp16 bundle, speaker/emotion/pause controls)
- **[F5-TTS](docs/models/f5-tts.md)** — Zero-shot voice cloning from a short reference clip + transcript (SWivid F5-TTS v1 Base, DiT flow matching + Vocos, MLX fp16, 24 kHz, English + Mandarin; non-commercial license)
- **[Higgs TTS 3](docs/models/higgs-tts.md)** — Conversational TTS with zero-shot voice cloning and inline emotion/style/SFX/prosody tags (Boson Higgs TTS 3, Qwen3-4B backbone, MLX bf16, 24 kHz, 100+ languages; research/non-commercial license)
- **[Kokoro TTS](https://soniqo.audio/guides/kokoro)** — TTS trên thiết bị (82M, CoreML/Neural Engine, 54 giọng, sẵn sàng cho iOS, 10 ngôn ngữ)
- **[VibeVoice TTS](https://soniqo.audio/guides/vibevoice)** — TTS định dạng dài / nhiều người nói (Microsoft VibeVoice Realtime-0.5B + 1.5B, MLX, tổng hợp podcast/sách nói lên đến 90 phút, EN/ZH)
- **[Magpie TTS](https://soniqo.audio/guides/magpie)** — TTS đa ngôn ngữ (NVIDIA Magpie-TTS Multilingual 357M, MLX INT8 411 MB hoặc CoreML INT8 342 MB, 9 ngôn ngữ, 5 giọng nói có sẵn, streaming trên MLX)
- **[Supertonic TTS](https://soniqo.audio/guides/supertonic)** — TTS flow-matching trên thiết bị (Supertone Supertonic-3 99M, CoreML/Neural Engine, 31 ngôn ngữ, 10 giọng, không cần G2P, 44,1 kHz)
- **[Chatterbox TTS](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16)** — TTS đa ngôn ngữ với nhân bản giọng nói zero-shot (Resemble AI Chatterbox Multilingual, MLX fp16 ~1,3 GB, 23 ngôn ngữ runtime; tiếng Hebrew cần niqqud, MIT)
- **[OmniVoice TTS](https://huggingface.co/aufklarer/OmniVoice-MLX-fp16)** — TTS khuếch tán phi tự hồi quy với nhân bản giọng nói zero-shot (k2-fsa OmniVoice, backbone Qwen3, MLX fp16 mặc định / có int8, 600+ ngôn ngữ, Apache-2.0)
- **[Indic-Mio](docs/models/indic-mio-tts.md)** — Hindi/Indic TTS with inline emotion markers and optional reference-voice cloning (MLX, 24 kHz)

**LLM và dịch thuật**

- **[Qwen3Chat](https://soniqo.audio/guides/chat)** — Chat LLM trên thiết bị (Qwen3.5-0.8B MLX/CoreML cùng các backend MLX dense Qwen3 4B và Gemma 4 E2B/E4B, token streaming)
- **[FunctionGemma](https://soniqo.audio/guides/function-calls)** — LLM trên thiết bị cho các lệnh gọi hàm / công cụ có cấu trúc (Gemma 3 270M, CoreML palette hóa 8-bit, Neural Engine, ~252 tok/s)
- **[MADLAD-400](https://soniqo.audio/guides/translate)** — Dịch nhiều-sang-nhiều giữa hơn 400 ngôn ngữ (3B, MLX INT4 + INT8, T5 v1.1, Apache 2.0)

**Speech-to-Speech và voice agents**

- **[Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate)** — Dịch giọng nói sang giọng nói streaming (FR/ES/PT/DE → EN, MLX INT4 + INT8, stack Kyutai Moshi/Mimi, CC-BY-4.0)
- **[PersonaPlex](https://soniqo.audio/guides/respond)** — Giọng nói sang giọng nói full-duplex (7B, audio vào → audio ra, 18 preset giọng nói)
- **[Audio2Face-3D](docs/models/audio2face3d.md)** — Hoạt ảnh khuôn mặt avatar điều khiển bằng giọng nói (NVIDIA Audio2Face-3D v2.3 Mark, 301 hệ số khuôn mặt, MLX)

**Nâng cấp, tách nguồn và tạo âm thanh**

- **[DeepFilterNet3](https://soniqo.audio/guides/denoise)** — Khử nhiễu thời gian thực (2.1M tham số, 48 kHz)
- **[Tách nguồn](https://soniqo.audio/guides/separate)** — Tách nguồn âm thanh nhạc qua HTDemucs (Demucs v4) + Open-Unmix (UMX-HQ / UMX-L, 4 stem: giọng hát/trống/bass/khác, 44,1 kHz stereo)
- **[MAGNeT](https://soniqo.audio/guides/compose)** — Tạo nhạc từ văn bản (Meta MAGNeT Small 300M / Medium 1.5B, MLX INT8, đoạn 30 giây ở 32 kHz mono, giải mã song song có mặt nạ)
- **[Stable Audio 3](docs/models/stable-audio-3.md)** — Text-to-audio/music generation (Stable Audio 3 Medium, MLX INT8/INT4, 44.1 kHz stereo, variable length)
- **[FlashSR](https://soniqo.audio/guides/upsample)** — Siêu phân giải âm thanh (FlashSR ICASSP 2025, MLX, 48 kHz mono, diffusion chưng cất 1 bước, INT4 363 MB / INT8 720 MB)

**Phát hiện lượt nói, diarization và định danh người nói**

- **[Từ kích hoạt](https://soniqo.audio/guides/wake-word)** — Phát hiện từ khóa trên thiết bị (KWS Zipformer 3M, CoreML, 26× thời gian thực, danh sách từ khóa có thể cấu hình)
- **[VAD](https://soniqo.audio/guides/vad)** — Phát hiện hoạt động giọng nói (Silero streaming, Pyannote ngoại tuyến, FireRedVAD hơn 100 ngôn ngữ)
- **[Phân tách người nói](https://soniqo.audio/guides/diarize)** — Ai nói khi nào (pipeline Pyannote, Sortformer end-to-end trên Neural Engine) — nay có phiên streaming tăng dần (ID người nói ổn định, cập nhật mỗi 480 ms)
- **[Embedding người nói](https://soniqo.audio/guides/embed-speaker)** — WeSpeaker ResNet34 (256 chiều), ReDimNet2-B6 để nhận dạng giọng nói có tên (192 chiều), CAM++ (192 chiều)

Papers: [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba) · [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba) · [Omnilingual ASR](https://arxiv.org/abs/2511.09690) (Meta) · [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA) · [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba) · [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2) · [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA) · [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai) · [Hibiki](https://arxiv.org/abs/2502.03382) (Kyutai) · [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## Tin tức

- **19 thg 4, 2026** — [MLX so với CoreML trên Apple Silicon — Hướng dẫn thực tế để chọn backend phù hợp](https://blog.ivan.digital/mlx-vs-coreml-on-apple-silicon-a-practical-guide-to-picking-the-right-backend-and-why-you-should-f77ddea7b27a)
- **20 thg 3, 2026** — [Chúng tôi đánh bại Whisper Large v3 bằng mô hình 600M chạy hoàn toàn trên Mac của bạn](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **26 thg 2, 2026** — [Phân tách người nói và phát hiện hoạt động giọng nói trên Apple Silicon — Swift gốc với MLX](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **23 thg 2, 2026** — [NVIDIA PersonaPlex 7B trên Apple Silicon — Giọng nói sang giọng nói full-duplex bằng Swift gốc với MLX](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **12 thg 2, 2026** — [Qwen3-ASR Swift: ASR + TTS trên thiết bị cho Apple Silicon — Kiến trúc và benchmark](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## Bắt đầu nhanh

Thêm gói vào `Package.swift` của bạn:

```swift
.package(url: "https://github.com/soniqo/speech-swift", branch: "main")
```

Chỉ import những module bạn cần — mỗi mô hình là một thư viện SPM riêng, nên bạn không phải trả giá cho những gì không dùng:

```swift
.product(name: "ParakeetStreamingASR", package: "speech-swift"),
.product(name: "SpeechUI",             package: "speech-swift"),  // các view SwiftUI tùy chọn
```

**Phiên âm một buffer âm thanh trong 3 dòng:**

```swift
import ParakeetStreamingASR

let model = try await ParakeetStreamingASRModel.fromPretrained()
let text = try model.transcribeAudio(audioSamples, sampleRate: 16000)
```

**Streaming trực tiếp với kết quả tạm thời:**

```swift
for await partial in model.transcribeStream(audio: samples, sampleRate: 16000) {
    print(partial.isFinal ? "FINAL: \(partial.text)" : "... \(partial.text)")
}
```

**View đọc chính tả SwiftUI trong khoảng 10 dòng:**

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

`SpeechUI` chỉ cung cấp `TranscriptionView` (kết quả cuối + tạm thời) và `TranscriptionStore` (adapter ASR streaming). Hãy dùng AVFoundation để hiển thị trực quan và phát lại âm thanh.

Các sản phẩm SPM có sẵn: `Qwen3ASR`, `WhisperASR`, `Qwen3TTS`, `Qwen3TTSCoreML`, `ParakeetASR`, `ParakeetStreamingASR`, `NemotronStreamingASR`, `OmnilingualASR`, `KokoroTTS`, `SupertonicTTS`, `VibeVoiceTTS`, `CosyVoiceTTS`, `VoxCPM2TTS`, `IndexTTS2TTS`, `F5TTS`, `HiggsTTS`, `ChatterboxTTS`, `OmniVoiceTTS`, `IndicMioTTS`, `FishAudioTTS`, `MagpieTTS`, `MagpieTTSCoreML`, `MAGNeTMusicGen`, `StableAudio3MusicGen`, `FlashSR`, `PersonaPlex`, `Audio2Face3D`, `HibikiTranslate`, `MADLADTranslation`, `SpeechVAD`, `SpeechWakeWord`, `SpeechEnhancement`, `SpeechRestoration`, `SourceSeparation`, `Qwen3Chat`, `FunctionGemma`, `SpeechCore`, `SpeechUI`, `AudioCommon`.

## Mô hình

Xem tổng quan gọn bên dưới. **[Danh mục mô hình đầy đủ với kích thước, lượng tử hóa, URL tải xuống và bảng bộ nhớ → soniqo.audio/architecture](https://soniqo.audio/architecture)**.

| Mô hình | Tác vụ | Backend | Kích thước | Ngôn ngữ |
|-------|------|----------|-------|-----------|
| [Qwen3-ASR](https://soniqo.audio/guides/transcribe) | Giọng nói → Văn bản | MLX, CoreML (hybrid) | 0.6B, 1.7B | 52 |
| [WhisperASR](docs/models/whisper-asr.md) | Speech → Text | CoreML (ANE) | Large-v3 Turbo | Multi |
| [Parakeet TDT](https://soniqo.audio/guides/parakeet) | Giọng nói → Văn bản | CoreML (ANE) | 0.6B | 25 ngôn ngữ châu Âu |
| [Parakeet EOU](https://soniqo.audio/guides/dictate) | Giọng nói → Văn bản (streaming) | CoreML (ANE) | 120M | 25 ngôn ngữ châu Âu |
| [Nemotron Streaming (Đa ngôn ngữ)](https://soniqo.audio/guides/nemotron) | Giọng nói → Văn bản (streaming, có dấu câu) | CoreML (ANE), MLX | 0.6B | **40** |
| [Nemotron Streaming (Tiếng Anh)](https://soniqo.audio/guides/nemotron) | Giọng nói → Văn bản (streaming, có dấu câu) | CoreML (ANE) | 0.6B | EN |
| [Omnilingual ASR](https://soniqo.audio/guides/omnilingual) | Giọng nói → Văn bản | CoreML (ANE), MLX | 300M / 1B / 3B / 7B | **[1.672](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)** |
| [Qwen3-ForcedAligner](https://soniqo.audio/guides/align) | Audio + Văn bản → Dấu thời gian | MLX, CoreML | 0.6B | Đa ngôn ngữ |
| [Qwen3-TTS](https://soniqo.audio/guides/speak) | Văn bản → Giọng nói | MLX, CoreML | 0.6B, 1.7B | 10 |
| [CosyVoice3](https://soniqo.audio/guides/cosyvoice) | Văn bản → Giọng nói | MLX | 0.5B | 9 |
| [VoxCPM2](https://soniqo.audio/speech-generation) | Văn bản → Giọng nói (48 kHz, thiết kế giọng + nhân bản) | MLX | 2B (bf16/int8) | 30 |
| [IndexTTS2](docs/models/indextts2.md) | Text → Speech (zero-shot voice cloning) | MLX | 1.5B-class (fp16) | EN/ZH |
| [F5-TTS](docs/models/f5-tts.md) | Text → Speech (zero-shot voice cloning) | MLX | 336M (fp16) | EN/ZH |
| [Higgs TTS 3](docs/models/higgs-tts.md) | Text → Speech (conversational, zero-shot voice cloning) | MLX | 4B (bf16) | 100+ |
| [Kokoro-82M](https://soniqo.audio/guides/kokoro) | Văn bản → Giọng nói | CoreML (ANE) | 82M | 10 |
| [Supertonic-3](https://soniqo.audio/guides/supertonic) | Văn bản → Giọng nói (44,1 kHz, flow-matching, không cần G2P) | CoreML (ANE) | 99M | 31 |
| [VibeVoice Realtime-0.5B](https://soniqo.audio/guides/vibevoice) | Văn bản → Giọng nói (định dạng dài, nhiều người nói) | MLX | 0.5B | EN/ZH |
| [VibeVoice 1.5B](https://soniqo.audio/guides/vibevoice) | Văn bản → Giọng nói (podcast đến 90 phút) | MLX | 1.5B | EN/ZH |
| [Magpie-TTS Multilingual](https://soniqo.audio/guides/magpie) | Văn bản → Giọng nói (5 giọng có sẵn, streaming) | MLX / CoreML | 357M (MLX INT8, CoreML INT8) | 9 (CoreML loại trừ JA) |
| [Chatterbox Multilingual](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16) | Văn bản → Giọng nói (nhân bản zero-shot) | MLX | 0.8B (fp16) | 23 (HE cần niqqud) |
| [OmniVoice](https://huggingface.co/aufklarer/OmniVoice-MLX-fp16) | Văn bản → Giọng nói (khuếch tán NAR, nhân bản zero-shot) | MLX | 0.8B (fp16 mặc định / int8) | **600+** |
| [Indic-Mio](docs/models/indic-mio-tts.md) | Text → Speech (Hindi/Indic, emotion tags, voice cloning) | MLX | fp16 | Hindi / Indic |
| [Fish Audio S2 Pro](docs/models/fish-audio-s2-pro.md) | Văn bản → Giọng nói (nhân bản zero-shot, marker phong cách tường minh) | MLX | 0.5B-class (fp16) | Đa ngôn ngữ |
| [Qwen3.5 Chat](docs/models/qwen35-chat.md) | Text → Text (LLM) | MLX, CoreML | 0.8B | Multi |
| [Qwen3 Dense Chat](docs/models/qwen3-dense-chat.md) | Text → Text (LLM) | MLX | 4B | Multi |
| [Gemma 4 Chat](docs/models/gemma4-chat.md) | Text → Text (LLM) | MLX | E2B / E4B (4-bit) | Multi |
| [FunctionGemma](docs/models/function-gemma.md) | Văn bản → Lệnh gọi công cụ (LLM) | CoreML | 270M | EN |
| [MADLAD-400](https://soniqo.audio/guides/translate) | Văn bản → Văn bản (Dịch) | MLX | 3B | **400+** |
| [Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate) | Giọng nói → Giọng nói (Dịch) | MLX | 3B | FR/ES/PT/DE → EN |
| [PersonaPlex](https://soniqo.audio/guides/respond) | Giọng nói → Giọng nói | MLX | 7B | EN |
| [Audio2Face-3D](docs/models/audio2face3d.md) | Giọng nói → Hoạt ảnh khuôn mặt | MLX | v2.3 Mark | Không phụ thuộc ngôn ngữ |
| [Silero VAD](https://soniqo.audio/guides/vad) | Phát hiện hoạt động giọng nói | MLX, CoreML | 309K | Không phụ thuộc ngôn ngữ |
| [KWS Zipformer](docs/models/kws-zipformer.md) | Audio → Wake word | CoreML (ANE) | 3M | EN/custom keywords |
| [Pyannote](https://soniqo.audio/guides/diarize) | VAD + Phân tách người nói | MLX | 1.5M | Không phụ thuộc ngôn ngữ |
| [Pyannote Community-1](https://huggingface.co/aufklarer/Pyannote-Community-1-CoreML) | Phân tách người nói + embedding người nói | CoreML (ANE) + Swift VBx | 8.35M | Không phụ thuộc ngôn ngữ |
| [Sortformer](https://huggingface.co/aufklarer/Sortformer-Diarization-CoreML) | [Phân tách người nói (E2E), streaming tăng dần](https://soniqo.audio/vi/guides/diarize) | CoreML (ANE) | 117M | Không phụ thuộc ngôn ngữ |
| [DeepFilterNet3](https://soniqo.audio/guides/denoise) | Cải thiện chất lượng giọng nói | CoreML | 2.1M | Không phụ thuộc ngôn ngữ |
| [Sidon](https://soniqo.audio/guides/restore) | Phục hồi giọng nói (khử nhiễu + khử vang, 48 kHz) | CoreML | w2v-BERT 2.0 + DAC (fp16/int8) | Không phụ thuộc ngôn ngữ |
| [HTDemucs (Demucs v4)](https://soniqo.audio/guides/separate) | Tách nguồn | MLX | 168M | Không phụ thuộc ngôn ngữ |
| [Open-Unmix](https://soniqo.audio/guides/separate) | Tách nguồn | MLX | 8.6M | Không phụ thuộc ngôn ngữ |
| [MAGNeT](https://soniqo.audio/guides/compose) | Văn bản → Nhạc (30s @ 32 kHz) | MLX | 300M / 1.5B (int4/int8) | Prompt EN |
| [Stable Audio 3](docs/models/stable-audio-3.md) | Text → Music/audio (44.1 kHz stereo) | MLX | Medium 1.4B (int4/int8) | EN prompts |
| [FlashSR](https://soniqo.audio/guides/upsample) | Siêu phân giải âm thanh (48 kHz) | MLX | 363 MB / 720 MB (int4/int8) | Không phụ thuộc ngôn ngữ |
| [WeSpeaker](https://soniqo.audio/guides/embed-speaker) | Embedding người nói | MLX, CoreML | 6.6M | Không phụ thuộc ngôn ngữ |
| [ReDimNet2-B6](https://huggingface.co/aufklarer/ReDimNet2-B6-CoreML) | Nhận dạng giọng nói có tên | CoreML | 12.3M | Không phụ thuộc ngôn ngữ |

## Cài đặt

### Homebrew

Yêu cầu Homebrew ARM gốc (`/opt/homebrew`). Homebrew Rosetta/x86_64 không được hỗ trợ.

```bash
brew install speech
```

Sau đó:

```bash
speech transcribe recording.wav
speech speak "Hello world"
speech translate "Hello, how are you?" --to es
speech respond --input question.wav --transcript
speech-server --port 8080            # máy chủ HTTP / WebSocket cục bộ (tương thích OpenAI /v1/realtime + /v1/audio/transcriptions)
```

**[Tài liệu CLI đầy đủ →](https://soniqo.audio/cli)**

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

Chỉ import những gì bạn cần — mỗi mô hình là target SPM riêng:

```swift
import Qwen3ASR             // Nhận dạng giọng nói (MLX)
import WhisperASR           // Whisper Large-v3 Turbo (CoreML)
import ParakeetASR          // Nhận dạng giọng nói (CoreML, batch)
import ParakeetStreamingASR // Đọc chính tả streaming với kết quả tạm thời + EOU
import NemotronStreamingASR // ASR streaming đa ngôn ngữ với dấu câu tự nhiên (0.6B, 40 ngôn ngữ)
import OmnilingualASR       // 1.672 ngôn ngữ (CoreML + MLX)
import Qwen3TTS             // Tổng hợp giọng nói
import CosyVoiceTTS         // Tổng hợp giọng nói với nhân bản giọng
import VoxCPM2TTS           // TTS 48 kHz với nhân bản giọng + thiết kế giọng (2B)
import IndexTTS2TTS         // Native MLX voice cloning from reference audio
import F5TTS                // Zero-shot voice cloning (DiT flow matching + Vocos)
import HiggsTTS             // Conversational TTS + cloning (Qwen3 backbone, control tags)
import KokoroTTS            // Tổng hợp giọng nói (sẵn sàng cho iOS)
import VibeVoiceTTS         // TTS định dạng dài / nhiều người nói (EN/ZH)
import MagpieTTS            // TTS đa ngôn ngữ (NVIDIA Magpie 357M, MLX, 9 ngôn ngữ)
import MagpieTTSCoreML      // Backend CoreML của Magpie (hybrid CoreML + MLX, 8 ngôn ngữ)
import FishAudioTTS         // Runtime Fish Audio S2 Pro thử nghiệm với nhân bản giọng
import Qwen3Chat            // Chat LLM trên thiết bị
import FunctionGemma    // LLM trên thiết bị cho lệnh gọi công cụ
import MADLADTranslation    // Dịch nhiều-sang-nhiều giữa hơn 400 ngôn ngữ
import HibikiTranslate      // Dịch giọng nói sang giọng nói streaming (FR/ES/PT/DE → EN)
import PersonaPlex          // Giọng nói sang giọng nói full-duplex
import SpeechVAD            // VAD + phân tách người nói + embedding
import SpeechEnhancement    // Khử nhiễu
import SpeechRestoration    // Phục hồi giọng nói — khử nhiễu + khử vang (Sidon, CoreML, 48 kHz)
import SourceSeparation     // Tách nguồn âm thanh nhạc (Open-Unmix, 4 stem)
import SpeechUI             // Component SwiftUI cho bản phiên âm streaming
import AudioCommon          // Giao thức và tiện ích dùng chung
```

### Yêu cầu

- Swift 6+, Xcode 16+ (kèm Metal Toolchain)
- macOS 15+ (Sequoia) hoặc iOS 18+, Apple Silicon (M1/M2/M3/M4)

Yêu cầu tối thiểu macOS 15 / iOS 18 đến từ [MLState](https://developer.apple.com/documentation/coreml/mlstate) — API trạng thái ANE bền vững của Apple mà các pipeline CoreML (Qwen3-ASR, Qwen3-Chat, Qwen3-TTS) sử dụng để giữ cache KV thường trú trên Neural Engine giữa các bước token.

### Build từ mã nguồn

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

`make build` biên dịch gói Swift **và** thư viện shader MLX Metal. Thư viện Metal là bắt buộc cho suy luận GPU — không có nó bạn sẽ thấy `Failed to load the default metallib` khi chạy. `make debug` cho bản build debug, `make test` cho bộ kiểm thử.

**[Hướng dẫn build và cài đặt đầy đủ →](https://soniqo.audio/getting-started)**

## Ứng dụng demo

- **[DictateDemo](Examples/DictateDemo/)** ([tài liệu](https://soniqo.audio/guides/dictate)) — Đọc chính tả streaming trên thanh menu macOS với kết quả tạm thời trực tiếp, phát hiện kết thúc phát ngôn dựa trên VAD và sao chép một cú nhấp. Chạy như agent nền (Parakeet-EOU-120M + Silero VAD).
- **[iOSEchoDemo](Examples/iOSEchoDemo/)** — Demo echo iOS (Parakeet ASR + Kokoro TTS). Thiết bị và simulator.
- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — Trợ lý giọng nói hội thoại với đầu vào micro, VAD và ngữ cảnh nhiều lượt. macOS. RTF ~0.94 trên M2 Max (nhanh hơn thời gian thực).
- **[SpeechDemo](Examples/SpeechDemo/)** — Đọc chính tả và tổng hợp TTS trong giao diện dạng tab. macOS.

README của từng demo có hướng dẫn build.

## Ví dụ mã

Các đoạn mã dưới đây cho thấy con đường tối thiểu cho mỗi lĩnh vực. Mỗi phần liên kết đến hướng dẫn đầy đủ trên [soniqo.audio](https://soniqo.audio) với tùy chọn cấu hình, nhiều backend, mẫu streaming và công thức CLI.

### Chuyển giọng nói thành văn bản — [hướng dẫn đầy đủ →](https://soniqo.audio/guides/transcribe)

```swift
import Qwen3ASR

let model = try await Qwen3ASRModel.fromPretrained()
let text = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

Backend thay thế: [WhisperASR](docs/inference/whisper-asr-inference.md) (Whisper Large-v3 Turbo, native CoreML), [Parakeet TDT](https://soniqo.audio/guides/parakeet) (CoreML, 32× thời gian thực), [Omnilingual ASR](https://soniqo.audio/guides/omnilingual) (1.672 ngôn ngữ, CoreML hoặc MLX), [Đọc chính tả streaming](https://soniqo.audio/guides/dictate) (kết quả tạm thời trực tiếp).

### Căn chỉnh cưỡng bức — [hướng dẫn đầy đủ →](https://soniqo.audio/guides/align)

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

### Tổng hợp giọng nói — [hướng dẫn đầy đủ →](https://soniqo.audio/guides/speak)

```swift
import Qwen3TTS
import AudioCommon

let model = try await Qwen3TTSModel.fromPretrained()
let audio = model.synthesize(text: "Hello world", language: "english")
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

Các engine TTS thay thế: [CosyVoice3](https://soniqo.audio/guides/cosyvoice) (streaming + nhân bản giọng + thẻ cảm xúc), [Kokoro-82M](https://soniqo.audio/guides/kokoro) (sẵn sàng cho iOS, 54 giọng), [VibeVoice](https://soniqo.audio/guides/vibevoice) (podcast định dạng dài / nhiều người nói, EN/ZH), [Fish Audio S2 Pro](docs/inference/fish-audio-s2-pro.md) (nhân bản zero-shot thử nghiệm + marker phong cách trong ngoặc vuông), [Nhân bản giọng nói](https://soniqo.audio/guides/voice-cloning).

### Giọng nói sang giọng nói — [hướng dẫn đầy đủ →](https://soniqo.audio/guides/respond)

```swift
import PersonaPlex

let model = try await PersonaPlexModel.fromPretrained()
let responseAudio = model.respond(userAudio: userSamples)
// Đầu ra Float32 mono 24 kHz sẵn sàng để phát lại
```

### Chat LLM — [hướng dẫn đầy đủ →](https://soniqo.audio/guides/chat)

```swift
import Qwen3Chat
import FunctionGemma

let chat = try await Qwen35MLXChat.fromPretrained()
chat.chat(messages: [(.user, "Explain MLX in one sentence")]) { token, isFinal in
    print(token, terminator: "")
}
```

### Dịch — [hướng dẫn đầy đủ →](https://soniqo.audio/guides/translate)

```swift
import MADLADTranslation

let translator = try await MADLADTranslator.fromPretrained()
let es = try translator.translate("Hello, how are you?", to: "es")
// → "Hola, ¿cómo estás?"
```

### Dịch giọng nói — [hướng dẫn đầy đủ →](https://soniqo.audio/guides/audio-translate)

```swift
import HibikiTranslate
import AudioCommon

let model = try await HibikiTranslateModel.fromPretrained()
let pcm = try AudioFileLoader.load(url: input, targetSampleRate: 24000)
let (englishAudio, textTokens) = model.translate(
    sourceAudio: pcm, sourceLanguage: .fr
)
// Hibiki Zero-3B — FR/ES/PT/DE → EN, trên thiết bị, codec Mimi streaming
```

### Phát hiện hoạt động giọng nói — [hướng dẫn đầy đủ →](https://soniqo.audio/guides/vad)

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
for s in segments { print("\(s.startTime)s → \(s.endTime)s") }
```

### Phân tách người nói — [hướng dẫn đầy đủ →](https://soniqo.audio/guides/diarize)

```swift
import SpeechVAD

let diarizer = try await DiarizationPipeline.fromPretrained()
let segments = diarizer.diarize(audio: samples, sampleRate: 16000)
for s in segments { print("Speaker \(s.speakerId): \(s.startTime)s - \(s.endTime)s") }
```

### Cải thiện chất lượng giọng nói — [hướng dẫn đầy đủ →](https://soniqo.audio/guides/denoise)

```swift
import SpeechEnhancement

let denoiser = try await DeepFilterNet3Model.fromPretrained()
let clean = try denoiser.enhance(audio: noisySamples, sampleRate: 48000)
```

### Phục hồi giọng nói — [hướng dẫn đầy đủ →](https://soniqo.audio/guides/restore)

Khử nhiễu **và** khử vang đồng thời với [Sidon](https://arxiv.org/abs/2509.17052) (bộ dự đoán w2v-BERT 2.0 + bộ vocoder DAC, Core ML). Khác với một bộ khử nhiễu thông thường, Sidon được huấn luyện để giữ nguyên đặc điểm nhận dạng người nói, nên rất phù hợp để làm sạch một mẫu tham chiếu nhân bản giọng nói bị nhiễu hoặc bị vang trước khi TTS. Đầu vào là 16 kHz; đầu ra là 48 kHz mono.

```swift
import SpeechRestoration

let restorer = try await SpeechRestorer.fromPretrained()          // .fp16 (default) or .int8
let clean = try restorer.restore(audio: noisySamples, sampleRate: 16000)  // → 48 kHz
```

Từ CLI:

```bash
speech restore noisy.wav -o clean.wav            # denoise + dereverb, 48 kHz output
speech restore noisy.wav --variant int8          # smaller, lower peak RAM

# Clean a voice-cloning reference before TTS (opt-in; preserves speaker identity):
speech speak "Hello" --engine voxcpm2 --voice-sample ref.wav --clean-reference
```

### Voice Pipeline (ASR → LLM → TTS) — [hướng dẫn đầy đủ →](https://soniqo.audio/voice-agents)

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

`VoicePipeline` là máy trạng thái voice-agent thời gian thực (được hỗ trợ bởi [speech-core](https://github.com/soniqo/speech-core)) với phát hiện lượt nói dựa trên VAD, xử lý ngắt lời và STT eager. Nó kết nối bất kỳ `SpeechRecognitionModel` + `SpeechGenerationModel` + `StreamingVADProvider`.

### Máy chủ API HTTP

```bash
speech-server --port 8080
```

Cung cấp mỗi mô hình thông qua endpoint HTTP REST + WebSocket, bao gồm các API tương thích OpenAI: Realtime WebSocket tại `/v1/realtime` và endpoint REST phiên âm tại `/v1/audio/transcriptions`. Xem [`Sources/AudioServer/`](Sources/AudioServer/).

## Kiến trúc

speech-swift được chia thành một target SPM cho mỗi mô hình để người dùng chỉ phải trả giá cho những gì họ import. Hạ tầng dùng chung nằm trong `AudioCommon` (giao thức, I/O âm thanh, trình tải HuggingFace, `SentencePieceModel`) và `MLXCommon` (tải trọng số, helper `QuantizedLinear`, helper attention multi-head `SDPA`).

**[Sơ đồ kiến trúc đầy đủ với backend, bảng bộ nhớ và bản đồ module → soniqo.audio/architecture](https://soniqo.audio/architecture)** · **[Tài liệu API → soniqo.audio/api](https://soniqo.audio/api)** · **[Benchmark → soniqo.audio/benchmarks](https://soniqo.audio/benchmarks)**

Tài liệu cục bộ (kho repo):
- **Mô hình:** [Qwen3-ASR](docs/models/asr-model.md) · [WhisperASR](docs/models/whisper-asr.md) · [Qwen3-TTS](docs/models/tts-model.md) · [CosyVoice](docs/models/cosyvoice-tts.md) · [Kokoro](docs/models/kokoro-tts.md) · [VoxCPM2](docs/models/voxcpm2-tts.md) · [IndexTTS2](docs/models/indextts2.md) · [F5-TTS](docs/models/f5-tts.md) · [Higgs TTS 3](docs/models/higgs-tts.md) · [VibeVoice](docs/models/vibevoice.md) · [Supertonic](docs/models/supertonic-tts.md) · [Chatterbox](docs/models/chatterbox-tts.md) · [Indic-Mio](docs/models/indic-mio-tts.md) · [Fish Audio S2 Pro](docs/models/fish-audio-s2-pro.md) · [Magpie TTS](docs/models/magpie-tts.md) · [Parakeet TDT](docs/models/parakeet-asr.md) · [Parakeet Streaming](docs/models/parakeet-streaming-asr.md) · [Nemotron Streaming](docs/models/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/models/omnilingual-asr.md) · [PersonaPlex](docs/models/personaplex.md) · [Hibiki](docs/models/hibiki.md) · [MADLAD-400](docs/models/madlad-translation.md) · [FunctionGemma](docs/models/function-gemma.md) · [Qwen3.5 Chat](docs/models/qwen35-chat.md) · [Gemma 4 Chat](docs/models/gemma4-chat.md) · [Qwen3 Dense Chat](docs/models/qwen3-dense-chat.md) · [FireRedVAD](docs/models/fireredvad.md) · [KWS Zipformer](docs/models/kws-zipformer.md) · [Sidon](docs/models/sidon.md) · [Source Separation](docs/models/source-separation.md) · [HTDemucs](docs/models/htdemucs.md) · [MAGNeT](docs/models/magnet-music-gen.md) · [Stable Audio 3](docs/models/stable-audio-3.md) · [FlashSR](docs/models/flashsr.md) · [Audio2Face-3D](docs/models/audio2face3d.md)
- **Suy luận:** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md) · [WhisperASR](docs/inference/whisper-asr-inference.md) · [Parakeet TDT](docs/inference/parakeet-asr-inference.md) · [Parakeet Streaming](docs/inference/parakeet-streaming-asr-inference.md) · [Nemotron Streaming](docs/inference/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/inference/omnilingual-asr-inference.md) · [TTS](docs/inference/qwen3-tts-inference.md) · [VoxCPM2](docs/inference/voxcpm2-inference.md) · [IndexTTS2](docs/inference/indextts2.md) · [F5-TTS](docs/inference/f5-tts.md) · [Higgs TTS 3](docs/inference/higgs-tts.md) · [VibeVoice](docs/inference/vibevoice-inference.md) · [Fish Audio S2 Pro](docs/inference/fish-audio-s2-pro.md) · [Magpie TTS](docs/inference/magpie-tts.md) · [Hibiki](docs/inference/hibiki-inference.md) · [MADLAD-400](docs/inference/madlad-translation.md) · [MAGNeT](docs/inference/magnet-music-gen.md) · [Stable Audio 3](docs/inference/stable-audio-3.md) · [FlashSR](docs/inference/flashsr.md) · [Forced Aligner](docs/inference/forced-aligner.md) · [Silero VAD](docs/inference/silero-vad.md) · [FireRedVAD](docs/inference/fireredvad.md) · [Wake-word](docs/inference/wake-word.md) · [Speaker Diarization](docs/inference/speaker-diarization.md) · [Speech Enhancement](docs/inference/speech-enhancement.md) · [Sidon](docs/inference/sidon.md) · [Cache/offline](docs/inference/cache-and-offline.md)
- **Tài liệu tham khảo:** [Giao thức dùng chung](docs/shared-protocols.md)

## Cấu hình cache

Trọng số mô hình tải xuống từ HuggingFace ở lần dùng đầu tiên và lưu cache vào `~/Library/Caches/qwen3-speech/`. Ghi đè bằng `QWEN3_CACHE_DIR` (CLI) hoặc `cacheDir:` (Swift API). Tất cả điểm vào `fromPretrained()` cũng chấp nhận `offlineMode: true` để bỏ qua mạng khi trọng số đã có trong cache.

Xem [`docs/inference/cache-and-offline.md`](docs/inference/cache-and-offline.md) để biết chi tiết đầy đủ, bao gồm đường dẫn container iOS sandboxed.

## Thư viện MLX Metal

Nếu bạn thấy `Failed to load the default metallib` khi chạy, thư viện shader Metal đang bị thiếu. Chạy `make build` hoặc `./scripts/build_mlx_metallib.sh release` sau khi `swift build` thủ công. Nếu thiếu Metal Toolchain, hãy cài đặt nó trước:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Kiểm thử

```bash
make test                            # bộ đầy đủ (unit + E2E với tải mô hình)
swift test --skip E2E                # chỉ unit (an toàn cho CI, không tải)
swift test --filter Qwen3ASRTests    # module cụ thể
```

Các lớp test E2E sử dụng tiền tố `E2E` để CI có thể loại bỏ chúng bằng `--skip E2E`. Xem [CLAUDE.md](CLAUDE.md#testing) để biết quy ước kiểm thử đầy đủ.

## Đóng góp

Hoan nghênh PR — sửa lỗi, tích hợp mô hình mới, tài liệu. Fork, tạo nhánh feature, `make build && make test`, mở PR vào `main`.

## Giấy phép

Apache 2.0
