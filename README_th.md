# Speech Swift

โมเดล AI สำหรับเสียงพูดบน Apple Silicon ขับเคลื่อนด้วย MLX Swift และ CoreML

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

การรู้จำเสียงพูด การสังเคราะห์เสียง และการทำความเข้าใจเสียงพูดบนอุปกรณ์สำหรับ Mac และ iOS ทำงานบนเครื่องด้วย Apple Silicon — ไม่ใช้คลาวด์ ไม่ต้องใช้คีย์ API และไม่มีข้อมูลใดออกจากอุปกรณ์ของคุณ

**[📚 เอกสารฉบับเต็ม →](https://soniqo.audio)** · **[🤗 โมเดลบน HuggingFace](https://huggingface.co/aufklarer)** · **[📝 Blog](https://blog.ivan.digital)** · **[💬 Discord](https://discord.gg/TnCryqEMgu)**

<p align="center">
  <a href="https://trendshift.io/repositories/24196?utm_source=trendshift-badge&amp;utm_medium=badge&amp;utm_campaign=badge-trendshift-24196" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/trendshift/repositories/24196/daily?language=Swift" alt="soniqo%2Fspeech-swift | Trendshift" width="250" height="55"/></a>
</p>

<p align="center">
  <a href="https://youtu.be/x9zgcaW0gUk">
    <img src="https://img.youtube.com/vi/x9zgcaW0gUk/maxresdefault.jpg" width="640" alt="Speech AI ที่ทำงานในเครื่องบน MacBook — ชมทัวร์ไลบรารีโอเพนซอร์สความยาว 4 นาทีบน YouTube">
  </a>
</p>
<p align="center"><em>Speech AI ที่ทำงานในเครื่องบน MacBook — ชมทัวร์ไลบรารีโอเพนซอร์สความยาว 4 นาทีบน YouTube</em></p>

**กรณีการใช้งาน:** [Voice Agents](https://soniqo.audio/voice-agents) · [การถอดเสียง](https://soniqo.audio/transcription) · [การสังเคราะห์เสียงพูด](https://soniqo.audio/speech-generation)

**กลุ่มความสามารถ:** STT / ASR · การจัดแนว · TTS · LLM และการแปล · Speech-to-Speech · การปรับปรุง / การฟื้นฟู · การแยกแหล่งเสียง · การสร้างดนตรี / เสียง · Wake word, VAD, diarization และอัตลักษณ์ผู้พูด

**STT / ASR**

- **[Qwen3-ASR](https://soniqo.audio/guides/transcribe)** — แปลงเสียงพูดเป็นข้อความ (การรู้จำเสียงพูดอัตโนมัติ รองรับ 52 ภาษา MLX + CoreML)
- **[WhisperASR](docs/models/whisper-asr.md)** — Whisper Large-v3 Turbo speech-to-text via native CoreML runtime (ANE, multilingual)
- **[Parakeet TDT](https://soniqo.audio/guides/parakeet)** — แปลงเสียงพูดเป็นข้อความผ่าน CoreML (Neural Engine, NVIDIA FastConformer + ตัวถอดรหัส TDT รองรับ 25 ภาษา)
- **[Omnilingual ASR](https://soniqo.audio/guides/omnilingual)** — แปลงเสียงพูดเป็นข้อความ (Meta wav2vec2 + CTC รองรับ **1,672 ภาษา** ครอบคลุม 32 ระบบอักษร, CoreML 300M + MLX 300M/1B/3B/7B)
- **[Streaming Dictation](https://soniqo.audio/guides/dictate)** — การเขียนตามคำบอกแบบเรียลไทม์พร้อมผลลัพธ์บางส่วนและการตรวจจับจุดจบของประโยค (Parakeet-EOU-120M)
- **[Nemotron Streaming (หลายภาษา)](https://soniqo.audio/guides/nemotron)** — ASR แบบสตรีมมิ่งที่มีความหน่วงต่ำ พร้อมเครื่องหมายวรรคตอนและตัวพิมพ์ใหญ่ในตัว (NVIDIA Nemotron-3.5-ASR-Streaming-0.6B, CoreML + MLX, **40 ภาษา-ตำแหน่ง**)
- **[Nemotron Streaming (อังกฤษ)](https://soniqo.audio/guides/nemotron)** — ASR แบบสตรีมมิ่งที่มีความหน่วงต่ำ พร้อมเครื่องหมายวรรคตอนและตัวพิมพ์ใหญ่ในตัว (NVIDIA Nemotron-Speech-Streaming-0.6B, CoreML, ภาษาอังกฤษเท่านั้น เล็กและเร็วกว่ารุ่นหลายภาษา)

**การจัดแนว**

- **[Qwen3-ForcedAligner](https://soniqo.audio/guides/align)** — การจัดเรียงเครื่องหมายเวลาในระดับคำ (เสียง + ข้อความ → เครื่องหมายเวลา)

**TTS / การสังเคราะห์เสียงพูด**

- **[Qwen3-TTS](https://soniqo.audio/guides/speak)** — การสังเคราะห์เสียงพูด (คุณภาพสูงสุด สตรีมมิ่ง ผู้พูดที่กำหนดเอง 10 ภาษา)
- **[CosyVoice TTS](https://soniqo.audio/guides/cosyvoice)** — TTS แบบสตรีมมิ่งพร้อมการโคลนเสียง บทสนทนาหลายผู้พูด และแท็กอารมณ์ (9 ภาษา)
- **[VoxCPM2](https://soniqo.audio/speech-generation)** — TTS คุณภาพระดับสตูดิโอที่ 48 kHz พร้อมการโคลนเสียง + การออกแบบเสียงผ่านคำสั่ง (2B, MLX bf16/int8, 30 ภาษา)
- **[IndexTTS2](docs/models/indextts2.md)** — Native MLX voice cloning from a reference voice (IndexTeam IndexTTS-2, 1.5B-class fp16 bundle, speaker/emotion/pause controls)
- **[F5-TTS](docs/models/f5-tts.md)** — Zero-shot voice cloning from a short reference clip + transcript (SWivid F5-TTS v1 Base, DiT flow matching + Vocos, MLX fp16, 24 kHz, English + Mandarin; non-commercial license)
- **[Higgs TTS 3](docs/models/higgs-tts.md)** — Conversational TTS with zero-shot voice cloning and inline emotion/style/SFX/prosody tags (Boson Higgs TTS 3, Qwen3-4B backbone, MLX bf16, 24 kHz, 100+ languages; research/non-commercial license)
- **[Kokoro TTS](https://soniqo.audio/guides/kokoro)** — TTS บนอุปกรณ์ (82M, CoreML/Neural Engine, 54 เสียง พร้อมใช้งานบน iOS, 10 ภาษา)
- **[VibeVoice TTS](https://soniqo.audio/guides/vibevoice)** — TTS แบบยาว / หลายผู้พูด (Microsoft VibeVoice Realtime-0.5B + 1.5B, MLX สังเคราะห์พอดแคสต์/หนังสือเสียงได้นานสูงสุด 90 นาที, EN/ZH)
- **[Magpie TTS](https://soniqo.audio/guides/magpie)** — TTS หลายภาษา (NVIDIA Magpie-TTS Multilingual 357M, MLX INT8 411 MB หรือ CoreML INT8 342 MB, 9 ภาษา, 5 ผู้พูดสำเร็จรูป สตรีมมิ่งบน MLX)
- **[Supertonic TTS](https://soniqo.audio/guides/supertonic)** — TTS แบบ flow-matching บนอุปกรณ์ (Supertone Supertonic-3 99M, CoreML/Neural Engine, 31 ภาษา, 10 เสียง, G2P-free, 44.1 kHz)
- **[Chatterbox TTS](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16)** — TTS หลายภาษาพร้อมการโคลนเสียงแบบ zero-shot (Resemble AI Chatterbox Multilingual, MLX fp16 ~1.3 GB, 23 ภาษาที่รันไทม์; ภาษาฮีบรูต้องมี niqqud, MIT)
- **[OmniVoice TTS](https://huggingface.co/aufklarer/OmniVoice-MLX-fp16)** — TTS แบบ diffusion non-autoregressive พร้อมการโคลนเสียงแบบ zero-shot (k2-fsa OmniVoice, backbone Qwen3, MLX fp16 เป็นค่าเริ่มต้น / มี int8, 600+ ภาษา, Apache-2.0)
- **[Indic-Mio](docs/models/indic-mio-tts.md)** — Hindi/Indic TTS with inline emotion markers and optional reference-voice cloning (MLX, 24 kHz)

**LLM และการแปล**

- **[Qwen3Chat](https://soniqo.audio/guides/chat)** — แชท LLM บนอุปกรณ์ (Qwen3.5-0.8B MLX/CoreML พร้อมแบ็กเอนด์ MLX ของ dense Qwen3 4B และ Gemma 4 E2B/E4B, สตรีมมิ่งโทเค็น)
- **[FunctionGemma](https://soniqo.audio/guides/function-calls)** — LLM บนอุปกรณ์สำหรับการเรียกฟังก์ชัน / เครื่องมือแบบมีโครงสร้าง (Gemma 3 270M, CoreML 8-bit palettize, Neural Engine, ~252 tok/s)
- **[MADLAD-400](https://soniqo.audio/guides/translate)** — การแปลแบบหลายต่อหลายระหว่างกว่า 400 ภาษา (3B, MLX INT4 + INT8, T5 v1.1, Apache 2.0)

**Speech-to-Speech และ voice agents**

- **[Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate)** — การแปลเสียงพูดสู่เสียงพูดแบบสตรีมมิ่ง (FR/ES/PT/DE → EN, MLX INT4 + INT8, สแต็ก Kyutai Moshi/Mimi, CC-BY-4.0)
- **[PersonaPlex](https://soniqo.audio/guides/respond)** — เสียงพูดสู่เสียงพูดแบบ full-duplex (7B, เสียงเข้า → เสียงออก, 18 พรีเซ็ตเสียง)
- **[Audio2Face-3D](docs/models/audio2face3d.md)** — แอนิเมชันใบหน้าอวาตาร์ขับเคลื่อนด้วยเสียงพูด (NVIDIA Audio2Face-3D v2.3 Mark, ค่าสัมประสิทธิ์ใบหน้า 301 ค่า, MLX)

**การปรับปรุง การแยก และการสร้างเสียง**

- **[DeepFilterNet3](https://soniqo.audio/guides/denoise)** — การลดเสียงรบกวนแบบเรียลไทม์ (2.1M พารามิเตอร์, 48 kHz)
- **[Source Separation](https://soniqo.audio/guides/separate)** — การแยกแหล่งกำเนิดดนตรีด้วย HTDemucs (Demucs v4) + Open-Unmix (UMX-HQ / UMX-L, 4 stems: เสียงร้อง/กลอง/เบส/อื่นๆ, 44.1 kHz สเตอริโอ)
- **[MAGNeT](https://soniqo.audio/guides/compose)** — การสร้างดนตรีจากข้อความ (Meta MAGNeT Small 300M / Medium 1.5B, MLX INT8, คลิป 30 วินาทีที่ 32 kHz โมโน, การถอดรหัสแบบขนานที่มีการมาสก์)
- **[Stable Audio 3](docs/models/stable-audio-3.md)** — Text-to-audio/music generation (Stable Audio 3 Medium, MLX INT8/INT4, 44.1 kHz stereo, variable length)
- **[FlashSR](https://soniqo.audio/guides/upsample)** — การเพิ่มความละเอียดเสียง (FlashSR ICASSP 2025, MLX, 48 kHz โมโน, diffusion แบบกลั่นใน 1 ขั้น, INT4 363 MB / INT8 720 MB)

**การตรวจจับเทิร์น, diarization และอัตลักษณ์ผู้พูด**

- **[Wake-word](https://soniqo.audio/guides/wake-word)** — การตรวจจับคำสั่งปลุกบนอุปกรณ์ (KWS Zipformer 3M, CoreML, เร็วกว่าเรียลไทม์ 26 เท่า, รายการคำสั่งปลุกปรับแต่งได้)
- **[VAD](https://soniqo.audio/guides/vad)** — การตรวจจับเสียงพูด (Silero streaming, Pyannote offline, FireRedVAD รองรับกว่า 100 ภาษา)
- **[Speaker Diarization](https://soniqo.audio/guides/diarize)** — ใครพูดเมื่อใด (pipeline Pyannote, Sortformer แบบ end-to-end บน Neural Engine)
- **[Speaker Embeddings](https://soniqo.audio/guides/embed-speaker)** — WeSpeaker ResNet34 (256 มิติ), CAM++ (192 มิติ)

Papers: [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba) · [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba) · [Omnilingual ASR](https://arxiv.org/abs/2511.09690) (Meta) · [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA) · [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba) · [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2) · [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA) · [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai) · [Hibiki](https://arxiv.org/abs/2502.03382) (Kyutai) · [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## ข่าวสาร

- **19 เม.ย. 2026** — [MLX vs CoreML on Apple Silicon — A Practical Guide to Picking the Right Backend](https://blog.ivan.digital/mlx-vs-coreml-on-apple-silicon-a-practical-guide-to-picking-the-right-backend-and-why-you-should-f77ddea7b27a)
- **20 มี.ค. 2026** — [We Beat Whisper Large v3 with a 600M Model Running Entirely on Your Mac](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **26 ก.พ. 2026** — [Speaker Diarization and Voice Activity Detection on Apple Silicon — Native Swift with MLX](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **23 ก.พ. 2026** — [NVIDIA PersonaPlex 7B on Apple Silicon — Full-Duplex Speech-to-Speech in Native Swift with MLX](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **12 ก.พ. 2026** — [Qwen3-ASR Swift: On-Device ASR + TTS for Apple Silicon — Architecture and Benchmarks](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## เริ่มต้นอย่างรวดเร็ว

เพิ่มแพ็กเกจลงใน `Package.swift` ของคุณ:

```swift
.package(url: "https://github.com/soniqo/speech-swift", branch: "main")
```

นำเข้าเฉพาะโมดูลที่คุณต้องการ — ทุกโมเดลเป็นไลบรารี SPM ของตัวเอง คุณจึงไม่ต้องจ่ายให้กับสิ่งที่ไม่ได้ใช้:

```swift
.product(name: "ParakeetStreamingASR", package: "speech-swift"),
.product(name: "SpeechUI",             package: "speech-swift"),  // มุมมอง SwiftUI เสริม
```

**ถอดเสียงจากบัฟเฟอร์เสียงในสามบรรทัด:**

```swift
import ParakeetStreamingASR

let model = try await ParakeetStreamingASRModel.fromPretrained()
let text = try model.transcribeAudio(audioSamples, sampleRate: 16000)
```

**สตรีมมิ่งสดพร้อมผลลัพธ์บางส่วน:**

```swift
for await partial in model.transcribeStream(audio: samples, sampleRate: 16000) {
    print(partial.isFinal ? "FINAL: \(partial.text)" : "... \(partial.text)")
}
```

**มุมมองการเขียนตามคำบอกด้วย SwiftUI ในประมาณ 10 บรรทัด:**

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

`SpeechUI` มาพร้อมเพียง `TranscriptionView` (ผลลัพธ์สุดท้าย + บางส่วน) และ `TranscriptionStore` (อะแดปเตอร์สำหรับ ASR แบบสตรีมมิ่ง) ใช้ AVFoundation สำหรับการแสดงผลภาพเสียงและการเล่นเสียง

ผลิตภัณฑ์ SPM ที่มีให้: `Qwen3ASR`, `WhisperASR`, `Qwen3TTS`, `Qwen3TTSCoreML`, `ParakeetASR`, `ParakeetStreamingASR`, `NemotronStreamingASR`, `OmnilingualASR`, `KokoroTTS`, `SupertonicTTS`, `VibeVoiceTTS`, `CosyVoiceTTS`, `VoxCPM2TTS`, `IndexTTS2TTS`, `F5TTS`, `HiggsTTS`, `ChatterboxTTS`, `OmniVoiceTTS`, `IndicMioTTS`, `FishAudioTTS`, `MagpieTTS`, `MagpieTTSCoreML`, `MAGNeTMusicGen`, `StableAudio3MusicGen`, `FlashSR`, `PersonaPlex`, `Audio2Face3D`, `HibikiTranslate`, `MADLADTranslation`, `SpeechVAD`, `SpeechWakeWord`, `SpeechEnhancement`, `SpeechRestoration`, `SourceSeparation`, `Qwen3Chat`, `FunctionGemma`, `SpeechCore`, `SpeechUI`, `AudioCommon`

## โมเดล

มุมมองแบบย่อด้านล่าง **[แคตตาล็อกโมเดลฉบับเต็มพร้อมขนาด การควอนไทซ์ URL ดาวน์โหลด และตารางหน่วยความจำ → soniqo.audio/architecture](https://soniqo.audio/architecture)**

| โมเดล | งาน | Backends | ขนาด | ภาษา |
|-------|------|----------|-------|-----------|
| [Qwen3-ASR](https://soniqo.audio/guides/transcribe) | เสียงพูด → ข้อความ | MLX, CoreML (ไฮบริด) | 0.6B, 1.7B | 52 |
| [WhisperASR](docs/models/whisper-asr.md) | Speech → Text | CoreML (ANE) | Large-v3 Turbo | Multi |
| [Parakeet TDT](https://soniqo.audio/guides/parakeet) | เสียงพูด → ข้อความ | CoreML (ANE) | 0.6B | 25 ยุโรป |
| [Parakeet EOU](https://soniqo.audio/guides/dictate) | เสียงพูด → ข้อความ (สตรีมมิ่ง) | CoreML (ANE) | 120M | 25 ยุโรป |
| [Nemotron Streaming (หลายภาษา)](https://soniqo.audio/guides/nemotron) | เสียงพูด → ข้อความ (สตรีมมิ่ง มีเครื่องหมายวรรคตอน) | CoreML (ANE), MLX | 0.6B | **40** |
| [Nemotron Streaming (อังกฤษ)](https://soniqo.audio/guides/nemotron) | เสียงพูด → ข้อความ (สตรีมมิ่ง มีเครื่องหมายวรรคตอน) | CoreML (ANE) | 0.6B | EN |
| [Omnilingual ASR](https://soniqo.audio/guides/omnilingual) | เสียงพูด → ข้อความ | CoreML (ANE), MLX | 300M / 1B / 3B / 7B | **[1,672](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)** |
| [Qwen3-ForcedAligner](https://soniqo.audio/guides/align) | เสียง + ข้อความ → เครื่องหมายเวลา | MLX, CoreML | 0.6B | หลายภาษา |
| [Qwen3-TTS](https://soniqo.audio/guides/speak) | ข้อความ → เสียงพูด | MLX, CoreML | 0.6B, 1.7B | 10 |
| [CosyVoice3](https://soniqo.audio/guides/cosyvoice) | ข้อความ → เสียงพูด | MLX | 0.5B | 9 |
| [VoxCPM2](https://soniqo.audio/speech-generation) | ข้อความ → เสียงพูด (48 kHz, ออกแบบเสียง + โคลนเสียง) | MLX | 2B (bf16/int8) | 30 |
| [IndexTTS2](docs/models/indextts2.md) | Text → Speech (zero-shot voice cloning) | MLX | 1.5B-class (fp16) | EN/ZH |
| [F5-TTS](docs/models/f5-tts.md) | Text → Speech (zero-shot voice cloning) | MLX | 336M (fp16) | EN/ZH |
| [Higgs TTS 3](docs/models/higgs-tts.md) | Text → Speech (conversational, zero-shot voice cloning) | MLX | 4B (bf16) | 100+ |
| [Kokoro-82M](https://soniqo.audio/guides/kokoro) | ข้อความ → เสียงพูด | CoreML (ANE) | 82M | 10 |
| [Supertonic-3](https://soniqo.audio/guides/supertonic) | ข้อความ → เสียงพูด (44.1 kHz, flow-matching, G2P-free) | CoreML (ANE) | 99M | 31 |
| [VibeVoice Realtime-0.5B](https://soniqo.audio/guides/vibevoice) | ข้อความ → เสียงพูด (รูปแบบยาว หลายผู้พูด) | MLX | 0.5B | EN/ZH |
| [VibeVoice 1.5B](https://soniqo.audio/guides/vibevoice) | ข้อความ → เสียงพูด (พอดแคสต์ยาวสุด 90 นาที) | MLX | 1.5B | EN/ZH |
| [Magpie-TTS Multilingual](https://soniqo.audio/guides/magpie) | ข้อความ → เสียงพูด (5 ผู้พูดสำเร็จรูป สตรีมมิ่ง) | MLX / CoreML | 357M (MLX INT8, CoreML INT8) | 9 (CoreML ไม่รวม JA) |
| [Chatterbox Multilingual](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16) | ข้อความ → เสียงพูด (โคลนแบบ zero-shot) | MLX | 0.8B (fp16) | 23 (HE ต้องมี niqqud) |
| [OmniVoice](https://huggingface.co/aufklarer/OmniVoice-MLX-fp16) | ข้อความ → เสียงพูด (diffusion NAR, การโคลนแบบ zero-shot) | MLX | 0.8B (fp16 ค่าเริ่มต้น / int8) | 600+ |
| [Indic-Mio](docs/models/indic-mio-tts.md) | Text → Speech (Hindi/Indic, emotion tags, voice cloning) | MLX | fp16 | Hindi / Indic |
| [Fish Audio S2 Pro](docs/models/fish-audio-s2-pro.md) | ข้อความ → เสียงพูด (การโคลนแบบ zero-shot, marker สไตล์แบบชัดเจน) | MLX | 0.5B-class (fp16) | หลายภาษา |
| [Qwen3.5 Chat](docs/models/qwen35-chat.md) | Text → Text (LLM) | MLX, CoreML | 0.8B | Multi |
| [Qwen3 Dense Chat](docs/models/qwen3-dense-chat.md) | Text → Text (LLM) | MLX | 4B | Multi |
| [Gemma 4 Chat](docs/models/gemma4-chat.md) | Text → Text (LLM) | MLX | E2B / E4B (4-bit) | Multi |
| [FunctionGemma](docs/models/function-gemma.md) | ข้อความ → การเรียกเครื่องมือ (LLM) | CoreML | 270M | EN |
| [MADLAD-400](https://soniqo.audio/guides/translate) | ข้อความ → ข้อความ (การแปล) | MLX | 3B | **400+** |
| [Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate) | เสียงพูด → เสียงพูด (การแปล) | MLX | 3B | FR/ES/PT/DE → EN |
| [PersonaPlex](https://soniqo.audio/guides/respond) | เสียงพูด → เสียงพูด | MLX | 7B | EN |
| [Audio2Face-3D](docs/models/audio2face3d.md) | เสียงพูด → แอนิเมชันใบหน้า | MLX | v2.3 Mark | ไม่จำกัดภาษา |
| [Silero VAD](https://soniqo.audio/guides/vad) | การตรวจจับเสียงพูด | MLX, CoreML | 309K | ไม่จำกัดภาษา |
| [KWS Zipformer](docs/models/kws-zipformer.md) | Audio → Wake word | CoreML (ANE) | 3M | EN/custom keywords |
| [Pyannote](https://soniqo.audio/guides/diarize) | VAD + การแยกผู้พูด | MLX | 1.5M | ไม่จำกัดภาษา |
| [Pyannote Community-1](https://huggingface.co/aufklarer/Pyannote-Community-1-CoreML) | การแยกผู้พูด + เอ็มเบดดิงผู้พูด | CoreML (ANE) + Swift VBx | 8.35M | ไม่จำกัดภาษา |
| [Sortformer](https://soniqo.audio/guides/diarize) | การแยกผู้พูด (E2E) | CoreML (ANE) | — | ไม่จำกัดภาษา |
| [DeepFilterNet3](https://soniqo.audio/guides/denoise) | การปรับปรุงเสียงพูด | CoreML | 2.1M | ไม่จำกัดภาษา |
| [Sidon](https://soniqo.audio/guides/restore) | การฟื้นฟูเสียงพูด (ลดเสียงรบกวน + ลดเสียงก้อง, 48 kHz) | CoreML | w2v-BERT 2.0 + DAC (fp16/int8) | ไม่จำกัดภาษา |
| [HTDemucs (Demucs v4)](https://soniqo.audio/guides/separate) | การแยกแหล่งกำเนิด | MLX | 168M | ไม่จำกัดภาษา |
| [Open-Unmix](https://soniqo.audio/guides/separate) | การแยกแหล่งกำเนิด | MLX | 8.6M | ไม่จำกัดภาษา |
| [MAGNeT](https://soniqo.audio/guides/compose) | ข้อความ → ดนตรี (30 วินาที @ 32 kHz) | MLX | 300M / 1.5B (int4/int8) | พรอมต์ EN |
| [Stable Audio 3](docs/models/stable-audio-3.md) | Text → Music/audio (44.1 kHz stereo) | MLX | Medium 1.4B (int4/int8) | EN prompts |
| [FlashSR](https://soniqo.audio/guides/upsample) | การเพิ่มความละเอียดเสียง (48 kHz) | MLX | 363 MB / 720 MB (int4/int8) | ไม่จำกัดภาษา |
| [WeSpeaker](https://soniqo.audio/guides/embed-speaker) | Embedding ของผู้พูด | MLX, CoreML | 6.6M | ไม่จำกัดภาษา |

## การติดตั้ง

### Homebrew

ต้องใช้ Homebrew ARM แบบเนทีฟ (`/opt/homebrew`) ไม่รองรับ Homebrew แบบ Rosetta/x86_64

```bash
brew install speech
```

จากนั้น:

```bash
speech transcribe recording.wav
speech speak "Hello world"
speech translate "Hello, how are you?" --to es
speech respond --input question.wav --transcript
speech-server --port 8080            # เซิร์ฟเวอร์ HTTP / WebSocket ท้องถิ่น (รองรับ /v1/realtime + /v1/audio/transcriptions ของ OpenAI)
```

**[คู่มืออ้างอิง CLI ฉบับเต็ม →](https://soniqo.audio/cli)**

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

นำเข้าเฉพาะสิ่งที่คุณต้องการ — ทุกโมเดลเป็น target SPM ของตัวเอง:

```swift
import Qwen3ASR             // การรู้จำเสียงพูด (MLX)
import WhisperASR           // Whisper Large-v3 Turbo (CoreML)
import ParakeetASR          // การรู้จำเสียงพูด (CoreML, batch)
import ParakeetStreamingASR // การเขียนตามคำบอกแบบสตรีมมิ่งพร้อม partials + EOU
import NemotronStreamingASR // ASR สตรีมมิ่งหลายภาษาพร้อมเครื่องหมายวรรคตอนในตัว (0.6B, 40 ภาษา)
import OmnilingualASR       // 1,672 ภาษา (CoreML + MLX)
import Qwen3TTS             // การสังเคราะห์เสียงพูด
import CosyVoiceTTS         // การสังเคราะห์เสียงพูดพร้อมการโคลนเสียง
import VoxCPM2TTS           // TTS 48 kHz พร้อมการโคลนเสียง + การออกแบบเสียง (2B)
import IndexTTS2TTS         // Native MLX voice cloning from reference audio
import F5TTS                // Zero-shot voice cloning (DiT flow matching + Vocos)
import HiggsTTS             // Conversational TTS + cloning (Qwen3 backbone, control tags)
import KokoroTTS            // การสังเคราะห์เสียงพูด (พร้อมใช้งานบน iOS)
import VibeVoiceTTS         // TTS รูปแบบยาว / หลายผู้พูด (EN/ZH)
import MagpieTTS            // TTS หลายภาษา (NVIDIA Magpie 357M, MLX, 9 ภาษา)
import MagpieTTSCoreML      // Backend CoreML ของ Magpie (ไฮบริด CoreML + MLX, 8 ภาษา)
import FishAudioTTS         // runtime Fish Audio S2 Pro แบบทดลองพร้อมการโคลนเสียง
import Qwen3Chat            // แชท LLM บนอุปกรณ์
import FunctionGemma    // LLM บนอุปกรณ์สำหรับการเรียกเครื่องมือ
import MADLADTranslation    // การแปลแบบหลายต่อหลายระหว่างกว่า 400 ภาษา
import HibikiTranslate      // การแปลเสียงพูดสู่เสียงพูดแบบสตรีมมิ่ง (FR/ES/PT/DE → EN)
import PersonaPlex          // เสียงพูดสู่เสียงพูดแบบ full-duplex
import SpeechVAD            // VAD + การแยกผู้พูด + embeddings
import SpeechEnhancement    // การลดเสียงรบกวน
import SpeechRestoration    // การฟื้นฟูเสียงพูด — ลดเสียงรบกวน + ลดเสียงก้อง (Sidon, CoreML, 48 kHz)
import SourceSeparation     // การแยกแหล่งกำเนิดดนตรี (Open-Unmix, 4 stems)
import SpeechUI             // คอมโพเนนต์ SwiftUI สำหรับการถอดเสียงแบบสตรีมมิ่ง
import AudioCommon          // โปรโตคอลและยูทิลิตี้ที่ใช้ร่วมกัน
```

### ความต้องการของระบบ

- Swift 6+, Xcode 16+ (พร้อม Metal Toolchain)
- macOS 15+ (Sequoia) หรือ iOS 18+, Apple Silicon (M1/M2/M3/M4)

ข้อกำหนดขั้นต่ำ macOS 15 / iOS 18 มาจาก [MLState](https://developer.apple.com/documentation/coreml/mlstate) — API สถานะถาวรของ ANE จาก Apple ที่ pipelines CoreML (Qwen3-ASR, Qwen3-Chat, Qwen3-TTS) ใช้เพื่อเก็บ KV caches ให้อยู่ใน Neural Engine ตลอดขั้นตอนของโทเค็น

### คอมไพล์จากซอร์สโค้ด

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

`make build` คอมไพล์แพ็กเกจ Swift **และ** ไลบรารี shader MLX Metal ไลบรารี Metal จำเป็นสำหรับการอนุมานบน GPU — หากไม่มีคุณจะเห็น `Failed to load the default metallib` ในระหว่างการทำงาน ใช้ `make debug` สำหรับ build แบบดีบัก และ `make test` สำหรับชุดทดสอบ

**[คู่มือคอมไพล์และติดตั้งฉบับเต็ม →](https://soniqo.audio/getting-started)**

## แอปตัวอย่าง

- **[DictateDemo](Examples/DictateDemo/)** ([เอกสาร](https://soniqo.audio/guides/dictate)) — การเขียนตามคำบอกแบบสตรีมมิ่งบนเมนูบาร์ของ macOS พร้อมผลลัพธ์บางส่วนแบบสด การตรวจจับจุดจบของประโยคโดย VAD และการคัดลอกในคลิกเดียว ทำงานเป็น background agent (Parakeet-EOU-120M + Silero VAD)
- **[iOSEchoDemo](Examples/iOSEchoDemo/)** — เดโม echo สำหรับ iOS (Parakeet ASR + Kokoro TTS) ใช้งานได้บนอุปกรณ์จริงและซิมูเลเตอร์
- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — ผู้ช่วยเสียงสนทนาพร้อมอินพุตจากไมโครโฟน VAD และบริบทหลายเทิร์น บน macOS RTF ~0.94 บน M2 Max (เร็วกว่าเรียลไทม์)
- **[SpeechDemo](Examples/SpeechDemo/)** — การเขียนตามคำบอกและการสังเคราะห์ TTS ในอินเทอร์เฟซแบบแท็บ บน macOS

README ของแต่ละเดโมมีคำแนะนำการคอมไพล์

## ตัวอย่างโค้ด

โค้ดด้านล่างแสดงเส้นทางที่สั้นที่สุดสำหรับแต่ละโดเมน ทุกหัวข้อมีลิงก์ไปยังคู่มือฉบับเต็มบน [soniqo.audio](https://soniqo.audio) พร้อมตัวเลือกการตั้งค่า backends หลายแบบ รูปแบบสตรีมมิ่ง และสูตร CLI

### การรู้จำเสียงพูด — [คู่มือฉบับเต็ม →](https://soniqo.audio/guides/transcribe)

```swift
import Qwen3ASR

let model = try await Qwen3ASRModel.fromPretrained()
let text = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

Backends ทางเลือก: [WhisperASR](docs/inference/whisper-asr-inference.md) (Whisper Large-v3 Turbo, native CoreML), [Parakeet TDT](https://soniqo.audio/guides/parakeet) (CoreML เร็วกว่าเรียลไทม์ 32 เท่า), [Omnilingual ASR](https://soniqo.audio/guides/omnilingual) (1,672 ภาษา, CoreML หรือ MLX), [Streaming dictation](https://soniqo.audio/guides/dictate) (ผลลัพธ์บางส่วนแบบสด)

### Forced Alignment — [คู่มือฉบับเต็ม →](https://soniqo.audio/guides/align)

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

### การสังเคราะห์เสียงพูด — [คู่มือฉบับเต็ม →](https://soniqo.audio/guides/speak)

```swift
import Qwen3TTS
import AudioCommon

let model = try await Qwen3TTSModel.fromPretrained()
let audio = model.synthesize(text: "Hello world", language: "english")
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

เอนจิน TTS ทางเลือก: [CosyVoice3](https://soniqo.audio/guides/cosyvoice) (สตรีมมิ่ง + การโคลนเสียง + แท็กอารมณ์), [Kokoro-82M](https://soniqo.audio/guides/kokoro) (พร้อมใช้งานบน iOS, 54 เสียง), [VibeVoice](https://soniqo.audio/guides/vibevoice) (พอดแคสต์รูปแบบยาว / หลายผู้พูด EN/ZH), [Fish Audio S2 Pro](docs/inference/fish-audio-s2-pro.md) (การโคลน zero-shot แบบทดลอง + marker สไตล์ในวงเล็บเหลี่ยม), [การโคลนเสียง](https://soniqo.audio/guides/voice-cloning)

### เสียงพูดสู่เสียงพูด — [คู่มือฉบับเต็ม →](https://soniqo.audio/guides/respond)

```swift
import PersonaPlex

let model = try await PersonaPlexModel.fromPretrained()
let responseAudio = model.respond(userAudio: userSamples)
// เอาต์พุต Float32 โมโน 24 kHz พร้อมเล่น
```

### แชท LLM — [คู่มือฉบับเต็ม →](https://soniqo.audio/guides/chat)

```swift
import Qwen3Chat
import FunctionGemma

let chat = try await Qwen35MLXChat.fromPretrained()
chat.chat(messages: [(.user, "Explain MLX in one sentence")]) { token, isFinal in
    print(token, terminator: "")
}
```

### การแปล — [คู่มือฉบับเต็ม →](https://soniqo.audio/guides/translate)

```swift
import MADLADTranslation

let translator = try await MADLADTranslator.fromPretrained()
let es = try translator.translate("Hello, how are you?", to: "es")
// → "Hola, ¿cómo estás?"
```

### การแปลเสียงพูด — [คู่มือฉบับเต็ม →](https://soniqo.audio/guides/audio-translate)

```swift
import HibikiTranslate
import AudioCommon

let model = try await HibikiTranslateModel.fromPretrained()
let pcm = try AudioFileLoader.load(url: input, targetSampleRate: 24000)
let (englishAudio, textTokens) = model.translate(
    sourceAudio: pcm, sourceLanguage: .fr
)
// Hibiki Zero-3B — FR/ES/PT/DE → EN, บนอุปกรณ์, สตรีมมิ่งด้วยตัวเข้ารหัส Mimi
```

### การตรวจจับเสียงพูด — [คู่มือฉบับเต็ม →](https://soniqo.audio/guides/vad)

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
for s in segments { print("\(s.startTime)s → \(s.endTime)s") }
```

### การแยกผู้พูด — [คู่มือฉบับเต็ม →](https://soniqo.audio/guides/diarize)

```swift
import SpeechVAD

let diarizer = try await DiarizationPipeline.fromPretrained()
let segments = diarizer.diarize(audio: samples, sampleRate: 16000)
for s in segments { print("Speaker \(s.speakerId): \(s.startTime)s - \(s.endTime)s") }
```

### การปรับปรุงเสียงพูด — [คู่มือฉบับเต็ม →](https://soniqo.audio/guides/denoise)

```swift
import SpeechEnhancement

let denoiser = try await DeepFilterNet3Model.fromPretrained()
let clean = try denoiser.enhance(audio: noisySamples, sampleRate: 48000)
```

### การฟื้นฟูเสียงพูด — [คู่มือฉบับเต็ม →](https://soniqo.audio/guides/restore)

ลดเสียงรบกวน **และ** ลดเสียงก้องไปพร้อมกันด้วย [Sidon](https://arxiv.org/abs/2509.17052) (ตัวทำนาย w2v-BERT 2.0 + โวโคเดอร์ DAC, Core ML) ต่างจากตัวลดเสียงรบกวนทั่วไป Sidon ถูกฝึกมาเพื่อรักษาเอกลักษณ์ของผู้พูดไว้ จึงเหมาะอย่างยิ่งสำหรับการทำความสะอาดเสียงอ้างอิงสำหรับการโคลนเสียงที่มีเสียงรบกวนหรือเสียงก้องก่อนทำ TTS อินพุตเป็น 16 kHz เอาต์พุตเป็นโมโน 48 kHz

```swift
import SpeechRestoration

let restorer = try await SpeechRestorer.fromPretrained()          // .fp16 (default) or .int8
let clean = try restorer.restore(audio: noisySamples, sampleRate: 16000)  // → 48 kHz
```

จาก CLI:

```bash
speech restore noisy.wav -o clean.wav            # denoise + dereverb, 48 kHz output
speech restore noisy.wav --variant int8          # smaller, lower peak RAM

# Clean a voice-cloning reference before TTS (opt-in; preserves speaker identity):
speech speak "Hello" --engine voxcpm2 --voice-sample ref.wav --clean-reference
```

### Voice Pipeline (ASR → LLM → TTS) — [คู่มือฉบับเต็ม →](https://soniqo.audio/voice-agents)

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

`VoicePipeline` คือสเตตแมชชีนสำหรับ voice agent แบบเรียลไทม์ (ขับเคลื่อนโดย [speech-core](https://github.com/soniqo/speech-core)) พร้อมการตรวจจับเทิร์นที่ขับเคลื่อนด้วย VAD การจัดการการขัดจังหวะ และ STT แบบ eager เชื่อมต่อ `SpeechRecognitionModel` + `SpeechGenerationModel` + `StreamingVADProvider` ใดก็ได้

### เซิร์ฟเวอร์ HTTP API

```bash
speech-server --port 8080
```

เปิดให้เข้าถึงทุกโมเดลผ่าน endpoints HTTP REST + WebSocket รวมถึง API ที่รองรับ OpenAI: Realtime WebSocket ที่ `/v1/realtime` และ REST endpoint สำหรับการถอดเสียงที่ `/v1/audio/transcriptions` ดู [`Sources/AudioServer/`](Sources/AudioServer/)

## สถาปัตยกรรม

speech-swift ถูกแบ่งเป็นหนึ่ง SPM target ต่อโมเดล เพื่อให้ผู้ใช้จ่ายเฉพาะสิ่งที่นำเข้าเท่านั้น โครงสร้างพื้นฐานที่ใช้ร่วมกันอยู่ใน `AudioCommon` (โปรโตคอล I/O ของเสียง ตัวดาวน์โหลดจาก HuggingFace `SentencePieceModel`) และ `MLXCommon` (การโหลดน้ำหนัก ตัวช่วย `QuantizedLinear` ตัวช่วย attention หลายหัว `SDPA`)

**[แผนภาพสถาปัตยกรรมฉบับเต็มพร้อม backends ตารางหน่วยความจำ และแผนผังโมดูล → soniqo.audio/architecture](https://soniqo.audio/architecture)** · **[API reference → soniqo.audio/api](https://soniqo.audio/api)** · **[Benchmarks → soniqo.audio/benchmarks](https://soniqo.audio/benchmarks)**

เอกสารในเครื่อง (repo):
- **โมเดล:** [Qwen3-ASR](docs/models/asr-model.md) · [WhisperASR](docs/models/whisper-asr.md) · [Qwen3-TTS](docs/models/tts-model.md) · [CosyVoice](docs/models/cosyvoice-tts.md) · [Kokoro](docs/models/kokoro-tts.md) · [VoxCPM2](docs/models/voxcpm2-tts.md) · [IndexTTS2](docs/models/indextts2.md) · [F5-TTS](docs/models/f5-tts.md) · [Higgs TTS 3](docs/models/higgs-tts.md) · [VibeVoice](docs/models/vibevoice.md) · [Supertonic](docs/models/supertonic-tts.md) · [Chatterbox](docs/models/chatterbox-tts.md) · [Indic-Mio](docs/models/indic-mio-tts.md) · [Fish Audio S2 Pro](docs/models/fish-audio-s2-pro.md) · [Magpie TTS](docs/models/magpie-tts.md) · [Parakeet TDT](docs/models/parakeet-asr.md) · [Parakeet Streaming](docs/models/parakeet-streaming-asr.md) · [Nemotron Streaming](docs/models/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/models/omnilingual-asr.md) · [PersonaPlex](docs/models/personaplex.md) · [Hibiki](docs/models/hibiki.md) · [MADLAD-400](docs/models/madlad-translation.md) · [FunctionGemma](docs/models/function-gemma.md) · [Qwen3.5 Chat](docs/models/qwen35-chat.md) · [Gemma 4 Chat](docs/models/gemma4-chat.md) · [Qwen3 Dense Chat](docs/models/qwen3-dense-chat.md) · [FireRedVAD](docs/models/fireredvad.md) · [KWS Zipformer](docs/models/kws-zipformer.md) · [Sidon](docs/models/sidon.md) · [Source Separation](docs/models/source-separation.md) · [HTDemucs](docs/models/htdemucs.md) · [MAGNeT](docs/models/magnet-music-gen.md) · [Stable Audio 3](docs/models/stable-audio-3.md) · [FlashSR](docs/models/flashsr.md) · [Audio2Face-3D](docs/models/audio2face3d.md)
- **การอนุมาน:** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md) · [WhisperASR](docs/inference/whisper-asr-inference.md) · [Parakeet TDT](docs/inference/parakeet-asr-inference.md) · [Parakeet Streaming](docs/inference/parakeet-streaming-asr-inference.md) · [Nemotron Streaming](docs/inference/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/inference/omnilingual-asr-inference.md) · [TTS](docs/inference/qwen3-tts-inference.md) · [VoxCPM2](docs/inference/voxcpm2-inference.md) · [IndexTTS2](docs/inference/indextts2.md) · [F5-TTS](docs/inference/f5-tts.md) · [Higgs TTS 3](docs/inference/higgs-tts.md) · [VibeVoice](docs/inference/vibevoice-inference.md) · [Fish Audio S2 Pro](docs/inference/fish-audio-s2-pro.md) · [Magpie TTS](docs/inference/magpie-tts.md) · [Hibiki](docs/inference/hibiki-inference.md) · [MADLAD-400](docs/inference/madlad-translation.md) · [MAGNeT](docs/inference/magnet-music-gen.md) · [Stable Audio 3](docs/inference/stable-audio-3.md) · [FlashSR](docs/inference/flashsr.md) · [Forced Aligner](docs/inference/forced-aligner.md) · [Silero VAD](docs/inference/silero-vad.md) · [FireRedVAD](docs/inference/fireredvad.md) · [Wake-word](docs/inference/wake-word.md) · [Speaker Diarization](docs/inference/speaker-diarization.md) · [Speech Enhancement](docs/inference/speech-enhancement.md) · [Sidon](docs/inference/sidon.md) · [Cache/offline](docs/inference/cache-and-offline.md)
- **อ้างอิง:** [Shared Protocols](docs/shared-protocols.md)

## การกำหนดค่าแคช

น้ำหนักของโมเดลจะถูกดาวน์โหลดจาก HuggingFace เมื่อใช้งานครั้งแรก และเก็บแคชไว้ที่ `~/Library/Caches/qwen3-speech/` สามารถเขียนทับด้วย `QWEN3_CACHE_DIR` (CLI) หรือ `cacheDir:` (Swift API) ทุก entry point ของ `fromPretrained()` ยังรับ `offlineMode: true` เพื่อข้ามการเชื่อมต่อเครือข่ายเมื่อมีน้ำหนักอยู่ในแคชแล้ว

ดู [`docs/inference/cache-and-offline.md`](docs/inference/cache-and-offline.md) สำหรับรายละเอียดทั้งหมด รวมถึงพาธของคอนเทนเนอร์ iOS แบบ sandboxed

## ไลบรารี MLX Metal

หากคุณเห็น `Failed to load the default metallib` ในระหว่างการทำงาน แสดงว่าไลบรารี shader ของ Metal หายไป ให้รัน `make build` หรือ `./scripts/build_mlx_metallib.sh release` หลังจาก `swift build` แบบ manual หากไม่มี Metal Toolchain ให้ติดตั้งก่อน:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## การทดสอบ

```bash
make test                            # ชุดเต็ม (unit + E2E พร้อมการดาวน์โหลดโมเดล)
swift test --skip E2E                # เฉพาะ unit (ปลอดภัยสำหรับ CI ไม่มีการดาวน์โหลด)
swift test --filter Qwen3ASRTests    # โมดูลเฉพาะ
```

คลาสทดสอบ E2E ใช้คำนำหน้า `E2E` เพื่อให้ CI สามารถกรองออกได้ด้วย `--skip E2E` ดู [CLAUDE.md](CLAUDE.md#testing) สำหรับข้อตกลงเรื่องการทดสอบฉบับเต็ม

## การมีส่วนร่วม

ยินดีรับ PRs — การแก้ไขบั๊ก การผสานรวมโมเดลใหม่ และเอกสาร Fork สร้าง branch ของฟีเจอร์ รัน `make build && make test` แล้วเปิด PR ไปที่ `main`

## สัญญาอนุญาต

Apache 2.0
