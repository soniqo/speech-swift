# Speech Swift

MLX Swift와 CoreML 기반의 Apple Silicon용 AI 음성 모델.

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

Mac과 iOS를 위한 온디바이스 음성 인식, 합성 및 이해. Apple Silicon에서 완전히 로컬로 실행됩니다 — 클라우드 없이, API 키 없이, 데이터가 기기 밖으로 나가지 않습니다.

**[📚 전체 문서 →](https://soniqo.audio/ko)** · **[🤗 HuggingFace 모델](https://huggingface.co/aufklarer)** · **[📝 블로그](https://blog.ivan.digital)**

<p align="center">
  <a href="https://formulae.brew.sh/formula/speech"><img src="https://img.shields.io/homebrew/installs/dm/speech.svg?logo=homebrew&amp;label=Homebrew%20installs&amp;color=FBB040" alt="Homebrew installs"></a>
  <a href="https://github.com/soniqo/speech-swift#built-with-speech-swift"><img src="https://img.shields.io/badge/verified%20public%20repositories-15-2ea44f?logo=github" alt="Verified public repositories: 15"></a>
</p>

<p align="center">
  <a href="https://trendshift.io/repositories/24196?utm_source=trendshift-badge&amp;utm_medium=badge&amp;utm_campaign=badge-trendshift-24196" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/trendshift/repositories/24196/daily?language=Swift" alt="soniqo%2Fspeech-swift | Trendshift" width="250" height="55"/></a>
</p>

<p align="center">
  <a href="https://youtu.be/x9zgcaW0gUk">
    <img src="https://img.youtube.com/vi/x9zgcaW0gUk/maxresdefault.jpg" width="640" alt="MacBook에서 동작하는 로컬 음성 AI — YouTube에서 4분 분량의 오픈소스 라이브러리 투어 시청">
  </a>
</p>
<p align="center"><em>MacBook에서 동작하는 로컬 음성 AI — YouTube에서 4분 분량의 오픈소스 라이브러리 투어 시청</em></p>

**사용 사례:** [음성 에이전트](https://soniqo.audio/ko/voice-agents) · [전사](https://soniqo.audio/ko/transcription) · [음성 합성](https://soniqo.audio/ko/speech-generation)

## Speech Swift로 구축

Speech Swift 패키지 참조를 공개 소스에서 확인할 수 있는 저장소 15개입니다.

[Palmier Pro](https://github.com/palmier-io/palmier-pro) · [Anarlog](https://github.com/fastrepl/anarlog) · [ClawdHome](https://github.com/ThinkInAIXYZ/clawdhome) · [Jabber](https://github.com/rselbach/jabber) · [Ora](https://github.com/wuwangzhang1216/ora) · [VoxFlow](https://github.com/xingbofeng/VoxFlow) · [LokalBot](https://github.com/stevyhacker/lokalbot) · [Voicey](https://github.com/jonathanKingston/voicey) · [HushType](https://github.com/felixfu824/HushType) · [DexDictate macOS](https://github.com/westkitty/DexDictate_MacOS) · [Watchtower](https://github.com/aiwatchtowers/watchtower) · [Wishper App](https://github.com/irangareddy/wishper-app) · [FriSpeak](https://github.com/KSubedi/FriSpeak) · [Scribe](https://github.com/itchat/Scribe) · [VoicePen](https://github.com/dot-sk/VoicePen)

**OpenAI 호환 앱:** [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) ([병합된 WAV 지원](https://github.com/Mintplex-Labs/anything-llm/pull/6012))

**기능 그룹:** STT / ASR · 정렬 · TTS · LLM 및 번역 · Speech-to-Speech · 향상 / 복원 · 소스 분리 · 음악 / 오디오 생성 · 웨이크워드, VAD, 화자 분리 및 화자 식별

**STT / ASR**

- **[Qwen3-ASR](https://soniqo.audio/ko/guides/transcribe)** — 음성-텍스트 변환 (자동 음성 인식, 52개 언어, MLX + CoreML)
- **[WhisperASR](docs/models/whisper-asr.md)** — Whisper Large-v3 Turbo speech-to-text via native CoreML runtime (ANE, multilingual)
- **[Parakeet TDT](https://soniqo.audio/ko/guides/parakeet)** — CoreML을 통한 음성-텍스트 변환 (Neural Engine, NVIDIA FastConformer + TDT 디코더, 25개 언어)
- **[Omnilingual ASR](https://soniqo.audio/ko/guides/omnilingual)** — 음성-텍스트 변환 (Meta wav2vec2 + CTC, **1,672개 언어**, 32개 문자 체계, CoreML 300M + MLX 300M/1B/3B/7B)
- **[스트리밍 받아쓰기](https://soniqo.audio/ko/guides/dictate)** — 부분 결과와 발화 종료 감지를 갖춘 실시간 받아쓰기 (Parakeet-EOU-120M)
- **[Nemotron 스트리밍 (다국어)](https://soniqo.audio/ko/guides/nemotron)** — 네이티브 구두점 및 대소문자 처리를 제공하는 저지연 스트리밍 ASR (NVIDIA Nemotron-3.5-ASR-Streaming-0.6B, CoreML + MLX, **40개 언어-로케일**)
- **[Nemotron 스트리밍 (영어)](https://soniqo.audio/guides/nemotron)** — 네이티브 구두점 및 대소문자 처리를 제공하는 저지연 스트리밍 ASR (NVIDIA Nemotron-Speech-Streaming-0.6B, CoreML, 영어 전용, 다국어 버전보다 가볍고 빠름)

**정렬**

- **[Qwen3-ForcedAligner](https://soniqo.audio/ko/guides/align)** — 단어 수준 타임스탬프 정렬 (오디오 + 텍스트 → 타임스탬프)

**TTS / 음성 생성**

- **[Qwen3-TTS](https://soniqo.audio/ko/guides/speak)** — 텍스트-음성 변환 (최고 품질, 스트리밍, 커스텀 화자, 10개 언어)
- **[CosyVoice TTS](https://soniqo.audio/ko/guides/cosyvoice)** — 음성 복제, 다화자 대화, 감정 태그를 지원하는 스트리밍 TTS (9개 언어)
- **[VoxCPM2](https://soniqo.audio/ko/speech-generation)** — 48 kHz 스튜디오 품질 TTS, 음성 복제 + 명령 기반 보이스 디자인 (2B, MLX bf16/int8, 30개 언어)
- **[IndexTTS2](docs/models/indextts2.md)** — Native MLX voice cloning from a reference voice (IndexTeam IndexTTS-2, 1.5B-class fp16 bundle, speaker/emotion/pause controls)
- **[F5-TTS](docs/models/f5-tts.md)** — Zero-shot voice cloning from a short reference clip + transcript (SWivid F5-TTS v1 Base, DiT flow matching + Vocos, MLX fp16, 24 kHz, English + Mandarin; non-commercial license)
- **[Higgs TTS 3](docs/models/higgs-tts.md)** — Conversational TTS with zero-shot voice cloning and inline emotion/style/SFX/prosody tags (Boson Higgs TTS 3, Qwen3-4B backbone, MLX bf16, 24 kHz, 100+ languages; research/non-commercial license)
- **[Kokoro TTS](https://soniqo.audio/ko/guides/kokoro)** — 온디바이스 TTS (82M, CoreML/Neural Engine, 54개 음색, iOS 지원, 10개 언어)
- **[VibeVoice TTS](https://soniqo.audio/ko/guides/vibevoice)** — 장문 / 멀티 스피커 TTS (Microsoft VibeVoice Realtime-0.5B + 1.5B, MLX, 최대 90분 팟캐스트 / 오디오북 합성, EN/ZH)
- **[Magpie TTS](https://soniqo.audio/ko/guides/magpie)** — 다국어 TTS (NVIDIA Magpie-TTS Multilingual 357M, MLX INT8 411 MB 또는 CoreML INT8 342 MB, 9개 언어, 5개 내장 스피커, MLX 스트리밍)
- **[Supertonic TTS](https://soniqo.audio/guides/supertonic)** — 온디바이스 flow-matching TTS (Supertone Supertonic-3 99M, CoreML/Neural Engine, 31개 언어, 10개 음색, G2P-free, 44.1 kHz)
- **[Chatterbox TTS](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16)** — 제로샷 음성 복제를 지원하는 다국어 TTS (Resemble AI Chatterbox Multilingual, MLX fp16 ~1.3 GB, 런타임 23개 언어, 히브리어는 niqqud 필요, MIT)
- **[OmniVoice TTS](https://huggingface.co/aufklarer/OmniVoice-MLX-fp16)** — 제로샷 음성 복제를 지원하는 비자기회귀 확산 TTS (k2-fsa OmniVoice, Qwen3 백본, MLX fp16 기본 / int8 사용 가능, 600+개 언어, Apache-2.0)
- **[Indic-Mio](docs/models/indic-mio-tts.md)** — Hindi/Indic TTS with inline emotion markers and optional reference-voice cloning (MLX, 24 kHz)

**LLM 및 번역**

- **[Qwen3Chat](https://soniqo.audio/ko/guides/chat)** — 온디바이스 LLM 채팅 (Qwen3.5-0.8B MLX/CoreML에 dense Qwen3 4B 및 Gemma 4 E2B/E4B MLX 백엔드 추가, 스트리밍 토큰)
- **[FunctionGemma](https://soniqo.audio/ko/guides/function-calls)** — 온디바이스 구조화된 함수 / 도구 호출 LLM (Gemma 3 270M, CoreML 8비트 팔레타이즈, Neural Engine, 약 252 tok/s)
- **[MADLAD-400](https://soniqo.audio/ko/guides/translate)** — 400+ 언어 간 다대다 번역 (3B, MLX INT4 + INT8, T5 v1.1, Apache 2.0)

**Speech-to-Speech 및 음성 에이전트**

- **[Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate)** — 스트리밍 음성-음성 번역 (FR/ES/PT/DE → EN, MLX INT4 + INT8, Kyutai Moshi/Mimi 스택, CC-BY-4.0)
- **[PersonaPlex](https://soniqo.audio/ko/guides/respond)** — 전이중 음성-음성 대화 (7B, 오디오 입력 → 오디오 출력, 18개 음색 프리셋)
- **[Audio2Face-3D](docs/models/audio2face3d.md)** — 음성 기반 아바타 얼굴 애니메이션 (NVIDIA Audio2Face-3D v2.3 Mark, 얼굴 계수 301개, MLX)

**향상, 분리 및 오디오 생성**

- **[DeepFilterNet3](https://soniqo.audio/ko/guides/denoise)** — 실시간 노이즈 억제 (2.1M 파라미터, 48 kHz). 60 s 단일 처리 한계를 초과하는 장시간 오디오는 crossfade로 자동 청크 처리 — `enhanceChunked(...)` API 참조
- **[LocalVQE v1.4-AEC](https://soniqo.audio/ko/guides/echo-cancellation)** — 분리되고 동기화된 마이크 및 재생 참조 스트림을 사용하는 스트리밍 음향 에코 제거(Core ML + 네이티브 적응형 필터, 16 kHz, 16 ms 알고리즘 지연)
- **[소스 분리](https://soniqo.audio/ko/guides/separate)** — HTDemucs (Demucs v4) + Open-Unmix 기반 음악 소스 분리 (UMX-HQ / UMX-L, 4개 스템: 보컬/드럼/베이스/기타, 44.1 kHz 스테레오)
- **[MAGNeT](https://soniqo.audio/ko/guides/compose)** — 텍스트 → 음악 생성 (Meta MAGNeT Small 300M / Medium 1.5B, MLX INT8, 30초 클립 32 kHz 모노, 마스크 병렬 디코딩)
- **[Stable Audio 3](docs/models/stable-audio-3.md)** — Text-to-audio/music generation (Stable Audio 3 Medium, MLX INT8/INT4, 44.1 kHz stereo, variable length)
- **[FlashSR](https://soniqo.audio/ko/guides/upsample)** — 오디오 초고해상도 (FlashSR ICASSP 2025, MLX, 48 kHz 모노, 1단계 증류 확산, INT4 363 MB / INT8 720 MB)

**턴 감지, 화자 분리 및 화자 식별**

- **[웨이크워드](https://soniqo.audio/ko/guides/wake-word)** — 온디바이스 키워드 감지 (KWS Zipformer 3M, CoreML, 실시간의 26배, 구성 가능한 키워드 목록)
- **[VAD](https://soniqo.audio/ko/guides/vad)** — 음성 활동 감지 (Silero 스트리밍, Pyannote 오프라인, FireRedVAD 100+ 개 언어)
- **[화자 분리](https://soniqo.audio/ko/guides/diarize)** — 누가 언제 말했는지 (Pyannote 파이프라인, Neural Engine 상의 엔드투엔드 Sortformer) — 이제 증분 스트리밍 세션 지원(안정적인 화자 ID, 480ms마다 업데이트)
- **[화자 임베딩](https://soniqo.audio/ko/guides/embed-speaker)** — WeSpeaker ResNet34 (256차원), ReDimNet2-B6 이름 지정 음성 식별 (192차원), CAM++ (192차원)

논문: [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba) · [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba) · [Omnilingual ASR](https://arxiv.org/abs/2511.09690) (Meta) · [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA) · [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba) · [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2) · [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA) · [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai) · [Hibiki](https://arxiv.org/abs/2502.03382) (Kyutai) · [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## 소식

- **2026년 4월 19일** — [Apple Silicon에서의 MLX와 CoreML — 올바른 백엔드 선택을 위한 실용 가이드](https://blog.ivan.digital/mlx-vs-coreml-on-apple-silicon-a-practical-guide-to-picking-the-right-backend-and-why-you-should-f77ddea7b27a)
- **2026년 3월 20일** — [600M 모델로 Mac에서 Whisper Large v3를 능가하다](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **2026년 2월 26일** — [Apple Silicon에서의 화자 분리 및 음성 활동 감지 — MLX 기반 네이티브 Swift](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **2026년 2월 23일** — [Apple Silicon에서 NVIDIA PersonaPlex 7B — MLX 기반 네이티브 Swift로 전이중 음성-음성 변환](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **2026년 2월 12일** — [Qwen3-ASR Swift: Apple Silicon용 온디바이스 ASR + TTS — 아키텍처 및 벤치마크](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## 빠른 시작

`Package.swift`에 패키지를 추가하세요:

```swift
.package(url: "https://github.com/soniqo/speech-swift", branch: "main")
```

필요한 모듈만 임포트하세요 — 모든 모델이 독립된 SPM 라이브러리이므로 사용하지 않는 것에 비용을 지불할 필요가 없습니다:

```swift
.product(name: "ParakeetStreamingASR", package: "speech-swift"),
.product(name: "SpeechUI",             package: "speech-swift"),  // 선택적 SwiftUI 뷰
```

**3줄로 오디오 버퍼 전사:**

```swift
import ParakeetStreamingASR

let model = try await ParakeetStreamingASRModel.fromPretrained()
let text = try model.transcribeAudio(audioSamples, sampleRate: 16000)
```

**부분 결과가 포함된 라이브 스트리밍:**

```swift
for await partial in model.transcribeStream(audio: samples, sampleRate: 16000) {
    print(partial.isFinal ? "FINAL: \(partial.text)" : "... \(partial.text)")
}
```

**약 10줄짜리 SwiftUI 받아쓰기 뷰:**

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

`SpeechUI`에는 `TranscriptionView`(파이널 + 파셜)와 `TranscriptionStore`(스트리밍 ASR 어댑터)만 포함됩니다. 오디오 시각화와 재생에는 AVFoundation을 사용하세요.

사용 가능한 SPM 프로덕트: `Qwen3ASR`, `WhisperASR`, `Qwen3TTS`, `Qwen3TTSCoreML`, `ParakeetASR`, `ParakeetStreamingASR`, `NemotronStreamingASR`, `OmnilingualASR`, `KokoroTTS`, `SupertonicTTS`, `VibeVoiceTTS`, `CosyVoiceTTS`, `VoxCPM2TTS`, `IndexTTS2TTS`, `F5TTS`, `HiggsTTS`, `ChatterboxTTS`, `OmniVoiceTTS`, `IndicMioTTS`, `FishAudioTTS`, `MagpieTTS`, `MagpieTTSCoreML`, `MAGNeTMusicGen`, `StableAudio3MusicGen`, `FlashSR`, `PersonaPlex`, `Audio2Face3D`, `HibikiTranslate`, `MADLADTranslation`, `SpeechVAD`, `SpeechWakeWord`, `SpeechEnhancement`, `SpeechRestoration`, `SourceSeparation`, `Qwen3Chat`, `FunctionGemma`, `SpeechCore`, `SpeechUI`, `AudioCommon`.

## 모델

아래는 컴팩트 뷰입니다. **[크기, 양자화, 다운로드 URL, 메모리 테이블을 포함한 전체 모델 카탈로그 → soniqo.audio/architecture](https://soniqo.audio/ko/architecture)**.

| 모델 | 작업 | 백엔드 | 크기 | 언어 |
|-------|------|----------|-------|-----------|
| [Qwen3-ASR](https://soniqo.audio/ko/guides/transcribe) | 음성 → 텍스트 | MLX, CoreML (하이브리드) | 0.6B, 1.7B | 52 |
| [WhisperASR](docs/models/whisper-asr.md) | Speech → Text | CoreML (ANE) | Large-v3 Turbo | Multi |
| [Parakeet TDT](https://soniqo.audio/ko/guides/parakeet) | 음성 → 텍스트 | CoreML (ANE) | 0.6B | 25개 유럽어 |
| [Parakeet EOU](https://soniqo.audio/ko/guides/dictate) | 음성 → 텍스트 (스트리밍) | CoreML (ANE) | 120M | 25개 유럽어 |
| [Nemotron Streaming (다국어)](https://soniqo.audio/ko/guides/nemotron) | 음성 → 텍스트 (스트리밍, 구두점 포함) | CoreML (ANE), MLX | 0.6B | **40** |
| [Nemotron Streaming (영어)](https://soniqo.audio/guides/nemotron) | 음성 → 텍스트 (스트리밍, 구두점 포함) | CoreML (ANE) | 0.6B | EN |
| [Omnilingual ASR](https://soniqo.audio/ko/guides/omnilingual) | 음성 → 텍스트 | CoreML (ANE), MLX | 300M / 1B / 3B / 7B | **[1,672](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)** |
| [Qwen3-ForcedAligner](https://soniqo.audio/ko/guides/align) | 오디오 + 텍스트 → 타임스탬프 | MLX, CoreML | 0.6B | 다언어 |
| [Qwen3-TTS](https://soniqo.audio/ko/guides/speak) | 텍스트 → 음성 | MLX, CoreML | 0.6B, 1.7B | 10 |
| [CosyVoice3](https://soniqo.audio/ko/guides/cosyvoice) | 텍스트 → 음성 | MLX | 0.5B | 9 |
| [VoxCPM2](https://soniqo.audio/ko/speech-generation) | 텍스트 → 음성 (48 kHz, 보이스 디자인 + 복제) | MLX | 2B (bf16/int8) | 30 |
| [IndexTTS2](docs/models/indextts2.md) | Text → Speech (zero-shot voice cloning) | MLX | 1.5B-class (fp16) | EN/ZH |
| [F5-TTS](docs/models/f5-tts.md) | Text → Speech (zero-shot voice cloning) | MLX | 336M (fp16) | EN/ZH |
| [Higgs TTS 3](docs/models/higgs-tts.md) | Text → Speech (conversational, zero-shot voice cloning) | MLX | 4B (bf16) | 100+ |
| [Kokoro-82M](https://soniqo.audio/ko/guides/kokoro) | 텍스트 → 음성 | CoreML (ANE) | 82M | 10 |
| [Supertonic-3](https://soniqo.audio/guides/supertonic) | 텍스트 → 음성 (44.1 kHz, flow-matching, G2P-free) | CoreML (ANE) | 99M | 31 |
| [VibeVoice Realtime-0.5B](https://soniqo.audio/ko/guides/vibevoice) | 텍스트 → 음성 (장문, 멀티 스피커) | MLX | 0.5B | EN/ZH |
| [VibeVoice 1.5B](https://soniqo.audio/ko/guides/vibevoice) | 텍스트 → 음성 (최대 90분 팟캐스트) | MLX | 1.5B | EN/ZH |
| [Magpie-TTS Multilingual](https://soniqo.audio/ko/guides/magpie) | 텍스트 → 음성 (5개 내장 스피커, 스트리밍) | MLX / CoreML | 357M (MLX INT8, CoreML INT8) | 9 (CoreML은 일본어 제외) |
| [Chatterbox Multilingual](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16) | 텍스트 → 음성 (제로샷 복제) | MLX | 0.8B (fp16) | 23 (HE는 niqqud 필요) |
| [OmniVoice](https://huggingface.co/aufklarer/OmniVoice-MLX-fp16) | 텍스트 → 음성 (NAR 확산, 제로샷 복제) | MLX | 0.8B (fp16 기본 / int8) | **600+** |
| [Indic-Mio](docs/models/indic-mio-tts.md) | Text → Speech (Hindi/Indic, emotion tags, voice cloning) | MLX | fp16 | Hindi / Indic |
| [Fish Audio S2 Pro](docs/models/fish-audio-s2-pro.md) | 텍스트 → 음성 (제로샷 복제, 명시적 스타일 마커) | MLX | 0.5B-class (fp16) | 다국어 |
| [Qwen3.5 Chat](docs/models/qwen35-chat.md) | Text → Text (LLM) | MLX, CoreML | 0.8B | Multi |
| [Qwen3 Dense Chat](docs/models/qwen3-dense-chat.md) | Text → Text (LLM) | MLX | 4B | Multi |
| [Gemma 4 Chat](docs/models/gemma4-chat.md) | Text → Text (LLM) | MLX | E2B / E4B (4-bit) | Multi |
| [FunctionGemma](https://soniqo.audio/ko/guides/function-calls) | 텍스트 → 도구 호출 (LLM) | CoreML | 270M | 영어 위주 |
| [MADLAD-400](https://soniqo.audio/ko/guides/translate) | 텍스트 → 텍스트 (번역) | MLX | 3B | **400+** |
| [Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate) | 음성 → 음성 (번역) | MLX | 3B | FR/ES/PT/DE → EN |
| [PersonaPlex](https://soniqo.audio/ko/guides/respond) | 음성 → 음성 | MLX | 7B | EN |
| [Audio2Face-3D](docs/models/audio2face3d.md) | 음성 → 얼굴 애니메이션 | MLX | v2.3 Mark | 언어 무관 |
| [Silero VAD](https://soniqo.audio/ko/guides/vad) | 음성 활동 감지 | MLX, CoreML | 309K | 언어 무관 |
| [KWS Zipformer](docs/models/kws-zipformer.md) | Audio → Wake word | CoreML (ANE) | 3M | EN/custom keywords |
| [Pyannote](https://soniqo.audio/ko/guides/diarize) | VAD + 화자 분리 | MLX | 1.5M | 언어 무관 |
| [Pyannote Community-1](https://huggingface.co/aufklarer/Pyannote-Community-1-CoreML) | 화자 분리 + 화자 임베딩 | CoreML (ANE) + Swift VBx | 8.35M | 언어 무관 |
| [Sortformer](https://huggingface.co/aufklarer/Sortformer-Diarization-CoreML) | [화자 분리 (E2E), 증분 스트리밍](https://soniqo.audio/ko/guides/diarize) | CoreML (ANE) | 117M | 언어 무관 |
| [DeepFilterNet3](https://soniqo.audio/ko/guides/denoise) | 음성 향상 | CoreML | 2.1M | 언어 무관 |
| [LocalVQE v1.4-AEC](https://soniqo.audio/ko/guides/echo-cancellation) | 음향 에코 제거 | CoreML + C++ | 200K + 2,742 | 언어 무관 |
| [Sidon](https://soniqo.audio/ko/guides/restore) | 음성 복원 (노이즈 제거 + 잔향 제거, 48 kHz) | CoreML | w2v-BERT 2.0 + DAC (fp16/int8) | 언어 무관 |
| [HTDemucs (Demucs v4)](https://soniqo.audio/ko/guides/separate) | 소스 분리 | MLX | 168M | Agnostic |
| [Open-Unmix](https://soniqo.audio/ko/guides/separate) | 소스 분리 | MLX | 8.6M | Agnostic |
| [MAGNeT](https://soniqo.audio/ko/guides/compose) | 텍스트 → 음악 (30초 @ 32 kHz) | MLX | 300M / 1.5B (int4/int8) | 영어 프롬프트 |
| [Stable Audio 3](docs/models/stable-audio-3.md) | Text → Music/audio (44.1 kHz stereo) | MLX | Medium 1.4B (int4/int8) | EN prompts |
| [FlashSR](https://soniqo.audio/ko/guides/upsample) | 오디오 초고해상도 (48 kHz) | MLX | 363 MB / 720 MB (int4/int8) | 언어 무관 |
| [WeSpeaker](https://soniqo.audio/ko/guides/embed-speaker) | 화자 임베딩 | MLX, CoreML | 6.6M | 언어 무관 |
| [ReDimNet2-B6](https://huggingface.co/aufklarer/ReDimNet2-B6-CoreML) | 이름 지정 음성 식별 | CoreML | 12.3M | 언어 무관 |

## 설치

### Homebrew

네이티브 ARM Homebrew(`/opt/homebrew`)가 필요합니다. Rosetta/x86_64 Homebrew는 지원되지 않습니다.

```bash
brew install speech
```

그런 다음:

```bash
speech transcribe recording.wav
speech speak "Hello world"
speech translate "Hello, how are you?" --to es
speech respond --input question.wav --transcript
speech-server --port 8080            # 로컬 HTTP / WebSocket 서버 (OpenAI 호환 /v1/realtime + /v1/audio/transcriptions)
```

**[전체 CLI 레퍼런스 →](https://soniqo.audio/ko/cli)**

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

필요한 것만 임포트하세요 — 모든 모델이 독립된 SPM 타겟입니다:

```swift
import Qwen3ASR             // 음성 인식 (MLX)
import WhisperASR           // Whisper Large-v3 Turbo (CoreML)
import ParakeetASR          // 음성 인식 (CoreML, 배치)
import ParakeetStreamingASR // 부분 결과 + EOU 포함 스트리밍 받아쓰기
import NemotronStreamingASR // 다국어 스트리밍 ASR, 네이티브 구두점 (0.6B, 40개 언어)
import OmnilingualASR       // 1,672개 언어 (CoreML + MLX)
import Qwen3TTS             // 텍스트-음성 변환
import CosyVoiceTTS         // 음성 복제 포함 텍스트-음성 변환
import VoxCPM2TTS           // 48 kHz TTS, 음성 복제 + 보이스 디자인 (2B)
import IndexTTS2TTS         // Native MLX voice cloning from reference audio
import F5TTS                // Zero-shot voice cloning (DiT flow matching + Vocos)
import HiggsTTS             // Conversational TTS + cloning (Qwen3 backbone, control tags)
import KokoroTTS            // 텍스트-음성 변환 (iOS 지원)
import VibeVoiceTTS         // 장문 / 멀티 스피커 TTS (EN/ZH)
import MagpieTTS            // 다국어 TTS (NVIDIA Magpie 357M, MLX, 9개 언어)
import MagpieTTSCoreML      // Magpie CoreML 백엔드 (CoreML + MLX 하이브리드, 8개 언어)
import FishAudioTTS         // 음성 복제 지원 실험적 Fish Audio S2 Pro 런타임
import Qwen3Chat            // 온디바이스 LLM 채팅
import FunctionGemma    // 온디바이스 함수 / 도구 호출 LLM
import MADLADTranslation    // 400+ 언어 간 다대다 번역
import HibikiTranslate      // 스트리밍 음성-음성 번역 (FR/ES/PT/DE → EN)
import PersonaPlex          // 전이중 음성-음성 변환
import SpeechVAD            // VAD + 화자 분리 + 임베딩
import SpeechEnhancement    // 노이즈 억제
import SpeechRestoration    // 음성 복원 — 노이즈 제거 + 잔향 제거 (Sidon, CoreML, 48 kHz)
import SourceSeparation     // 음악 소스 분리 (Open-Unmix, 4 스템)
import MAGNeTMusicGen      // 텍스트 → 음악 생성 (30초, 32 kHz)
import FlashSR             // 오디오 초고해상도 (48 kHz, 1단계 확산)
import SpeechUI             // 스트리밍 전사를 위한 SwiftUI 컴포넌트
import AudioCommon          // 공유 프로토콜 및 유틸리티
```

### 요구사항

- Swift 6+, Xcode 16+ (Metal Toolchain 포함)
- macOS 15+ (Sequoia) 또는 iOS 18+, Apple Silicon (M1/M2/M3/M4)

macOS 15 / iOS 18 최소 요구사항은 [MLState](https://developer.apple.com/documentation/coreml/mlstate) —— Apple의 영속적 ANE 상태 API —— 에서 비롯됩니다. CoreML 파이프라인(Qwen3-ASR, Qwen3-Chat, Qwen3-TTS)은 MLState를 사용해 KV 캐시를 토큰 스텝 간 Neural Engine에 상주시킵니다.

### 소스 빌드

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

`make build`는 Swift 패키지**와** MLX Metal 셰이더 라이브러리를 함께 컴파일합니다. Metal 라이브러리는 GPU 추론에 필요합니다 — 없으면 런타임에 `Failed to load the default metallib`이 발생합니다. 디버그 빌드는 `make debug`, 테스트 스위트는 `make test`로 실행합니다.

**[전체 빌드 및 설치 가이드 →](https://soniqo.audio/ko/getting-started)**

## 데모 앱

- **[DictateDemo](Examples/DictateDemo/)** ([문서](https://soniqo.audio/ko/guides/dictate)) — macOS 메뉴 바 스트리밍 받아쓰기. 라이브 파셜, VAD 기반 발화 종료 감지, 원클릭 복사. 백그라운드 agent로 실행됩니다 (Parakeet-EOU-120M + Silero VAD).
- **[iOSEchoDemo](Examples/iOSEchoDemo/)** — iOS 에코 데모 (Parakeet ASR + Kokoro TTS). 기기 및 시뮬레이터 지원.
- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — 마이크 입력, VAD, 멀티턴 컨텍스트를 지원하는 대화형 음성 어시스턴트. macOS. M2 Max에서 RTF 약 0.94 (실시간보다 빠름).
- **[SpeechDemo](Examples/SpeechDemo/)** — 탭형 인터페이스에서 받아쓰기와 TTS 합성. macOS.

각 데모의 README에 빌드 방법이 있습니다.

## 코드 예제

아래 스니펫은 각 도메인의 최소 사용 경로를 보여줍니다. 각 섹션은 [soniqo.audio](https://soniqo.audio/ko)의 전체 가이드로 링크되어 있으며, 설정 옵션, 다양한 백엔드, 스트리밍 패턴 및 CLI 레시피를 다룹니다.

### 음성-텍스트 변환 — [전체 가이드 →](https://soniqo.audio/ko/guides/transcribe)

```swift
import Qwen3ASR

let model = try await Qwen3ASRModel.fromPretrained()
let text = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

대체 백엔드: [WhisperASR](docs/inference/whisper-asr-inference.md) (Whisper Large-v3 Turbo, native CoreML), [Parakeet TDT](https://soniqo.audio/ko/guides/parakeet) (CoreML, 32× 실시간), [Omnilingual ASR](https://soniqo.audio/ko/guides/omnilingual) (1,672개 언어, CoreML 또는 MLX), [스트리밍 받아쓰기](https://soniqo.audio/ko/guides/dictate) (라이브 파셜).

### 강제 정렬 — [전체 가이드 →](https://soniqo.audio/ko/guides/align)

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

### 텍스트-음성 변환 — [전체 가이드 →](https://soniqo.audio/ko/guides/speak)

```swift
import Qwen3TTS
import AudioCommon

let model = try await Qwen3TTSModel.fromPretrained()
let audio = model.synthesize(text: "Hello world", language: "english")
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

대체 TTS 엔진: [CosyVoice3](https://soniqo.audio/ko/guides/cosyvoice) (스트리밍 + 음성 복제 + 감정 태그), [Kokoro-82M](https://soniqo.audio/ko/guides/kokoro) (iOS 지원, 54개 음색), [VibeVoice](https://soniqo.audio/ko/guides/vibevoice) (장문 팟캐스트 / 멀티 스피커, EN/ZH), [Fish Audio S2 Pro](docs/inference/fish-audio-s2-pro.md) (실험적 제로샷 복제 + 대괄호 스타일 마커), [음성 복제](https://soniqo.audio/ko/guides/voice-cloning).

### 음성-음성 변환 — [전체 가이드 →](https://soniqo.audio/ko/guides/respond)

```swift
import PersonaPlex

let model = try await PersonaPlexModel.fromPretrained()
let responseAudio = model.respond(userAudio: userSamples)
// 24 kHz 모노 Float32 출력, 재생 준비 완료
```

### LLM 채팅 — [전체 가이드 →](https://soniqo.audio/ko/guides/chat)

```swift
import Qwen3Chat
import FunctionGemma

let chat = try await Qwen35MLXChat.fromPretrained()
chat.chat(messages: [(.user, "Explain MLX in one sentence")]) { token, isFinal in
    print(token, terminator: "")
}
```

### 번역 — [전체 가이드 →](https://soniqo.audio/ko/guides/translate)

```swift
import MADLADTranslation

let translator = try await MADLADTranslator.fromPretrained()
let es = try translator.translate("Hello, how are you?", to: "es")
// → "Hola, ¿cómo estás?"
```

### 음성 번역 — [전체 가이드 →](https://soniqo.audio/guides/audio-translate)

```swift
import HibikiTranslate
import AudioCommon

let model = try await HibikiTranslateModel.fromPretrained()
let pcm = try AudioFileLoader.load(url: input, targetSampleRate: 24000)
let (englishAudio, textTokens) = model.translate(
    sourceAudio: pcm, sourceLanguage: .fr
)
// Hibiki Zero-3B — FR/ES/PT/DE → EN, 온디바이스, 스트리밍 Mimi 코덱
```

### 음성 활동 감지 — [전체 가이드 →](https://soniqo.audio/ko/guides/vad)

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
for s in segments { print("\(s.startTime)s → \(s.endTime)s") }
```

### 화자 분리 — [전체 가이드 →](https://soniqo.audio/ko/guides/diarize)

```swift
import SpeechVAD

let diarizer = try await DiarizationPipeline.fromPretrained()
let segments = diarizer.diarize(audio: samples, sampleRate: 16000)
for s in segments { print("Speaker \(s.speakerId): \(s.startTime)s - \(s.endTime)s") }
```

### 음성 향상 — [전체 가이드 →](https://soniqo.audio/ko/guides/denoise)

```swift
import SpeechEnhancement

let denoiser = try await DeepFilterNet3Model.fromPretrained()
let clean = try denoiser.enhance(audio: noisySamples, sampleRate: 48000)
```

### 음향 에코 제거 — [전체 가이드 →](https://soniqo.audio/ko/guides/echo-cancellation)

```swift
import SpeechEnhancement

let aec = try await LocalVQEEchoCanceller.fromPretrained()
let cleanMicrophone = try aec.processFrame(
    microphone: microphoneFrame,
    reference: playbackReferenceFrame
)
```

### 음성 복원 — [전체 가이드 →](https://soniqo.audio/ko/guides/restore)

[Sidon](https://arxiv.org/abs/2509.17052) (w2v-BERT 2.0 예측기 + DAC 보코더, Core ML)으로 노이즈 제거 **및** 잔향 제거를 동시에 수행합니다. 일반적인 노이즈 억제기와 달리 Sidon은 화자의 정체성을 보존하도록 학습되어, TTS 이전에 노이즈가 많거나 잔향이 있는 음성 복제 레퍼런스를 정리하는 데 적합합니다. 입력은 16 kHz이고 출력은 48 kHz 모노입니다.

```swift
import SpeechRestoration

let restorer = try await SpeechRestorer.fromPretrained()          // .fp16 (기본값) 또는 .int8
let clean = try restorer.restore(audio: noisySamples, sampleRate: 16000)  // → 48 kHz
```

CLI에서:

```bash
speech restore noisy.wav -o clean.wav            # 노이즈 제거 + 잔향 제거, 48 kHz 출력
speech restore noisy.wav --variant int8          # 더 작은 크기, 낮은 최대 RAM

# TTS 이전에 음성 복제 레퍼런스 정리 (옵트인; 화자 정체성 보존):
speech speak "Hello" --engine voxcpm2 --voice-sample ref.wav --clean-reference
```

### 음성 파이프라인 (ASR → LLM → TTS) — [전체 가이드 →](https://soniqo.audio/ko/voice-agents)

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

`VoicePipeline`은 실시간 음성 agent 상태 머신으로 ([speech-core](https://github.com/soniqo/speech-core)로 구동), VAD 기반 턴 감지, 인터럽션 처리, 이거(eager) STT를 지원합니다. 임의의 `SpeechRecognitionModel` + `SpeechGenerationModel` + `StreamingVADProvider`를 연결할 수 있습니다.

### HTTP API 서버

```bash
speech-server --port 8080
```

HTTP REST + WebSocket 엔드포인트로 모든 모델을 공개합니다. OpenAI 호환 API로 `/v1/realtime`의 Realtime WebSocket과 `/v1/audio/transcriptions`의 음성 인식 REST 엔드포인트가 포함됩니다. [`Sources/AudioServer/`](Sources/AudioServer/)를 참조하세요.

## 아키텍처

speech-swift는 모델당 하나의 SPM 타겟으로 분리되어 있어 사용자는 임포트한 것에 대해서만 비용을 지불합니다. 공유 인프라는 `AudioCommon` (프로토콜, 오디오 I/O, HuggingFace 다운로더, `SentencePieceModel`)과 `MLXCommon` (웨이트 로딩, `QuantizedLinear` 헬퍼, `SDPA` 멀티헤드 어텐션 헬퍼)에 있습니다.

**[백엔드, 메모리 테이블, 모듈 맵이 포함된 전체 아키텍처 다이어그램 → soniqo.audio/architecture](https://soniqo.audio/ko/architecture)** · **[API 레퍼런스 → soniqo.audio/api](https://soniqo.audio/ko/api)** · **[벤치마크 → soniqo.audio/benchmarks](https://soniqo.audio/ko/benchmarks)**

로컬 문서 (리포지토리):
- **모델:** [Qwen3-ASR](docs/models/asr-model.md) · [WhisperASR](docs/models/whisper-asr.md) · [Qwen3-TTS](docs/models/tts-model.md) · [CosyVoice](docs/models/cosyvoice-tts.md) · [Kokoro](docs/models/kokoro-tts.md) · [VoxCPM2](docs/models/voxcpm2-tts.md) · [IndexTTS2](docs/models/indextts2.md) · [F5-TTS](docs/models/f5-tts.md) · [Higgs TTS 3](docs/models/higgs-tts.md) · [VibeVoice](docs/models/vibevoice.md) · [Supertonic](docs/models/supertonic-tts.md) · [Chatterbox](docs/models/chatterbox-tts.md) · [Indic-Mio](docs/models/indic-mio-tts.md) · [Fish Audio S2 Pro](docs/models/fish-audio-s2-pro.md) · [Magpie TTS](docs/models/magpie-tts.md) · [Parakeet TDT](docs/models/parakeet-asr.md) · [Parakeet Streaming](docs/models/parakeet-streaming-asr.md) · [Nemotron Streaming](docs/models/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/models/omnilingual-asr.md) · [PersonaPlex](docs/models/personaplex.md) · [Hibiki](docs/models/hibiki.md) · [MADLAD-400](docs/models/madlad-translation.md) · [FunctionGemma](docs/models/function-gemma.md) · [Qwen3.5 Chat](docs/models/qwen35-chat.md) · [Gemma 4 Chat](docs/models/gemma4-chat.md) · [Qwen3 Dense Chat](docs/models/qwen3-dense-chat.md) · [FireRedVAD](docs/models/fireredvad.md) · [KWS Zipformer](docs/models/kws-zipformer.md) · [Sidon](docs/models/sidon.md) · [Source Separation](docs/models/source-separation.md) · [HTDemucs](docs/models/htdemucs.md) · [MAGNeT](docs/models/magnet-music-gen.md) · [Stable Audio 3](docs/models/stable-audio-3.md) · [FlashSR](docs/models/flashsr.md) · [Audio2Face-3D](docs/models/audio2face3d.md)
- **추론:** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md) · [WhisperASR](docs/inference/whisper-asr-inference.md) · [Parakeet TDT](docs/inference/parakeet-asr-inference.md) · [Parakeet Streaming](docs/inference/parakeet-streaming-asr-inference.md) · [Nemotron Streaming](docs/inference/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/inference/omnilingual-asr-inference.md) · [TTS](docs/inference/qwen3-tts-inference.md) · [VoxCPM2](docs/inference/voxcpm2-inference.md) · [IndexTTS2](docs/inference/indextts2.md) · [F5-TTS](docs/inference/f5-tts.md) · [Higgs TTS 3](docs/inference/higgs-tts.md) · [VibeVoice](docs/inference/vibevoice-inference.md) · [Fish Audio S2 Pro](docs/inference/fish-audio-s2-pro.md) · [Magpie TTS](docs/inference/magpie-tts.md) · [Hibiki](docs/inference/hibiki-inference.md) · [MADLAD-400](docs/inference/madlad-translation.md) · [MAGNeT](docs/inference/magnet-music-gen.md) · [Stable Audio 3](docs/inference/stable-audio-3.md) · [FlashSR](docs/inference/flashsr.md) · [Forced Aligner](docs/inference/forced-aligner.md) · [Silero VAD](docs/inference/silero-vad.md) · [FireRedVAD](docs/inference/fireredvad.md) · [Wake-word](docs/inference/wake-word.md) · [Speaker Diarization](docs/inference/speaker-diarization.md) · [Speech Enhancement](docs/inference/speech-enhancement.md) · [Sidon](docs/inference/sidon.md) · [Cache/offline](docs/inference/cache-and-offline.md)
- **에코 제거:** [LocalVQE AEC](docs/inference/echo-cancellation.md)
- **레퍼런스:** [공유 프로토콜](docs/shared-protocols.md)

## 캐시 설정

모델 웨이트는 첫 사용 시 HuggingFace에서 다운로드되어 `~/Library/Caches/qwen3-speech/`에 캐시됩니다. `QWEN3_CACHE_DIR` (CLI) 또는 `cacheDir:` (Swift API)로 재정의할 수 있습니다. 모든 `fromPretrained()` 엔트리 포인트는 `offlineMode: true`를 지원하여 웨이트가 이미 캐시되어 있으면 네트워크를 건너뜁니다.

중국 본토 사용자(또는 `huggingface.co` 접속이 느리거나 차단된 지역)는 `HF_ENDPOINT`를 설정하여 미러에서 다운로드할 수 있습니다. 예: `export HF_ENDPOINT=https://hf-mirror.com`.

샌드박스 iOS 컨테이너 경로를 포함한 자세한 내용은 [`docs/inference/cache-and-offline.md`](docs/inference/cache-and-offline.md)를 참조하세요.

## MLX Metal 라이브러리

런타임에 `Failed to load the default metallib`이 보이면 Metal 셰이더 라이브러리가 없습니다. 수동 `swift build` 후에 `make build` 또는 `./scripts/build_mlx_metallib.sh release`를 실행하세요. Metal Toolchain이 없다면 먼저 설치하세요:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## 테스트

```bash
make test                            # 전체 스위트 (단위 + 모델 다운로드 포함 E2E)
swift test --skip E2E                # 단위만 (CI 안전, 다운로드 없음)
swift test --filter Qwen3ASRTests    # 특정 모듈
```

E2E 테스트 클래스는 `E2E` 접두사를 사용하므로 CI는 `--skip E2E`로 필터링할 수 있습니다. 전체 테스트 규약은 [CLAUDE.md](CLAUDE.md#testing)를 참조하세요.

## 기여

PR 환영합니다 — 버그 수정, 새 모델 통합, 문서. fork, 피처 브랜치 생성, `make build && make test`, `main`에 대해 PR을 여세요.

## 라이선스

Apache 2.0
