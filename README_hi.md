# Speech Swift

Apple Silicon के लिए AI स्पीच मॉडल, MLX Swift और CoreML द्वारा संचालित।

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

Mac और iOS के लिए ऑन-डिवाइस स्पीच रिकग्निशन, सिंथेसिस और समझ। Apple Silicon पर पूरी तरह लोकली चलता है — कोई क्लाउड नहीं, कोई API key नहीं, कोई डेटा डिवाइस से बाहर नहीं जाता।

**[📚 पूर्ण डॉक्यूमेंटेशन →](https://soniqo.audio/hi)** · **[🤗 HuggingFace मॉडल](https://huggingface.co/aufklarer)** · **[📝 ब्लॉग](https://blog.ivan.digital)**

<p align="center">
  <a href="https://formulae.brew.sh/formula/speech"><img src="https://img.shields.io/homebrew/installs/dm/speech.svg?logo=homebrew&amp;label=Homebrew%20installs&amp;color=FBB040" alt="Homebrew installs"></a>
  <a href="https://github.com/soniqo/speech-swift#built-with-speech-swift"><img src="https://img.shields.io/badge/verified%20public%20repositories-15-2ea44f?logo=github" alt="Verified public repositories: 15"></a>
</p>

<p align="center">
  <a href="https://trendshift.io/repositories/24196?utm_source=trendshift-badge&amp;utm_medium=badge&amp;utm_campaign=badge-trendshift-24196" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/trendshift/repositories/24196/daily?language=Swift" alt="soniqo%2Fspeech-swift | Trendshift" width="250" height="55"/></a>
</p>

<p align="center">
  <a href="https://youtu.be/x9zgcaW0gUk">
    <img src="https://img.youtube.com/vi/x9zgcaW0gUk/maxresdefault.jpg" width="640" alt="MacBook पर लोकल स्पीच AI — YouTube पर चार मिनट का ओपन-सोर्स लाइब्रेरी टूर देखें">
  </a>
</p>
<p align="center"><em>MacBook पर लोकल स्पीच AI — YouTube पर चार मिनट का ओपन-सोर्स लाइब्रेरी टूर देखें</em></p>

**यूज़-केस:** [वॉइस एजेंट](https://soniqo.audio/hi/voice-agents) · [ट्रांसक्रिप्शन](https://soniqo.audio/hi/transcription) · [स्पीच जनरेशन](https://soniqo.audio/hi/speech-generation)

## Speech Swift के साथ निर्मित

Speech Swift पैकेज के सत्यापन योग्य संदर्भ वाले 15 सार्वजनिक रिपॉज़िटरी।

[Palmier Pro](https://github.com/palmier-io/palmier-pro) · [Anarlog](https://github.com/fastrepl/anarlog) · [ClawdHome](https://github.com/ThinkInAIXYZ/clawdhome) · [Jabber](https://github.com/rselbach/jabber) · [Ora](https://github.com/wuwangzhang1216/ora) · [VoxFlow](https://github.com/xingbofeng/VoxFlow) · [LokalBot](https://github.com/stevyhacker/lokalbot) · [Voicey](https://github.com/jonathanKingston/voicey) · [HushType](https://github.com/felixfu824/HushType) · [DexDictate macOS](https://github.com/westkitty/DexDictate_MacOS) · [Watchtower](https://github.com/aiwatchtowers/watchtower) · [Wishper App](https://github.com/irangareddy/wishper-app) · [FriSpeak](https://github.com/KSubedi/FriSpeak) · [Scribe](https://github.com/itchat/Scribe) · [VoicePen](https://github.com/dot-sk/VoicePen)

**OpenAI-संगत ऐप्स:** [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) ([मर्ज किया गया WAV समर्थन](https://github.com/Mintplex-Labs/anything-llm/pull/6012))

**क्षमता समूह:** STT / ASR · अलाइनमेंट · TTS · LLM और अनुवाद · स्पीच-टू-स्पीच · एन्हांसमेंट / रिस्टोरेशन · सोर्स सेपरेशन · संगीत / ऑडियो जनरेशन · वेक वर्ड, VAD, डायराइज़ेशन और स्पीकर पहचान

**STT / ASR**

- **[Qwen3-ASR](https://soniqo.audio/hi/guides/transcribe)** — स्पीच-टू-टेक्स्ट (ऑटोमैटिक स्पीच रिकग्निशन, 52 भाषाएँ, MLX + CoreML)
- **[WhisperASR](docs/models/whisper-asr.md)** — Whisper Large-v3 Turbo speech-to-text via native CoreML runtime (ANE, multilingual)
- **[Parakeet TDT](https://soniqo.audio/hi/guides/parakeet)** — CoreML के माध्यम से स्पीच-टू-टेक्स्ट (Neural Engine, NVIDIA FastConformer + TDT decoder, 25 भाषाएँ)
- **[Omnilingual ASR](https://soniqo.audio/hi/guides/omnilingual)** — स्पीच-टू-टेक्स्ट (Meta wav2vec2 + CTC, **1,672 भाषाएँ** 32 लिपियों में, CoreML 300M + MLX 300M/1B/3B/7B)
- **[Streaming Dictation](https://soniqo.audio/hi/guides/dictate)** — पार्शियल्स और एंड-ऑफ-अटरन्स डिटेक्शन के साथ रियल-टाइम डिक्टेशन (Parakeet-EOU-120M)
- **[Nemotron Streaming (बहुभाषी)](https://soniqo.audio/hi/guides/nemotron)** — नेटिव विराम चिह्न और कैपिटलाइज़ेशन के साथ लो-लेटेंसी स्ट्रीमिंग ASR (NVIDIA Nemotron-3.5-ASR-Streaming-0.6B, CoreML + MLX, **40 भाषा-लोकेल**)
- **[Nemotron Streaming (अंग्रेज़ी)](https://soniqo.audio/guides/nemotron)** — नेटिव विराम चिह्न और कैपिटलाइज़ेशन के साथ लो-लेटेंसी स्ट्रीमिंग ASR (NVIDIA Nemotron-Speech-Streaming-0.6B, CoreML, केवल अंग्रेज़ी, बहुभाषी संस्करण से छोटा और तेज़)

**अलाइनमेंट**

- **[Qwen3-ForcedAligner](https://soniqo.audio/hi/guides/align)** — शब्द-स्तरीय टाइमस्टैम्प अलाइनमेंट (ऑडियो + टेक्स्ट → टाइमस्टैम्प)

**TTS / स्पीच जनरेशन**

- **[Qwen3-TTS](https://soniqo.audio/hi/guides/speak)** — टेक्स्ट-टू-स्पीच (सर्वोच्च गुणवत्ता, स्ट्रीमिंग, कस्टम स्पीकर, 10 भाषाएँ)
- **[CosyVoice TTS](https://soniqo.audio/hi/guides/cosyvoice)** — वॉयस क्लोनिंग, मल्टी-स्पीकर डायलॉग, इमोशन टैग के साथ स्ट्रीमिंग TTS (9 भाषाएँ)
- **[VoxCPM2](https://soniqo.audio/hi/speech-generation)** — वॉयस क्लोनिंग + निर्देश-आधारित वॉयस डिज़ाइन के साथ 48 kHz स्टूडियो-गुणवत्ता TTS (2B, MLX bf16/int8, 30 भाषाएँ)
- **[IndexTTS2](docs/models/indextts2.md)** — Native MLX voice cloning from a reference voice (IndexTeam IndexTTS-2, 1.5B-class fp16 bundle, speaker/emotion/pause controls)
- **[F5-TTS](docs/models/f5-tts.md)** — Zero-shot voice cloning from a short reference clip + transcript (SWivid F5-TTS v1 Base, DiT flow matching + Vocos, MLX fp16, 24 kHz, English + Mandarin; non-commercial license)
- **[Higgs TTS 3](docs/models/higgs-tts.md)** — Conversational TTS with zero-shot voice cloning and inline emotion/style/SFX/prosody tags (Boson Higgs TTS 3, Qwen3-4B backbone, MLX bf16, 24 kHz, 100+ languages; research/non-commercial license)
- **[Kokoro TTS](https://soniqo.audio/hi/guides/kokoro)** — ऑन-डिवाइस TTS (82M, CoreML/Neural Engine, 54 वॉयस, iOS-ready, 10 भाषाएँ)
- **[VibeVoice TTS](https://soniqo.audio/hi/guides/vibevoice)** — लंबे-रूप / बहु-वक्ता TTS (Microsoft VibeVoice Realtime-0.5B + 1.5B, MLX, 90 मिनट तक के पॉडकास्ट / ऑडियोबुक संश्लेषण, EN/ZH)
- **[Magpie TTS](https://soniqo.audio/hi/guides/magpie)** — बहुभाषी TTS (NVIDIA Magpie-TTS Multilingual 357M, MLX INT8 411 MB या CoreML INT8 342 MB, 9 भाषाएँ, 5 पूर्व-निर्धारित वक्ता, MLX पर स्ट्रीमिंग)
- **[Supertonic TTS](https://soniqo.audio/guides/supertonic)** — ऑन-डिवाइस फ़्लो-मैचिंग TTS (Supertone Supertonic-3 99M, CoreML/Neural Engine, 31 भाषाएँ, 10 वॉयस, G2P-free, 44.1 kHz)
- **[Chatterbox TTS](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16)** — ज़ीरो-शॉट वॉयस क्लोनिंग के साथ बहुभाषी TTS (Resemble AI Chatterbox Multilingual, MLX fp16 ~1.3 GB, 23 रनटाइम भाषाएँ; हिब्रू के लिए niqqud आवश्यक, MIT)
- **[OmniVoice TTS](https://huggingface.co/aufklarer/OmniVoice-MLX-fp16)** — ज़ीरो-शॉट वॉयस क्लोनिंग के साथ नॉन-ऑटोरिग्रेसिव डिफ्यूज़न TTS (k2-fsa OmniVoice, Qwen3 बैकबोन, MLX fp16 डिफ़ॉल्ट / int8 उपलब्ध, 600+ भाषाएँ, Apache-2.0)
- **[Indic-Mio](docs/models/indic-mio-tts.md)** — Hindi/Indic TTS with inline emotion markers and optional reference-voice cloning (MLX, 24 kHz)

**LLM और अनुवाद**

- **[Qwen3Chat](https://soniqo.audio/hi/guides/chat)** — ऑन-डिवाइस LLM चैट (Qwen3.5-0.8B MLX/CoreML के साथ dense Qwen3 4B और Gemma 4 E2B/E4B MLX बैकएंड, स्ट्रीमिंग टोकन)
- **[FunctionGemma](https://soniqo.audio/hi/guides/function-calls)** — संरचित फ़ंक्शन / टूल कॉल के लिए ऑन-डिवाइस LLM (Gemma 3 270M, CoreML 8-बिट पैलेटाइज़, Neural Engine, ~252 tok/s)
- **[MADLAD-400](https://soniqo.audio/hi/guides/translate)** — 400+ भाषाओं में बहु-दिशात्मक अनुवाद (3B, MLX INT4 + INT8, T5 v1.1, Apache 2.0)

**स्पीच-टू-स्पीच और वॉइस एजेंट**

- **[Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate)** — स्ट्रीमिंग स्पीच-टू-स्पीच अनुवाद (FR/ES/PT/DE → EN, MLX INT4 + INT8, Kyutai Moshi/Mimi स्टैक, CC-BY-4.0)
- **[PersonaPlex](https://soniqo.audio/hi/guides/respond)** — फुल-डुप्लेक्स स्पीच-टू-स्पीच (7B, ऑडियो इन → ऑडियो आउट, 18 वॉयस प्रीसेट)
- **[Audio2Face-3D](docs/models/audio2face3d.md)** — वाक्-चालित अवतार चेहरा एनीमेशन (NVIDIA Audio2Face-3D v2.3 Mark, 301 फेशियल कोएफ़िशिएंट, MLX)

**एन्हांसमेंट, सेपरेशन और ऑडियो जनरेशन**

- **[DeepFilterNet3](https://soniqo.audio/hi/guides/denoise)** — रियल-टाइम नॉइज़ सप्रेशन (2.1M params, 48 kHz)। 60 s सिंगल-शॉट सीमा से अधिक लंबे ऑडियो को crossfade के साथ स्वतः चंक किया जाता है — `enhanceChunked(...)` देखें
- **[सोर्स सेपरेशन](https://soniqo.audio/hi/guides/separate)** — HTDemucs (Demucs v4) + Open-Unmix के ज़रिए म्यूज़िक सोर्स सेपरेशन (UMX-HQ / UMX-L, 4 स्टेम: वोकल/ड्रम्स/बेस/अन्य, 44.1 kHz स्टीरियो)
- **[MAGNeT](https://soniqo.audio/hi/guides/compose)** — टेक्स्ट से संगीत निर्माण (Meta MAGNeT Small 300M / Medium 1.5B, MLX INT8, 30 सेकंड क्लिप 32 kHz मोनो, मास्क्ड पैरलल डिकोडिंग)
- **[Stable Audio 3](docs/models/stable-audio-3.md)** — Text-to-audio/music generation (Stable Audio 3 Medium, MLX INT8/INT4, 44.1 kHz stereo, variable length)
- **[FlashSR](https://soniqo.audio/hi/guides/upsample)** — ऑडियो सुपर-रेज़ोल्यूशन (FlashSR ICASSP 2025, MLX, 48 kHz मोनो, 1-स्टेप डिस्टिल्ड डिफ्यूज़न, INT4 363 MB / INT8 720 MB)

**टर्न डिटेक्शन, डायराइज़ेशन और स्पीकर पहचान**

- **[वेक-वर्ड](https://soniqo.audio/hi/guides/wake-word)** — ऑन-डिवाइस कीवर्ड स्पॉटिंग (KWS Zipformer 3M, CoreML, 26× रियल-टाइम, कॉन्फ़िगरेबल कीवर्ड सूची)
- **[VAD](https://soniqo.audio/hi/guides/vad)** — वॉयस एक्टिविटी डिटेक्शन (Silero स्ट्रीमिंग, Pyannote ऑफ़लाइन, FireRedVAD 100+ भाषाएँ)
- **[Speaker Diarization](https://soniqo.audio/hi/guides/diarize)** — कौन कब बोला (Pyannote पाइपलाइन, Neural Engine पर एंड-टू-एंड Sortformer) — अब इंक्रीमेंटल स्ट्रीमिंग सत्र के साथ (स्थिर स्पीकर ID, हर 480 ms पर अपडेट)
- **[Speaker Embeddings](https://soniqo.audio/hi/guides/embed-speaker)** — WeSpeaker ResNet34 (256-dim), ReDimNet2-B6 नामित आवाज़ पहचान (192-dim), CAM++ (192-dim)

पेपर: [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba) · [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba) · [Omnilingual ASR](https://arxiv.org/abs/2511.09690) (Meta) · [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA) · [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba) · [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2) · [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA) · [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai) · [Hibiki](https://arxiv.org/abs/2502.03382) (Kyutai) · [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## समाचार

- **19 Apr 2026** — [MLX vs CoreML on Apple Silicon — A Practical Guide to Picking the Right Backend](https://blog.ivan.digital/mlx-vs-coreml-on-apple-silicon-a-practical-guide-to-picking-the-right-backend-and-why-you-should-f77ddea7b27a)
- **20 Mar 2026** — [We Beat Whisper Large v3 with a 600M Model Running Entirely on Your Mac](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **26 Feb 2026** — [Speaker Diarization and Voice Activity Detection on Apple Silicon — Native Swift with MLX](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **23 Feb 2026** — [NVIDIA PersonaPlex 7B on Apple Silicon — Full-Duplex Speech-to-Speech in Native Swift with MLX](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **12 Feb 2026** — [Qwen3-ASR Swift: On-Device ASR + TTS for Apple Silicon — Architecture and Benchmarks](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## त्वरित प्रारंभ

अपने `Package.swift` में पैकेज जोड़ें:

```swift
.package(url: "https://github.com/soniqo/speech-swift", branch: "main")
```

केवल वही मॉड्यूल इम्पोर्ट करें जिनकी आपको ज़रूरत है — प्रत्येक मॉडल अपनी अलग SPM लाइब्रेरी है, इसलिए आप केवल उसी के लिए भुगतान करते हैं जो आप उपयोग करते हैं:

```swift
.product(name: "ParakeetStreamingASR", package: "speech-swift"),
.product(name: "SpeechUI",             package: "speech-swift"),  // वैकल्पिक SwiftUI व्यू
```

**3 लाइनों में ऑडियो बफ़र ट्रांसक्राइब करें:**

```swift
import ParakeetStreamingASR

let model = try await ParakeetStreamingASRModel.fromPretrained()
let text = try model.transcribeAudio(audioSamples, sampleRate: 16000)
```

**पार्शियल्स के साथ लाइव स्ट्रीमिंग:**

```swift
for await partial in model.transcribeStream(audio: samples, sampleRate: 16000) {
    print(partial.isFinal ? "FINAL: \(partial.text)" : "... \(partial.text)")
}
```

**~10 लाइनों में SwiftUI डिक्टेशन व्यू:**

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

`SpeechUI` केवल `TranscriptionView` (finals + partials) और `TranscriptionStore` (स्ट्रीमिंग ASR एडाप्टर) प्रदान करता है। ऑडियो विज़ुअलाइज़ेशन और प्लेबैक के लिए AVFoundation का उपयोग करें।

उपलब्ध SPM उत्पाद: `Qwen3ASR`, `WhisperASR`, `Qwen3TTS`, `Qwen3TTSCoreML`, `ParakeetASR`, `ParakeetStreamingASR`, `NemotronStreamingASR`, `OmnilingualASR`, `KokoroTTS`, `SupertonicTTS`, `VibeVoiceTTS`, `CosyVoiceTTS`, `VoxCPM2TTS`, `IndexTTS2TTS`, `F5TTS`, `HiggsTTS`, `ChatterboxTTS`, `OmniVoiceTTS`, `IndicMioTTS`, `FishAudioTTS`, `MagpieTTS`, `MagpieTTSCoreML`, `MAGNeTMusicGen`, `StableAudio3MusicGen`, `FlashSR`, `PersonaPlex`, `Audio2Face3D`, `HibikiTranslate`, `MADLADTranslation`, `SpeechVAD`, `SpeechWakeWord`, `SpeechEnhancement`, `SpeechRestoration`, `SourceSeparation`, `Qwen3Chat`, `FunctionGemma`, `SpeechCore`, `SpeechUI`, `AudioCommon`.

## मॉडल

नीचे संक्षिप्त दृश्य। **[पूर्ण मॉडल कैटलॉग (आकार, क्वांटिज़ेशन, डाउनलोड URL, मेमोरी टेबल्स) → soniqo.audio/architecture](https://soniqo.audio/hi/architecture)**.

| मॉडल | कार्य | बैकएंड | आकार | भाषाएँ |
|------|------|--------|------|--------|
| [Qwen3-ASR](https://soniqo.audio/hi/guides/transcribe) | स्पीच → टेक्स्ट | MLX, CoreML (हाइब्रिड) | 0.6B, 1.7B | 52 |
| [WhisperASR](docs/models/whisper-asr.md) | Speech → Text | CoreML (ANE) | Large-v3 Turbo | Multi |
| [Parakeet TDT](https://soniqo.audio/hi/guides/parakeet) | स्पीच → टेक्स्ट | CoreML (ANE) | 0.6B | 25 यूरोपीय |
| [Parakeet EOU](https://soniqo.audio/hi/guides/dictate) | स्पीच → टेक्स्ट (स्ट्रीमिंग) | CoreML (ANE) | 120M | 25 यूरोपीय |
| [Nemotron Streaming (बहुभाषी)](https://soniqo.audio/hi/guides/nemotron) | भाषण → टेक्स्ट (स्ट्रीमिंग, विराम चिह्नों के साथ) | CoreML (ANE), MLX | 0.6B | **40** |
| [Nemotron Streaming (अंग्रेज़ी)](https://soniqo.audio/guides/nemotron) | भाषण → टेक्स्ट (स्ट्रीमिंग, विराम चिह्नों के साथ) | CoreML (ANE) | 0.6B | EN |
| [Omnilingual ASR](https://soniqo.audio/hi/guides/omnilingual) | स्पीच → टेक्स्ट | CoreML (ANE), MLX | 300M / 1B / 3B / 7B | **[1,672](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)** |
| [Qwen3-ForcedAligner](https://soniqo.audio/hi/guides/align) | ऑडियो + टेक्स्ट → टाइमस्टैम्प | MLX, CoreML | 0.6B | बहुभाषी |
| [Qwen3-TTS](https://soniqo.audio/hi/guides/speak) | टेक्स्ट → स्पीच | MLX, CoreML | 0.6B, 1.7B | 10 |
| [CosyVoice3](https://soniqo.audio/hi/guides/cosyvoice) | टेक्स्ट → स्पीच | MLX | 0.5B | 9 |
| [VoxCPM2](https://soniqo.audio/hi/speech-generation) | टेक्स्ट → स्पीच (48 kHz, वॉयस डिज़ाइन + क्लोनिंग) | MLX | 2B (bf16/int8) | 30 |
| [IndexTTS2](docs/models/indextts2.md) | Text → Speech (zero-shot voice cloning) | MLX | 1.5B-class (fp16) | EN/ZH |
| [F5-TTS](docs/models/f5-tts.md) | Text → Speech (zero-shot voice cloning) | MLX | 336M (fp16) | EN/ZH |
| [Higgs TTS 3](docs/models/higgs-tts.md) | Text → Speech (conversational, zero-shot voice cloning) | MLX | 4B (bf16) | 100+ |
| [Chatterbox Multilingual](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16) | टेक्स्ट → स्पीच (ज़ीरो-शॉट क्लोनिंग) | MLX | 0.8B (fp16) | 23 (HE के लिए niqqud आवश्यक) |
| [OmniVoice](https://huggingface.co/aufklarer/OmniVoice-MLX-fp16) | टेक्स्ट → स्पीच (NAR डिफ्यूज़न, ज़ीरो-शॉट क्लोनिंग) | MLX | 0.8B (fp16 डिफ़ॉल्ट / int8) | 600+ |
| [Indic-Mio](docs/models/indic-mio-tts.md) | Text → Speech (Hindi/Indic, emotion tags, voice cloning) | MLX | fp16 | Hindi / Indic |
| [Fish Audio S2 Pro](docs/models/fish-audio-s2-pro.md) | टेक्स्ट → स्पीच (ज़ीरो-शॉट क्लोनिंग, स्पष्ट स्टाइल मार्कर) | MLX | 0.5B-class (fp16) | बहुभाषी |
| [Kokoro-82M](https://soniqo.audio/hi/guides/kokoro) | टेक्स्ट → स्पीच | CoreML (ANE) | 82M | 10 |
| [Supertonic-3](https://soniqo.audio/guides/supertonic) | टेक्स्ट → स्पीच (44.1 kHz, फ़्लो-मैचिंग, G2P-free) | CoreML (ANE) | 99M | 31 |
| [VibeVoice Realtime-0.5B](https://soniqo.audio/hi/guides/vibevoice) | टेक्स्ट → स्पीच (लंबे-रूप, बहु-वक्ता) | MLX | 0.5B | EN/ZH |
| [VibeVoice 1.5B](https://soniqo.audio/hi/guides/vibevoice) | टेक्स्ट → स्पीच (90 मिनट तक पॉडकास्ट) | MLX | 1.5B | EN/ZH |
| [Magpie-TTS Multilingual](https://soniqo.audio/hi/guides/magpie) | टेक्स्ट → वाक् (5 पूर्व-निर्धारित वक्ता, स्ट्रीमिंग) | MLX / CoreML | 357M (MLX INT8, CoreML INT8) | 9 (CoreML में JA नहीं) |
| [Qwen3.5 Chat](docs/models/qwen35-chat.md) | Text → Text (LLM) | MLX, CoreML | 0.8B | Multi |
| [Qwen3 Dense Chat](docs/models/qwen3-dense-chat.md) | Text → Text (LLM) | MLX | 4B | Multi |
| [Gemma 4 Chat](docs/models/gemma4-chat.md) | Text → Text (LLM) | MLX | E2B / E4B (4-bit) | Multi |
| [FunctionGemma](https://soniqo.audio/hi/guides/function-calls) | टेक्स्ट → टूल कॉल (LLM) | CoreML | 270M | EN |
| [MADLAD-400](https://soniqo.audio/hi/guides/translate) | टेक्स्ट → टेक्स्ट (अनुवाद) | MLX | 3B | **400+** |
| [Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate) | स्पीच → स्पीच (अनुवाद) | MLX | 3B | FR/ES/PT/DE → EN |
| [PersonaPlex](https://soniqo.audio/hi/guides/respond) | स्पीच → स्पीच | MLX | 7B | EN |
| [Audio2Face-3D](docs/models/audio2face3d.md) | स्पीच → चेहरा एनीमेशन | MLX | v2.3 Mark | भाषा-तटस्थ |
| [Silero VAD](https://soniqo.audio/hi/guides/vad) | वॉयस एक्टिविटी डिटेक्शन | MLX, CoreML | 309K | भाषा-तटस्थ |
| [KWS Zipformer](docs/models/kws-zipformer.md) | Audio → Wake word | CoreML (ANE) | 3M | EN/custom keywords |
| [Pyannote](https://soniqo.audio/hi/guides/diarize) | VAD + Diarization | MLX | 1.5M | भाषा-तटस्थ |
| [Pyannote Community-1](https://huggingface.co/aufklarer/Pyannote-Community-1-CoreML) | Diarization + स्पीकर embeddings | CoreML (ANE) + Swift VBx | 8.35M | भाषा-तटस्थ |
| [Sortformer](https://huggingface.co/aufklarer/Sortformer-Diarization-CoreML) | [Diarization (E2E), इंक्रीमेंटल स्ट्रीमिंग](https://soniqo.audio/hi/guides/diarize) | CoreML (ANE) | 117M | भाषा-तटस्थ |
| [DeepFilterNet3](https://soniqo.audio/hi/guides/denoise) | स्पीच एन्हांसमेंट | CoreML | 2.1M | भाषा-तटस्थ |
| [Sidon](https://soniqo.audio/hi/guides/restore) | स्पीच रिस्टोरेशन (नॉइज़ हटाना + डीरीवर्ब, 48 kHz) | CoreML | w2v-BERT 2.0 + DAC (fp16/int8) | भाषा-तटस्थ |
| [HTDemucs (Demucs v4)](https://soniqo.audio/hi/guides/separate) | सोर्स सेपरेशन | MLX | 168M | Agnostic |
| [Open-Unmix](https://soniqo.audio/hi/guides/separate) | सोर्स सेपरेशन | MLX | 8.6M | Agnostic |
| [MAGNeT](https://soniqo.audio/hi/guides/compose) | टेक्स्ट → संगीत (30 सेकंड @ 32 kHz) | MLX | 300M / 1.5B (int4/int8) | EN प्रॉम्प्ट्स |
| [Stable Audio 3](docs/models/stable-audio-3.md) | Text → Music/audio (44.1 kHz stereo) | MLX | Medium 1.4B (int4/int8) | EN prompts |
| [FlashSR](https://soniqo.audio/hi/guides/upsample) | ऑडियो सुपर-रेज़ोल्यूशन (48 kHz) | MLX | 363 MB / 720 MB (int4/int8) | भाषा-निरपेक्ष |
| [WeSpeaker](https://soniqo.audio/hi/guides/embed-speaker) | स्पीकर एम्बेडिंग | MLX, CoreML | 6.6M | भाषा-तटस्थ |
| [ReDimNet2-B6](https://huggingface.co/aufklarer/ReDimNet2-B6-CoreML) | नामित आवाज़ पहचान | CoreML | 12.3M | भाषा-तटस्थ |

## इंस्टॉलेशन

### Homebrew

नेटिव ARM Homebrew (`/opt/homebrew`) आवश्यक है। Rosetta/x86_64 Homebrew समर्थित नहीं है।

```bash
brew install speech
```

फिर:

```bash
speech transcribe recording.wav
speech speak "Hello world"
speech translate "Hello, how are you?" --to es
speech respond --input question.wav --transcript
speech-server --port 8080            # स्थानीय HTTP / WebSocket सर्वर (OpenAI-compatible /v1/realtime + /v1/audio/transcriptions)
```

**[पूर्ण CLI संदर्भ →](https://soniqo.audio/hi/cli)**

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

केवल वही इम्पोर्ट करें जो आपको चाहिए — प्रत्येक मॉडल अपना SPM target है:

```swift
import Qwen3ASR             // स्पीच रिकग्निशन (MLX)
import WhisperASR           // Whisper Large-v3 Turbo (CoreML)
import ParakeetASR          // स्पीच रिकग्निशन (CoreML, बैच)
import ParakeetStreamingASR // पार्शियल्स + EOU के साथ स्ट्रीमिंग डिक्टेशन
import NemotronStreamingASR // बहुभाषी स्ट्रीमिंग ASR नेटिव विराम चिह्न के साथ (0.6B, 40 भाषाएँ)
import OmnilingualASR       // 1,672 भाषाएँ (CoreML + MLX)
import Qwen3TTS             // टेक्स्ट-टू-स्पीच
import CosyVoiceTTS         // वॉयस क्लोनिंग के साथ TTS
import VoxCPM2TTS           // 48 kHz TTS, वॉयस क्लोनिंग + वॉयस डिज़ाइन (2B)
import IndexTTS2TTS         // Native MLX voice cloning from reference audio
import F5TTS                // Zero-shot voice cloning (DiT flow matching + Vocos)
import HiggsTTS             // Conversational TTS + cloning (Qwen3 backbone, control tags)
import KokoroTTS            // टेक्स्ट-टू-स्पीच (iOS-ready)
import VibeVoiceTTS         // लंबे-रूप / बहु-वक्ता TTS (EN/ZH)
import MagpieTTS            // बहुभाषी TTS (NVIDIA Magpie 357M, MLX, 9 भाषाएँ)
import MagpieTTSCoreML      // Magpie CoreML बैकएंड (CoreML + MLX हाइब्रिड, 8 भाषाएँ)
import FishAudioTTS         // वॉयस क्लोनिंग वाला प्रयोगात्मक Fish Audio S2 Pro runtime
import Qwen3Chat            // ऑन-डिवाइस LLM चैट
import FunctionGemma    // ऑन-डिवाइस टूल-कॉल LLM
import MADLADTranslation    // 400+ भाषाओं में बहु-दिशात्मक अनुवाद
import HibikiTranslate      // स्ट्रीमिंग स्पीच-टू-स्पीच अनुवाद (FR/ES/PT/DE → EN)
import PersonaPlex          // फुल-डुप्लेक्स स्पीच-टू-स्पीच
import SpeechVAD            // VAD + स्पीकर डायराइज़ेशन + एम्बेडिंग
import SpeechEnhancement    // नॉइज़ सप्रेशन
import SpeechRestoration    // स्पीच रिस्टोरेशन — नॉइज़ हटाना + डीरीवर्ब (Sidon, CoreML, 48 kHz)
import SourceSeparation     // म्यूज़िक सोर्स सेपरेशन (Open-Unmix, 4 स्टेम)
import MAGNeTMusicGen      // टेक्स्ट से संगीत निर्माण (30 सेकंड, 32 kHz)
import FlashSR             // ऑडियो सुपर-रेज़ोल्यूशन (48 kHz, 1-स्टेप डिफ्यूज़न)
import SpeechUI             // स्ट्रीमिंग ट्रांसक्रिप्ट के लिए SwiftUI कॉम्पोनेंट
import AudioCommon          // शेयर्ड प्रोटोकॉल और यूटिलिटीज़
```

### आवश्यकताएँ

- Swift 6+, Xcode 16+ (Metal Toolchain के साथ)
- macOS 15+ (Sequoia) या iOS 18+, Apple Silicon (M1/M2/M3/M4)

macOS 15 / iOS 18 न्यूनतम आवश्यकता [MLState](https://developer.apple.com/documentation/coreml/mlstate) से आती है —— Apple की परसिस्टेंट ANE स्टेट API —— जिसका उपयोग CoreML पाइपलाइन (Qwen3-ASR, Qwen3-Chat, Qwen3-TTS) टोकन स्टेप्स के बीच KV कैश को Neural Engine पर रखने के लिए करती हैं।

### सोर्स से बिल्ड

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

`make build` Swift पैकेज **और** MLX Metal shader library दोनों को कंपाइल करता है। GPU इन्फ़रेंस के लिए Metal library आवश्यक है — इसके बिना आपको रनटाइम पर `Failed to load the default metallib` दिखेगा। `make debug` डीबग बिल्ड के लिए, `make test` टेस्ट सूट के लिए।

**[पूर्ण बिल्ड और इंस्टॉल गाइड →](https://soniqo.audio/hi/getting-started)**

## डेमो ऐप्स

- **[DictateDemo](Examples/DictateDemo/)** ([डॉक्स](https://soniqo.audio/hi/guides/dictate)) — macOS मेनू-बार स्ट्रीमिंग डिक्टेशन, लाइव पार्शियल्स, VAD-संचालित एंड-ऑफ-अटरन्स डिटेक्शन, और वन-क्लिक कॉपी। बैकग्राउंड एजेंट के रूप में चलता है (Parakeet-EOU-120M + Silero VAD)।
- **[iOSEchoDemo](Examples/iOSEchoDemo/)** — iOS इको डेमो (Parakeet ASR + Kokoro TTS)। डिवाइस और सिम्युलेटर।
- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — माइक इनपुट, VAD, और मल्टी-टर्न संदर्भ के साथ संवादात्मक वॉयस असिस्टेंट। macOS। M2 Max पर RTF ~0.94 (रियल-टाइम से तेज़)।
- **[SpeechDemo](Examples/SpeechDemo/)** — टैब्ड इंटरफ़ेस में डिक्टेशन और TTS सिंथेसिस। macOS।

प्रत्येक डेमो के README में बिल्ड निर्देश हैं।

## कोड उदाहरण

नीचे दिए गए स्निपेट्स प्रत्येक डोमेन के लिए न्यूनतम पथ दिखाते हैं। प्रत्येक अनुभाग [soniqo.audio](https://soniqo.audio/hi) पर पूर्ण गाइड से लिंक होता है जिसमें कॉन्फ़िगरेशन विकल्प, कई बैकएंड, स्ट्रीमिंग पैटर्न, और CLI रेसिपी शामिल हैं।

### स्पीच-टू-टेक्स्ट — [पूर्ण गाइड →](https://soniqo.audio/hi/guides/transcribe)

```swift
import Qwen3ASR

let model = try await Qwen3ASRModel.fromPretrained()
let text = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

वैकल्पिक बैकएंड: [WhisperASR](docs/inference/whisper-asr-inference.md) (Whisper Large-v3 Turbo, native CoreML), [Parakeet TDT](https://soniqo.audio/hi/guides/parakeet) (CoreML, 32× रियल-टाइम), [Omnilingual ASR](https://soniqo.audio/hi/guides/omnilingual) (1,672 भाषाएँ, CoreML या MLX), [स्ट्रीमिंग डिक्टेशन](https://soniqo.audio/hi/guides/dictate) (लाइव पार्शियल्स)।

### फ़ोर्स्ड अलाइनमेंट — [पूर्ण गाइड →](https://soniqo.audio/hi/guides/align)

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

### टेक्स्ट-टू-स्पीच — [पूर्ण गाइड →](https://soniqo.audio/hi/guides/speak)

```swift
import Qwen3TTS
import AudioCommon

let model = try await Qwen3TTSModel.fromPretrained()
let audio = model.synthesize(text: "Hello world", language: "english")
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

वैकल्पिक TTS इंजन: [CosyVoice3](https://soniqo.audio/hi/guides/cosyvoice) (स्ट्रीमिंग + वॉयस क्लोनिंग + इमोशन टैग), [Kokoro-82M](https://soniqo.audio/hi/guides/kokoro) (iOS-ready, 54 वॉयस), [VibeVoice](https://soniqo.audio/hi/guides/vibevoice) (लंबे-रूप पॉडकास्ट / बहु-वक्ता, EN/ZH), [Fish Audio S2 Pro](docs/inference/fish-audio-s2-pro.md) (प्रयोगात्मक ज़ीरो-शॉट क्लोनिंग + ब्रैकेट स्टाइल मार्कर), [वॉयस क्लोनिंग](https://soniqo.audio/hi/guides/voice-cloning)।

### स्पीच-टू-स्पीच — [पूर्ण गाइड →](https://soniqo.audio/hi/guides/respond)

```swift
import PersonaPlex

let model = try await PersonaPlexModel.fromPretrained()
let responseAudio = model.respond(userAudio: userSamples)
// 24 kHz मोनो Float32 आउटपुट — प्लेबैक के लिए तैयार
```

### LLM चैट — [पूर्ण गाइड →](https://soniqo.audio/hi/guides/chat)

```swift
import Qwen3Chat
import FunctionGemma

let chat = try await Qwen35MLXChat.fromPretrained()
chat.chat(messages: [(.user, "Explain MLX in one sentence")]) { token, isFinal in
    print(token, terminator: "")
}
```

### अनुवाद — [पूरी गाइड →](https://soniqo.audio/hi/guides/translate)

```swift
import MADLADTranslation

let translator = try await MADLADTranslator.fromPretrained()
let es = try translator.translate("Hello, how are you?", to: "es")
// → "Hola, ¿cómo estás?"
```

### स्पीच अनुवाद — [पूरी गाइड →](https://soniqo.audio/guides/audio-translate)

```swift
import HibikiTranslate
import AudioCommon

let model = try await HibikiTranslateModel.fromPretrained()
let pcm = try AudioFileLoader.load(url: input, targetSampleRate: 24000)
let (englishAudio, textTokens) = model.translate(
    sourceAudio: pcm, sourceLanguage: .fr
)
// Hibiki Zero-3B — FR/ES/PT/DE → EN, ऑन-डिवाइस, स्ट्रीमिंग Mimi कोडेक
```

### वॉयस एक्टिविटी डिटेक्शन — [पूर्ण गाइड →](https://soniqo.audio/hi/guides/vad)

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
for s in segments { print("\(s.startTime)s → \(s.endTime)s") }
```

### स्पीकर डायराइज़ेशन — [पूर्ण गाइड →](https://soniqo.audio/hi/guides/diarize)

```swift
import SpeechVAD

let diarizer = try await DiarizationPipeline.fromPretrained()
let segments = diarizer.diarize(audio: samples, sampleRate: 16000)
for s in segments { print("Speaker \(s.speakerId): \(s.startTime)s - \(s.endTime)s") }
```

### स्पीच एन्हांसमेंट — [पूर्ण गाइड →](https://soniqo.audio/hi/guides/denoise)

```swift
import SpeechEnhancement

let denoiser = try await DeepFilterNet3Model.fromPretrained()
let clean = try denoiser.enhance(audio: noisySamples, sampleRate: 48000)
```

### स्पीच रिस्टोरेशन — [पूर्ण गाइड →](https://soniqo.audio/hi/guides/restore)

[Sidon](https://arxiv.org/abs/2509.17052) (w2v-BERT 2.0 प्रेडिक्टर + DAC वोकोडर, Core ML) के साथ संयुक्त नॉइज़ हटाना **और** डीरीवर्ब। एक सामान्य नॉइज़ सप्रेसर के विपरीत, Sidon को स्पीकर की पहचान संरक्षित रखने के लिए ट्रेन किया गया है, इसलिए यह TTS से पहले किसी शोरयुक्त या रीवर्बरेंट वॉयस-क्लोनिंग रेफ़रेंस को साफ़ करने के लिए उपयुक्त है। इनपुट 16 kHz है; आउटपुट 48 kHz मोनो है।

```swift
import SpeechRestoration

let restorer = try await SpeechRestorer.fromPretrained()          // .fp16 (default) or .int8
let clean = try restorer.restore(audio: noisySamples, sampleRate: 16000)  // → 48 kHz
```

CLI से:

```bash
speech restore noisy.wav -o clean.wav            # denoise + dereverb, 48 kHz output
speech restore noisy.wav --variant int8          # smaller, lower peak RAM

# Clean a voice-cloning reference before TTS (opt-in; preserves speaker identity):
speech speak "Hello" --engine voxcpm2 --voice-sample ref.wav --clean-reference
```

### वॉयस पाइपलाइन (ASR → LLM → TTS) — [पूर्ण गाइड →](https://soniqo.audio/hi/voice-agents)

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

`VoicePipeline` रियल-टाइम वॉयस-एजेंट स्टेट मशीन है ([speech-core](https://github.com/soniqo/speech-core) द्वारा संचालित) जो VAD-संचालित टर्न डिटेक्शन, इंटरप्शन हैंडलिंग, और eager STT के साथ आता है। यह किसी भी `SpeechRecognitionModel` + `SpeechGenerationModel` + `StreamingVADProvider` को कनेक्ट करता है।

### HTTP API सर्वर

```bash
speech-server --port 8080
```

सभी मॉडलों को HTTP REST + WebSocket endpoints के माध्यम से एक्सपोज़ करता है, जिसमें OpenAI-संगत APIs शामिल हैं: `/v1/realtime` पर Realtime WebSocket और `/v1/audio/transcriptions` पर ट्रांसक्रिप्शन REST endpoint। देखें [`Sources/AudioServer/`](Sources/AudioServer/)।

## आर्किटेक्चर

speech-swift प्रति मॉडल एक SPM टारगेट में विभाजित है ताकि उपभोक्ता केवल उसी के लिए भुगतान करें जो वे इम्पोर्ट करते हैं। साझा इन्फ़्रास्ट्रक्चर `AudioCommon` (प्रोटोकॉल, ऑडियो I/O, HuggingFace डाउनलोडर, `SentencePieceModel`) और `MLXCommon` (वेट लोडिंग, `QuantizedLinear` हेल्पर्स, multi-head attention के लिए `SDPA` हेल्पर) में रहता है।

**[बैकएंड, मेमोरी टेबल्स, और मॉड्यूल मैप के साथ पूर्ण आर्किटेक्चर डायग्राम → soniqo.audio/architecture](https://soniqo.audio/hi/architecture)** · **[API संदर्भ → soniqo.audio/api](https://soniqo.audio/hi/api)** · **[बेंचमार्क → soniqo.audio/benchmarks](https://soniqo.audio/hi/benchmarks)**

स्थानीय डॉक्स (रिपॉज़िटरी):
- **मॉडल:** [Qwen3-ASR](docs/models/asr-model.md) · [WhisperASR](docs/models/whisper-asr.md) · [Qwen3-TTS](docs/models/tts-model.md) · [CosyVoice](docs/models/cosyvoice-tts.md) · [Kokoro](docs/models/kokoro-tts.md) · [VoxCPM2](docs/models/voxcpm2-tts.md) · [IndexTTS2](docs/models/indextts2.md) · [F5-TTS](docs/models/f5-tts.md) · [Higgs TTS 3](docs/models/higgs-tts.md) · [VibeVoice](docs/models/vibevoice.md) · [Supertonic](docs/models/supertonic-tts.md) · [Chatterbox](docs/models/chatterbox-tts.md) · [Indic-Mio](docs/models/indic-mio-tts.md) · [Fish Audio S2 Pro](docs/models/fish-audio-s2-pro.md) · [Magpie TTS](docs/models/magpie-tts.md) · [Parakeet TDT](docs/models/parakeet-asr.md) · [Parakeet Streaming](docs/models/parakeet-streaming-asr.md) · [Nemotron Streaming](docs/models/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/models/omnilingual-asr.md) · [PersonaPlex](docs/models/personaplex.md) · [Hibiki](docs/models/hibiki.md) · [MADLAD-400](docs/models/madlad-translation.md) · [FunctionGemma](docs/models/function-gemma.md) · [Qwen3.5 Chat](docs/models/qwen35-chat.md) · [Gemma 4 Chat](docs/models/gemma4-chat.md) · [Qwen3 Dense Chat](docs/models/qwen3-dense-chat.md) · [FireRedVAD](docs/models/fireredvad.md) · [KWS Zipformer](docs/models/kws-zipformer.md) · [Sidon](docs/models/sidon.md) · [Source Separation](docs/models/source-separation.md) · [HTDemucs](docs/models/htdemucs.md) · [MAGNeT](docs/models/magnet-music-gen.md) · [Stable Audio 3](docs/models/stable-audio-3.md) · [FlashSR](docs/models/flashsr.md) · [Audio2Face-3D](docs/models/audio2face3d.md)
- **इन्फ़रेंस:** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md) · [WhisperASR](docs/inference/whisper-asr-inference.md) · [Parakeet TDT](docs/inference/parakeet-asr-inference.md) · [Parakeet Streaming](docs/inference/parakeet-streaming-asr-inference.md) · [Nemotron Streaming](docs/inference/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/inference/omnilingual-asr-inference.md) · [TTS](docs/inference/qwen3-tts-inference.md) · [VoxCPM2](docs/inference/voxcpm2-inference.md) · [IndexTTS2](docs/inference/indextts2.md) · [F5-TTS](docs/inference/f5-tts.md) · [Higgs TTS 3](docs/inference/higgs-tts.md) · [VibeVoice](docs/inference/vibevoice-inference.md) · [Fish Audio S2 Pro](docs/inference/fish-audio-s2-pro.md) · [Magpie TTS](docs/inference/magpie-tts.md) · [Hibiki](docs/inference/hibiki-inference.md) · [MADLAD-400](docs/inference/madlad-translation.md) · [MAGNeT](docs/inference/magnet-music-gen.md) · [Stable Audio 3](docs/inference/stable-audio-3.md) · [FlashSR](docs/inference/flashsr.md) · [Forced Aligner](docs/inference/forced-aligner.md) · [Silero VAD](docs/inference/silero-vad.md) · [FireRedVAD](docs/inference/fireredvad.md) · [Wake-word](docs/inference/wake-word.md) · [Speaker Diarization](docs/inference/speaker-diarization.md) · [Speech Enhancement](docs/inference/speech-enhancement.md) · [Sidon](docs/inference/sidon.md) · [Cache/offline](docs/inference/cache-and-offline.md)
- **संदर्भ:** [शेयर्ड प्रोटोकॉल](docs/shared-protocols.md)

## कैश कॉन्फ़िगरेशन

मॉडल वेट पहले उपयोग पर HuggingFace से डाउनलोड होते हैं और `~/Library/Caches/qwen3-speech/` में कैश होते हैं। `QWEN3_CACHE_DIR` (CLI) या `cacheDir:` (Swift API) से ओवरराइड करें। सभी `fromPretrained()` एंट्री पॉइंट `offlineMode: true` भी स्वीकार करते हैं ताकि वेट कैश होने पर नेटवर्क स्किप किया जा सके।

मुख्यभूमि चीन के उपयोगकर्ता (या जहाँ भी `huggingface.co` धीमा/अवरुद्ध है) `HF_ENDPOINT` सेट करके मिरर से डाउनलोड कर सकते हैं, जैसे `export HF_ENDPOINT=https://hf-mirror.com`।

सैंडबॉक्स्ड iOS कंटेनर पाथ सहित पूर्ण विवरण के लिए [`docs/inference/cache-and-offline.md`](docs/inference/cache-and-offline.md) देखें।

## MLX Metal library

यदि आपको रनटाइम पर `Failed to load the default metallib` दिखता है, तो Metal shader library गुम है। मैनुअल `swift build` के बाद `make build` या `./scripts/build_mlx_metallib.sh release` चलाएँ। यदि Metal Toolchain गुम है, तो पहले इसे इंस्टॉल करें:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## टेस्टिंग

```bash
make test                            # पूर्ण सुइट (यूनिट + मॉडल डाउनलोड के साथ E2E)
swift test --skip E2E                # केवल यूनिट (CI-सुरक्षित, कोई डाउनलोड नहीं)
swift test --filter Qwen3ASRTests    # विशिष्ट मॉड्यूल
```

E2E टेस्ट क्लासेस `E2E` उपसर्ग का उपयोग करती हैं ताकि CI उन्हें `--skip E2E` से फ़िल्टर कर सके। पूर्ण टेस्टिंग नियम के लिए [CLAUDE.md](CLAUDE.md#testing) देखें।

## योगदान

PRs का स्वागत है — बग फ़िक्स, नए मॉडल इंटीग्रेशन, डॉक्यूमेंटेशन। फ़ॉर्क करें, feature ब्रांच बनाएँ, `make build && make test`, `main` के विरुद्ध PR खोलें।

## लाइसेंस

Apache 2.0
