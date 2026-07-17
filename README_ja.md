# Speech Swift

Apple Silicon向けAI音声モデル。MLX SwiftとCoreMLで動作します。

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

Mac・iOS向けのオンデバイス音声認識・合成・理解。Apple Silicon上で完全にローカル動作します——クラウド不要、APIキー不要、データはデバイスから外部に送信されません。

**[📚 ドキュメント →](https://soniqo.audio/ja)** · **[🤗 HuggingFaceモデル](https://huggingface.co/aufklarer)** · **[📝 ブログ](https://blog.ivan.digital)**

<p align="center">
  <a href="https://formulae.brew.sh/formula/speech"><img src="https://img.shields.io/homebrew/installs/dm/speech.svg?logo=homebrew&amp;label=Homebrew%20installs&amp;color=FBB040" alt="Homebrew installs"></a>
  <a href="https://github.com/soniqo/speech-swift#built-with-speech-swift"><img src="https://img.shields.io/badge/verified%20public%20repositories-29-2ea44f?logo=github" alt="Verified public repositories: 29"></a>
</p>

<p align="center">
  <a href="https://trendshift.io/repositories/24196?utm_source=trendshift-badge&amp;utm_medium=badge&amp;utm_campaign=badge-trendshift-24196" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/trendshift/repositories/24196/daily?language=Swift" alt="soniqo%2Fspeech-swift | Trendshift" width="250" height="55"/></a>
</p>

<p align="center">
  <a href="https://youtu.be/x9zgcaW0gUk">
    <img src="https://img.youtube.com/vi/x9zgcaW0gUk/maxresdefault.jpg" width="640" alt="MacBook で動くローカル音声 AI ―― YouTube で 4 分間のオープンソースライブラリツアーを視聴">
  </a>
</p>
<p align="center"><em>MacBook で動くローカル音声 AI ―― YouTube で 4 分間のオープンソースライブラリツアーを視聴</em></p>

**ユースケース：** [音声エージェント](https://soniqo.audio/ja/voice-agents) · [文字起こし](https://soniqo.audio/ja/transcription) · [音声合成](https://soniqo.audio/ja/speech-generation)

## Speech Swift で構築

Speech Swift パッケージへの参照を公開ソースで確認できる 29 個のリポジトリです。

[Palmier Pro](https://github.com/palmier-io/palmier-pro) · [Anarlog](https://github.com/fastrepl/anarlog) · [ClawdHome](https://github.com/ThinkInAIXYZ/clawdhome) · [Jabber](https://github.com/rselbach/jabber) · [Ora](https://github.com/wuwangzhang1216/ora) · [VoxFlow](https://github.com/xingbofeng/VoxFlow) · [LokalBot](https://github.com/stevyhacker/lokalbot) · [Voicey](https://github.com/jonathanKingston/voicey) · [HushType](https://github.com/felixfu824/HushType) · [DexDictate macOS](https://github.com/westkitty/DexDictate_MacOS) · [Watchtower](https://github.com/aiwatchtowers/watchtower) · [Wishper App](https://github.com/irangareddy/wishper-app) · [FriSpeak](https://github.com/KSubedi/FriSpeak) · [Scribe](https://github.com/itchat/Scribe) · [VoicePen](https://github.com/dot-sk/VoicePen) · [Anything Voice](https://github.com/jakemaly/anything-voice) · [Conversational MLX](https://github.com/ottokafka/conversational_mlx) · [HachiSpeak](https://github.com/sarinali/hachispeak) · [JustTalk](https://github.com/d0zingcat/JustTalk) · [Kioku](https://github.com/matthewmorrone/Kioku) · [Luxicon](https://github.com/DavidsonCollege/luxicon) · [Mako](https://github.com/bn-l/mako) · [Meeting Emo Transcriber](https://github.com/kouko/meeting-emo-transcriber) · [MeetingSummary](https://github.com/a9650615/MeetingSummary) · [Stenograf](https://github.com/ivan-digital/stenograf) · [Toast](https://github.com/drbh/toast) · [TxtVoiceApp](https://github.com/2mauis/TxtVoiceApp) · [video_to_srt](https://github.com/dogacan/video_to_srt) · [Warmth iOS](https://github.com/molyleelatham/gtmhackathon)

**機能グループ：** STT / ASR · アラインメント · TTS · LLM と翻訳 · Speech-to-Speech · 強化 / 復元 · 音源分離 · 音楽 / オーディオ生成 · ウェイクワード、VAD、話者分離、話者識別

**STT / ASR**

- **[Qwen3-ASR](https://soniqo.audio/ja/guides/transcribe)** — 音声認識（自動音声認識、52言語、MLX + CoreML）
- **[WhisperASR](docs/models/whisper-asr.md)** — Whisper Large-v3 Turbo speech-to-text via native CoreML runtime (ANE, multilingual)
- **[Parakeet TDT](https://soniqo.audio/ja/guides/parakeet)** — CoreMLによる音声認識（Neural Engine、NVIDIA FastConformer + TDTデコーダー、25言語）
- **[Omnilingual ASR](https://soniqo.audio/ja/guides/omnilingual)** — 音声認識（Meta wav2vec2 + CTC、**1,672言語**、32文字体系、CoreML 300M + MLX 300M/1B/3B/7B）
- **[ストリーミングディクテーション](https://soniqo.audio/ja/guides/dictate)** — 部分結果と発話終端検出付きのリアルタイムディクテーション（Parakeet-EOU-120M）
- **[Nemotron ストリーミング (多言語)](https://soniqo.audio/ja/guides/nemotron)** — ネイティブな句読点と大文字化を備えた低レイテンシストリーミングASR（NVIDIA Nemotron-3.5-ASR-Streaming-0.6B、CoreML + MLX、**40言語ロケール**）
- **[Nemotron ストリーミング (英語)](https://soniqo.audio/guides/nemotron)** — ネイティブな句読点と大文字化を備えた低レイテンシストリーミングASR （NVIDIA Nemotron-Speech-Streaming-0.6B、CoreML、英語のみ、多言語版より小型・高速）

**アラインメント**

- **[Qwen3-ForcedAligner](https://soniqo.audio/ja/guides/align)** — 単語レベルのタイムスタンプ整列（音声 + テキスト → タイムスタンプ）

**TTS / 音声生成**

- **[Qwen3-TTS](https://soniqo.audio/ja/guides/speak)** — 音声合成（最高品質、ストリーミング、カスタムスピーカー、10言語）
- **[CosyVoice TTS](https://soniqo.audio/ja/guides/cosyvoice)** — 音声クローン、マルチスピーカー対話、感情タグを備えたストリーミングTTS（9言語）
- **[VoxCPM2](https://soniqo.audio/ja/speech-generation)** — 48 kHz スタジオ品質の TTS。音声クローンと指示ベースのボイスデザインに対応（2B、MLX bf16/int8、30言語）
- **[IndexTTS2](docs/models/indextts2.md)** — Native MLX voice cloning from a reference voice (IndexTeam IndexTTS-2, 1.5B-class fp16 bundle, speaker/emotion/pause controls)
- **[F5-TTS](docs/models/f5-tts.md)** — Zero-shot voice cloning from a short reference clip + transcript (SWivid F5-TTS v1 Base, DiT flow matching + Vocos, MLX fp16, 24 kHz, English + Mandarin; non-commercial license)
- **[Higgs TTS 3](docs/models/higgs-tts.md)** — Conversational TTS with zero-shot voice cloning and inline emotion/style/SFX/prosody tags (Boson Higgs TTS 3, Qwen3-4B backbone, MLX bf16, 24 kHz, 100+ languages; research/non-commercial license)
- **[Kokoro TTS](https://soniqo.audio/ja/guides/kokoro)** — オンデバイスTTS（82M、CoreML/Neural Engine、54ボイス、iOS対応、10言語）
- **[VibeVoice TTS](https://soniqo.audio/ja/guides/vibevoice)** — 長尺・マルチスピーカーTTS（Microsoft VibeVoice Realtime-0.5B + 1.5B、MLX、最長90分のポッドキャスト／オーディオブック生成、EN/ZH）
- **[Magpie TTS](https://soniqo.audio/ja/guides/magpie)** — 多言語 TTS（NVIDIA Magpie-TTS Multilingual 357M、MLX INT8 411 MB または CoreML INT8 342 MB、9 言語、5 つの組み込みスピーカー、MLX でストリーミング）
- **[Supertonic TTS](https://soniqo.audio/guides/supertonic)** — オンデバイスのフローマッチングTTS（Supertone Supertonic-3 99M、CoreML/Neural Engine、31言語、10ボイス、G2P-free、44.1 kHz）
- **[Chatterbox TTS](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16)** — ゼロショット音声クローン対応の多言語TTS（Resemble AI Chatterbox Multilingual、MLX fp16 ~1.3 GB、23ランタイム言語、ヘブライ語はニクダーが必要、MIT）
- **[OmniVoice TTS](https://huggingface.co/aufklarer/OmniVoice-MLX-fp16)** — ゼロショット音声クローンに対応した非自己回帰拡散TTS（k2-fsa OmniVoice、Qwen3バックボーン、MLX fp16デフォルト / int8利用可、600+言語、Apache-2.0）
- **[Indic-Mio](docs/models/indic-mio-tts.md)** — Hindi/Indic TTS with inline emotion markers and optional reference-voice cloning (MLX, 24 kHz)

**LLM と翻訳**

- **[Qwen3Chat](https://soniqo.audio/ja/guides/chat)** — オンデバイスLLMチャット（Qwen3.5-0.8BのMLX/CoreMLに加え、dense Qwen3 4BとGemma 4 E2B/E4BのMLXバックエンド、ストリーミングトークン）
- **[FunctionGemma](https://soniqo.audio/ja/guides/function-calls)** — オンデバイスの構造化された関数 / ツール呼び出し用 LLM（Gemma 3 270M、CoreML 8-bit パレタイズ、Neural Engine、約 252 tok/s）
- **[MADLAD-400](https://soniqo.audio/ja/guides/translate)** — 400+言語間の多対多翻訳（3B、MLX INT4 + INT8、T5 v1.1、Apache 2.0）

**Speech-to-Speech と音声エージェント**

- **[Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate)** — ストリーミング音声間翻訳（FR/ES/PT/DE → EN、MLX INT4 + INT8、Kyutai Moshi/Mimi スタック、CC-BY-4.0）
- **[PersonaPlex](https://soniqo.audio/ja/guides/respond)** — 全二重音声間会話（7B、音声入力 → 音声出力、18種類のボイスプリセット）
- **[Audio2Face-3D](docs/models/audio2face3d.md)** — 音声駆動のアバター表情アニメーション（NVIDIA Audio2Face-3D v2.3 Mark、顔係数 301 次元、MLX）

**強化、分離、オーディオ生成**

- **[DeepFilterNet3](https://soniqo.audio/ja/guides/denoise)** — リアルタイムノイズ抑制（2.1Mパラメーター、48 kHz）。60 s のシングルショット上限を超える長尺音声は crossfade 付きで自動チャンク化 — `enhanceChunked(...)` を参照
- **[音源分離](https://soniqo.audio/ja/guides/separate)** — HTDemucs (Demucs v4) + Open-Unmix による音楽音源分離（UMX-HQ / UMX-L、4 ステム：ボーカル／ドラム／ベース／その他、44.1 kHz ステレオ）
- **[MAGNeT](https://soniqo.audio/ja/guides/compose)** — テキスト→音楽生成（Meta MAGNeT Small 300M / Medium 1.5B、MLX INT8、30 秒 32 kHz モノラル、マスク並列デコーディング）
- **[Stable Audio 3](docs/models/stable-audio-3.md)** — Text-to-audio/music generation (Stable Audio 3 Medium, MLX INT8/INT4, 44.1 kHz stereo, variable length)
- **[FlashSR](https://soniqo.audio/ja/guides/upsample)** — オーディオ超解像（FlashSR ICASSP 2025、MLX、48 kHz モノラル、1ステップ蒸留拡散、INT4 363 MB / INT8 720 MB）

**ターン検出、話者分離、話者識別**

- **[ウェイクワード](https://soniqo.audio/ja/guides/wake-word)** — オンデバイスのキーワード検出（KWS Zipformer 3M、CoreML、リアルタイムの26倍、キーワードリスト設定可能）
- **[VAD](https://soniqo.audio/ja/guides/vad)** — 音声区間検出（Sileroストリーミング、Pyannoteオフライン、FireRedVAD 100以上の言語）
- **[話者ダイアライゼーション](https://soniqo.audio/ja/guides/diarize)** — 誰がいつ話したか（Pyannoteパイプライン、Neural Engine上のエンドツーエンドSortformer） — インクリメンタルなストリーミングセッションに対応（話者 ID は安定、480 ミリ秒ごとに更新）
- **[話者埋め込み](https://soniqo.audio/ja/guides/embed-speaker)** — WeSpeaker ResNet34（256次元）、ReDimNet2-B6による名前付き音声識別（192次元）、CAM++（192次元）

論文：[Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba) · [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba) · [Omnilingual ASR](https://arxiv.org/abs/2511.09690) (Meta) · [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA) · [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba) · [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2) · [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA) · [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai) · [Hibiki](https://arxiv.org/abs/2502.03382) (Kyutai) · [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## ニュース

- **2026年4月19日** — [Apple SiliconにおけるMLXとCoreML — 正しいバックエンドを選ぶための実践ガイド](https://blog.ivan.digital/mlx-vs-coreml-on-apple-silicon-a-practical-guide-to-picking-the-right-backend-and-why-you-should-f77ddea7b27a)
- **2026年3月20日** — [600MモデルだけでMac上でWhisper Large v3を超えた](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **2026年2月26日** — [Apple Silicon上の話者ダイアライゼーションと音声区間検出 — ネイティブSwift + MLX](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **2026年2月23日** — [Apple Silicon上のNVIDIA PersonaPlex 7B — ネイティブSwift + MLXによる全二重音声間変換](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **2026年2月12日** — [Qwen3-ASR Swift: Apple Silicon向けオンデバイスASR + TTS — アーキテクチャとベンチマーク](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## クイックスタート

`Package.swift` にパッケージを追加します：

```swift
.package(url: "https://github.com/soniqo/speech-swift", branch: "main")
```

必要なモジュールだけをインポートします。各モデルは個別のSPMライブラリなので、使わないものにコストを払う必要はありません：

```swift
.product(name: "ParakeetStreamingASR", package: "speech-swift"),
.product(name: "SpeechUI",             package: "speech-swift"),  // オプションのSwiftUIビュー
```

**3行で音声バッファを文字起こし：**

```swift
import ParakeetStreamingASR

let model = try await ParakeetStreamingASRModel.fromPretrained()
let text = try model.transcribeAudio(audioSamples, sampleRate: 16000)
```

**部分結果付きのライブストリーミング：**

```swift
for await partial in model.transcribeStream(audio: samples, sampleRate: 16000) {
    print(partial.isFinal ? "FINAL: \(partial.text)" : "... \(partial.text)")
}
```

**約10行のSwiftUIディクテーションビュー：**

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

`SpeechUI` には `TranscriptionView`（確定 + 部分）と `TranscriptionStore`（ストリーミングASRアダプター）のみが含まれます。音声の可視化や再生には AVFoundation をお使いください。

利用可能なSPMプロダクト：`Qwen3ASR`, `WhisperASR`, `Qwen3TTS`, `Qwen3TTSCoreML`, `ParakeetASR`, `ParakeetStreamingASR`, `NemotronStreamingASR`, `OmnilingualASR`, `KokoroTTS`, `SupertonicTTS`, `VibeVoiceTTS`, `CosyVoiceTTS`, `VoxCPM2TTS`, `IndexTTS2TTS`, `F5TTS`, `HiggsTTS`, `ChatterboxTTS`, `OmniVoiceTTS`, `IndicMioTTS`, `FishAudioTTS`, `MagpieTTS`, `MagpieTTSCoreML`, `MAGNeTMusicGen`, `StableAudio3MusicGen`, `FlashSR`, `PersonaPlex`, `Audio2Face3D`, `HibikiTranslate`, `MADLADTranslation`, `SpeechVAD`, `SpeechWakeWord`, `SpeechEnhancement`, `SpeechRestoration`, `SourceSeparation`, `Qwen3Chat`, `FunctionGemma`, `SpeechCore`, `SpeechUI`, `AudioCommon`。

## モデル

以下はコンパクト表示です。**[サイズ、量子化、ダウンロードURL、メモリ表を含む完全なモデルカタログ → soniqo.audio/architecture](https://soniqo.audio/ja/architecture)**。

| モデル | タスク | バックエンド | サイズ | 言語 |
|-------|------|----------|-------|-----------|
| [Qwen3-ASR](https://soniqo.audio/ja/guides/transcribe) | 音声 → テキスト | MLX、CoreML（ハイブリッド） | 0.6B、1.7B | 52 |
| [WhisperASR](docs/models/whisper-asr.md) | Speech → Text | CoreML (ANE) | Large-v3 Turbo | Multi |
| [Parakeet TDT](https://soniqo.audio/ja/guides/parakeet) | 音声 → テキスト | CoreML (ANE) | 0.6B | 25欧州言語 |
| [Parakeet EOU](https://soniqo.audio/ja/guides/dictate) | 音声 → テキスト（ストリーミング） | CoreML (ANE) | 120M | 25欧州言語 |
| [Nemotron Streaming (多言語)](https://soniqo.audio/ja/guides/nemotron) | 音声 → テキスト（ストリーミング、句読点付き） | CoreML (ANE), MLX | 0.6B | **40** |
| [Nemotron Streaming (英語)](https://soniqo.audio/guides/nemotron) | 音声 → テキスト（ストリーミング、句読点付き） | CoreML (ANE) | 0.6B | EN |
| [Omnilingual ASR](https://soniqo.audio/ja/guides/omnilingual) | 音声 → テキスト | CoreML (ANE)、MLX | 300M / 1B / 3B / 7B | **[1,672](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)** |
| [Qwen3-ForcedAligner](https://soniqo.audio/ja/guides/align) | 音声 + テキスト → タイムスタンプ | MLX、CoreML | 0.6B | 多言語 |
| [Qwen3-TTS](https://soniqo.audio/ja/guides/speak) | テキスト → 音声 | MLX、CoreML | 0.6B、1.7B | 10 |
| [CosyVoice3](https://soniqo.audio/ja/guides/cosyvoice) | テキスト → 音声 | MLX | 0.5B | 9 |
| [VoxCPM2](https://soniqo.audio/ja/speech-generation) | テキスト → 音声 (48 kHz、ボイスデザイン + クローン) | MLX | 2B (bf16/int8) | 30 |
| [IndexTTS2](docs/models/indextts2.md) | Text → Speech (zero-shot voice cloning) | MLX | 1.5B-class (fp16) | EN/ZH |
| [F5-TTS](docs/models/f5-tts.md) | Text → Speech (zero-shot voice cloning) | MLX | 336M (fp16) | EN/ZH |
| [Higgs TTS 3](docs/models/higgs-tts.md) | Text → Speech (conversational, zero-shot voice cloning) | MLX | 4B (bf16) | 100+ |
| [Kokoro-82M](https://soniqo.audio/ja/guides/kokoro) | テキスト → 音声 | CoreML (ANE) | 82M | 10 |
| [Supertonic-3](https://soniqo.audio/guides/supertonic) | テキスト → 音声（44.1 kHz、フローマッチング、G2P-free） | CoreML (ANE) | 99M | 31 |
| [VibeVoice Realtime-0.5B](https://soniqo.audio/ja/guides/vibevoice) | テキスト → 音声（長尺・マルチスピーカー） | MLX | 0.5B | EN/ZH |
| [VibeVoice 1.5B](https://soniqo.audio/ja/guides/vibevoice) | テキスト → 音声（最長90分のポッドキャスト） | MLX | 1.5B | EN/ZH |
| [Magpie-TTS Multilingual](https://soniqo.audio/ja/guides/magpie) | テキスト → 音声（5 つの組み込みスピーカー、ストリーミング） | MLX / CoreML | 357M (MLX INT8, CoreML INT8) | 9（CoreML は日本語を除く） |
| [Chatterbox Multilingual](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16) | テキスト → 音声（ゼロショットクローン） | MLX | 0.8B (fp16) | 23（HEはニクダー必須） |
| [OmniVoice](https://huggingface.co/aufklarer/OmniVoice-MLX-fp16) | テキスト → 音声（NAR 拡散、ゼロショットクローン） | MLX | 0.8B (fp16デフォルト / int8) | **600+** |
| [Indic-Mio](docs/models/indic-mio-tts.md) | Text → Speech (Hindi/Indic, emotion tags, voice cloning) | MLX | fp16 | Hindi / Indic |
| [Fish Audio S2 Pro](docs/models/fish-audio-s2-pro.md) | テキスト → 音声（ゼロショットクローン、明示的スタイルマーカー） | MLX | 0.5B-class (fp16) | 多言語 |
| [Qwen3.5 Chat](docs/models/qwen35-chat.md) | Text → Text (LLM) | MLX, CoreML | 0.8B | Multi |
| [Qwen3 Dense Chat](docs/models/qwen3-dense-chat.md) | Text → Text (LLM) | MLX | 4B | Multi |
| [Gemma 4 Chat](docs/models/gemma4-chat.md) | Text → Text (LLM) | MLX | E2B / E4B (4-bit) | Multi |
| [FunctionGemma](https://soniqo.audio/ja/guides/function-calls) | テキスト → ツール呼び出し（LLM） | CoreML | 270M | 英語主体 |
| [MADLAD-400](https://soniqo.audio/ja/guides/translate) | テキスト → テキスト（翻訳） | MLX | 3B | **400+** |
| [Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate) | 音声 → 音声（翻訳） | MLX | 3B | FR/ES/PT/DE → EN |
| [PersonaPlex](https://soniqo.audio/ja/guides/respond) | 音声 → 音声 | MLX | 7B | EN |
| [Audio2Face-3D](docs/models/audio2face3d.md) | 音声 → 表情アニメーション | MLX | v2.3 Mark | 言語非依存 |
| [Silero VAD](https://soniqo.audio/ja/guides/vad) | 音声区間検出 | MLX、CoreML | 309K | 言語非依存 |
| [KWS Zipformer](docs/models/kws-zipformer.md) | Audio → Wake word | CoreML (ANE) | 3M | EN/custom keywords |
| [Pyannote](https://soniqo.audio/ja/guides/diarize) | VAD + ダイアライゼーション | MLX | 1.5M | 言語非依存 |
| [Pyannote Community-1](https://huggingface.co/aufklarer/Pyannote-Community-1-CoreML) | ダイアライゼーション + 話者embedding | CoreML (ANE) + Swift VBx | 8.35M | 言語非依存 |
| [Sortformer](https://huggingface.co/aufklarer/Sortformer-Diarization-CoreML) | [ダイアライゼーション（E2E）, インクリメンタルストリーミング](https://soniqo.audio/ja/guides/diarize) | CoreML (ANE) | 117M | 言語非依存 |
| [DeepFilterNet3](https://soniqo.audio/ja/guides/denoise) | 音声強調 | CoreML | 2.1M | 言語非依存 |
| [Sidon](https://soniqo.audio/ja/guides/restore) | 音声修復（ノイズ抑制 + 残響除去、48 kHz） | CoreML | w2v-BERT 2.0 + DAC (fp16/int8) | 言語非依存 |
| [HTDemucs (Demucs v4)](https://soniqo.audio/ja/guides/separate) | 音源分離 | MLX | 168M | Agnostic |
| [Open-Unmix](https://soniqo.audio/ja/guides/separate) | 音源分離 | MLX | 8.6M | Agnostic |
| [MAGNeT](https://soniqo.audio/ja/guides/compose) | テキスト → 音楽 (30 秒 @ 32 kHz) | MLX | 300M / 1.5B (int4/int8) | 英語プロンプト |
| [Stable Audio 3](docs/models/stable-audio-3.md) | Text → Music/audio (44.1 kHz stereo) | MLX | Medium 1.4B (int4/int8) | EN prompts |
| [FlashSR](https://soniqo.audio/ja/guides/upsample) | オーディオ超解像 (48 kHz) | MLX | 363 MB / 720 MB (int4/int8) | 言語非依存 |
| [WeSpeaker](https://soniqo.audio/ja/guides/embed-speaker) | 話者埋め込み | MLX、CoreML | 6.6M | 言語非依存 |
| [ReDimNet2-B6](https://huggingface.co/aufklarer/ReDimNet2-B6-CoreML) | 名前付き音声識別 | CoreML | 12.3M | 言語非依存 |

## インストール

### Homebrew

ネイティブARM Homebrew（`/opt/homebrew`）が必要です。Rosetta/x86_64 Homebrewはサポートされません。

```bash
brew install speech
```

その後：

```bash
speech transcribe recording.wav
speech speak "Hello world"
speech translate "Hello, how are you?" --to es
speech respond --input question.wav --transcript
speech-server --port 8080            # ローカル HTTP / WebSocket サーバー（OpenAI 互換 /v1/realtime + /v1/audio/transcriptions）
```

**[完全なCLIリファレンス →](https://soniqo.audio/ja/cli)**

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

必要なものだけをインポート。各モデルは個別のSPMターゲットです：

```swift
import Qwen3ASR             // 音声認識 (MLX)
import WhisperASR           // Whisper Large-v3 Turbo (CoreML)
import ParakeetASR          // 音声認識 (CoreML、バッチ)
import ParakeetStreamingASR // 部分結果 + EOU付きストリーミングディクテーション
import NemotronStreamingASR // 多言語ストリーミングASR、ネイティブ句読点付き（0.6B、40言語）
import OmnilingualASR       // 1,672言語 (CoreML + MLX)
import Qwen3TTS             // 音声合成
import CosyVoiceTTS         // 音声クローン付き音声合成
import VoxCPM2TTS           // 48 kHz TTS、音声クローン + ボイスデザイン (2B)
import IndexTTS2TTS         // Native MLX voice cloning from reference audio
import F5TTS                // Zero-shot voice cloning (DiT flow matching + Vocos)
import HiggsTTS             // Conversational TTS + cloning (Qwen3 backbone, control tags)
import KokoroTTS            // 音声合成 (iOS対応)
import VibeVoiceTTS         // 長尺・マルチスピーカーTTS（EN/ZH）
import MagpieTTS            // 多言語 TTS（NVIDIA Magpie 357M、MLX、9 言語）
import MagpieTTSCoreML      // Magpie CoreML バックエンド(CoreML + MLX のハイブリッド、8 言語)
import FishAudioTTS         // 音声クローン対応の実験的 Fish Audio S2 Pro ランタイム
import Qwen3Chat            // オンデバイスLLMチャット
import FunctionGemma    // オンデバイスの関数 / ツール呼び出しLLM
import MADLADTranslation    // 400+ 言語間の多対多翻訳
import HibikiTranslate      // ストリーミング音声間翻訳（FR/ES/PT/DE → EN）
import PersonaPlex          // 全二重音声間変換
import SpeechVAD            // VAD + 話者ダイアライゼーション + 埋め込み
import SpeechEnhancement    // ノイズ抑制
import SpeechRestoration    // 音声修復 — ノイズ抑制 + 残響除去（Sidon、CoreML、48 kHz）
import SourceSeparation     // 音楽音源分離（Open-Unmix、4 ステム）
import MAGNeTMusicGen      // テキスト→音楽生成（30 秒、32 kHz）
import FlashSR             // オーディオ超解像（48 kHz、1 ステップ拡散）
import SpeechUI             // ストリーミングトランスクリプト用SwiftUIコンポーネント
import AudioCommon          // 共有プロトコルとユーティリティ
```

### 動作要件

- Swift 6+、Xcode 16+（Metal Toolchainを含む）
- macOS 15+（Sequoia）または iOS 18+、Apple Silicon（M1/M2/M3/M4）

macOS 15 / iOS 18 の最小要件は [MLState](https://developer.apple.com/documentation/coreml/mlstate) —— Apple の永続的 ANE ステート API —— に由来します。CoreML パイプライン（Qwen3-ASR、Qwen3-Chat、Qwen3-TTS）は MLState を用いて、KV キャッシュをトークンステップ間で Neural Engine 上に常駐させます。

### ソースからビルド

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

`make build` はSwiftパッケージ**と** MLX Metalシェーダーライブラリを同時にコンパイルします。Metalライブラリは GPU 推論に必要です——これがないと実行時に `Failed to load the default metallib` が出ます。`make debug` でデバッグビルド、`make test` でテストスイートを実行します。

**[完全なビルド・インストールガイド →](https://soniqo.audio/ja/getting-started)**

## デモアプリ

- **[DictateDemo](Examples/DictateDemo/)**（[ドキュメント](https://soniqo.audio/ja/guides/dictate)）— macOSメニューバーのストリーミングディクテーション。ライブ部分結果、VADベースの発話終端検出、ワンクリックコピー付き。バックグラウンドagentとして動作（Parakeet-EOU-120M + Silero VAD）。
- **[iOSEchoDemo](Examples/iOSEchoDemo/)** — iOSエコーデモ（Parakeet ASR + Kokoro TTS）。実機・シミュレーター対応。
- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** — マイク入力、VAD、マルチターンコンテキスト付きの対話型音声アシスタント。macOS。M2 MaxでRTF約0.94（リアルタイムより高速）。
- **[SpeechDemo](Examples/SpeechDemo/)** — タブ形式インターフェース上でのディクテーションとTTS合成。macOS。

各デモのREADMEにビルド手順があります。

## コード例

以下のスニペットは、各領域の最小限の使い方を示しています。各セクションは [soniqo.audio](https://soniqo.audio/ja) 上の完全ガイドにリンクしており、設定オプション、複数のバックエンド、ストリーミングパターン、CLIレシピが載っています。

### 音声認識 — [完全ガイド →](https://soniqo.audio/ja/guides/transcribe)

```swift
import Qwen3ASR

let model = try await Qwen3ASRModel.fromPretrained()
let text = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

代替バックエンド：[WhisperASR](docs/inference/whisper-asr-inference.md)（Whisper Large-v3 Turbo、native CoreML）、[Parakeet TDT](https://soniqo.audio/ja/guides/parakeet)（CoreML、32×リアルタイム）、[Omnilingual ASR](https://soniqo.audio/ja/guides/omnilingual)（1,672言語、CoreMLまたはMLX）、[ストリーミングディクテーション](https://soniqo.audio/ja/guides/dictate)（ライブ部分結果）。

### 強制整列 — [完全ガイド →](https://soniqo.audio/ja/guides/align)

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

### 音声合成 — [完全ガイド →](https://soniqo.audio/ja/guides/speak)

```swift
import Qwen3TTS
import AudioCommon

let model = try await Qwen3TTSModel.fromPretrained()
let audio = model.synthesize(text: "Hello world", language: "english")
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

代替TTSエンジン：[CosyVoice3](https://soniqo.audio/ja/guides/cosyvoice)（ストリーミング + 音声クローン + 感情タグ）、[Kokoro-82M](https://soniqo.audio/ja/guides/kokoro)（iOS対応、54ボイス）、[VibeVoice](https://soniqo.audio/ja/guides/vibevoice)（長尺ポッドキャスト・マルチスピーカー、EN/ZH）、[Fish Audio S2 Pro](docs/inference/fish-audio-s2-pro.md)（実験的ゼロショットクローン + 角括弧スタイルマーカー）、[音声クローン](https://soniqo.audio/ja/guides/voice-cloning)。

### 音声間変換 — [完全ガイド →](https://soniqo.audio/ja/guides/respond)

```swift
import PersonaPlex

let model = try await PersonaPlexModel.fromPretrained()
let responseAudio = model.respond(userAudio: userSamples)
// 24 kHz モノラル Float32 出力、再生可能
```

### LLMチャット — [完全ガイド →](https://soniqo.audio/ja/guides/chat)

```swift
import Qwen3Chat
import FunctionGemma

let chat = try await Qwen35MLXChat.fromPretrained()
chat.chat(messages: [(.user, "Explain MLX in one sentence")]) { token, isFinal in
    print(token, terminator: "")
}
```

### 翻訳 — [完全ガイド →](https://soniqo.audio/ja/guides/translate)

```swift
import MADLADTranslation

let translator = try await MADLADTranslator.fromPretrained()
let es = try translator.translate("Hello, how are you?", to: "es")
// → "Hola, ¿cómo estás?"
```

### 音声翻訳 — [完全ガイド →](https://soniqo.audio/guides/audio-translate)

```swift
import HibikiTranslate
import AudioCommon

let model = try await HibikiTranslateModel.fromPretrained()
let pcm = try AudioFileLoader.load(url: input, targetSampleRate: 24000)
let (englishAudio, textTokens) = model.translate(
    sourceAudio: pcm, sourceLanguage: .fr
)
// Hibiki Zero-3B — FR/ES/PT/DE → EN、オンデバイス、ストリーミング Mimi コーデック
```

### 音声区間検出 — [完全ガイド →](https://soniqo.audio/ja/guides/vad)

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
for s in segments { print("\(s.startTime)s → \(s.endTime)s") }
```

### 話者ダイアライゼーション — [完全ガイド →](https://soniqo.audio/ja/guides/diarize)

```swift
import SpeechVAD

let diarizer = try await DiarizationPipeline.fromPretrained()
let segments = diarizer.diarize(audio: samples, sampleRate: 16000)
for s in segments { print("Speaker \(s.speakerId): \(s.startTime)s - \(s.endTime)s") }
```

### 音声強調 — [完全ガイド →](https://soniqo.audio/ja/guides/denoise)

```swift
import SpeechEnhancement

let denoiser = try await DeepFilterNet3Model.fromPretrained()
let clean = try denoiser.enhance(audio: noisySamples, sampleRate: 48000)
```

### 音声修復 — [完全ガイド →](https://soniqo.audio/ja/guides/restore)

[Sidon](https://arxiv.org/abs/2509.17052)（w2v-BERT 2.0 予測器 + DAC ボコーダー、Core ML）によるノイズ抑制**と**残響除去の同時処理。汎用のノイズサプレッサーと異なり、Sidon は話者の同一性を保持するよう訓練されているため、TTS の前にノイズや残響のある音声クローン用リファレンスをクリーンにする用途に適しています。入力は 16 kHz、出力は 48 kHz モノラルです。

```swift
import SpeechRestoration

let restorer = try await SpeechRestorer.fromPretrained()          // .fp16（デフォルト）または .int8
let clean = try restorer.restore(audio: noisySamples, sampleRate: 16000)  // → 48 kHz
```

CLI から：

```bash
speech restore noisy.wav -o clean.wav            # ノイズ抑制 + 残響除去、48 kHz 出力
speech restore noisy.wav --variant int8          # より小型、ピーク RAM 削減

# TTS の前に音声クローン用リファレンスをクリーンにする（オプトイン、話者の同一性を保持）：
speech speak "Hello" --engine voxcpm2 --voice-sample ref.wav --clean-reference
```

### 音声パイプライン（ASR → LLM → TTS）— [完全ガイド →](https://soniqo.audio/ja/voice-agents)

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

`VoicePipeline` はリアルタイム音声agentのステートマシンで（[speech-core](https://github.com/soniqo/speech-core)が駆動）、VADベースのターン検出、割り込み処理、イーガーSTTを備えています。任意の `SpeechRecognitionModel` + `SpeechGenerationModel` + `StreamingVADProvider` を接続できます。

### HTTP APIサーバー

```bash
speech-server --port 8080
```

HTTP REST + WebSocketエンドポイントですべてのモデルを公開します。OpenAI 互換 API として `/v1/realtime` の Realtime WebSocket と `/v1/audio/transcriptions` の文字起こし REST エンドポイントを含みます。[`Sources/AudioServer/`](Sources/AudioServer/) を参照してください。

## アーキテクチャ

speech-swift はモデルごとに1つのSPMターゲットに分割されており、利用者はインポートした分だけコストを負担します。共有インフラは `AudioCommon`（プロトコル、音声I/O、HuggingFaceダウンローダー、`SentencePieceModel`）と `MLXCommon`（ウェイトローディング、`QuantizedLinear` ヘルパー、`SDPA` マルチヘッドアテンションヘルパー）にあります。

**[バックエンド、メモリ表、モジュールマップ付きの完全なアーキテクチャ図 → soniqo.audio/architecture](https://soniqo.audio/ja/architecture)** · **[APIリファレンス → soniqo.audio/api](https://soniqo.audio/ja/api)** · **[ベンチマーク → soniqo.audio/benchmarks](https://soniqo.audio/ja/benchmarks)**

ローカルドキュメント（リポジトリ内）：
- **モデル：** [Qwen3-ASR](docs/models/asr-model.md) · [WhisperASR](docs/models/whisper-asr.md) · [Qwen3-TTS](docs/models/tts-model.md) · [CosyVoice](docs/models/cosyvoice-tts.md) · [Kokoro](docs/models/kokoro-tts.md) · [VoxCPM2](docs/models/voxcpm2-tts.md) · [IndexTTS2](docs/models/indextts2.md) · [F5-TTS](docs/models/f5-tts.md) · [Higgs TTS 3](docs/models/higgs-tts.md) · [VibeVoice](docs/models/vibevoice.md) · [Supertonic](docs/models/supertonic-tts.md) · [Chatterbox](docs/models/chatterbox-tts.md) · [Indic-Mio](docs/models/indic-mio-tts.md) · [Fish Audio S2 Pro](docs/models/fish-audio-s2-pro.md) · [Magpie TTS](docs/models/magpie-tts.md) · [Parakeet TDT](docs/models/parakeet-asr.md) · [Parakeet Streaming](docs/models/parakeet-streaming-asr.md) · [Nemotron Streaming](docs/models/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/models/omnilingual-asr.md) · [PersonaPlex](docs/models/personaplex.md) · [Hibiki](docs/models/hibiki.md) · [MADLAD-400](docs/models/madlad-translation.md) · [FunctionGemma](docs/models/function-gemma.md) · [Qwen3.5 Chat](docs/models/qwen35-chat.md) · [Gemma 4 Chat](docs/models/gemma4-chat.md) · [Qwen3 Dense Chat](docs/models/qwen3-dense-chat.md) · [FireRedVAD](docs/models/fireredvad.md) · [KWS Zipformer](docs/models/kws-zipformer.md) · [Sidon](docs/models/sidon.md) · [Source Separation](docs/models/source-separation.md) · [HTDemucs](docs/models/htdemucs.md) · [MAGNeT](docs/models/magnet-music-gen.md) · [Stable Audio 3](docs/models/stable-audio-3.md) · [FlashSR](docs/models/flashsr.md) · [Audio2Face-3D](docs/models/audio2face3d.md)
- **推論：** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md) · [WhisperASR](docs/inference/whisper-asr-inference.md) · [Parakeet TDT](docs/inference/parakeet-asr-inference.md) · [Parakeet Streaming](docs/inference/parakeet-streaming-asr-inference.md) · [Nemotron Streaming](docs/inference/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/inference/omnilingual-asr-inference.md) · [TTS](docs/inference/qwen3-tts-inference.md) · [VoxCPM2](docs/inference/voxcpm2-inference.md) · [IndexTTS2](docs/inference/indextts2.md) · [F5-TTS](docs/inference/f5-tts.md) · [Higgs TTS 3](docs/inference/higgs-tts.md) · [VibeVoice](docs/inference/vibevoice-inference.md) · [Fish Audio S2 Pro](docs/inference/fish-audio-s2-pro.md) · [Magpie TTS](docs/inference/magpie-tts.md) · [Hibiki](docs/inference/hibiki-inference.md) · [MADLAD-400](docs/inference/madlad-translation.md) · [MAGNeT](docs/inference/magnet-music-gen.md) · [Stable Audio 3](docs/inference/stable-audio-3.md) · [FlashSR](docs/inference/flashsr.md) · [Forced Aligner](docs/inference/forced-aligner.md) · [Silero VAD](docs/inference/silero-vad.md) · [FireRedVAD](docs/inference/fireredvad.md) · [Wake-word](docs/inference/wake-word.md) · [Speaker Diarization](docs/inference/speaker-diarization.md) · [Speech Enhancement](docs/inference/speech-enhancement.md) · [Sidon](docs/inference/sidon.md) · [Cache/offline](docs/inference/cache-and-offline.md)
- **リファレンス：** [共有プロトコル](docs/shared-protocols.md)

## キャッシュ設定

モデルの重みは初回使用時にHuggingFaceからダウンロードされ、`~/Library/Caches/qwen3-speech/` にキャッシュされます。`QWEN3_CACHE_DIR`（CLI）または `cacheDir:`（Swift API）で上書き可能です。すべての `fromPretrained()` エントリーポイントは `offlineMode: true` を受け付け、重みがすでにキャッシュされている場合はネットワークをスキップします。

中国本土のユーザー（または `huggingface.co` が遅い・ブロックされている地域）は、`HF_ENDPOINT` を設定することでミラーからダウンロードできます。例: `export HF_ENDPOINT=https://hf-mirror.com`。

iOSサンドボックスコンテナのパスを含む詳細は [`docs/inference/cache-and-offline.md`](docs/inference/cache-and-offline.md) を参照してください。

## MLX Metalライブラリ

実行時に `Failed to load the default metallib` と表示された場合は、Metalシェーダーライブラリが不足しています。`make build` または手動 `swift build` の後に `./scripts/build_mlx_metallib.sh release` を実行してください。Metal Toolchainがない場合はまずインストールします：

```bash
xcodebuild -downloadComponent MetalToolchain
```

## テスト

```bash
make test                            # 完全スイート（ユニット + モデルダウンロード付きE2E）
swift test --skip E2E                # ユニットのみ（CIセーフ、ダウンロードなし）
swift test --filter Qwen3ASRTests    # 指定モジュール
```

E2Eテストクラスは `E2E` プレフィックスを使うため、CIは `--skip E2E` でそれらを除外できます。完全なテスト規約は [CLAUDE.md](CLAUDE.md#testing) を参照してください。

## コントリビューション

PR歓迎 — バグ修正、新しいモデル統合、ドキュメント改善。fork、フィーチャーブランチ作成、`make build && make test`、`main` に対してPRを開いてください。

## ライセンス

Apache 2.0
