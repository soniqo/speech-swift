# Speech Swift

Modeles IA de parole pour Apple Silicon, propulses par MLX Swift et CoreML.

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

Reconnaissance, synthese et comprehension vocale embarquees pour Mac et iOS. S'execute entierement en local sur Apple Silicon -- sans cloud, sans cle API, aucune donnee ne quitte l'appareil.

**[📚 Documentation complete →](https://soniqo.audio/fr)** · **[🤗 Modeles HuggingFace](https://huggingface.co/aufklarer)** · **[📝 Blog](https://blog.ivan.digital)** · **[💬 Discord](https://discord.gg/TnCryqEMgu)**

<p align="center">
  <a href="https://formulae.brew.sh/formula/speech"><img src="https://img.shields.io/homebrew/installs/dm/speech.svg?logo=homebrew&amp;label=Homebrew%20installs&amp;color=FBB040" alt="Homebrew installs"></a>
  <a href="https://github.com/soniqo/speech-swift#built-with-speech-swift"><img src="https://img.shields.io/badge/verified%20public%20repositories-15-2ea44f?logo=github" alt="Verified public repositories: 15"></a>
</p>

<p align="center">
  <a href="https://trendshift.io/repositories/24196?utm_source=trendshift-badge&amp;utm_medium=badge&amp;utm_campaign=badge-trendshift-24196" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/trendshift/repositories/24196/daily?language=Swift" alt="soniqo%2Fspeech-swift | Trendshift" width="250" height="55"/></a>
</p>

<p align="center">
  <a href="https://youtu.be/x9zgcaW0gUk">
    <img src="https://img.youtube.com/vi/x9zgcaW0gUk/maxresdefault.jpg" width="640" alt="IA vocale locale sur un MacBook — regarder sur YouTube la visite guidée de quatre minutes de la bibliothèque open source">
  </a>
</p>
<p align="center"><em>IA vocale locale sur un MacBook — regarder sur YouTube la visite guidée de quatre minutes de la bibliothèque open source</em></p>

**Cas d'usage :** [Agents vocaux](https://soniqo.audio/fr/voice-agents) · [Transcription](https://soniqo.audio/fr/transcription) · [Synthese vocale](https://soniqo.audio/fr/speech-generation)

## Projets créés avec Speech Swift

15 dépôts publics contenant une référence vérifiable au paquet Speech Swift.

[Palmier Pro](https://github.com/palmier-io/palmier-pro) · [Anarlog](https://github.com/fastrepl/anarlog) · [ClawdHome](https://github.com/ThinkInAIXYZ/clawdhome) · [Jabber](https://github.com/rselbach/jabber) · [Ora](https://github.com/wuwangzhang1216/ora) · [VoxFlow](https://github.com/xingbofeng/VoxFlow) · [LokalBot](https://github.com/stevyhacker/lokalbot) · [Voicey](https://github.com/jonathanKingston/voicey) · [HushType](https://github.com/felixfu824/HushType) · [DexDictate macOS](https://github.com/westkitty/DexDictate_MacOS) · [Watchtower](https://github.com/aiwatchtowers/watchtower) · [Wishper App](https://github.com/irangareddy/wishper-app) · [FriSpeak](https://github.com/KSubedi/FriSpeak) · [Scribe](https://github.com/itchat/Scribe) · [VoicePen](https://github.com/dot-sk/VoicePen)

**Applications compatibles OpenAI :** [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) ([prise en charge WAV fusionnée](https://github.com/Mintplex-Labs/anything-llm/pull/6012))

**Groupes de capacités :** STT / ASR · Alignement · TTS · LLM et traduction · Speech-to-Speech · Amélioration / restauration · Séparation de sources · Génération musique / audio · Mot de réveil, VAD, diarisation et identité de locuteur

**STT / ASR**

- **[Qwen3-ASR](https://soniqo.audio/fr/guides/transcribe)** -- Reconnaissance vocale (reconnaissance automatique de la parole, 52 langues, MLX + CoreML)
- **[WhisperASR](docs/models/whisper-asr.md)** — Whisper Large-v3 Turbo speech-to-text via native CoreML runtime (ANE, multilingual)
- **[MOSS Transcribe Diarize](https://soniqo.audio/fr/guides/moss)** — Transcription CoreML native par lots avec étiquettes de locuteur et horodatages générés par le modèle (INT8 par défaut, FP16 de référence)
- **[Parakeet TDT](https://soniqo.audio/fr/guides/parakeet)** -- Reconnaissance vocale via CoreML (Neural Engine, NVIDIA FastConformer + decodeur TDT, 25 langues)
- **[Omnilingual ASR](https://soniqo.audio/fr/guides/omnilingual)** -- Reconnaissance vocale (Meta wav2vec2 + CTC, **1 672 langues** reparties dans 32 ecritures, CoreML 300M + MLX 300M/1B/3B/7B)
- **[Dictee en streaming](https://soniqo.audio/fr/guides/dictate)** -- Dictee en temps reel avec resultats partiels et detection de fin d'enonce (Parakeet-EOU-120M)
- **[Nemotron Streaming (Multilingue)](https://soniqo.audio/fr/guides/nemotron)** — ASR en streaming à faible latence avec ponctuation et majuscules natives (NVIDIA Nemotron-3.5-ASR-Streaming-0.6B, CoreML + MLX, **40 paramètres régionaux**)
- **[Nemotron Streaming (Anglais)](https://soniqo.audio/guides/nemotron)** — ASR en streaming à faible latence avec ponctuation et majuscules natives (NVIDIA Nemotron-Speech-Streaming-0.6B, CoreML, anglais uniquement, plus compact et rapide que la variante multilingue)

**Alignement**

- **[Qwen3-ForcedAligner](https://soniqo.audio/fr/guides/align)** -- Alignement temporel au niveau du mot (audio + texte → horodatages)

**TTS / Synthèse vocale**

- **[Qwen3-TTS](https://soniqo.audio/fr/guides/speak)** -- Synthese vocale (qualite maximale, streaming, locuteurs personnalises, 10 langues)
- **[CosyVoice TTS](https://soniqo.audio/fr/guides/cosyvoice)** -- TTS en streaming avec clonage vocal, dialogue multi-locuteurs, balises d'emotion (9 langues)
- **[VoxCPM2](https://soniqo.audio/fr/speech-generation)** -- TTS qualite studio 48 kHz avec clonage vocal et conception de voix par instruction (2B, MLX bf16/int8, 30 langues)
- **[IndexTTS2](docs/models/indextts2.md)** — Native MLX voice cloning from a reference voice (IndexTeam IndexTTS-2, 1.5B-class fp16 bundle, speaker/emotion/pause controls)
- **[F5-TTS](docs/models/f5-tts.md)** — Zero-shot voice cloning from a short reference clip + transcript (SWivid F5-TTS v1 Base, DiT flow matching + Vocos, MLX fp16, 24 kHz, English + Mandarin; non-commercial license)
- **[Higgs TTS 3](docs/models/higgs-tts.md)** — Conversational TTS with zero-shot voice cloning and inline emotion/style/SFX/prosody tags (Boson Higgs TTS 3, Qwen3-4B backbone, MLX bf16, 24 kHz, 100+ languages; research/non-commercial license)
- **[Kokoro TTS](https://soniqo.audio/fr/guides/kokoro)** -- TTS embarque (82M, CoreML/Neural Engine, 54 voix, compatible iOS, 10 langues)
- **[VibeVoice TTS](https://soniqo.audio/fr/guides/vibevoice)** -- TTS long format / multi-locuteurs (Microsoft VibeVoice Realtime-0.5B + 1.5B, MLX, synthese de podcast/livre audio jusqu'a 90 min, EN/ZH)
- **[Magpie TTS](https://soniqo.audio/fr/guides/magpie)** — TTS multilingue (NVIDIA Magpie-TTS Multilingual 357M, MLX INT8 411 Mo ou CoreML INT8 342 Mo, 9 langues, 5 voix prédéfinies, streaming sur MLX)
- **[Supertonic TTS](https://soniqo.audio/guides/supertonic)** — TTS embarque par appariement de flux (Supertone Supertonic-3 99M, CoreML/Neural Engine, 31 langues, 10 voix, sans G2P, 44.1 kHz)
- **[Chatterbox TTS](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16)** — TTS multilingue avec clonage vocal zero-shot (Resemble AI Chatterbox Multilingual, MLX fp16 ~1,3 Go, 23 langues runtime; l'hébreu exige le niqqud, MIT)
- **[OmniVoice TTS](https://huggingface.co/aufklarer/OmniVoice-MLX-fp16)** — TTS à diffusion non autorégressive avec clonage vocal zero-shot (k2-fsa OmniVoice, backbone Qwen3, MLX fp16 par défaut / int8 disponible, 600+ langues, Apache-2.0)
- **[Indic-Mio](docs/models/indic-mio-tts.md)** — Hindi/Indic TTS with inline emotion markers and optional reference-voice cloning (MLX, 24 kHz)

**LLM et traduction**

- **[Qwen3Chat](https://soniqo.audio/fr/guides/chat)** -- Chat LLM embarque (Qwen3.5-0.8B MLX/CoreML plus backends MLX Qwen3 dense 4B et Gemma 4 E2B/E4B, tokens en streaming)
- **[FunctionGemma](https://soniqo.audio/fr/guides/function-calls)** — LLM embarque pour les appels structures de fonctions / outils (Gemma 3 270M, CoreML palettisation 8 bits, Neural Engine, ~252 tok/s)
- **[MADLAD-400](https://soniqo.audio/fr/guides/translate)** — Traduction multidirectionnelle entre 400+ langues (3B, MLX INT4 + INT8, T5 v1.1, Apache 2.0)

**Speech-to-Speech et agents vocaux**

- **[Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate)** — Traduction parole-a-parole en streaming (FR/ES/PT/DE → EN, MLX INT4 + INT8, pile Kyutai Moshi/Mimi, CC-BY-4.0)
- **[PersonaPlex](https://soniqo.audio/fr/guides/respond)** -- Parole-a-parole en full-duplex (7B, audio entrant → audio sortant, 18 preselections de voix)
- **[Audio2Face-3D](docs/models/audio2face3d.md)** — Animation faciale d'avatar pilotée par la voix (NVIDIA Audio2Face-3D v2.3 Mark, 301 coefficients faciaux, MLX)

**Amélioration, séparation et génération audio**

- **[DeepFilterNet3](https://soniqo.audio/fr/guides/denoise)** -- Suppression de bruit en temps reel (2,1M parametres, 48 kHz). L'audio long depassant la limite de 60 s en un seul passage est decoupe automatiquement en blocs avec crossfade -- voir l'API `enhanceChunked(...)`
- **[LocalVQE v1.4-AEC](https://soniqo.audio/fr/guides/echo-cancellation)** — Annulation d'écho acoustique en streaming à partir de flux microphone et référence de lecture séparés et synchronisés (Core ML + filtre adaptatif natif, 16 kHz, latence algorithmique de 16 ms)
- **[Séparation de sources](https://soniqo.audio/fr/guides/separate)** — Séparation de sources musicales avec HTDemucs (Demucs v4) + Open-Unmix (UMX-HQ / UMX-L, 4 stems : voix/batterie/basse/autres, 44,1 kHz stéréo)
- **[MAGNeT](https://soniqo.audio/fr/guides/compose)** — Génération de musique à partir de texte (Meta MAGNeT Small 300M / Medium 1.5B, MLX INT8, clips de 30 s à 32 kHz mono, décodage masqué en parallèle)
- **[Stable Audio 3](docs/models/stable-audio-3.md)** — Text-to-audio/music generation (Stable Audio 3 Medium, MLX INT8/INT4, 44.1 kHz stereo, variable length)
- **[FlashSR](https://soniqo.audio/fr/guides/upsample)** — Super-résolution audio (FlashSR ICASSP 2025, MLX, 48 kHz mono, diffusion distillée en 1 étape, INT4 363 Mo / INT8 720 Mo)

**Détection de tour, diarisation et identité de locuteur**

- **[Mot de reveil](https://soniqo.audio/fr/guides/wake-word)** -- Detection de mots-cles sur appareil (KWS Zipformer 3M, CoreML, 26x temps reel, liste de mots-cles configurable)
- **[VAD](https://soniqo.audio/fr/guides/vad)** -- Detection d'activite vocale (Silero streaming, Pyannote hors ligne, FireRedVAD 100+ langues)
- **[Diarisation de locuteurs](https://soniqo.audio/fr/guides/diarize)** -- Qui a parle quand (pipeline Pyannote, Sortformer de bout en bout sur Neural Engine) — désormais avec une session de streaming incrémental (IDs de locuteurs stables, mises à jour toutes les 480 ms)
- **[Empreintes de locuteur](https://soniqo.audio/fr/guides/embed-speaker)** -- WeSpeaker ResNet34 (256 dim), ReDimNet2-B6 pour l’identité vocale nommée (192 dim), CAM++ (192 dim)

Articles : [Qwen3-ASR](https://arxiv.org/abs/2601.21337) (Alibaba) · [Qwen3-TTS](https://arxiv.org/abs/2601.15621) (Alibaba) · [Omnilingual ASR](https://arxiv.org/abs/2511.09690) (Meta) · [Parakeet TDT](https://arxiv.org/abs/2304.06795) (NVIDIA) · [CosyVoice 3](https://arxiv.org/abs/2505.17589) (Alibaba) · [Kokoro](https://arxiv.org/abs/2301.01695) (StyleTTS 2) · [PersonaPlex](https://arxiv.org/abs/2602.06053) (NVIDIA) · [Mimi](https://arxiv.org/abs/2410.00037) (Kyutai) · [Hibiki](https://arxiv.org/abs/2502.03382) (Kyutai) · [Sortformer](https://arxiv.org/abs/2409.06656) (NVIDIA)

## Actualites

- **19 avr. 2026** -- [MLX vs CoreML sur Apple Silicon -- guide pratique pour choisir le bon backend](https://blog.ivan.digital/mlx-vs-coreml-on-apple-silicon-a-practical-guide-to-picking-the-right-backend-and-why-you-should-f77ddea7b27a)
- **20 mars 2026** -- [Nous battons Whisper Large v3 avec un modele de 600M tournant entierement sur votre Mac](https://blog.ivan.digital/we-beat-whisper-large-v3-with-a-600m-model-running-entirely-on-your-mac-20e6ce191174)
- **26 fev. 2026** -- [Diarisation de locuteurs et detection d'activite vocale sur Apple Silicon -- Swift natif avec MLX](https://blog.ivan.digital/speaker-diarization-and-voice-activity-detection-on-apple-silicon-native-swift-with-mlx-92ea0c9aca0f)
- **23 fev. 2026** -- [NVIDIA PersonaPlex 7B sur Apple Silicon -- parole-a-parole full-duplex en Swift natif avec MLX](https://blog.ivan.digital/nvidia-personaplex-7b-on-apple-silicon-full-duplex-speech-to-speech-in-native-swift-with-mlx-0aa5276f2e23)
- **12 fev. 2026** -- [Qwen3-ASR Swift : ASR + TTS embarques pour Apple Silicon -- architecture et benchmarks](https://blog.ivan.digital/qwen3-asr-swift-on-device-asr-tts-for-apple-silicon-architecture-and-benchmarks-27cbf1e4463f)

## Demarrage rapide

Ajoutez le package a votre `Package.swift` :

```swift
.package(url: "https://github.com/soniqo/speech-swift", branch: "main")
```

N'importez que les modules dont vous avez besoin -- chaque modele est une bibliotheque SPM independante, vous ne payez donc pas pour ce que vous n'utilisez pas :

```swift
.product(name: "ParakeetStreamingASR", package: "speech-swift"),
.product(name: "SpeechUI",             package: "speech-swift"),  // vues SwiftUI optionnelles
```

**Transcrire un tampon audio en 3 lignes :**

```swift
import ParakeetStreamingASR

let model = try await ParakeetStreamingASRModel.fromPretrained()
let text = try model.transcribeAudio(audioSamples, sampleRate: 16000)
```

**Streaming en direct avec resultats partiels :**

```swift
for await partial in model.transcribeStream(audio: samples, sampleRate: 16000) {
    print(partial.isFinal ? "FINAL: \(partial.text)" : "... \(partial.text)")
}
```

**Vue de dictee SwiftUI en ~10 lignes :**

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

`SpeechUI` ne fournit que `TranscriptionView` (finaux + partiels) et `TranscriptionStore` (adaptateur ASR en streaming). Utilisez AVFoundation pour la visualisation et la lecture audio.

Produits SPM disponibles : `Qwen3ASR`, `WhisperASR`, `MossTranscribe`, `Qwen3TTS`, `Qwen3TTSCoreML`, `ParakeetASR`, `ParakeetStreamingASR`, `NemotronStreamingASR`, `OmnilingualASR`, `KokoroTTS`, `SupertonicTTS`, `VibeVoiceTTS`, `CosyVoiceTTS`, `VoxCPM2TTS`, `IndexTTS2TTS`, `F5TTS`, `HiggsTTS`, `ChatterboxTTS`, `OmniVoiceTTS`, `IndicMioTTS`, `FishAudioTTS`, `MagpieTTS`, `MagpieTTSCoreML`, `MAGNeTMusicGen`, `StableAudio3MusicGen`, `FlashSR`, `PersonaPlex`, `Audio2Face3D`, `HibikiTranslate`, `MADLADTranslation`, `SpeechVAD`, `SpeechWakeWord`, `SpeechEnhancement`, `SpeechRestoration`, `SourceSeparation`, `Qwen3Chat`, `FunctionGemma`, `SpeechCore`, `SpeechUI`, `AudioCommon`.

## Modeles

Vue compacte ci-dessous. **[Catalogue complet des modeles avec tailles, quantifications, URLs de telechargement et tableaux de memoire → soniqo.audio/architecture](https://soniqo.audio/fr/architecture)**.

| Modele | Tache | Backends | Tailles | Langues |
|-------|------|----------|-------|-----------|
| [Qwen3-ASR](https://soniqo.audio/fr/guides/transcribe) | Parole → Texte | MLX, CoreML (hybride) | 0.6B, 1.7B | 52 |
| [WhisperASR](docs/models/whisper-asr.md) | Speech → Text | CoreML (ANE) | Large-v3 Turbo | Multi |
| [MOSS Transcribe Diarize](https://soniqo.audio/fr/guides/moss) | Audio → Texte + horodatages par locuteur | CoreML | 0.9B (INT8 / FP16) | Multilingue |
| [Parakeet TDT](https://soniqo.audio/fr/guides/parakeet) | Parole → Texte | CoreML (ANE) | 0.6B | 25 europeennes |
| [Parakeet EOU](https://soniqo.audio/fr/guides/dictate) | Parole → Texte (streaming) | CoreML (ANE) | 120M | 25 europeennes |
| [Nemotron Streaming (Multilingue)](https://soniqo.audio/fr/guides/nemotron) | Voix → Texte (streaming, ponctué) | CoreML (ANE), MLX | 0.6B | **40** |
| [Nemotron Streaming (Anglais)](https://soniqo.audio/guides/nemotron) | Voix → Texte (streaming, ponctué) | CoreML (ANE) | 0.6B | EN |
| [Omnilingual ASR](https://soniqo.audio/fr/guides/omnilingual) | Parole → Texte | CoreML (ANE), MLX | 300M / 1B / 3B / 7B | **[1 672](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)** |
| [Qwen3-ForcedAligner](https://soniqo.audio/fr/guides/align) | Audio + Texte → Horodatages | MLX, CoreML | 0.6B | Multi |
| [Qwen3-TTS](https://soniqo.audio/fr/guides/speak) | Texte → Parole | MLX, CoreML | 0.6B, 1.7B | 10 |
| [CosyVoice3](https://soniqo.audio/fr/guides/cosyvoice) | Texte → Parole | MLX | 0.5B | 9 |
| [VoxCPM2](https://soniqo.audio/fr/speech-generation) | Texte → Parole (48 kHz, conception vocale + clonage) | MLX | 2B (bf16/int8) | 30 |
| [IndexTTS2](docs/models/indextts2.md) | Text → Speech (zero-shot voice cloning) | MLX | 1.5B-class (fp16) | EN/ZH |
| [F5-TTS](docs/models/f5-tts.md) | Text → Speech (zero-shot voice cloning) | MLX | 336M (fp16) | EN/ZH |
| [Higgs TTS 3](docs/models/higgs-tts.md) | Text → Speech (conversational, zero-shot voice cloning) | MLX | 4B (bf16) | 100+ |
| [Chatterbox Multilingual](https://huggingface.co/aufklarer/Chatterbox-Multilingual-MLX-fp16) | Texte → Voix (clonage zero-shot) | MLX | 0.8B (fp16) | 23 (HE exige le niqqud) |
| [OmniVoice](https://huggingface.co/aufklarer/OmniVoice-MLX-fp16) | Texte → Voix (diffusion NAR, clonage zero-shot) | MLX | 0.8B (fp16 par défaut / int8) | 600+ |
| [Indic-Mio](docs/models/indic-mio-tts.md) | Text → Speech (Hindi/Indic, emotion tags, voice cloning) | MLX | fp16 | Hindi / Indic |
| [Fish Audio S2 Pro](docs/models/fish-audio-s2-pro.md) | Texte → Voix (clonage zero-shot, marqueurs de style explicites) | MLX | 0.5B-class (fp16) | Multilingue |
| [Kokoro-82M](https://soniqo.audio/fr/guides/kokoro) | Texte → Parole | CoreML (ANE) | 82M | 10 |
| [Supertonic-3](https://soniqo.audio/guides/supertonic) | Texte → Parole (44.1 kHz, appariement de flux, sans G2P) | CoreML (ANE) | 99M | 31 |
| [VibeVoice Realtime-0.5B](https://soniqo.audio/fr/guides/vibevoice) | Texte → Parole (long format, multi-locuteurs) | MLX | 0.5B | EN/ZH |
| [VibeVoice 1.5B](https://soniqo.audio/fr/guides/vibevoice) | Texte → Parole (podcast jusqu'a 90 min) | MLX | 1.5B | EN/ZH |
| [Magpie-TTS Multilingual](https://soniqo.audio/fr/guides/magpie) | Texte → Voix (5 voix prédéfinies, streaming) | MLX / CoreML | 357M (MLX INT8, CoreML INT8) | 9 (CoreML sans JA) |
| [Qwen3.5 Chat](docs/models/qwen35-chat.md) | Text → Text (LLM) | MLX, CoreML | 0.8B | Multi |
| [Qwen3 Dense Chat](docs/models/qwen3-dense-chat.md) | Text → Text (LLM) | MLX | 4B | Multi |
| [Gemma 4 Chat](docs/models/gemma4-chat.md) | Text → Text (LLM) | MLX | E2B / E4B (4-bit) | Multi |
| [FunctionGemma](https://soniqo.audio/fr/guides/function-calls) | Texte → Appels d'outils (LLM) | CoreML | 270M | EN |
| [MADLAD-400](https://soniqo.audio/fr/guides/translate) | Texte → Texte (Traduction) | MLX | 3B | **400+** |
| [Hibiki Zero-3B](https://soniqo.audio/guides/audio-translate) | Parole → Parole (Traduction) | MLX | 3B | FR/ES/PT/DE → EN |
| [PersonaPlex](https://soniqo.audio/fr/guides/respond) | Parole → Parole | MLX | 7B | EN |
| [Audio2Face-3D](docs/models/audio2face3d.md) | Parole → Animation faciale | MLX | v2.3 Mark | Agnostique |
| [Silero VAD](https://soniqo.audio/fr/guides/vad) | Detection d'activite vocale | MLX, CoreML | 309K | Agnostique |
| [KWS Zipformer](docs/models/kws-zipformer.md) | Audio → Wake word | CoreML (ANE) | 3M | EN/custom keywords |
| [Pyannote](https://soniqo.audio/fr/guides/diarize) | VAD + Diarisation | MLX | 1.5M | Agnostique |
| [Pyannote Community-1](https://huggingface.co/aufklarer/Pyannote-Community-1-CoreML) | Diarisation + empreintes de locuteur | CoreML (ANE) + Swift VBx | 8.35M | Agnostique |
| [Sortformer](https://huggingface.co/aufklarer/Sortformer-Diarization-CoreML) | [Diarisation (E2E), streaming incrémental](https://soniqo.audio/fr/guides/diarize) | CoreML (ANE) | 117M | Agnostique |
| [DeepFilterNet3](https://soniqo.audio/fr/guides/denoise) | Amelioration de la parole | CoreML | 2.1M | Agnostique |
| [LocalVQE v1.4-AEC](https://soniqo.audio/fr/guides/echo-cancellation) | Annulation d'écho acoustique | CoreML + C++ | 200K + 2,742 | Agnostique |
| [Sidon](https://soniqo.audio/fr/guides/restore) | Restauration de la parole (debruitage + dereverberation, 48 kHz) | CoreML | w2v-BERT 2.0 + DAC (fp16/int8) | Agnostique |
| [HTDemucs (Demucs v4)](https://soniqo.audio/fr/guides/separate) | Séparation de sources | MLX | 168M | Agnostic |
| [Open-Unmix](https://soniqo.audio/fr/guides/separate) | Séparation de sources | MLX | 8.6M | Agnostic |
| [MAGNeT](https://soniqo.audio/fr/guides/compose) | Texte → Musique (30 s @ 32 kHz) | MLX | 300M / 1.5B (int4/int8) | Prompts EN |
| [Stable Audio 3](docs/models/stable-audio-3.md) | Text → Music/audio (44.1 kHz stereo) | MLX | Medium 1.4B (int4/int8) | EN prompts |
| [FlashSR](https://soniqo.audio/fr/guides/upsample) | Super-résolution audio (48 kHz) | MLX | 363 Mo / 720 Mo (int4/int8) | Agnostique |
| [WeSpeaker](https://soniqo.audio/fr/guides/embed-speaker) | Empreinte de locuteur | MLX, CoreML | 6.6M | Agnostique |
| [ReDimNet2-B6](https://huggingface.co/aufklarer/ReDimNet2-B6-CoreML) | Identité vocale nommée | CoreML | 12.3M | Agnostique |

## Installation

### Homebrew

Necessite un Homebrew ARM natif (`/opt/homebrew`). Homebrew Rosetta/x86_64 n'est pas supporte.

```bash
brew install speech
```

Ensuite :

```bash
speech transcribe recording.wav
speech transcribe recording.wav --engine moss
speech speak "Hello world"
speech translate "Hello, how are you?" --to es
speech respond --input question.wav --transcript
speech-server --port 8080            # serveur HTTP / WebSocket local (OpenAI-compatible /v1/realtime + /v1/audio/transcriptions)
```

**[Reference CLI complete →](https://soniqo.audio/fr/cli)**

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
]
```

N'importez que ce dont vous avez besoin -- chaque modele est sa propre cible SPM :

```swift
import Qwen3ASR             // Reconnaissance vocale (MLX)
import WhisperASR           // Whisper Large-v3 Turbo (CoreML)
import MossTranscribe       // MOSS transcription with timestamps + speaker labels (CoreML)
import ParakeetASR          // Reconnaissance vocale (CoreML, batch)
import ParakeetStreamingASR // Dictee en streaming avec partiels + EOU
import NemotronStreamingASR // ASR streaming multilingue avec ponctuation native (0.6B, 40 langues)
import OmnilingualASR       // 1 672 langues (CoreML + MLX)
import Qwen3TTS             // Synthese vocale
import CosyVoiceTTS         // Synthese vocale avec clonage
import VoxCPM2TTS           // TTS 48 kHz, clonage vocal + conception de voix (2B)
import IndexTTS2TTS         // Native MLX voice cloning from reference audio
import F5TTS                // Zero-shot voice cloning (DiT flow matching + Vocos)
import HiggsTTS             // Conversational TTS + cloning (Qwen3 backbone, control tags)
import KokoroTTS            // Synthese vocale (compatible iOS)
import VibeVoiceTTS         // TTS long format / multi-locuteurs (EN/ZH)
import MagpieTTS            // TTS multilingue (NVIDIA Magpie 357M, MLX, 9 langues)
import MagpieTTSCoreML      // Backend CoreML de Magpie (hybride CoreML + MLX, 8 langues)
import FishAudioTTS         // Runtime experimentale Fish Audio S2 Pro avec clonage vocal
import Qwen3Chat            // Chat LLM embarque
import FunctionGemma    // LLM embarque pour appels d'outils
import MADLADTranslation    // Traduction multidirectionnelle entre 400+ langues
import HibikiTranslate      // Traduction parole-a-parole en streaming (FR/ES/PT/DE → EN)
import PersonaPlex          // Parole-a-parole full-duplex
import SpeechVAD            // VAD + diarisation + empreintes
import SpeechEnhancement    // Suppression de bruit
import SpeechRestoration    // Restauration de la parole — debruitage + dereverberation (Sidon, CoreML, 48 kHz)
import SourceSeparation     // Séparation de sources musicales (Open-Unmix, 4 stems)
import MAGNeTMusicGen      // Génération de musique depuis du texte (30 s, 32 kHz)
import FlashSR             // Super-résolution audio (48 kHz, diffusion en 1 étape)
import SpeechUI             // Composants SwiftUI pour transcriptions en streaming
import AudioCommon          // Protocoles et utilitaires partages
```

### Prerequis

- Swift 6+, Xcode 16+ (avec Metal Toolchain)
- macOS 15+ (Sequoia) ou iOS 18+, Apple Silicon (M1/M2/M3/M4)

Le minimum macOS 15 / iOS 18 vient de [MLState](https://developer.apple.com/documentation/coreml/mlstate) —— l'API d'état persistant ANE d'Apple —— que les pipelines CoreML (Qwen3-ASR, Qwen3-Chat, Qwen3-TTS) utilisent pour garder les caches KV résidents sur le Neural Engine entre les pas de token.

### Compiler depuis les sources

```bash
git clone https://github.com/soniqo/speech-swift
cd speech-swift
make build
```

`make build` compile le package Swift **et** la bibliotheque de shaders MLX Metal. La bibliotheque Metal est necessaire pour l'inference GPU -- sans elle, vous verrez `Failed to load the default metallib` a l'execution. `make debug` pour les builds de debug, `make test` pour la suite de tests.

**[Guide complet de compilation et installation →](https://soniqo.audio/fr/getting-started)**

## Applications de demonstration

- **[DictateDemo](Examples/DictateDemo/)** ([docs](https://soniqo.audio/fr/guides/dictate)) -- Dictee en streaming dans la barre de menus macOS avec partiels en direct, detection de fin d'enonce basee sur VAD et copie en un clic. S'execute comme agent en arriere-plan (Parakeet-EOU-120M + Silero VAD).
- **[iOSEchoDemo](Examples/iOSEchoDemo/)** -- Demo d'echo iOS (Parakeet ASR + Kokoro TTS). Appareil et simulateur.
- **[PersonaPlexDemo](Examples/PersonaPlexDemo/)** -- Assistant vocal conversationnel avec entree micro, VAD et contexte multi-tours. macOS. RTF ~0,94 sur M2 Max (plus rapide que le temps reel).
- **[SpeechDemo](Examples/SpeechDemo/)** -- Dictee et synthese TTS dans une interface a onglets. macOS.

Le README de chaque demo contient les instructions de compilation.

## Exemples de code

Les extraits ci-dessous montrent le chemin minimal pour chaque domaine. Chaque section renvoie vers un guide complet sur [soniqo.audio](https://soniqo.audio/fr) avec les options de configuration, plusieurs backends, les patrons de streaming et les recettes CLI.

### Reconnaissance vocale -- [guide complet →](https://soniqo.audio/fr/guides/transcribe)

```swift
import Qwen3ASR

let model = try await Qwen3ASRModel.fromPretrained()
let text = model.transcribe(audio: audioSamples, sampleRate: 16000)
```

Backends alternatifs : [WhisperASR](docs/inference/whisper-asr-inference.md) (Whisper Large-v3 Turbo, native CoreML), [Parakeet TDT](https://soniqo.audio/fr/guides/parakeet) (CoreML, 32× temps reel), [Omnilingual ASR](https://soniqo.audio/fr/guides/omnilingual) (1 672 langues, CoreML ou MLX), [Dictee en streaming](https://soniqo.audio/fr/guides/dictate) (partiels en direct).

### Alignement force -- [guide complet →](https://soniqo.audio/fr/guides/align)

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

### Synthese vocale -- [guide complet →](https://soniqo.audio/fr/guides/speak)

```swift
import Qwen3TTS
import AudioCommon

let model = try await Qwen3TTSModel.fromPretrained()
let audio = model.synthesize(text: "Hello world", language: "english")
try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
```

Moteurs TTS alternatifs : [CosyVoice3](https://soniqo.audio/fr/guides/cosyvoice) (streaming + clonage + balises d'emotion), [Kokoro-82M](https://soniqo.audio/fr/guides/kokoro) (compatible iOS, 54 voix), [VibeVoice](https://soniqo.audio/fr/guides/vibevoice) (podcast long format / multi-locuteurs, EN/ZH), [Fish Audio S2 Pro](docs/inference/fish-audio-s2-pro.md) (clonage zero-shot experimental + marqueurs de style entre crochets), [Clonage vocal](https://soniqo.audio/fr/guides/voice-cloning).

### Parole-a-parole -- [guide complet →](https://soniqo.audio/fr/guides/respond)

```swift
import PersonaPlex

let model = try await PersonaPlexModel.fromPretrained()
let responseAudio = model.respond(userAudio: userSamples)
// Sortie Float32 mono 24 kHz prete pour la lecture
```

### Chat LLM -- [guide complet →](https://soniqo.audio/fr/guides/chat)

```swift
import Qwen3Chat
import FunctionGemma

let chat = try await Qwen35MLXChat.fromPretrained()
chat.chat(messages: [(.user, "Explain MLX in one sentence")]) { token, isFinal in
    print(token, terminator: "")
}
```

### Traduction — [guide complet →](https://soniqo.audio/fr/guides/translate)

```swift
import MADLADTranslation

let translator = try await MADLADTranslator.fromPretrained()
let es = try translator.translate("Hello, how are you?", to: "es")
// → "Hola, ¿cómo estás?"
```

### Traduction vocale — [guide complet →](https://soniqo.audio/guides/audio-translate)

```swift
import HibikiTranslate
import AudioCommon

let model = try await HibikiTranslateModel.fromPretrained()
let pcm = try AudioFileLoader.load(url: input, targetSampleRate: 24000)
let (englishAudio, textTokens) = model.translate(
    sourceAudio: pcm, sourceLanguage: .fr
)
// Hibiki Zero-3B — FR/ES/PT/DE → EN, sur appareil, codec Mimi en streaming
```

### Detection d'activite vocale -- [guide complet →](https://soniqo.audio/fr/guides/vad)

```swift
import SpeechVAD

let vad = try await SileroVADModel.fromPretrained()
let segments = vad.detectSpeech(audio: samples, sampleRate: 16000)
for s in segments { print("\(s.startTime)s → \(s.endTime)s") }
```

### Diarisation de locuteurs -- [guide complet →](https://soniqo.audio/fr/guides/diarize)

```swift
import SpeechVAD

let diarizer = try await DiarizationPipeline.fromPretrained()
let segments = diarizer.diarize(audio: samples, sampleRate: 16000)
for s in segments { print("Speaker \(s.speakerId): \(s.startTime)s - \(s.endTime)s") }
```

### Amelioration de la parole -- [guide complet →](https://soniqo.audio/fr/guides/denoise)

```swift
import SpeechEnhancement

let denoiser = try await DeepFilterNet3Model.fromPretrained()
let clean = try denoiser.enhance(audio: noisySamples, sampleRate: 48000)
```

### Annulation d'écho acoustique — [guide complet →](https://soniqo.audio/fr/guides/echo-cancellation)

```swift
import SpeechEnhancement

let aec = try await LocalVQEEchoCanceller.fromPretrained()
let cleanMicrophone = try aec.processFrame(
    microphone: microphoneFrame,
    reference: playbackReferenceFrame
)
```

### Restauration de la parole -- [guide complet →](https://soniqo.audio/fr/guides/restore)

Debruitage **et** dereverberation conjoints avec [Sidon](https://arxiv.org/abs/2509.17052) (predicteur w2v-BERT 2.0 + vocodeur DAC, Core ML). Contrairement a un suppresseur de bruit generique, Sidon est entraine pour preserver l'identite du locuteur, ce qui le rend bien adapte au nettoyage d'une voix de reference bruitee ou reverberee pour le clonage vocal avant la synthese TTS. L'entree est en 16 kHz ; la sortie en 48 kHz mono.

```swift
import SpeechRestoration

let restorer = try await SpeechRestorer.fromPretrained()          // .fp16 (defaut) ou .int8
let clean = try restorer.restore(audio: noisySamples, sampleRate: 16000)  // → 48 kHz
```

Depuis la CLI :

```bash
speech restore noisy.wav -o clean.wav            # debruitage + dereverberation, sortie 48 kHz
speech restore noisy.wav --variant int8          # plus compact, pic de RAM plus faible

# Nettoyer une voix de reference pour le clonage vocal avant la synthese TTS (optionnel ; preserve l'identite du locuteur) :
speech speak "Hello" --engine voxcpm2 --voice-sample ref.wav --clean-reference
```

### Voice Pipeline (ASR → LLM → TTS) -- [guide complet →](https://soniqo.audio/fr/voice-agents)

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

`VoicePipeline` est la machine a etats agent vocal temps reel (propulsee par [speech-core](https://github.com/soniqo/speech-core)) avec detection de tours basee sur VAD, gestion des interruptions et STT eager. Elle connecte n'importe quel `SpeechRecognitionModel` + `SpeechGenerationModel` + `StreamingVADProvider`.

### Serveur API HTTP

```bash
speech-server --port 8080
```

Expose chaque modele via des endpoints HTTP REST + WebSocket, y compris des APIs compatibles OpenAI : un WebSocket Realtime sur `/v1/realtime` et un endpoint REST de transcription sur `/v1/audio/transcriptions`. Voir [`Sources/AudioServer/`](Sources/AudioServer/).

## Architecture

speech-swift est decoupe en une cible SPM par modele, de sorte que les consommateurs ne paient que ce qu'ils importent. L'infrastructure partagee reside dans `AudioCommon` (protocoles, I/O audio, telechargeur HuggingFace, `SentencePieceModel`) et `MLXCommon` (chargement de poids, aides `QuantizedLinear`, aide d'attention multi-tete `SDPA`).

**[Diagramme d'architecture complet avec backends, tableaux de memoire et carte des modules → soniqo.audio/architecture](https://soniqo.audio/fr/architecture)** · **[Reference d'API → soniqo.audio/api](https://soniqo.audio/fr/api)** · **[Benchmarks → soniqo.audio/benchmarks](https://soniqo.audio/fr/benchmarks)**

Docs locales (depot) :
- **Modeles :** [Qwen3-ASR](docs/models/asr-model.md) · [WhisperASR](docs/models/whisper-asr.md) · [MOSS Transcribe Diarize](docs/models/moss-transcribe-diarize.md) · [Qwen3-TTS](docs/models/tts-model.md) · [CosyVoice](docs/models/cosyvoice-tts.md) · [Kokoro](docs/models/kokoro-tts.md) · [VoxCPM2](docs/models/voxcpm2-tts.md) · [IndexTTS2](docs/models/indextts2.md) · [F5-TTS](docs/models/f5-tts.md) · [Higgs TTS 3](docs/models/higgs-tts.md) · [VibeVoice](docs/models/vibevoice.md) · [Supertonic](docs/models/supertonic-tts.md) · [Chatterbox](docs/models/chatterbox-tts.md) · [Indic-Mio](docs/models/indic-mio-tts.md) · [Fish Audio S2 Pro](docs/models/fish-audio-s2-pro.md) · [Magpie TTS](docs/models/magpie-tts.md) · [Parakeet TDT](docs/models/parakeet-asr.md) · [Parakeet Streaming](docs/models/parakeet-streaming-asr.md) · [Nemotron Streaming](docs/models/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/models/omnilingual-asr.md) · [PersonaPlex](docs/models/personaplex.md) · [Hibiki](docs/models/hibiki.md) · [MADLAD-400](docs/models/madlad-translation.md) · [FunctionGemma](docs/models/function-gemma.md) · [Qwen3.5 Chat](docs/models/qwen35-chat.md) · [Gemma 4 Chat](docs/models/gemma4-chat.md) · [Qwen3 Dense Chat](docs/models/qwen3-dense-chat.md) · [FireRedVAD](docs/models/fireredvad.md) · [KWS Zipformer](docs/models/kws-zipformer.md) · [Sidon](docs/models/sidon.md) · [Source Separation](docs/models/source-separation.md) · [HTDemucs](docs/models/htdemucs.md) · [MAGNeT](docs/models/magnet-music-gen.md) · [Stable Audio 3](docs/models/stable-audio-3.md) · [FlashSR](docs/models/flashsr.md) · [Audio2Face-3D](docs/models/audio2face3d.md)
- **Inference :** [Qwen3-ASR](docs/inference/qwen3-asr-inference.md) · [WhisperASR](docs/inference/whisper-asr-inference.md) · [MOSS Transcribe Diarize](docs/inference/moss-transcribe-diarize.md) · [Parakeet TDT](docs/inference/parakeet-asr-inference.md) · [Parakeet Streaming](docs/inference/parakeet-streaming-asr-inference.md) · [Nemotron Streaming](docs/inference/nemotron-asr-streaming.md) · [Omnilingual ASR](docs/inference/omnilingual-asr-inference.md) · [TTS](docs/inference/qwen3-tts-inference.md) · [VoxCPM2](docs/inference/voxcpm2-inference.md) · [IndexTTS2](docs/inference/indextts2.md) · [F5-TTS](docs/inference/f5-tts.md) · [Higgs TTS 3](docs/inference/higgs-tts.md) · [VibeVoice](docs/inference/vibevoice-inference.md) · [Fish Audio S2 Pro](docs/inference/fish-audio-s2-pro.md) · [Magpie TTS](docs/inference/magpie-tts.md) · [Hibiki](docs/inference/hibiki-inference.md) · [MADLAD-400](docs/inference/madlad-translation.md) · [MAGNeT](docs/inference/magnet-music-gen.md) · [Stable Audio 3](docs/inference/stable-audio-3.md) · [FlashSR](docs/inference/flashsr.md) · [Forced Aligner](docs/inference/forced-aligner.md) · [Silero VAD](docs/inference/silero-vad.md) · [FireRedVAD](docs/inference/fireredvad.md) · [Wake-word](docs/inference/wake-word.md) · [Speaker Diarization](docs/inference/speaker-diarization.md) · [Speech Enhancement](docs/inference/speech-enhancement.md) · [Sidon](docs/inference/sidon.md) · [Cache/offline](docs/inference/cache-and-offline.md)
- **Annulation d'écho :** [LocalVQE AEC](docs/inference/echo-cancellation.md)
- **Reference :** [Protocoles partages](docs/shared-protocols.md)

## Configuration du cache

Les poids de modele sont telecharges depuis HuggingFace lors de la premiere utilisation et mis en cache dans `~/Library/Caches/qwen3-speech/`. Surchargez avec `QWEN3_CACHE_DIR` (CLI) ou `cacheDir:` (API Swift). Tous les points d'entree `fromPretrained()` acceptent `offlineMode: true` pour sauter le reseau lorsque les poids sont deja en cache.

Les utilisateurs en Chine continentale (ou partout ou `huggingface.co` est lent ou bloque) peuvent telecharger depuis un miroir en definissant `HF_ENDPOINT`, par ex. `export HF_ENDPOINT=https://hf-mirror.com`.

Voir [`docs/inference/cache-and-offline.md`](docs/inference/cache-and-offline.md) pour les details complets incluant les chemins de container iOS sandboxes.

## Bibliotheque MLX Metal

Si vous voyez `Failed to load the default metallib` a l'execution, la bibliotheque de shaders Metal est manquante. Executez `make build` ou `./scripts/build_mlx_metallib.sh release` apres un `swift build` manuel. Si le Metal Toolchain est absent, installez-le d'abord :

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Tests

```bash
make test                            # suite complete (unite + E2E avec telechargements de modeles)
swift test --skip E2E                # unite uniquement (CI-safe, sans telechargements)
swift test --filter Qwen3ASRTests    # module specifique
```

Les classes de test E2E utilisent le prefixe `E2E` pour que la CI puisse les filtrer avec `--skip E2E`. Voir [CLAUDE.md](CLAUDE.md#testing) pour la convention complete de tests.

## Contribuer

PR bienvenues -- corrections de bugs, nouvelles integrations de modeles, documentation. Fork, creez une branche de fonctionnalite, `make build && make test`, ouvrez une PR vers `main`.

## Licence

Apache 2.0
