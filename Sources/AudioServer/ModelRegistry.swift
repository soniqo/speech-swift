import Foundation
import Qwen3ASR
import Qwen3TTS
import Qwen3TTSCoreML
import CosyVoiceTTS
import ParakeetASR
import ParakeetStreamingASR
import NemotronStreamingASR
import OmnilingualASR
import KokoroTTS
import VoxCPM2TTS
import IndicMioTTS
import MagpieTTS
import MagpieTTSCoreML
import VibeVoiceTTS
import PersonaPlex
import HibikiTranslate
import SpeechEnhancement
import SpeechVAD
import SourceSeparation
import FlashSR
import MAGNeTMusicGen
import StableAudio3MusicGen

// MARK: - Model registry

/// A single selectable model variant.
///
/// Each row carries the exact identity the server will load:
///   - `name` — canonical name echoed back on session.updated
///   - `engine` — dispatch slot ("qwen3-asr", "parakeet", "kokoro", "hibiki", …)
///   - `modelId` — full HuggingFace model identifier passed to `fromPretrained`
///   - `aliases` — short-form names accepted on session.update (`"qwen3"` → 0.6B INT4 by default)
///   - `kind` — which session slot this variant updates (ASR / TTS / S2S)
///
/// Adding a new variant is one row in `MODEL_REGISTRY`. No new resolver code,
/// no new dispatch arm, no surprises.
public struct ModelVariant: Sendable, Equatable {
    public let name: String
    public let engine: String
    public let modelId: String
    public let aliases: [String]
    public let kind: Kind

    public enum Kind: String, Sendable, Equatable, CaseIterable {
        /// Speech recognition (audio → text).
        case asr
        /// Speech synthesis (text → audio).
        case tts
        /// Speech-to-speech (audio → audio in one shot).
        case s2s
        /// Speech enhancement / noise suppression (audio → cleaned audio).
        case enhance
        /// Music or sound-effect generation (text → audio).
        case music
        /// Voice activity detection (audio → speech/silence timeline).
        case vad
        /// Speaker diarization (audio → speaker-labelled segments).
        case diarize
        /// Speaker-embedding extractor (audio → fixed-size vector).
        case speaker
        /// Source separation (audio → per-stem audio).
        case separate
        /// Speech super-resolution (low-rate audio → high-rate audio).
        case sr
    }
}

/// Voice profile shape a TTS model can consume.
public enum TTSVoiceProfileMode: String, Sendable, Equatable, CaseIterable {
    case referenceClone = "reference-clone"
    case presetVoice = "preset-voice"
    case designedVoice = "designed-voice"
}

/// Text-side style control exposed by a TTS model.
public enum TTSStyleMode: String, Sendable, Equatable, CaseIterable {
    /// Natural-language instruction/caption.
    case instruction
    /// Scalar expressiveness control, not a specific emotion.
    case intensity
    /// The model accepts tags appended to the utterance, e.g. `text <happy>`.
    case suffixTag = "suffix-tag"
    /// The model accepts bracket tags in text, e.g. `[excited] text`.
    case bracketTag = "bracket-tag"
    /// No explicit style/emotion control is exposed.
    case none
}

/// Runtime-relevant capabilities for a selectable TTS model.
///
/// This mirrors the product-facing model registry shape: clients should be
/// able to decide whether a voice profile is compatible, which languages to
/// offer, and whether style markers/instructions are meaningful without
/// hardcoding per-engine behavior.
public struct TTSModelCapabilities: Sendable, Equatable {
    public let modelName: String
    public let displayName: String
    public let modelSize: String
    public let languages: [String]
    public let voiceProfileModes: [TTSVoiceProfileMode]
    public let requiresReferenceAudio: Bool
    public let requiresReferenceTranscript: Bool
    public let supportsInstruct: Bool
    public let styleMode: TTSStyleMode
    public let supportedMarkers: [String]
    public let needsTrim: Bool
    public let usePolicy: String
    public let readiness: String

    public init(
        modelName: String,
        displayName: String,
        modelSize: String,
        languages: [String],
        voiceProfileModes: [TTSVoiceProfileMode],
        requiresReferenceAudio: Bool,
        requiresReferenceTranscript: Bool,
        supportsInstruct: Bool,
        styleMode: TTSStyleMode,
        supportedMarkers: [String],
        needsTrim: Bool,
        usePolicy: String = "commercial-safe",
        readiness: String = "production"
    ) {
        self.modelName = modelName
        self.displayName = displayName
        self.modelSize = modelSize
        self.languages = languages
        self.voiceProfileModes = voiceProfileModes
        self.requiresReferenceAudio = requiresReferenceAudio
        self.requiresReferenceTranscript = requiresReferenceTranscript
        self.supportsInstruct = supportsInstruct
        self.styleMode = styleMode
        self.supportedMarkers = supportedMarkers
        self.needsTrim = needsTrim
        self.usePolicy = usePolicy
        self.readiness = readiness
    }
}

/// Every model name the Realtime API accepts.
///
/// `modelId` values reference each engine module's `defaultModelId` /
/// `largeModelId` / etc. constants — the SSOT lives with the engine,
/// the registry just aggregates names. Adding a variant means publishing
/// the HF slug as a constant on the engine module, then one row here.
///
/// Order matters only for aliases that collide — within each kind, the
/// first row that owns a given alias wins. That lets "kokoro" → the
/// default Kokoro variant, "qwen3" → the 0.6B INT4 default, etc.
public let MODEL_REGISTRY: [ModelVariant] = [
    // ─── ASR ───────────────────────────────────────────────────────────────
    .init(name: "qwen3-asr-0.6b-mlx-int4",
          engine: "qwen3-asr",
          modelId: Qwen3ASRModel.defaultModelId,
          aliases: ["qwen3", "qwen3-asr", "qwen3-0.6b"],
          kind: .asr),
    .init(name: "qwen3-asr-1.7b-mlx-int8",
          engine: "qwen3-asr",
          modelId: Qwen3ASRModel.largeModelId,
          aliases: ["qwen3-1.7b", "qwen3-asr-1.7b"],
          kind: .asr),
    .init(name: "qwen3-asr-coreml",
          engine: "qwen3-asr",
          modelId: Qwen3ASRModel.coreMLModelId,
          aliases: ["qwen3-coreml", "qwen3-asr-0.6b-coreml"],
          kind: .asr),
    .init(name: "parakeet-tdt-v3-coreml-int8-30s",
          engine: "parakeet",
          modelId: ParakeetASRModel.defaultModelId,
          aliases: ["parakeet", "parakeet-tdt", "parakeet-tdt-v3"],
          kind: .asr),
    .init(name: "parakeet-tdt-v3-coreml-int8-ios-5s",
          engine: "parakeet",
          modelId: ParakeetASRModel.iosModelId,
          aliases: ["parakeet-ios", "parakeet-5s"],
          kind: .asr),
    .init(name: "parakeet-eou-120m-coreml-int8",
          engine: "parakeet-streaming",
          modelId: ParakeetStreamingASRModel.defaultModelId,
          aliases: ["parakeet-streaming", "parakeet-eou", "parakeet-120m"],
          kind: .asr),
    .init(name: "nemotron-3.5-asr-streaming-0.6b-coreml-int8",
          engine: "nemotron",
          modelId: NemotronStreamingASRModel.defaultModelId,
          aliases: ["nemotron", "nemotron-3.5", "nemotron-streaming"],
          kind: .asr),
    .init(name: "omnilingual-asr-ctc-300m-coreml-int8-10s",
          engine: "omnilingual",
          modelId: OmnilingualASRModel.defaultModelId,
          aliases: ["omnilingual", "omnilingual-300m", "omnilingual-coreml"],
          kind: .asr),

    // ─── TTS ───────────────────────────────────────────────────────────────
    .init(name: "kokoro-82m-coreml",
          engine: "kokoro",
          modelId: KokoroTTSModel.defaultModelId,
          aliases: ["kokoro", "kokoro-82m"],
          kind: .tts),
    .init(name: "cosyvoice-3-0.5b-mlx-bf16",
          engine: "cosyvoice",
          modelId: CosyVoiceTTSModel.defaultModelId,
          aliases: [
              "cosyvoice", "cosyvoice-3", "cosyvoice-0.5b",
              "cosyvoice-bf16", "cosyvoice-16bit", "cosyvoice-unquantized",
          ],
          kind: .tts),
    .init(name: "cosyvoice-3-0.5b-mlx-8bit",
          engine: "cosyvoice",
          modelId: "aufklarer/CosyVoice3-0.5B-MLX-8bit",
          aliases: ["cosyvoice-8bit"],
          kind: .tts),
    .init(name: "cosyvoice-3-0.5b-mlx-8bit-full",
          engine: "cosyvoice",
          modelId: "aufklarer/CosyVoice3-0.5B-MLX-8bit-full",
          aliases: ["cosyvoice-8bit-full"],
          kind: .tts),
    .init(name: "voxcpm2-mlx-bf16",
          engine: "voxcpm2",
          modelId: VoxCPM2TTSModel.defaultModelId,
          aliases: ["voxcpm2", "voxcpm2-bf16"],
          kind: .tts),
    .init(name: "voxcpm2-mlx-int8",
          engine: "voxcpm2",
          modelId: VoxCPM2TTSModel.int8ModelId,
          aliases: ["voxcpm2-int8"],
          kind: .tts),
    .init(name: "indic-mio-mlx-fp16",
          engine: "indic-mio",
          modelId: IndicMioTTSModel.defaultModelId,
          aliases: ["indic-mio", "mio", "hindi-emotion"],
          kind: .tts),
    .init(name: "qwen3-tts-1.7b-mlx-bf16",
          engine: "qwen3-tts",
          modelId: Qwen3TTSModel.defaultModelId,
          // "qwen3" is shared with the ASR variant — bare "qwen3" lands in
          // both slots so naming the family pairs ASR + TTS in one update.
          aliases: [
              "qwen3", "qwen3-tts", "qwen3-speech", "qwen3-tts-1.7b",
              "qwen3-tts-base-1.7b", "qwen3-tts-1.7b-bf16",
          ],
          kind: .tts),
    .init(name: "qwen3-tts-0.6b-base-mlx-8bit",
          engine: "qwen3-tts",
          modelId: TTSModelVariant.base.rawValue,
          aliases: ["qwen3-tts-0.6b", "qwen3-tts-base-0.6b", "qwen3-speech-0.6b"],
          kind: .tts),
    // Magpie ships as a fixed bundle today (the `MagpieTTSVariant.int8`
    // default; int4 was decommissioned). Listing it here keeps it selectable
    // via the protocol; the dispatch site ignores the modelId because
    // MagpieTTS.fromPretrained builds the model from its own variant enum.
    .init(name: "magpie-tts-multilingual-mlx-int8",
          engine: "magpie",
          modelId: MagpieTTSVariant.int8.huggingFaceRepoId,
          aliases: ["magpie", "magpie-tts"],
          kind: .tts),
    .init(name: "magpie-tts-multilingual-357m-coreml-int8",
          engine: "magpie-coreml",
          modelId: MagpieCoreMLConstants.huggingFaceRepo,
          aliases: ["magpie-coreml", "magpie-357m-coreml"],
          kind: .tts),
    .init(name: "qwen3-tts-coreml",
          engine: "qwen3-tts-coreml",
          modelId: Qwen3TTSCoreMLModel.defaultModelId,
          aliases: ["qwen3-tts-coreml", "qwen3-speech-coreml"],
          kind: .tts),
    .init(name: "vibevoice-realtime-0.5b-mlx-int4",
          engine: "vibevoice",
          modelId: VibeVoiceTTSModel.defaultModelId,
          aliases: ["vibevoice", "vibevoice-realtime", "vibevoice-0.5b"],
          kind: .tts),
    .init(name: "vibevoice-1.5b-mlx-int4",
          engine: "vibevoice-1.5b",
          modelId: VibeVoice15BTTSModel.defaultModelId,
          aliases: ["vibevoice-1.5b", "vibevoice-large"],
          kind: .tts),

    // ─── Speech-to-speech (input audio → output audio in one shot) ─────────
    .init(name: "personaplex-7b-mlx-4bit",
          engine: "personaplex",
          modelId: PersonaPlexModel.defaultModelId,
          aliases: ["personaplex", "personaplex-7b"],
          kind: .s2s),
    .init(name: "personaplex-7b-mlx-8bit",
          engine: "personaplex",
          modelId: PersonaPlexModel.modelId8bit,
          aliases: ["personaplex-8bit"],
          kind: .s2s),
    .init(name: "hibiki-zero-3b-mlx-4bit",
          engine: "hibiki",
          modelId: HibikiTranslateModel.defaultModelId,
          aliases: ["hibiki", "hibiki-zero", "hibiki-3b"],
          kind: .s2s),
    .init(name: "hibiki-zero-3b-mlx-8bit",
          engine: "hibiki",
          modelId: HibikiTranslateModel.modelId8bit,
          aliases: ["hibiki-8bit"],
          kind: .s2s),

    // ─── Enhance (noise suppression / cleanup) ─────────────────────────────
    .init(name: "deepfilternet3-coreml",
          engine: "deepfilternet3",
          modelId: SpeechEnhancer.defaultModelId,
          aliases: ["deepfilternet3", "denoise", "dfn3"],
          kind: .enhance),

    // ─── Music / SFX generation ────────────────────────────────────────────
    .init(name: "magnet-small-30s-mlx-int4",
          engine: "magnet",
          modelId: MAGNeTVariant.smallInt4.huggingFaceRepoId,
          aliases: ["magnet", "magnet-small", "magnet-small-int4"],
          kind: .music),
    .init(name: "magnet-small-30s-mlx-int8",
          engine: "magnet",
          modelId: MAGNeTVariant.smallInt8.huggingFaceRepoId,
          aliases: ["magnet-small-int8"],
          kind: .music),
    .init(name: "magnet-medium-30s-mlx-int4",
          engine: "magnet",
          modelId: MAGNeTVariant.mediumInt4.huggingFaceRepoId,
          aliases: ["magnet-medium", "magnet-medium-int4"],
          kind: .music),
    .init(name: "magnet-medium-30s-mlx-int8",
          engine: "magnet",
          modelId: MAGNeTVariant.mediumInt8.huggingFaceRepoId,
          aliases: ["magnet-medium-int8"],
          kind: .music),
    .init(name: "stable-audio-3-dit-medium-mlx-int4",
          engine: "stable-audio-3",
          modelId: StableAudio3Variant.mediumInt4.huggingFaceRepoId,
          aliases: ["stable-audio-3", "sa3", "sa3-medium"],
          kind: .music),
    .init(name: "stable-audio-3-dit-medium-mlx-int8",
          engine: "stable-audio-3",
          modelId: StableAudio3Variant.mediumInt8.huggingFaceRepoId,
          aliases: ["sa3-medium-int8"],
          kind: .music),
    .init(name: "stable-audio-3-dit-small-music-mlx-int4",
          engine: "stable-audio-3",
          modelId: StableAudio3Variant.smallMusicInt4.huggingFaceRepoId,
          aliases: ["sa3-small-music"],
          kind: .music),
    .init(name: "stable-audio-3-dit-small-music-mlx-int8",
          engine: "stable-audio-3",
          modelId: StableAudio3Variant.smallMusicInt8.huggingFaceRepoId,
          aliases: ["sa3-small-music-int8"],
          kind: .music),
    .init(name: "stable-audio-3-dit-small-sfx-mlx-int4",
          engine: "stable-audio-3",
          modelId: StableAudio3Variant.smallSFXInt4.huggingFaceRepoId,
          aliases: ["sa3-small-sfx"],
          kind: .music),
    .init(name: "stable-audio-3-dit-small-sfx-mlx-int8",
          engine: "stable-audio-3",
          modelId: StableAudio3Variant.smallSFXInt8.huggingFaceRepoId,
          aliases: ["sa3-small-sfx-int8"],
          kind: .music),

    // ─── Voice activity detection ──────────────────────────────────────────
    .init(name: "silero-vad-v5-mlx",
          engine: "silero-vad",
          modelId: SileroVADModel.defaultModelId,
          aliases: ["silero", "silero-vad"],
          kind: .vad),
    .init(name: "silero-vad-v6.2.1-coreml",
          engine: "silero-vad",
          modelId: SileroVADModel.defaultCoreMLModelId,
          aliases: ["silero-coreml", "silero-vad-coreml", "silero-vad-v5-coreml"],
          kind: .vad),
    .init(name: "pyannote-segmentation-mlx",
          engine: "pyannote-segmentation",
          modelId: PyannoteVADModel.defaultModelId,
          aliases: ["pyannote", "pyannote-segmentation"],
          kind: .vad),
    .init(name: "firered-vad-coreml",
          engine: "firered-vad",
          modelId: FireRedVADModel.defaultModelId,
          aliases: ["firered", "firered-vad"],
          kind: .vad),

    // ─── Diarization ───────────────────────────────────────────────────────
    .init(name: "sortformer-diarization-coreml",
          engine: "sortformer",
          modelId: SortformerDiarizer.defaultModelId,
          aliases: ["sortformer", "sortformer-diarization"],
          kind: .diarize),

    // ─── Speaker embedding ─────────────────────────────────────────────────
    .init(name: "wespeaker-resnet34-lm-mlx",
          engine: "wespeaker",
          modelId: WeSpeakerModel.defaultModelId,
          aliases: ["wespeaker", "wespeaker-mlx"],
          kind: .speaker),
    .init(name: "wespeaker-resnet34-lm-coreml",
          engine: "wespeaker",
          modelId: WeSpeakerModel.defaultCoreMLModelId,
          aliases: ["wespeaker-coreml"],
          kind: .speaker),

    // ─── Source separation (vocals / drums / bass / other) ─────────────────
    .init(name: "openunmix-hq-mlx",
          engine: "openunmix",
          modelId: SourceSeparator.defaultModelId,
          aliases: ["openunmix", "openunmix-hq"],
          kind: .separate),
    .init(name: "openunmix-l-mlx",
          engine: "openunmix",
          modelId: SourceSeparator.largeModelId,
          aliases: ["openunmix-l", "openunmix-large"],
          kind: .separate),
    .init(name: "htdemucs-ft-mlx",
          engine: "htdemucs",
          modelId: HTDemucsSeparator.defaultModelId,
          aliases: ["htdemucs", "demucs"],
          kind: .separate),

    // ─── Speech super-resolution ───────────────────────────────────────────
    .init(name: "flashsr-mlx-int4",
          engine: "flashsr",
          modelId: FlashSRVariant.int4.huggingFaceRepoId,
          aliases: ["flashsr", "flashsr-int4"],
          kind: .sr),
    .init(name: "flashsr-mlx-int8",
          engine: "flashsr",
          modelId: FlashSRVariant.int8.huggingFaceRepoId,
          aliases: ["flashsr-int8"],
          kind: .sr),
]

private let qwenTTSLanguages = ["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"]

/// Capability metadata for TTS variants in `MODEL_REGISTRY`.
///
/// This is intentionally keyed by canonical `ModelVariant.name` so protocol
/// responses can attach capabilities after name resolution. Aliases remain
/// routing-only.
public let TTS_MODEL_CAPABILITIES: [String: TTSModelCapabilities] = [
    "kokoro-82m-coreml": .init(
        modelName: "kokoro-82m-coreml",
        displayName: "Kokoro 82M CoreML",
        modelSize: "82M",
        languages: ["en", "es", "fr", "hi", "it", "pt", "ja", "zh"],
        voiceProfileModes: [.presetVoice],
        requiresReferenceAudio: false,
        requiresReferenceTranscript: false,
        supportsInstruct: false,
        styleMode: .none,
        supportedMarkers: [],
        needsTrim: false
    ),
    "cosyvoice-3-0.5b-mlx-bf16": .init(
        modelName: "cosyvoice-3-0.5b-mlx-bf16",
        displayName: "CosyVoice 3 0.5B MLX bf16",
        modelSize: "0.5B",
        languages: ["en", "zh"],
        voiceProfileModes: [.referenceClone],
        requiresReferenceAudio: true,
        requiresReferenceTranscript: true,
        supportsInstruct: true,
        styleMode: .instruction,
        supportedMarkers: ["angry", "calm", "excited", "happy", "sad", "serious", "soft", "whispering"],
        needsTrim: false
    ),
    "cosyvoice-3-0.5b-mlx-8bit": .init(
        modelName: "cosyvoice-3-0.5b-mlx-8bit",
        displayName: "CosyVoice 3 0.5B MLX 8bit",
        modelSize: "0.5B",
        languages: ["en", "zh"],
        voiceProfileModes: [.referenceClone],
        requiresReferenceAudio: true,
        requiresReferenceTranscript: true,
        supportsInstruct: true,
        styleMode: .instruction,
        supportedMarkers: ["angry", "calm", "excited", "happy", "sad", "serious", "soft", "whispering"],
        needsTrim: false
    ),
    "cosyvoice-3-0.5b-mlx-8bit-full": .init(
        modelName: "cosyvoice-3-0.5b-mlx-8bit-full",
        displayName: "CosyVoice 3 0.5B MLX 8bit full",
        modelSize: "0.5B",
        languages: ["en", "zh"],
        voiceProfileModes: [.referenceClone],
        requiresReferenceAudio: true,
        requiresReferenceTranscript: true,
        supportsInstruct: true,
        styleMode: .instruction,
        supportedMarkers: ["angry", "calm", "excited", "happy", "sad", "serious", "soft", "whispering"],
        needsTrim: false
    ),
    "voxcpm2-mlx-bf16": .init(
        modelName: "voxcpm2-mlx-bf16",
        displayName: "VoxCPM2 MLX bf16",
        modelSize: "1.7B",
        languages: ["en", "zh"],
        voiceProfileModes: [.referenceClone],
        requiresReferenceAudio: true,
        requiresReferenceTranscript: false,
        supportsInstruct: true,
        styleMode: .instruction,
        supportedMarkers: ["angry", "calm", "excited", "happy", "sad", "serious", "soft", "whispering"],
        needsTrim: false
    ),
    "voxcpm2-mlx-int8": .init(
        modelName: "voxcpm2-mlx-int8",
        displayName: "VoxCPM2 MLX int8",
        modelSize: "1.7B",
        languages: ["en", "zh"],
        voiceProfileModes: [.referenceClone],
        requiresReferenceAudio: true,
        requiresReferenceTranscript: false,
        supportsInstruct: true,
        styleMode: .instruction,
        supportedMarkers: ["angry", "calm", "excited", "happy", "sad", "serious", "soft", "whispering"],
        needsTrim: false
    ),
    "indic-mio-mlx-fp16": .init(
        modelName: "indic-mio-mlx-fp16",
        displayName: "Indic-Mio MLX fp16",
        modelSize: "0.6B",
        languages: ["hi", "en"],
        voiceProfileModes: [.referenceClone],
        requiresReferenceAudio: true,
        requiresReferenceTranscript: false,
        supportsInstruct: false,
        styleMode: .suffixTag,
        supportedMarkers: ["<happy>", "<sad>", "<angry>", "<disgust>", "<fear>", "<surprise>"],
        needsTrim: false
    ),
    "qwen3-tts-1.7b-mlx-bf16": .init(
        modelName: "qwen3-tts-1.7b-mlx-bf16",
        displayName: "Qwen3-TTS 1.7B Base MLX bf16",
        modelSize: "1.7B",
        languages: qwenTTSLanguages,
        voiceProfileModes: [.referenceClone],
        requiresReferenceAudio: true,
        requiresReferenceTranscript: true,
        supportsInstruct: false,
        styleMode: .none,
        supportedMarkers: [],
        needsTrim: false
    ),
    "qwen3-tts-0.6b-base-mlx-8bit": .init(
        modelName: "qwen3-tts-0.6b-base-mlx-8bit",
        displayName: "Qwen3-TTS 0.6B Base MLX 8bit",
        modelSize: "0.6B",
        languages: qwenTTSLanguages,
        voiceProfileModes: [.referenceClone],
        requiresReferenceAudio: true,
        requiresReferenceTranscript: true,
        supportsInstruct: false,
        styleMode: .none,
        supportedMarkers: [],
        needsTrim: false,
        readiness: "experimental"
    ),
    "magpie-tts-multilingual-mlx-int8": .init(
        modelName: "magpie-tts-multilingual-mlx-int8",
        displayName: "Magpie TTS Multilingual MLX int8",
        modelSize: "multilingual",
        languages: ["en"],
        voiceProfileModes: [.presetVoice],
        requiresReferenceAudio: false,
        requiresReferenceTranscript: false,
        supportsInstruct: false,
        styleMode: .none,
        supportedMarkers: [],
        needsTrim: false
    ),
    "magpie-tts-multilingual-357m-coreml-int8": .init(
        modelName: "magpie-tts-multilingual-357m-coreml-int8",
        displayName: "Magpie TTS 357M CoreML int8",
        modelSize: "357M",
        languages: ["en"],
        voiceProfileModes: [.presetVoice],
        requiresReferenceAudio: false,
        requiresReferenceTranscript: false,
        supportsInstruct: false,
        styleMode: .none,
        supportedMarkers: [],
        needsTrim: false
    ),
    "qwen3-tts-coreml": .init(
        modelName: "qwen3-tts-coreml",
        displayName: "Qwen3-TTS CoreML",
        modelSize: "0.6B",
        languages: qwenTTSLanguages,
        voiceProfileModes: [.presetVoice],
        requiresReferenceAudio: false,
        requiresReferenceTranscript: false,
        supportsInstruct: false,
        styleMode: .none,
        supportedMarkers: [],
        needsTrim: false
    ),
    "vibevoice-realtime-0.5b-mlx-int4": .init(
        modelName: "vibevoice-realtime-0.5b-mlx-int4",
        displayName: "VibeVoice Realtime 0.5B MLX int4",
        modelSize: "0.5B",
        languages: ["en"],
        voiceProfileModes: [.presetVoice],
        requiresReferenceAudio: false,
        requiresReferenceTranscript: false,
        supportsInstruct: false,
        styleMode: .none,
        supportedMarkers: [],
        needsTrim: false,
        readiness: "experimental"
    ),
    "vibevoice-1.5b-mlx-int4": .init(
        modelName: "vibevoice-1.5b-mlx-int4",
        displayName: "VibeVoice 1.5B MLX int4",
        modelSize: "1.5B",
        languages: ["en"],
        voiceProfileModes: [.presetVoice],
        requiresReferenceAudio: false,
        requiresReferenceTranscript: false,
        supportsInstruct: false,
        styleMode: .none,
        supportedMarkers: [],
        needsTrim: false,
        readiness: "experimental"
    ),
]

public func ttsCapabilities(for variant: ModelVariant) -> TTSModelCapabilities? {
    guard variant.kind == .tts else { return nil }
    return TTS_MODEL_CAPABILITIES[variant.name]
}

/// Look up a model name (canonical or alias) in the registry.
///
/// Case-insensitive. Returns `nil` for empty / unknown names so the
/// session.update handler can apply forward-compat fall-through (keep the
/// current engines, echo the unknown name back).
public func resolveModelVariant(_ name: String) -> ModelVariant? {
    let lower = name.lowercased()
    guard !lower.isEmpty else { return nil }
    // First pass: exact canonical name match.
    if let exact = MODEL_REGISTRY.first(where: { $0.name == lower }) {
        return exact
    }
    // Second pass: alias match. Within a kind, the first row that owns the
    // alias wins — that's how "qwen3" maps to the 0.6B INT4 default rather
    // than the 1.7B INT8 variant.
    return MODEL_REGISTRY.first(where: { variant in
        variant.aliases.contains(lower)
    })
}

/// Find the default variant for an engine string. Used by the session
/// defaults — we want to point at the lightweight Parakeet/Kokoro variants
/// without hard-coding their HF slugs in `RealtimeSession`.
public func defaultVariant(forEngine engine: String, kind: ModelVariant.Kind) -> ModelVariant {
    if let v = MODEL_REGISTRY.first(where: { $0.engine == engine && $0.kind == kind }) {
        return v
    }
    fatalError("No registered variant for engine '\(engine)' (kind: \(kind.rawValue))")
}
