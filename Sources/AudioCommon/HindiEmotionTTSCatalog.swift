import Foundation

/// License posture for a candidate model.
///
/// This is intentionally separate from the raw license string because some
/// Hugging Face cards use `other` even when the model card explains a stricter
/// research-only license.
enum ModelUsePolicy: String, Sendable, Equatable {
    /// Model can be considered for default product/runtime integration.
    case commercialSafe
    /// Model is useful for quality benchmarking but not for default product use.
    case researchOnly
    /// Model metadata is incomplete or still needs legal/product review.
    case needsReview
}

/// How a model expects emotion or style control to be represented in text.
enum EmotionMarkerSyntax: String, Sendable, Equatable {
    /// Tags are appended to the utterance, e.g. `text <happy>`.
    case suffixAngleTag
    /// Tags are embedded inline, e.g. `[whisper] text [sad]`.
    case inlineBracketTag
    /// Control is supplied as a descriptive prompt/caption rather than a strict tag.
    case descriptivePrompt
    /// No explicit emotion marker support is documented today.
    case noneDocumented
}

/// How close a candidate is to a shippable speech-swift runtime.
enum CandidateReadiness: String, Sendable, Equatable {
    /// Candidate should be evaluated first for a runtime port.
    case primaryPortCandidate
    /// Candidate is promising, but needs API or quality validation before a port.
    case secondaryPortCandidate
    /// Candidate should be used for comparisons only because licensing blocks shipping.
    case benchmarkOnly
    /// Candidate is tracked for context, but is not an emotion-marker solution today.
    case trackedOnly
}

/// Voice conditioning style exposed by the upstream project.
enum VoiceConditioningMode: String, Sendable, Equatable {
    /// Runtime takes a short reference audio / speaker embedding path.
    case referenceOrEmbedding
    /// Runtime exposes fixed or described voices, not true zero-shot cloning.
    case presetOrDescriptiveVoice
    /// Runtime primarily expects fine-tuning or adaptation for new voices.
    case adaptationPath
    /// Voice-conditioning status is not clear enough to rely on.
    case unknown
}

/// A Hindi-capable TTS model candidate with explicit emotion/style control.
struct HindiEmotionTTSCandidate: Sendable, Equatable {
    let id: String
    let displayName: String
    let modelId: String
    let upstreamURL: URL
    let license: String
    let usePolicy: ModelUsePolicy
    let readiness: CandidateReadiness
    let supportsHindi: Bool
    let supportsExplicitEmotionMarkers: Bool
    let markerSyntax: EmotionMarkerSyntax
    let supportedMarkers: [String]
    let voiceConditioning: VoiceConditioningMode
    let notes: [String]

    init(
        id: String,
        displayName: String,
        modelId: String,
        upstreamURL: URL,
        license: String,
        usePolicy: ModelUsePolicy,
        readiness: CandidateReadiness,
        supportsHindi: Bool,
        supportsExplicitEmotionMarkers: Bool,
        markerSyntax: EmotionMarkerSyntax,
        supportedMarkers: [String],
        voiceConditioning: VoiceConditioningMode,
        notes: [String]
    ) {
        self.id = id
        self.displayName = displayName
        self.modelId = modelId
        self.upstreamURL = upstreamURL
        self.license = license
        self.usePolicy = usePolicy
        self.readiness = readiness
        self.supportsHindi = supportsHindi
        self.supportsExplicitEmotionMarkers = supportsExplicitEmotionMarkers
        self.markerSyntax = markerSyntax
        self.supportedMarkers = supportedMarkers
        self.voiceConditioning = voiceConditioning
        self.notes = notes
    }

    var isEligibleForDefaultIntegration: Bool {
        supportsHindi
            && supportsExplicitEmotionMarkers
            && usePolicy == .commercialSafe
            && readiness != .trackedOnly
            && markerSyntax != .noneDocumented
    }
}

/// Hindi emotion-TTS candidates tracked before committing to a full runtime port.
enum HindiEmotionTTSCatalog {
    static let all: [HindiEmotionTTSCandidate] = [
        .init(
            id: "indic-mio",
            displayName: "Indic-Mio",
            modelId: "SPRINGLab/Indic-Mio",
            upstreamURL: URL(string: "https://huggingface.co/SPRINGLab/Indic-Mio")!,
            license: "Apache-2.0",
            usePolicy: .commercialSafe,
            readiness: .primaryPortCandidate,
            supportsHindi: true,
            supportsExplicitEmotionMarkers: true,
            markerSyntax: .suffixAngleTag,
            supportedMarkers: ["<happy>", "<sad>", "<angry>", "<disgust>", "<fear>", "<surprise>"],
            voiceConditioning: .referenceOrEmbedding,
            notes: [
                "0.6B Qwen-style TTS model for all 22 scheduled Indian languages plus English.",
                "Model card documents end-of-sentence emotion/style tags for Indian languages.",
                "Model card describes zero-shot voice cloning via speaker embeddings in the codec.",
            ]
        ),
        .init(
            id: "svara-tts-v1",
            displayName: "Svara-TTS v1",
            modelId: "kenpath/svara-tts-v1",
            upstreamURL: URL(string: "https://huggingface.co/kenpath/svara-tts-v1")!,
            license: "Apache-2.0",
            usePolicy: .commercialSafe,
            readiness: .secondaryPortCandidate,
            supportsHindi: true,
            supportsExplicitEmotionMarkers: true,
            markerSyntax: .suffixAngleTag,
            supportedMarkers: ["<happy>", "<sad>", "<anger>", "<fear>"],
            voiceConditioning: .adaptationPath,
            notes: [
                "Indic-focused model covering 19 Indian languages including Hindi.",
                "Model card documents lightweight emotion/style tags and zero-shot adaptation paths.",
                "Speaker similarity needs validation before treating it as a drop-in clone engine.",
            ]
        ),
        .init(
            id: "fish-audio-s2-pro",
            displayName: "Fish Audio S2 Pro",
            modelId: "fishaudio/s2-pro",
            upstreamURL: URL(string: "https://huggingface.co/fishaudio/s2-pro")!,
            license: "Fish Audio Research License",
            usePolicy: .researchOnly,
            readiness: .benchmarkOnly,
            supportsHindi: true,
            supportsExplicitEmotionMarkers: true,
            markerSyntax: .inlineBracketTag,
            supportedMarkers: [
                "[pause]", "[emphasis]", "[laughing]", "[excited]", "[angry]",
                "[whisper]", "[screaming]", "[shouting]", "[surprised]", "[sad]",
            ],
            voiceConditioning: .referenceOrEmbedding,
            notes: [
                "Strongest benchmark for fine-grained inline prosody/emotion control.",
                "Model card lists Hindi among supported languages.",
                "Public weights are research/non-commercial; commercial use requires a separate license.",
            ]
        ),
        .init(
            id: "indic-parler-tts",
            displayName: "Indic Parler-TTS",
            modelId: "ai4bharat/indic-parler-tts",
            upstreamURL: URL(string: "https://huggingface.co/ai4bharat/indic-parler-tts")!,
            license: "Apache-2.0",
            usePolicy: .commercialSafe,
            readiness: .secondaryPortCandidate,
            supportsHindi: true,
            supportsExplicitEmotionMarkers: true,
            markerSyntax: .descriptivePrompt,
            supportedMarkers: [
                "anger", "disgust", "fear", "happy", "neutral", "sad", "surprise",
            ],
            voiceConditioning: .presetOrDescriptiveVoice,
            notes: [
                "Caption-driven expressive TTS rather than strict inline tags.",
                "Model card says Hindi is supported, but Hindi emotion rendering is not one of the officially tested emotion languages.",
                "Useful as a comparison point if description-prompt control is acceptable.",
            ]
        ),
        .init(
            id: "indicf5",
            displayName: "IndicF5",
            modelId: "ai4bharat/IndicF5",
            upstreamURL: URL(string: "https://huggingface.co/ai4bharat/IndicF5")!,
            license: "MIT",
            usePolicy: .commercialSafe,
            readiness: .trackedOnly,
            supportsHindi: true,
            supportsExplicitEmotionMarkers: false,
            markerSyntax: .noneDocumented,
            supportedMarkers: [],
            voiceConditioning: .referenceOrEmbedding,
            notes: [
                "Commercial-safe Indic voice-cloning baseline.",
                "No explicit emotion-marker interface is documented, so it is not an answer to marker-driven acting.",
            ]
        ),
        .init(
            id: "orpheus-tts-hi",
            displayName: "Orpheus TTS Hindi",
            modelId: "SachinTelecmi/Orpheus-tts-hi",
            upstreamURL: URL(string: "https://huggingface.co/SachinTelecmi/Orpheus-tts-hi")!,
            license: "Apache-2.0",
            usePolicy: .needsReview,
            readiness: .trackedOnly,
            supportsHindi: true,
            supportsExplicitEmotionMarkers: false,
            markerSyntax: .noneDocumented,
            supportedMarkers: [],
            voiceConditioning: .unknown,
            notes: [
                "Hindi Orpheus derivative with documented non-verbal tokens such as breath/laugh style controls.",
                "Model card lists emotion/prosody control tokens as future work, not current capability.",
                "Card metadata references proprietary data, so review before product integration.",
            ]
        ),
    ]

    static var implementationCandidates: [HindiEmotionTTSCandidate] {
        all.filter(\.isEligibleForDefaultIntegration)
    }

    static var researchBenchmarks: [HindiEmotionTTSCandidate] {
        all.filter { $0.readiness == .benchmarkOnly }
    }

    static func candidate(id: String) -> HindiEmotionTTSCandidate? {
        all.first { $0.id == id || $0.modelId.caseInsensitiveCompare(id) == .orderedSame }
    }
}
