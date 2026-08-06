import AudioCommon
import Foundation
import Hub
import MLX

/// Complete local VoiceChat 11B runtime.
///
/// All five published components are loaded as one unit so a bundle missing
/// EAR-TTS or codec weights fails immediately instead of degrading to the
/// understanding-only path.
public final class VoiceChatModel {
    public static let defaultModelID =
        "aufklarer/VoiceChat-11B-Perception-MLX-int8"

    public let perception: VoiceChatPerception
    public let languageModel: VoiceChatLanguageModel
    public let speechDecoder: VoiceChatSpeechDecoder
    public let codec: VoiceChatCodec
    public let tokenizer: VoiceChatTokenizer
    public let transcriber: VoiceChatTranscriber

    private init(
        perception: VoiceChatPerception,
        languageModel: VoiceChatLanguageModel,
        speechDecoder: VoiceChatSpeechDecoder,
        codec: VoiceChatCodec,
        tokenizer: VoiceChatTokenizer,
        transcriber: VoiceChatTranscriber
    ) {
        self.perception = perception
        self.languageModel = languageModel
        self.speechDecoder = speechDecoder
        self.codec = codec
        self.tokenizer = tokenizer
        self.transcriber = transcriber
    }

    public static func load(from root: URL) async throws -> VoiceChatModel {
        let encoderDirectory = root.appendingPathComponent("encoder")
        let languageDirectory = root.appendingPathComponent("llm")
        let speechDirectory = root.appendingPathComponent("tts")

        let tokenizer = try await VoiceChatTokenizer.load(from: languageDirectory)

        let encoderWeightsURL = encoderDirectory.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: encoderWeightsURL.path) else {
            throw VoiceChatLoadError.missingWeights(encoderWeightsURL)
        }
        let encoderWeights = try MLX.loadArrays(url: encoderWeightsURL)
        let perception = try VoiceChatPerception.load(
            configuration: VoiceChatPerceptionConfig.load(from: encoderDirectory),
            weights: encoderWeights)
        let transcriber = try VoiceChatTranscriber(
            perception: perception, weights: encoderWeights)

        let languageModel = try VoiceChatLanguageModel.load(from: languageDirectory)
        guard languageModel.functionHead != nil else {
            throw VoiceChatLoadError.unexpectedKeys(["function_head.weight"])
        }

        let speechConfiguration = try VoiceChatSpeechConfiguration.load(from: speechDirectory)
        let speechWeightsURL = speechDirectory.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: speechWeightsURL.path) else {
            throw VoiceChatLoadError.missingWeights(speechWeightsURL)
        }
        let speechWeights = try MLX.loadArrays(url: speechWeightsURL)
        let speechDecoder = try VoiceChatSpeechDecoder(
            weights: speechWeights,
            configuration: speechConfiguration,
            tokenizer: tokenizer)
        let codec = try VoiceChatCodec(weights: speechWeights)

        // Seconds here save hours later: this catches missing/promoted codec
        // tensors and the historic magnitude/phase inversion before a session.
        let silence = codec.verifySilence()
        guard silence.passed else {
            throw VoiceChatGenerationError.invalidSpeechConfiguration(
                "canonical silence decoded to RMS \(silence.rms), peak \(silence.peak)")
        }

        return VoiceChatModel(
            perception: perception,
            languageModel: languageModel,
            speechDecoder: speechDecoder,
            codec: codec,
            tokenizer: tokenizer,
            transcriber: transcriber)
    }

    /// Download a complete bundle from Hugging Face and load it locally.
    public static func loadFromHub(
        _ modelID: String = defaultModelID,
        revision: String = "main",
        progressHandler: @escaping (Progress) -> Void = { _ in }
    ) async throws -> VoiceChatModel {
        let hub = HubApi(endpoint: HuggingFaceDownloader.resolvedEndpoint())
        let directory = try await hub.snapshot(
            from: modelID, revision: revision, progressHandler: progressHandler)
        return try await load(from: directory)
    }

    /// Start a stateful full-duplex conversation. Generation caches are owned
    /// by the returned session, so the loaded weights can be reused.
    public func startSession(
        systemPrompt: String = VoiceChatSession.defaultSystemPrompt,
        sampling: VoiceChatTextSamplingParameters = .init(),
        speech: VoiceChatSpeechGenerationParameters = .init()
    ) async throws -> VoiceChatSession {
        try await VoiceChatSession.create(
            model: self, systemPrompt: systemPrompt,
            sampling: sampling, speechParameters: speech)
    }
}
