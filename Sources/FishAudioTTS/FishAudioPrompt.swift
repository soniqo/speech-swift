import Foundation

public enum FishAudioEmotionMarker {
    public static let supported: [String] = [
        "[pause]",
        "[emphasis]",
        "[laughing]",
        "[excited]",
        "[angry]",
        "[whisper]",
        "[screaming]",
        "[shouting]",
        "[surprised]",
        "[sad]",
    ]
}

public struct FishAudioReferencePrompt: Sendable, Equatable {
    public let text: String
    public let codes: [[Int]]

    public init(text: String, codes: [[Int]]) throws {
        guard !codes.isEmpty else {
            throw FishAudioError.invalidCodebookShape("reference codes must not be empty")
        }
        let width = codes[0].count
        guard width > 0 else {
            throw FishAudioError.invalidCodebookShape("reference codebook frames must not be empty")
        }
        guard codes.allSatisfy({ $0.count == width }) else {
            throw FishAudioError.invalidCodebookShape("all codebooks must have the same frame count")
        }
        self.text = text
        self.codes = codes
    }
}

public enum FishAudioPrompt {
    public static func systemPrompt(referenceTexts: [String]) -> String {
        guard !referenceTexts.isEmpty else {
            return "convert the provided text to speech"
        }
        let referenceText = referenceTexts.enumerated().map { index, text in
            text.contains("<|speaker:") ? text : "<|speaker:\(index)|>\(text)"
        }.joined(separator: "\n")
        return """
        convert the provided text to speech reference to the following:

        Text:
        \(referenceText)

        Speech:
        """
    }

    public static func userPrompt(_ text: String) -> String {
        text
    }

    public static func chatTemplate(system: String, user: String) -> String {
        """
        \(FishAudioToken.imStart)system
        \(system)\(FishAudioToken.imEnd)
        \(FishAudioToken.imStart)user
        \(user)\(FishAudioToken.imEnd)
        \(FishAudioToken.imStart)assistant
        \(FishAudioToken.voiceModality)
        """
    }
}
