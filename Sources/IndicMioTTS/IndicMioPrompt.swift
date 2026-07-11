import Foundation

public enum IndicMioPrompt {
    public static let endOfTextTokenId = 151_643
    public static let imStartTokenId = 151_644
    public static let imEndTokenId = 151_645

    public static let indianLanguageEmotionMarkers = [
        "<happy>", "<sad>", "<angry>", "<disgust>", "<fear>", "<surprise>",
    ]

    public static func validateMarkers(in text: String) throws {
        let pattern = #"<[^>\s]+>"#
        let regex = try NSRegularExpression(pattern: pattern)
        let ns = text as NSString
        for match in regex.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            let marker = ns.substring(with: match.range)
            if !indianLanguageEmotionMarkers.contains(marker) {
                throw IndicMioError.unsupportedMarker(marker)
            }
        }
    }

    public static func chatPrompt(for text: String) throws -> String {
        try validateMarkers(in: text)
        return "<|im_start|>user\n\(text)<|im_end|>\n<|im_start|>assistant\n"
    }
}

public enum IndicMioSpeechTokens {
    public static let offset = 151_669
    public static let count = 12_800
    public static let upperBound = offset + count

    public static func speechCode(from tokenId: Int) -> Int? {
        guard tokenId >= offset && tokenId < upperBound else { return nil }
        return tokenId - offset
    }

    public static func speechCodes(from tokenIds: [Int]) -> [Int] {
        tokenIds.compactMap(speechCode(from:))
    }
}

public struct IndicMioSamplingConfig: Sendable, Equatable {
    public var maxNewTokens: Int
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    public var repetitionPenalty: Float

    public init(
        maxNewTokens: Int = 1_024,
        temperature: Float = 0.9,
        topK: Int = 50,
        topP: Float = 0.9,
        repetitionPenalty: Float = 1.0
    ) {
        self.maxNewTokens = maxNewTokens
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
    }

    public static let `default` = IndicMioSamplingConfig()
    public static let greedy = IndicMioSamplingConfig(
        maxNewTokens: 1_024,
        temperature: 0,
        topK: 1,
        topP: 1,
        repetitionPenalty: 1
    )
}
