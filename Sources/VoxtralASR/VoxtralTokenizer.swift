import Foundation

private struct VoxtralTekkenFile: Decodable {
    struct Config: Decodable {
        let defaultNumSpecialTokens: Int

        enum CodingKeys: String, CodingKey {
            case defaultNumSpecialTokens = "default_num_special_tokens"
        }
    }
    struct SpecialToken: Decodable {
        let rank: Int
        let tokenString: String

        enum CodingKeys: String, CodingKey {
            case rank
            case tokenString = "token_str"
        }
    }
    struct VocabEntry: Decodable {
        let tokenBytes: String

        enum CodingKeys: String, CodingKey {
            case tokenBytes = "token_bytes"
        }
    }
    let config: Config
    let specialTokens: [SpecialToken]
    let vocab: [VocabEntry]

    enum CodingKeys: String, CodingKey {
        case config, vocab
        case specialTokens = "special_tokens"
    }
}

public final class VoxtralTokenizer: @unchecked Sendable {
    public static let supportedLanguages = ["en", "fr", "de", "es", "it", "pt", "nl", "hi"]

    private let vocab: [VoxtralTekkenFile.VocabEntry]
    private let specialTokens: [String: Int]
    private let specialIDs: Set<Int>
    private let numberOfSpecialTokens: Int
    private var byteCache: [Int: [UInt8]] = [:]

    public init(tekkenURL: URL) throws {
        let parsed = try JSONDecoder().decode(
            VoxtralTekkenFile.self,
            from: Data(contentsOf: tekkenURL))
        vocab = parsed.vocab
        numberOfSpecialTokens = parsed.config.defaultNumSpecialTokens
        specialTokens = Dictionary(uniqueKeysWithValues: parsed.specialTokens.map {
            ($0.tokenString, $0.rank)
        })
        specialIDs = Set(parsed.specialTokens.map(\.rank))
    }

    public func transcriptionPrompt(audioTokenCount: Int, language: String?) -> [Int] {
        let bos = specialTokens["<s>"] ?? 1
        let beginInstruction = specialTokens["[INST]"] ?? 3
        let endInstruction = specialTokens["[/INST]"] ?? 4
        let beginAudio = specialTokens["[BEGIN_AUDIO]"] ?? 25
        let audio = specialTokens["[AUDIO]"] ?? 24
        let transcribe = specialTokens["[TRANSCRIBE]"] ?? 34
        var tokens = [bos, beginInstruction, beginAudio]
        tokens.append(contentsOf: repeatElement(audio, count: max(0, audioTokenCount)))
        tokens.append(endInstruction)
        if let language, let hint = Self.languagePromptTokens(language) {
            tokens.append(contentsOf: hint)
        }
        tokens.append(transcribe)
        return tokens
    }

    /// The checkpoint's code for an ISO code, a `language-REGION` tag or an
    /// English language name, or `nil` for a language the prompt cannot name.
    public static func languageCode(_ language: String) -> String? {
        let aliases = [
            "english": "en", "french": "fr", "german": "de", "spanish": "es",
            "italian": "it", "portuguese": "pt", "dutch": "nl", "hindi": "hi",
        ]
        let raw = language.lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .split(whereSeparator: { $0 == "-" || $0 == "_" })
            .first.map(String.init) ?? ""
        let code = aliases[raw] ?? raw
        return supportedLanguages.contains(code) ? code : nil
    }

    public static func supportsLanguage(_ language: String) -> Bool {
        languageCode(language) != nil
    }

    /// The `lang:<code>` tokens for a supported language, or `nil` for any
    /// other. Earlier builds mapped an unknown code to English, which told
    /// the model a Russian or Chinese utterance was English; leaving the hint
    /// out lets the model choose, which is what it does for those languages.
    public static func languagePromptTokens(_ language: String) -> [Int]? {
        guard let code = languageCode(language) else { return nil }
        let table: [String: [Int]] = [
            "en": [9_909, 1_058, 1_262],
            "fr": [9_909, 1_058, 7_064],
            "de": [9_909, 1_058, 1_558],
            "es": [9_909, 1_058, 1_264],
            "it": [9_909, 1_058, 1_276],
            "pt": [9_909, 1_058, 1_515],
            "nl": [9_909, 24_082, 1_108],
            "hi": [9_909, 1_058, 8_101],
        ]
        return table[code]
    }

    public func decode(_ tokenIDs: [Int]) -> String {
        var bytes: [UInt8] = []
        for id in tokenIDs {
            guard id >= numberOfSpecialTokens, !specialIDs.contains(id) else { continue }
            bytes.append(contentsOf: tokenBytes(id))
        }
        return String(decoding: bytes, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func tokenBytes(_ tokenID: Int) -> [UInt8] {
        if let cached = byteCache[tokenID] { return cached }
        let index = tokenID - numberOfSpecialTokens
        guard index >= 0, index < vocab.count else { return [] }
        let bytes = [UInt8](Data(base64Encoded: vocab[index].tokenBytes) ?? Data())
        byteCache[tokenID] = bytes
        return bytes
    }
}
