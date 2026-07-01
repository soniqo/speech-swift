import AudioCommon
import Foundation

struct WhisperGenerationConfig: Sendable {
    let startOfTranscriptToken: Int
    let endToken: Int
    let englishToken: Int
    let transcribeToken: Int
    let noTimestampsToken: Int
    let specialTokenBegin: Int
    let beginSuppressTokens: Set<Int>
    let suppressTokens: Set<Int>
    let languageTokensByCode: [String: Int]
    let languageCodesByToken: [Int: String]

    var languageTokenSet: Set<Int> {
        Set(languageTokensByCode.values)
    }

    var transcribeTaskIndex: Int {
        max(0, transcribeToken - 50_359)
    }

    var prefillCacheTokenCount: Int {
        3
    }

    static func load(from modelFolder: URL) throws -> WhisperGenerationConfig {
        let url = modelFolder.appendingPathComponent("generation_config.json")
        let raw: RawGenerationConfig
        do {
            raw = try JSONDecoder().decode(RawGenerationConfig.self, from: Data(contentsOf: url))
        } catch {
            throw AudioModelError.weightLoadingFailed(path: url.path, underlying: error)
        }

        let languages = raw.langToId.reduce(into: [String: Int]()) { result, entry in
            let code = entry.key
                .replacingOccurrences(of: "<|", with: "")
                .replacingOccurrences(of: "|>", with: "")
                .lowercased()
            result[code] = entry.value
        }
        let inverse = Dictionary(uniqueKeysWithValues: languages.map { ($0.value, $0.key) })

        return WhisperGenerationConfig(
            startOfTranscriptToken: raw.decoderStartTokenId ?? 50_258,
            endToken: raw.eosTokenId ?? 50_257,
            englishToken: languages["en"] ?? 50_259,
            transcribeToken: raw.taskToId["transcribe"] ?? 50_360,
            noTimestampsToken: raw.noTimestampsTokenId ?? 50_364,
            specialTokenBegin: raw.eosTokenId ?? 50_257,
            beginSuppressTokens: Set(raw.beginSuppressTokens ?? [220, raw.eosTokenId ?? 50_257]),
            // Match WhisperKit's default DecodingOptions. The broad exported
            // suppress list over-constrains greedy text decoding for this model.
            suppressTokens: [],
            languageTokensByCode: languages,
            languageCodesByToken: inverse)
    }

    func languageToken(for hint: String) -> Int? {
        let normalized = hint
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .split(separator: "-", maxSplits: 1)
            .first
            .map(String.init)
        guard let normalized else { return nil }
        return languageTokensByCode[normalized]
    }

    func languageCode(forToken token: Int) -> String? {
        languageCodesByToken[token]
    }
}

private struct RawGenerationConfig: Decodable {
    let decoderStartTokenId: Int?
    let eosTokenId: Int?
    let noTimestampsTokenId: Int?
    let beginSuppressTokens: [Int]?
    let suppressTokens: [Int]?
    let langToId: [String: Int]
    let taskToId: [String: Int]

    enum CodingKeys: String, CodingKey {
        case decoderStartTokenId = "decoder_start_token_id"
        case eosTokenId = "eos_token_id"
        case noTimestampsTokenId = "no_timestamps_token_id"
        case beginSuppressTokens = "begin_suppress_tokens"
        case suppressTokens = "suppress_tokens"
        case langToId = "lang_to_id"
        case taskToId = "task_to_id"
    }
}
