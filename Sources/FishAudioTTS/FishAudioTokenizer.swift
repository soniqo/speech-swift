import Foundation
import Hub
import Tokenizers

public enum FishAudioToken {
    public static let eos = "<|endoftext|>"
    public static let pad = "<|pad|>"
    public static let imStart = "<|im_start|>"
    public static let imEnd = "<|im_end|>"
    public static let textModality = "<|text|>"
    public static let voiceModality = "<|voice|>"
    public static let interleaveModality = "<|interleave|>"
    public static let audioStart = "<|audio_start|>"
    public static let audioEnd = "<|audio_end|>"
    public static let audioPad = "<|audio_pad|>"

    public static func semantic(_ index: Int) -> String {
        "<|semantic:\(index)|>"
    }
}

public final class FishAudioTokenizer: @unchecked Sendable {
    private let tokenizer: Tokenizer
    public let metadata: FishAudioTokenizerMetadata

    public init(tokenizer: Tokenizer, metadata: FishAudioTokenizerMetadata) {
        self.tokenizer = tokenizer
        self.metadata = metadata
    }

    public static func load(from directory: URL) async throws -> FishAudioTokenizer {
        do {
            let metadata = try FishAudioTokenizerMetadata.load(from: directory)
            let dataURL = directory.appendingPathComponent("tokenizer.json")
            let configURL = directory.appendingPathComponent("tokenizer_config.json")
            guard FileManager.default.fileExists(atPath: dataURL.path) else {
                return FishAudioTokenizer(
                    tokenizer: try await AutoTokenizer.from(modelFolder: directory, strict: false),
                    metadata: metadata)
            }

            let hubApi = HubApi()
            let tokenizerData = try hubApi.configuration(fileURL: dataURL)
            let tokenizerConfig: Config
            if FileManager.default.fileExists(atPath: configURL.path) {
                tokenizerConfig = try hubApi.configuration(fileURL: configURL)
            } else {
                tokenizerConfig = Config([String: Config]())
            }
            return FishAudioTokenizer(
                tokenizer: try AutoTokenizer.from(
                    tokenizerConfig: tokenizerConfig,
                    tokenizerData: tokenizerData,
                    strict: false),
                metadata: metadata)
        } catch let error as FishAudioError {
            throw error
        } catch {
            throw FishAudioError.tokenizerLoadFailed(error.localizedDescription)
        }
    }

    public func tokenId(_ token: String) -> Int? {
        tokenizer.convertTokenToId(token)
    }

    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }

    public func decode(_ ids: [Int], skipSpecialTokens: Bool = false) -> String {
        tokenizer.decode(tokens: ids, skipSpecialTokens: skipSpecialTokens)
    }
}

public struct FishAudioTokenizerMetadata: Sendable, Equatable {
    public let vocab: [String: Int]
    public let semanticBeginId: Int
    public let semanticEndId: Int
    public let semanticTokenIds: [Int: Int]
    public let specialTokenIds: [String: Int]

    public static func load(from directory: URL) throws -> FishAudioTokenizerMetadata {
        let url = directory.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw FishAudioError.missingFile(url)
        }
        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = root["model"] as? [String: Any],
              let rawVocab = model["vocab"] as? [String: Any] else {
            throw FishAudioError.malformedTokenizerJSON
        }
        var vocab: [String: Int] = [:]
        vocab.reserveCapacity(rawVocab.count + FishAudioDefaults.codebookSize)
        for (token, value) in rawVocab {
            guard let id = (value as? NSNumber)?.intValue else {
                throw FishAudioError.malformedTokenizerJSON
            }
            vocab[token] = id
        }
        if let addedTokens = root["added_tokens"] as? [[String: Any]] {
            for token in addedTokens {
                guard let content = token["content"] as? String,
                      let id = (token["id"] as? NSNumber)?.intValue else {
                    throw FishAudioError.malformedTokenizerJSON
                }
                vocab[content] = id
            }
        }
        return try parse(vocab: vocab)
    }

    public static func parse(vocab: [String: Int]) throws -> FishAudioTokenizerMetadata {
        var semanticTokenIds: [Int: Int] = [:]
        var validIds: [Int] = []
        semanticTokenIds.reserveCapacity(FishAudioDefaults.codebookSize)

        for code in 0..<FishAudioDefaults.codebookSize {
            let token = FishAudioToken.semantic(code)
            guard let tokenId = vocab[token] else {
                throw FishAudioError.missingToken(token)
            }
            semanticTokenIds[code] = tokenId
            validIds.append(tokenId)
        }
        let begin = validIds.min() ?? 0
        let end = validIds.max() ?? 0

        let required = [
            FishAudioToken.eos,
            FishAudioToken.pad,
            FishAudioToken.imStart,
            FishAudioToken.imEnd,
            FishAudioToken.textModality,
            FishAudioToken.voiceModality,
            FishAudioToken.interleaveModality,
            FishAudioToken.audioStart,
            FishAudioToken.audioEnd,
            FishAudioToken.audioPad,
        ]
        var special: [String: Int] = [:]
        for token in required {
            guard let id = vocab[token] else {
                throw FishAudioError.missingToken(token)
            }
            special[token] = id
        }

        return FishAudioTokenizerMetadata(
            vocab: vocab,
            semanticBeginId: begin,
            semanticEndId: end,
            semanticTokenIds: semanticTokenIds,
            specialTokenIds: special
        )
    }

    public func tokenId(_ token: String) throws -> Int {
        if let id = vocab[token] {
            return id
        }
        throw FishAudioError.missingToken(token)
    }

    public func semanticTokenId(for code: Int) throws -> Int {
        guard let id = semanticTokenIds[code] else {
            throw FishAudioError.missingToken(FishAudioToken.semantic(code))
        }
        return id
    }
}

public enum FishAudioDefaults {
    public static let modelId = "aufklarer/Fish-Audio-S2-Pro-MLX-fp16"
    public static let sourceModelId = "fishaudio/s2-pro"
    public static let codebookSize = 4_096
    public static let numCodebooks = 10
}
