import Foundation

public enum GLiNERError: Error, LocalizedError {
    case invalidConfiguration(String), invalidSchema(String), missingWeight(String), inputTooLong(Int)
    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let s), .invalidSchema(let s), .missingWeight(let s): return s
        case .inputTooLong(let n): return "Input has \(n) tokens; limit is 512."
        }
    }
}

public struct GLiNERConfig: Codable, Sendable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let intermediateSize: Int
    public let vocabSize: Int
    public let positionBuckets: Int
    public let maxPositionEmbeddings: Int
    public let layerNormEps: Float
    public let relativeAttention: Bool
    public let shareAttKey: Bool
    public let positionBiasedInput: Bool
    public let posAttType: [String]
    public let normRelEbd: String
    public let typeVocabSize: Int
    public var convKernelSize: Int?
    public var hiddenAct: String?
    public func validate() throws {
        guard hiddenSize > 0, numHiddenLayers > 0, numAttentionHeads > 0,
              hiddenSize % numAttentionHeads == 0, intermediateSize > 0, vocabSize > 0,
              layerNormEps.isFinite, layerNormEps > 0,
              positionBuckets > 2, maxPositionEmbeddings > positionBuckets / 2 + 1,
              relativeAttention, shareAttKey, !positionBiasedInput,
              Set(posAttType) == Set(["p2c", "c2p"]), normRelEbd == "layer_norm",
              typeVocabSize == 0, (convKernelSize ?? 0) == 0,
              (hiddenAct ?? "gelu") == "gelu" else {
            throw GLiNERError.invalidConfiguration("Unsupported DeBERTa configuration; expected GLiNER2 span encoder with shared relative attention.")
        }
    }
    public static func load(from url: URL) throws -> Self {
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        let config = try decoder.decode(Self.self, from: Data(contentsOf: url))
        try config.validate(); return config
    }
}

public struct GLiNERChoice: Codable, Sendable {
    public let label: String
    public let probability: Float
}

public struct GLiNERSpan: Codable, Sendable {
    public let text: String
    public let score: Float
    /// UTF-16 offsets into the original input; suitable for NSRange.
    public let start: Int
    public let end: Int
}

struct GLiNERQuantization: Codable {
    let bits: Int
    let groupSize: Int
    let mode: String
    let quantizedKeys: [String]
    func validate() throws {
        guard bits == 8, groupSize == 64, mode == "affine", !quantizedKeys.isEmpty,
              Set(quantizedKeys).count == quantizedKeys.count,
              quantizedKeys.allSatisfy({ $0 == "encoder.embeddings.word_embeddings.weight" ||
                  ($0.hasPrefix("encoder.encoder.layers.") && $0.hasSuffix(".weight")) }) else {
            throw GLiNERError.invalidConfiguration("Expected unique affine INT8 encoder weights with group size 64.")
        }
    }
    static func load(from url: URL) throws -> Self {
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        let value = try decoder.decode(Self.self,from:Data(contentsOf:url))
        try value.validate(); return value
    }
}
