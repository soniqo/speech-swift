import Foundation
import MLX

public struct FishAudioModelInput: Sendable, Equatable {
    public let rows: [[Int32]]

    public var rowCount: Int { rows.count }
    public var tokenCount: Int { rows.first?.count ?? 0 }

    public init(rows: [[Int32]]) throws {
        guard !rows.isEmpty else {
            throw FishAudioError.invalidCodebookShape("model input must have at least one row")
        }
        let width = rows[0].count
        guard rows.allSatisfy({ $0.count == width }) else {
            throw FishAudioError.invalidCodebookShape("model input rows must have the same length")
        }
        self.rows = rows
    }

    public func asMLXArray() -> MLXArray {
        MLXArray(rows.flatMap { $0 }).reshaped([1, rowCount, tokenCount])
    }
}

public enum FishAudioInputBuilder {
    public static func build(
        text: String,
        references: [FishAudioReferencePrompt] = [],
        tokenizer: FishAudioTokenizer,
        config: FishAudioConfig
    ) throws -> FishAudioModelInput {
        var builder = RowBuilder(numCodebooks: config.audioDecoder.numCodebooks)

        if references.isEmpty {
            builder.appendText(
                tokenizer.encode("\(FishAudioToken.imStart)system\n\(FishAudioPrompt.systemPrompt(referenceTexts: []))\(FishAudioToken.imEnd)\n"))
        } else {
            let referenceTexts = references.map(\.text)
            let systemPrefix = """
            \(FishAudioToken.imStart)system
            convert the provided text to speech reference to the following:

            Text:
            """
            builder.appendText(tokenizer.encode(systemPrefix))
            builder.appendText(tokenizer.encode(taggedReferenceText(referenceTexts)))
            builder.appendText(tokenizer.encode("\n\nSpeech:\n"))
            for reference in references {
                try builder.appendVQ(reference.codes, semanticStartTokenId: config.semanticStartTokenId)
            }
            builder.appendText(tokenizer.encode("\(FishAudioToken.imEnd)\n"))
        }

        builder.appendText(tokenizer.encode("\(FishAudioToken.imStart)user\n\(text)\(FishAudioToken.imEnd)\n"))
        builder.appendText(tokenizer.encode("\(FishAudioToken.imStart)assistant\n\(FishAudioToken.voiceModality)"))
        return try builder.build()
    }

    static func taggedReferenceText(_ texts: [String]) -> String {
        texts.enumerated().map { index, text in
            text.contains("<|speaker:") ? text : "<|speaker:\(index)|>\(text)"
        }.joined(separator: "\n")
    }
}

private struct RowBuilder {
    private var rows: [[Int32]]

    init(numCodebooks: Int) {
        self.rows = Array(repeating: [], count: numCodebooks + 1)
    }

    mutating func appendText(_ tokenIds: [Int]) {
        for tokenId in tokenIds {
            rows[0].append(Int32(tokenId))
            for row in 1..<rows.count {
                rows[row].append(0)
            }
        }
    }

    mutating func appendVQ(_ codes: [[Int]], semanticStartTokenId: Int) throws {
        guard codes.count == rows.count - 1 else {
            throw FishAudioError.invalidCodebookShape(
                "reference codebook count \(codes.count) does not match model count \(rows.count - 1)")
        }
        guard let frameCount = codes.first?.count, frameCount > 0 else {
            throw FishAudioError.invalidCodebookShape("reference codebook frames must not be empty")
        }
        guard codes.allSatisfy({ $0.count == frameCount }) else {
            throw FishAudioError.invalidCodebookShape("all codebooks must have the same frame count")
        }

        for frame in 0..<frameCount {
            let semanticCode = codes[0][frame]
            rows[0].append(Int32(semanticStartTokenId + semanticCode))
            for codebook in 0..<codes.count {
                rows[codebook + 1].append(Int32(codes[codebook][frame]))
            }
        }
    }

    func build() throws -> FishAudioModelInput {
        try FishAudioModelInput(rows: rows)
    }
}
