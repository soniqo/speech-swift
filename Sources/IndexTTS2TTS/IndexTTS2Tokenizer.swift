import AudioCommon
import Foundation

public struct IndexTTS2Token: Equatable, Sendable {
    public let id: Int
    public let piece: String

    public init(id: Int, piece: String) {
        self.id = id
        self.piece = piece
    }
}

public enum IndexTTS2TokenizerError: Error, LocalizedError, Equatable {
    case emptyVocabulary
    case textTooLong(maxScalars: Int)
    case unencodableText(String)

    public var errorDescription: String? {
        switch self {
        case .emptyVocabulary:
            return "IndexTTS2 tokenizer vocabulary is empty."
        case .textTooLong(let maxScalars):
            return "IndexTTS2 tokenizer input is too long; max \(maxScalars) Unicode scalars."
        case .unencodableText(let text):
            return "IndexTTS2 tokenizer could not encode text: \(text)"
        }
    }
}

/// SentencePiece tokenizer scaffold for IndexTTS2's `bpe.model`.
///
/// Upstream applies additional Chinese text normalization before SentencePiece.
/// This Swift port starts with the model-compatible Unigram Viterbi path and
/// mirrors the upstream Latin uppercasing used before BPE lookup.
public struct IndexTTS2Tokenizer: Sendable {
    public static let maxInputScalars = 2_048

    private let pieces: [SentencePieceModel.Piece]
    private let tokenToId: [String: Int]

    public init(modelURL: URL) throws {
        let model = try SentencePieceModel(contentsOf: modelURL)
        try self.init(pieces: model.pieces)
    }

    init(pieces: [SentencePieceModel.Piece]) throws {
        guard !pieces.isEmpty else {
            throw IndexTTS2TokenizerError.emptyVocabulary
        }
        self.pieces = pieces
        self.tokenToId = Dictionary(
            uniqueKeysWithValues: pieces.enumerated().map { ($0.element.text, $0.offset) }
        )
    }

    public func tokenize(_ text: String) throws -> [IndexTTS2Token] {
        try encode(text).map { id in
            IndexTTS2Token(id: id, piece: pieces[id].text)
        }
    }

    public func encode(_ text: String) throws -> [Int] {
        let normalized = normalizedPieceText(for: text)
        guard !normalized.isEmpty else { return [] }

        let scalars = Array(normalized.unicodeScalars)
        guard scalars.count <= Self.maxInputScalars else {
            throw IndexTTS2TokenizerError.textTooLong(maxScalars: Self.maxInputScalars)
        }

        var bestScores = [Float](repeating: -.infinity, count: scalars.count + 1)
        var backPointer = [(start: Int, id: Int)?](repeating: nil, count: scalars.count + 1)
        bestScores[0] = 0

        for start in 0..<scalars.count where bestScores[start].isFinite {
            for end in (start + 1)...scalars.count {
                let piece = String(String.UnicodeScalarView(scalars[start..<end]))
                guard let id = tokenToId[piece] else { continue }
                guard !pieces[id].isControlOrUnknown else { continue }

                let score = bestScores[start] + pieces[id].score
                if score > bestScores[end] || (score == bestScores[end] && isTieBreakBetter(id, start, than: backPointer[end])) {
                    bestScores[end] = score
                    backPointer[end] = (start, id)
                }
            }
        }

        guard bestScores[scalars.count].isFinite else {
            throw IndexTTS2TokenizerError.unencodableText(text)
        }

        var ids: [Int] = []
        var cursor = scalars.count
        while cursor > 0, let pointer = backPointer[cursor] {
            ids.append(pointer.id)
            cursor = pointer.start
        }
        return ids.reversed()
    }

    public func decode(_ ids: [Int]) -> String {
        ids.compactMap { id in
            guard id >= 0, id < pieces.count else { return nil }
            let piece = pieces[id]
            return piece.isControlOrUnknown ? nil : piece.text
        }
        .joined()
        .replacingOccurrences(of: "▁", with: " ")
        .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func normalizedPieceText(for text: String) -> String {
        let normalized = (text as NSString)
            .precomposedStringWithCompatibilityMapping
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .split(whereSeparator: { $0.isWhitespace })
            .joined(separator: " ")
            .uppercased()
        guard !normalized.isEmpty else { return "" }
        return "▁" + normalized.replacingOccurrences(of: " ", with: "▁")
    }

    private func isTieBreakBetter(
        _ candidateId: Int,
        _ candidateStart: Int,
        than existing: (start: Int, id: Int)?
    ) -> Bool {
        guard let existing else { return true }
        return candidateId < existing.id ||
            (candidateId == existing.id && candidateStart > existing.start)
    }
}
