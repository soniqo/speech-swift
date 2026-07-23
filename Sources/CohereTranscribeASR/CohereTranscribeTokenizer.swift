import AudioCommon
import Foundation

public final class CohereTranscribeTokenizer: @unchecked Sendable {
    private let pieces: [SentencePieceModel.Piece]
    private let specialTokenToID: [String: Int]
    private let specialIDs: Set<Int>

    public init(modelDirectory: URL) throws {
        let sentencePiece = try SentencePieceModel(
            contentsOf: modelDirectory.appendingPathComponent("tokenizer.model"))
        pieces = sentencePiece.pieces

        let configURL = modelDirectory.appendingPathComponent("tokenizer_config.json")
        let config = try JSONDecoder().decode(
            CohereTokenizerConfig.self,
            from: Data(contentsOf: configURL))
        specialTokenToID = config.addedTokensDecoder.reduce(into: [:]) { result, entry in
            if let id = Int(entry.key) { result[entry.value.content] = id }
        }
        specialIDs = Set(specialTokenToID.values)
    }

    init(pieces: [SentencePieceModel.Piece], specialTokenToID: [String: Int]) {
        self.pieces = pieces
        self.specialTokenToID = specialTokenToID
        self.specialIDs = Set(specialTokenToID.values)
    }

    public var eosTokenID: Int {
        specialTokenToID["<|endoftext|>"] ?? 0
    }

    public func decode(tokens: [Int]) -> String {
        var bytes: [UInt8] = []
        for id in tokens {
            guard !specialIDs.contains(id), id >= 0, id < pieces.count else { continue }
            let piece = pieces[id]
            guard !piece.isControlOrUnknown || piece.pieceType == .byte else { continue }
            if piece.pieceType == .byte {
                if let byte = Self.decodeBytePiece(piece.text) { bytes.append(byte) }
            } else {
                bytes.append(contentsOf: piece.text.utf8)
            }
        }
        return String(decoding: bytes, as: UTF8.self)
        .replacingOccurrences(of: "▁", with: " ")
        .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public func buildPromptTokens(
        language: String,
        usePunctuation: Bool = true,
        useTimestamps: Bool = false
    ) -> [Int] {
        let languageToken = Self.languageToken(for: language)
        return [
            "<|startofcontext|>",
            "<|startoftranscript|>",
            "<|emo:undefined|>",
            languageToken,
            languageToken,
            usePunctuation ? "<|pnc|>" : "<|nopnc|>",
            "<|noitn|>",
            useTimestamps ? "<|timestamp|>" : "<|notimestamp|>",
            "<|nodiarize|>",
        ].compactMap { specialTokenToID[$0] }
    }

    public static func languageToken(for language: String) -> String {
        let code = language.lowercased().split(separator: "-").first.map(String.init) ?? "en"
        let aliases: [String: String] = [
            "english": "en", "french": "fr", "german": "de", "spanish": "es",
            "italian": "it", "portuguese": "pt", "dutch": "nl", "polish": "pl",
            "greek": "el", "arabic": "ar", "japanese": "ja", "chinese": "zh",
            "vietnamese": "vi", "korean": "ko",
        ]
        let supported = Set(["en", "fr", "de", "es", "it", "pt", "nl", "pl", "el", "ar", "ja", "zh", "vi", "ko"])
        let resolved = aliases[code] ?? (supported.contains(code) ? code : "en")
        return "<|\(resolved)|>"
    }

    private static func decodeBytePiece(_ piece: String) -> UInt8? {
        guard piece.hasPrefix("<0x"), piece.hasSuffix(">"), piece.count == 6 else { return nil }
        let start = piece.index(piece.startIndex, offsetBy: 3)
        let end = piece.index(start, offsetBy: 2)
        return UInt8(piece[start..<end], radix: 16)
    }
}

private struct CohereTokenizerConfig: Decodable {
    struct AddedToken: Decodable { let content: String }
    let addedTokensDecoder: [String: AddedToken]

    enum CodingKeys: String, CodingKey {
        case addedTokensDecoder = "added_tokens_decoder"
    }
}
