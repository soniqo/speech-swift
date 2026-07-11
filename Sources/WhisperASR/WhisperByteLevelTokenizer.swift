import AudioCommon
import Foundation

struct WhisperByteLevelTokenizer: Sendable {
    private let idToToken: [Int: String]
    private let specialIds: Set<Int>
    private let byteDecoder: [UInt32: UInt8]

    init(modelFolder: URL) throws {
        let url = modelFolder.appendingPathComponent("tokenizer.json")
        let raw: RawTokenizer
        do {
            raw = try JSONDecoder().decode(RawTokenizer.self, from: Data(contentsOf: url))
        } catch {
            throw AudioModelError.weightLoadingFailed(path: url.path, underlying: error)
        }

        var tokens = raw.model.vocab.reduce(into: [Int: String]()) { result, entry in
            result[entry.value] = entry.key
        }
        var specials = Set<Int>()
        for token in raw.addedTokens ?? [] {
            tokens[token.id] = token.content
            if token.special == true {
                specials.insert(token.id)
            }
        }

        self.idToToken = tokens
        self.specialIds = specials
        self.byteDecoder = Self.makeByteDecoder()
    }

    func decode(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(ids.count * 4)

        for id in ids {
            guard !specialIds.contains(id),
                  let token = idToToken[id],
                  !token.isEmpty
            else {
                continue
            }

            for scalar in token.unicodeScalars {
                if let byte = byteDecoder[scalar.value] {
                    bytes.append(byte)
                } else {
                    bytes.append(contentsOf: String(scalar).utf8)
                }
            }
        }

        return String(decoding: bytes, as: UTF8.self)
    }

    private static func makeByteDecoder() -> [UInt32: UInt8] {
        var bytes = Array(33...126) + Array(161...172) + Array(174...255)
        var chars = bytes
        var next = 0

        for byte in 0...255 where !bytes.contains(byte) {
            bytes.append(byte)
            chars.append(256 + next)
            next += 1
        }

        var decoder: [UInt32: UInt8] = [:]
        for (byte, char) in zip(bytes, chars) {
            decoder[UInt32(char)] = UInt8(byte)
        }
        return decoder
    }
}

private struct RawTokenizer: Decodable {
    let model: RawTokenizerModel
    let addedTokens: [RawAddedToken]?

    enum CodingKeys: String, CodingKey {
        case model
        case addedTokens = "added_tokens"
    }
}

private struct RawTokenizerModel: Decodable {
    let vocab: [String: Int]
}

private struct RawAddedToken: Decodable {
    let id: Int
    let content: String
    let special: Bool?
}
