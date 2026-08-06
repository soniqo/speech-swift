import Foundation
import Tokenizers

/// Nemotron-Nano tokenizer with VoiceChat's runtime control-token roles.
///
/// The published tokenizer metadata advertises <SPECIAL_12> as EOS, but NeMo
/// overrides the channel to PAD=<SPECIAL_12>, BOS=<s>, EOS=</s>. Resolve and
/// round-trip the literal tokens so metadata cannot silently swap silence and
/// end-of-turn again.
public final class VoiceChatTokenizer: @unchecked Sendable {
    public static let padToken = "<SPECIAL_12>"
    public static let bosToken = "<s>"
    public static let eosToken = "</s>"

    private let tokenizer: Tokenizer
    let subwordToCharacters: [Int: [Int32]]

    public let padID: Int
    public let bosID: Int
    public let eosID: Int
    public let specialIDs: Set<Int>

    private init(
        tokenizer: Tokenizer,
        subwordToCharacters: [Int: [Int32]],
        padID: Int,
        bosID: Int,
        eosID: Int
    ) {
        self.tokenizer = tokenizer
        self.subwordToCharacters = subwordToCharacters
        self.padID = padID
        self.bosID = bosID
        self.eosID = eosID

        var special = Set([padID, bosID, eosID])
        for token in ["<unk>", "<pad>"] {
            if let id = tokenizer.convertTokenToId(token),
               tokenizer.convertIdToToken(id) == token {
                special.insert(id)
            }
        }
        self.specialIDs = special
    }

    public static func load(from directory: URL) async throws -> VoiceChatTokenizer {
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory, strict: false)

        func exactID(_ token: String) throws -> Int {
            guard let id = tokenizer.convertTokenToId(token),
                  tokenizer.convertIdToToken(id) == token else {
                throw VoiceChatLoadError.unexpectedKeys(["tokenizer token \(token)"])
            }
            return id
        }

        let pad = try exactID(padToken)
        let bos = try exactID(bosToken)
        let eos = try exactID(eosToken)
        guard (pad, bos, eos) == (12, 1, 2) else {
            throw VoiceChatLoadError.unexpectedKeys([
                "text channel roles PAD/BOS/EOS = \(pad)/\(bos)/\(eos), expected 12/1/2"
            ])
        }

        let mapping = try buildSubwordCharacters(
            tokenizerJSON: directory.appendingPathComponent("tokenizer.json"))
        return VoiceChatTokenizer(
            tokenizer: tokenizer, subwordToCharacters: mapping,
            padID: pad, bosID: bos, eosID: eos)
    }

    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }

    public func decode(_ ids: [Int], skipSpecialTokens: Bool = true) -> String {
        tokenizer.decode(tokens: ids, skipSpecialTokens: skipSpecialTokens)
    }

    public func tokenID(_ token: String) -> Int? {
        tokenizer.convertTokenToId(token)
    }

    /// Reproduce NeMo's character vocabulary: every one-code-point token,
    /// densely renumbered by original token id, then each subword mapped to the
    /// code points present in that vocabulary.
    private static func buildSubwordCharacters(tokenizerJSON: URL) throws -> [Int: [Int32]] {
        let data = try Data(contentsOf: tokenizerJSON)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = root["model"] as? [String: Any],
              let rawVocabulary = model["vocab"] as? [String: Any] else {
            throw VoiceChatLoadError.unexpectedKeys(["tokenizer.json model.vocab"])
        }

        var vocabulary: [String: Int] = [:]
        vocabulary.reserveCapacity(rawVocabulary.count)
        for (token, rawID) in rawVocabulary {
            if let id = rawID as? Int {
                vocabulary[token] = id
            } else if let number = rawID as? NSNumber {
                vocabulary[token] = number.intValue
            }
        }

        let singles = vocabulary
            .filter { $0.key.unicodeScalars.count == 1 }
            .sorted { $0.value < $1.value }
        var characterIDs: [Unicode.Scalar: Int32] = [:]
        characterIDs.reserveCapacity(singles.count)
        for (denseID, entry) in singles.enumerated() {
            if let scalar = entry.key.unicodeScalars.first {
                characterIDs[scalar] = Int32(denseID)
            }
        }

        var mapping: [Int: [Int32]] = [:]
        mapping.reserveCapacity(vocabulary.count)
        for (subword, tokenID) in vocabulary {
            let characters = subword.unicodeScalars.compactMap { characterIDs[$0] }
            if !characters.isEmpty {
                mapping[tokenID] = characters
            }
        }
        return mapping
    }
}
