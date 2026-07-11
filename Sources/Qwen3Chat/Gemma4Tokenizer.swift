import Foundation

/// SentencePiece-style BPE tokenizer for the Gemma 4 vocabulary.
///
/// The Gemma tokenizer differs from the GPT-2 byte-level scheme used by `ChatTokenizer`:
///   • space → `▁` (U+2581) normalizer; `▁` is a regular vocab character.
///   • `byte_fallback: true` — bytes with no single-char token are emitted as `<0xXX>` tokens.
///   • decode = Replace(`▁`→space) → ByteFallback(`<0xXX>`→byte) → Fuse.
///
/// This loads `tokenizer.json` (BPE vocab + merge ranks + added/special tokens) and implements
/// encode (merge-rank BPE), decode, and the streaming `tokenBytes` surface the generation loop needs.
public final class Gemma4Tokenizer: @unchecked Sendable {
    private var tokenToId: [String: Int] = [:]
    private var idToToken: [Int: String] = [:]
    private var mergeRanks: [String: Int] = [:]
    /// content → id for the explicit added/control tokens (e.g. `<|turn>`, `<bos>`).
    private var specials: [(content: String, id: Int)] = []
    private var specialIds: Set<Int> = []
    /// `<0xXX>` byte-fallback token id for each byte value, when present.
    private var byteFallbackId: [Int] = Array(repeating: -1, count: 256)

    public private(set) var bosTokenId: Int = 2
    /// Tokens that terminate a model turn: `<eos>`(1), `<turn|>`(106), `<|tool_response>`(50).
    public private(set) var eosTokenIds: Set<Int> = [1, 106, 50]

    public init() {}

    public func load(from directory: URL) throws {
        let url = directory.appendingPathComponent("tokenizer.json")
        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = root["model"] as? [String: Any],
              let vocab = model["vocab"] as? [String: Int] else {
            throw ChatModelError.tokenizerLoadFailed("Invalid Gemma tokenizer.json")
        }
        tokenToId = vocab
        idToToken = Dictionary(uniqueKeysWithValues: vocab.map { ($1, $0) })

        if let merges = model["merges"] as? [[String]] {
            for (i, pair) in merges.enumerated() where pair.count == 2 {
                mergeRanks["\(pair[0]) \(pair[1])"] = i
            }
        } else if let merges = model["merges"] as? [String] {
            for (i, m) in merges.enumerated() {
                let p = m.split(separator: " ", maxSplits: 1)
                if p.count == 2 { mergeRanks["\(p[0]) \(p[1])"] = i }
            }
        }

        if let added = root["added_tokens"] as? [[String: Any]] {
            for a in added {
                guard let content = a["content"] as? String, let id = a["id"] as? Int else { continue }
                tokenToId[content] = id
                idToToken[id] = content
                if (a["special"] as? Bool) ?? false {
                    specials.append((content, id))
                    specialIds.insert(id)
                }
            }
        }
        // Longest content first so e.g. `<|turn>` matches before any shorter prefix.
        specials.sort { $0.content.count > $1.content.count }

        for b in 0..<256 {
            if let id = tokenToId[String(format: "<0x%02X>", b)] { byteFallbackId[b] = id }
        }

        let eosFromConfig = directory.appendingPathComponent("generation_config.json")
        if let d = try? Data(contentsOf: eosFromConfig),
           let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any] {
            if let arr = j["eos_token_id"] as? [NSNumber] { eosTokenIds = Set(arr.map { $0.intValue }) }
            else if let n = j["eos_token_id"] as? NSNumber { eosTokenIds = [n.intValue] }
            if let n = j["bos_token_id"] as? NSNumber { bosTokenId = n.intValue }
        }
    }

    public func isSpecialToken(_ id: Int) -> Bool { specialIds.contains(id) }

    // MARK: - Encode

    /// Encode text. Splits out explicit special tokens, then BPE-encodes plain runs.
    public func encode(_ text: String) -> [Int] {
        if text.isEmpty { return [] }
        var out: [Int] = []
        encodeSegment(Substring(text), into: &out)
        return out
    }

    private func encodeSegment(_ text: Substring, into out: inout [Int]) {
        if text.isEmpty { return }
        // Find the earliest special-token occurrence.
        var bestRange: Range<Substring.Index>? = nil
        var bestId = -1
        for (content, id) in specials {
            if let r = text.range(of: content) {
                if bestRange == nil || r.lowerBound < bestRange!.lowerBound {
                    bestRange = r; bestId = id
                }
            }
        }
        guard let r = bestRange else {
            bpeEncode(String(text), into: &out)
            return
        }
        if r.lowerBound > text.startIndex { bpeEncode(String(text[text.startIndex..<r.lowerBound]), into: &out) }
        out.append(bestId)
        encodeSegment(text[r.upperBound...], into: &out)
    }

    /// SentencePiece BPE: normalize spaces→`▁`, split into characters (byte-fallback for any char
    /// with no single-char token), then merge by rank.
    private func bpeEncode(_ raw: String, into out: inout [Int]) {
        if raw.isEmpty { return }
        let normalized = raw.replacingOccurrences(of: " ", with: "\u{2581}")

        // Initial symbols: one per character, falling back to per-byte `<0xXX>` symbols.
        var symbols: [String] = []
        for ch in normalized {
            let s = String(ch)
            if tokenToId[s] != nil {
                symbols.append(s)
            } else {
                for b in Array(s.utf8) { symbols.append("<0x" + String(format: "%02X", b) + ">") }
            }
        }
        if symbols.isEmpty { return }

        while symbols.count > 1 {
            var bestRank = Int.max
            var bestIdx = -1
            for i in 0..<(symbols.count - 1) {
                if let rank = mergeRanks["\(symbols[i]) \(symbols[i + 1])"], rank < bestRank {
                    bestRank = rank; bestIdx = i
                }
            }
            if bestIdx < 0 { break }
            symbols.replaceSubrange(bestIdx...bestIdx + 1, with: [symbols[bestIdx] + symbols[bestIdx + 1]])
        }

        for s in symbols {
            if let id = tokenToId[s] {
                out.append(id)
            } else {
                // Should not happen (byte-fallback guarantees coverage), but stay lossless.
                for b in Array(s.utf8) where byteFallbackId[Int(b)] >= 0 {
                    out.append(byteFallbackId[Int(b)])
                }
            }
        }
    }

    // MARK: - Decode

    /// Raw byte sequence a token maps to: `<0xXX>` → the byte; otherwise the token text with
    /// `▁`→space, as UTF-8. Streaming decoders accumulate these and decode UTF-8 on the buffer.
    public func tokenBytes(_ id: Int) -> [UInt8] {
        guard let piece = idToToken[id] else { return [] }
        if piece.count == 6, piece.hasPrefix("<0x"), piece.hasSuffix(">"),
           let b = UInt8(piece.dropFirst(3).dropLast(), radix: 16) {
            return [b]
        }
        return Array(piece.replacingOccurrences(of: "\u{2581}", with: " ").utf8)
    }

    /// Decode a full token list to text (byte-fallback aware, `▁`→space, UTF-8 fuse).
    public func decode(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []
        for id in ids { bytes.append(contentsOf: tokenBytes(id)) }
        return String(decoding: bytes, as: UTF8.self)
    }
}
