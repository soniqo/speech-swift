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
    /// Merge rules keyed by the two symbol ids packed into one `UInt64`, valued by the rank and the
    /// resulting symbol id packed the same way. Keying merges by `"\(left) \(right)"` cost a string
    /// allocation and a hash of it for every pair the merge loop looked at, over half a million rules.
    private var pairMerges: [UInt64: UInt64] = [:]
    /// content → id for the explicit added/control tokens (e.g. `<|turn>`, `<bos>`).
    private var specials: [(content: String, id: Int)] = []
    private var specialIds: Set<Int> = []
    /// `<0xXX>` byte-fallback token id for each byte value, when present.
    private var byteFallbackId: [Int] = Array(repeating: -1, count: 256)
    /// Symbol id of each `<0xXX>` piece — its vocab id wherever the checkpoint carries one.
    private var byteSymbolId: [Int32] = Array(repeating: -1, count: 256)
    /// A symbol id is the piece's vocab id. A merge part or result the vocab does not carry — no
    /// shipped Gemma checkpoint has one, but `tokenizer.json` does not forbid it — takes an id above
    /// the vocab, so the rule still applies and the piece still byte-falls-back when it is emitted.
    private var extraSymbols: [String] = []
    private var extraSymbolIds: [String: Int32] = [:]
    private var symbolIdCeiling: Int32 = 0

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

        // Symbol ids pack two to a `UInt64` in the merge table, so the vocab has to fit in 32 bits.
        let highestId = idToToken.keys.max() ?? -1
        guard highestId < Int(Int32.max) else {
            throw ChatModelError.tokenizerLoadFailed("Gemma vocab ids exceed the 32-bit symbol space")
        }
        symbolIdCeiling = Int32(highestId + 1)
        // Merges resolve last so every part takes the id the *final* vocab gives it, added tokens
        // included; interning a part before an added token could reassign it would split one piece
        // across two symbol ids.
        buildMergeTable(model["merges"])
        for b in 0..<256 { byteSymbolId[b] = internSymbol(String(format: "<0x%02X>", b)) }

        let eosFromConfig = directory.appendingPathComponent("generation_config.json")
        if let d = try? Data(contentsOf: eosFromConfig),
           let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any] {
            if let arr = j["eos_token_id"] as? [NSNumber] { eosTokenIds = Set(arr.map { $0.intValue }) }
            else if let n = j["eos_token_id"] as? NSNumber { eosTokenIds = [n.intValue] }
            if let n = j["bos_token_id"] as? NSNumber { bosTokenId = n.intValue }
        }
    }

    public func isSpecialToken(_ id: Int) -> Bool { specialIds.contains(id) }

    // MARK: - Merge table

    /// `tokenizer.json` carries merges either as `[[left, right]]` or as `["left right"]`.
    private func buildMergeTable(_ merges: Any?) {
        func record(_ left: String, _ right: String, rank: Int) {
            let key = Self.pairKey(internSymbol(left), internSymbol(right))
            let merged = internSymbol(left + right)
            pairMerges[key] = (UInt64(UInt32(rank)) << 32) | UInt64(UInt32(bitPattern: merged))
        }
        if let merges = merges as? [[String]] {
            pairMerges.reserveCapacity(merges.count)
            for (i, pair) in merges.enumerated() where pair.count == 2 {
                record(pair[0], pair[1], rank: i)
            }
        } else if let merges = merges as? [String] {
            pairMerges.reserveCapacity(merges.count)
            for (i, m) in merges.enumerated() {
                let p = m.split(separator: " ", maxSplits: 1)
                if p.count == 2 { record(String(p[0]), String(p[1]), rank: i) }
            }
        }
    }

    private func internSymbol(_ piece: String) -> Int32 {
        if let id = tokenToId[piece] { return Int32(id) }
        if let id = extraSymbolIds[piece] { return id }
        let id = symbolIdCeiling + Int32(extraSymbols.count)
        extraSymbols.append(piece)
        extraSymbolIds[piece] = id
        return id
    }

    @inline(__always)
    private static func pairKey(_ left: Int32, _ right: Int32) -> UInt64 {
        (UInt64(UInt32(bitPattern: left)) << 32) | UInt64(UInt32(bitPattern: right))
    }

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
    ///
    /// Merging walks a linked list of symbols with a queue of candidate pairs. Re-scanning every
    /// adjacent pair after every merge, splicing an array of growing strings, made this quadratic in
    /// text length: measured against this vocabulary, 32k characters took 32.3s to tokenize and the
    /// growth exponent over 2k…32k was 1.98. The queue picks the same merges in the same order — it
    /// is ordered by `(rank, position)` and a merged symbol keeps its left operand's position, so the
    /// lowest rank still wins and equal ranks still resolve leftmost.
    ///
    /// Splitting the text into words first, the usual bound on BPE cost, is *not* safe against this
    /// vocabulary: 466 merge rules join a piece to one starting with `▁`, so runs of spaces (and
    /// `>` before `▁</`) tokenize differently once a word boundary is imposed.
    private func bpeEncode(_ raw: String, into out: inout [Int]) {
        if raw.isEmpty { return }
        let normalized = raw.replacingOccurrences(of: " ", with: "\u{2581}")

        // Initial symbols: one per character, falling back to per-byte `<0xXX>` symbols.
        var symbols: [Int32] = []
        symbols.reserveCapacity(normalized.count)
        for ch in normalized {
            let piece = String(ch)
            if let id = tokenToId[piece] {
                symbols.append(Int32(id))
            } else {
                for b in piece.utf8 where byteSymbolId[Int(b)] >= 0 {
                    symbols.append(byteSymbolId[Int(b)])
                }
            }
        }
        let count = symbols.count
        if count == 0 { return }

        var prev = [Int32](repeating: 0, count: count)
        var next = [Int32](repeating: 0, count: count)
        var alive = [Bool](repeating: true, count: count)
        for i in 0..<count {
            prev[i] = Int32(i - 1)
            next[i] = i + 1 < count ? Int32(i + 1) : -1
        }

        var queue = MergeQueue()
        queue.reserveCapacity(count * 2)
        for i in 0..<(count - 1) {
            if let merge = pairMerges[Self.pairKey(symbols[i], symbols[i + 1])] {
                queue.push((merge >> 32) << 32 | UInt64(UInt32(i)))
            }
        }

        while let candidate = queue.pop() {
            let left = Int(UInt32(truncatingIfNeeded: candidate))
            guard alive[left] else { continue }
            let right = Int(next[left])
            guard right >= 0, let merge = pairMerges[Self.pairKey(symbols[left], symbols[right])],
                  merge >> 32 == candidate >> 32 else { continue }
            // A rank identifies exactly one pair, so an entry whose rank still matches the pair now
            // at that position is the entry this pair pushed — anything else is a superseded copy.
            symbols[left] = Int32(bitPattern: UInt32(truncatingIfNeeded: merge))
            alive[right] = false
            let after = next[right]
            next[left] = after
            if after >= 0 { prev[Int(after)] = Int32(left) }

            let before = prev[left]
            if before >= 0,
               let m = pairMerges[Self.pairKey(symbols[Int(before)], symbols[left])] {
                queue.push((m >> 32) << 32 | UInt64(UInt32(bitPattern: before)))
            }
            if after >= 0,
               let m = pairMerges[Self.pairKey(symbols[left], symbols[Int(after)])] {
                queue.push((m >> 32) << 32 | UInt64(UInt32(left)))
            }
        }

        var cursor = 0
        while cursor >= 0 {
            let symbol = symbols[cursor]
            if symbol < symbolIdCeiling {
                out.append(Int(symbol))
            } else {
                // Should not happen (byte-fallback guarantees coverage), but stay lossless.
                for b in extraSymbols[Int(symbol - symbolIdCeiling)].utf8
                where byteFallbackId[Int(b)] >= 0 {
                    out.append(byteFallbackId[Int(b)])
                }
            }
            cursor = Int(next[cursor])
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

/// Binary min-heap of merge candidates packed as `(rank << 32) | position`. Unsigned order over that
/// one word is lowest rank first and, between two positions holding the same rule, leftmost first.
/// Entries are never removed when a merge invalidates them; the pop side re-checks the rank instead,
/// which is cheaper than finding them and bounds the heap at three entries per symbol.
private struct MergeQueue {
    private var items: [UInt64] = []

    mutating func reserveCapacity(_ n: Int) { items.reserveCapacity(n) }

    mutating func push(_ item: UInt64) {
        items.append(item)
        var child = items.count - 1
        while child > 0 {
            let parent = (child - 1) / 2
            if items[parent] <= items[child] { break }
            items.swapAt(parent, child)
            child = parent
        }
    }

    mutating func pop() -> UInt64? {
        guard let top = items.first else { return nil }
        let tail = items.removeLast()
        if !items.isEmpty {
            items[0] = tail
            var parent = 0
            while true {
                let left = 2 * parent + 1
                guard left < items.count else { break }
                let right = left + 1
                let child = (right < items.count && items[right] < items[left]) ? right : left
                if items[parent] <= items[child] { break }
                items.swapAt(parent, child)
                parent = child
            }
        }
        return top
    }
}
