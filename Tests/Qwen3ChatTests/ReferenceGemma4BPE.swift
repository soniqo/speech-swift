import Foundation

/// The Gemma 4 BPE encoder exactly as it stood before the linked-list/priority-queue rewrite,
/// kept verbatim so parity is asserted against behaviour that actually shipped rather than against
/// a transcription of what it was believed to do. A faster tokenizer that reads differently is
/// worse than a slow one — nothing in the pipeline would report the difference.
///
/// Loads its own tables from the same `tokenizer.json`, so it shares no state with the encoder
/// under test.
final class ReferenceGemma4BPE {
    private var tokenToId: [String: Int] = [:]
    private var mergeRanks: [String: Int] = [:]
    private var specials: [(content: String, id: Int)] = []
    private var byteFallbackId: [Int] = Array(repeating: -1, count: 256)

    enum LoadError: Error { case invalid(String) }

    func load(from directory: URL) throws {
        let url = directory.appendingPathComponent("tokenizer.json")
        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = root["model"] as? [String: Any],
              let vocab = model["vocab"] as? [String: Int] else {
            throw LoadError.invalid("Invalid Gemma tokenizer.json")
        }
        tokenToId = vocab

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
                if (a["special"] as? Bool) ?? false { specials.append((content, id)) }
            }
        }
        specials.sort { $0.content.count > $1.content.count }

        for b in 0..<256 {
            if let id = tokenToId[String(format: "<0x%02X>", b)] { byteFallbackId[b] = id }
        }
    }

    func encode(_ text: String) -> [Int] {
        if text.isEmpty { return [] }
        var out: [Int] = []
        encodeSegment(Substring(text), into: &out)
        return out
    }

    private func encodeSegment(_ text: Substring, into out: inout [Int]) {
        if text.isEmpty { return }
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
        if r.lowerBound > text.startIndex {
            bpeEncode(String(text[text.startIndex..<r.lowerBound]), into: &out)
        }
        out.append(bestId)
        encodeSegment(text[r.upperBound...], into: &out)
    }

    private func bpeEncode(_ raw: String, into out: inout [Int]) {
        if raw.isEmpty { return }
        let normalized = raw.replacingOccurrences(of: " ", with: "\u{2581}")

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
            symbols.replaceSubrange(bestIdx...bestIdx + 1,
                                    with: [symbols[bestIdx] + symbols[bestIdx + 1]])
        }

        for s in symbols {
            if let id = tokenToId[s] {
                out.append(id)
            } else {
                for b in Array(s.utf8) where byteFallbackId[Int(b)] >= 0 {
                    out.append(byteFallbackId[Int(b)])
                }
            }
        }
    }
}
