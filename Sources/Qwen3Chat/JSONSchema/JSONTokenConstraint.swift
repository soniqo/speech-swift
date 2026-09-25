import Foundation
import MLX

/// A tokenizer vocabulary indexed for constrained decoding. Built once per model.
///
/// Every admissible token is a byte string, so the tokens a matcher state allows are found by
/// walking a trie of those strings with the matcher, pruning a branch the moment its bytes stop
/// being a valid prefix. That walk is cheap everywhere except inside a free JSON string, where
/// nearly the whole vocabulary is admissible: 262,144 Gemma tokens would be walked every step.
/// So tokens are split once. A *clean* token is complete UTF-8 with no `"`, no `\` and no
/// control character; inside an unpatterned string it is admissible exactly when its characters
/// fit the remaining `maxLength`, which one on-device comparison against ``cleanChars`` decides.
/// Only the rest — tokens that can close or escape a string — are walked, from their own trie.
final class JSONTokenVocabulary: @unchecked Sendable {
    /// Width of the logits row the masks apply to.
    let size: Int
    /// Byte string of each admissible token; empty for tokens the constraint never allows
    /// (special and control tokens, empty tokens, ids past the tokenizer).
    let tokenBytes: [[UInt8]]
    /// Character count of each clean token; `Int16.max` for every other id.
    let cleanCharsHost: [Int16]
    let allTrie: ByteTrie
    let uncleanTrie: ByteTrie
    private let lock = NSLock()
    private var cleanCharsArray: MLXArray?

    /// - Parameters:
    ///   - size: logits width; ids at or beyond `tokens.count` are never admissible.
    ///   - tokens: byte string per id, `nil` for ids the constraint must never admit.
    init(size: Int, tokens: [[UInt8]?]) {
        self.size = size
        var bytes = [[UInt8]](repeating: [], count: size)
        var clean = [Int16](repeating: .max, count: size)
        var all: [(bytes: [UInt8], id: Int32)] = []
        var unclean: [(bytes: [UInt8], id: Int32)] = []
        all.reserveCapacity(size)
        for id in 0 ..< min(size, tokens.count) {
            guard let t = tokens[id], !t.isEmpty else { continue }
            bytes[id] = t
            all.append((t, Int32(id)))
            if let chars = Self.cleanCharacterCount(t), chars < Int(Int16.max) {
                clean[id] = Int16(chars)
            } else {
                unclean.append((t, Int32(id)))
            }
        }
        self.tokenBytes = bytes
        self.cleanCharsHost = clean
        self.allTrie = ByteTrie(all)
        self.uncleanTrie = ByteTrie(unclean)
    }

    /// Gemma 4: every non-special id, as the bytes `Gemma4Tokenizer.tokenBytes` streams.
    convenience init(gemma tokenizer: Gemma4Tokenizer, size: Int) {
        var tokens = [[UInt8]?](repeating: nil, count: size)
        for id in 0 ..< size where !tokenizer.isSpecialToken(id) {
            let b = tokenizer.tokenBytes(id)
            if !b.isEmpty { tokens[id] = b }
        }
        self.init(size: size, tokens: tokens)
    }

    /// ``cleanCharsHost`` on device, uploaded on first use.
    var cleanChars: MLXArray {
        lock.lock()
        defer { lock.unlock() }
        if let cleanCharsArray { return cleanCharsArray }
        let array = MLXArray(cleanCharsHost)
        cleanCharsArray = array
        return array
    }

    /// Character count if `bytes` is whole UTF-8 characters free of `"`, `\` and controls.
    static func cleanCharacterCount(_ bytes: [UInt8]) -> Int? {
        var chars = 0
        var i = 0
        while i < bytes.count {
            let b = bytes[i]
            let need: Int
            switch b {
            case 0x22, 0x5C, 0 ..< 0x20: return nil
            case 0x20 ..< 0x80: need = 0
            case 0xC2 ... 0xDF: need = 1
            case 0xE0 ... 0xEF: need = 2
            case 0xF0 ... 0xF4: need = 3
            default: return nil
            }
            guard i + need < bytes.count || need == 0 else { return nil }
            if need > 0 {
                // Validate through Swift's decoder rather than restating the overlong rules.
                let slice = Array(bytes[i ... i + need])
                guard String(validating: slice, as: UTF8.self) != nil else { return nil }
            }
            i += need + 1
            chars += 1
        }
        return chars
    }
}

/// Byte-string trie over token ids, stored flat. A node's children are contiguous.
struct ByteTrie {
    private(set) var childStart: [Int32] = [0]
    private(set) var childCount: [Int32] = [0]
    private(set) var byte: [UInt8] = [0]
    private(set) var tokenStart: [Int32] = [0]
    private(set) var tokenCount: [Int32] = [0]
    private(set) var tokenIds: [Int32] = []

    init(_ entries: [(bytes: [UInt8], id: Int32)]) {
        let order = entries.indices.sorted {
            entries[$0].bytes.lexicographicallyPrecedes(entries[$1].bytes)
        }
        build(entries, order, 0 ..< order.count, depth: 0, node: 0)
    }

    private mutating func build(
        _ entries: [(bytes: [UInt8], id: Int32)], _ order: [Int], _ range: Range<Int>,
        depth: Int, node: Int
    ) {
        var i = range.lowerBound
        tokenStart[node] = Int32(tokenIds.count)
        // Shorter strings sort first, so the tokens ending at this node lead the range.
        while i < range.upperBound, entries[order[i]].bytes.count == depth {
            tokenIds.append(entries[order[i]].id)
            i += 1
        }
        tokenCount[node] = Int32(tokenIds.count) - tokenStart[node]
        var groups: [(UInt8, Range<Int>)] = []
        while i < range.upperBound {
            let b = entries[order[i]].bytes[depth]
            var j = i + 1
            while j < range.upperBound, entries[order[j]].bytes[depth] == b { j += 1 }
            groups.append((b, i ..< j))
            i = j
        }
        childStart[node] = Int32(byte.count)
        childCount[node] = Int32(groups.count)
        for (b, _) in groups {
            byte.append(b)
            childStart.append(0)
            childCount.append(0)
            tokenStart.append(0)
            tokenCount.append(0)
        }
        let first = Int(childStart[node])
        for (k, g) in groups.enumerated() {
            build(entries, order, g.1, depth: depth + 1, node: first + k)
        }
    }
}

/// The tokens admissible at one decode step.
struct JSONTokenAllowance {
    /// All clean tokens of at most this many characters are admissible (see
    /// ``JSONTokenVocabulary``); `nil` when none are admitted wholesale.
    var cleanUpTo: Int?
    /// Individually admitted ids, end-of-turn ids included when the document may end.
    var ids: [Int32]

    var isEmpty: Bool { (cleanUpTo ?? 0) <= 0 && ids.isEmpty }
}

/// One constrained decode: the matcher state and the admissible set it implies.
struct JSONTokenConstraint {
    let matcher: JSONSchemaMatcher
    let vocabulary: JSONTokenVocabulary
    let endTokens: [Int32]
    private(set) var states: [JSONSchemaMatcher.Stack]

    init(grammar: JSONSchemaGrammar, vocabulary: JSONTokenVocabulary, endTokens: [Int]) {
        self.matcher = JSONSchemaMatcher(grammar: grammar)
        self.vocabulary = vocabulary
        self.endTokens = endTokens.map(Int32.init)
        self.states = matcher.initial
    }

    /// Continue from other matcher states (used to step a constraint byte by byte).
    mutating func replaceStates(_ states: [JSONSchemaMatcher.Stack]) { self.states = states }

    /// The document is complete and nothing may follow it.
    var isDone: Bool { matcher.isDone(states) }

    /// Consume a sampled token. Returns false if it was not admissible (the state is unchanged).
    @discardableResult
    mutating func accept(_ id: Int) -> Bool {
        guard id >= 0, id < vocabulary.size else { return false }
        let b = vocabulary.tokenBytes[id]
        guard !b.isEmpty else { return false }
        let next = matcher.advance(states, bytes: b)
        guard !next.isEmpty else { return false }
        states = next
        return true
    }

    func allowance() -> JSONTokenAllowance {
        var ids: [Int32] = []
        var cleanUpTo: Int?
        if let room = freeStringRoom() {
            cleanUpTo = room
            walk(vocabulary.uncleanTrie, node: 0, states, &ids)
        } else {
            walk(vocabulary.allTrie, node: 0, states, &ids)
        }
        if matcher.canEnd(states) { ids += endTokens }
        return JSONTokenAllowance(cleanUpTo: cleanUpTo, ids: ids)
    }

    /// Host-side list of every admissible id — the reference the tests compare the fast path to.
    func allowedIDs() -> Set<Int> {
        let a = allowance()
        var out = Set(a.ids.map(Int.init))
        if let limit = a.cleanUpTo {
            for (id, c) in vocabulary.cleanCharsHost.enumerated() where Int(c) <= limit { out.insert(id) }
        }
        return out
    }

    /// When every stack sits at a character boundary inside an unpatterned string, the most
    /// characters any of them may still take; otherwise nil.
    private func freeStringRoom() -> Int? {
        var room = 0
        for s in states {
            guard let top = s.last, top.kind == .string, top.phase == 0, top.b & 0xF == 0,
                  case .string(let spec) = matcher.grammar.nodes[Int(top.node)], spec.pattern == nil
            else { return nil }
            room = max(room, spec.maxLength.map { $0 - Int(top.a) } ?? Int(Int16.max) - 1)
        }
        return room
    }

    private func walk(
        _ trie: ByteTrie, node: Int, _ states: [JSONSchemaMatcher.Stack], _ ids: inout [Int32]
    ) {
        let start = Int(trie.childStart[node])
        for child in start ..< start + Int(trie.childCount[node]) {
            let next = matcher.advance(states, byte: trie.byte[child])
            if next.isEmpty { continue }
            let t = Int(trie.tokenStart[child])
            ids.append(contentsOf: trie.tokenIds[t ..< t + Int(trie.tokenCount[child])])
            walk(trie, node: child, next, &ids)
        }
    }
}
