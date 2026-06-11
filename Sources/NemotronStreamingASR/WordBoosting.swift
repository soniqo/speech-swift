import Foundation

/// Decoder-time word boosting for Nemotron RNN-T greedy decoding: shallow
/// fusion that biases token selection toward configured phrases.
public struct WordBoostingConfig: Sendable, Equatable {
    public struct Phrase: Sendable, Equatable {
        public var text: String
        public var boost: Float

        public init(_ text: String, boost: Float) {
            self.text = text
            self.boost = boost
        }
    }

    public var phrases: [Phrase]
    public var contextScore: Float
    public var depthScaling: Float

    public init(
        phrases: [String],
        boost: Float = 0.75,
        contextScore: Float = 1.0,
        depthScaling: Float = 2.0
    ) {
        self.phrases = phrases.map { Phrase($0, boost: boost) }
        self.contextScore = contextScore
        self.depthScaling = depthScaling
    }

    public init(
        phrases: [Phrase],
        contextScore: Float = 1.0,
        depthScaling: Float = 2.0
    ) {
        self.phrases = phrases
        self.contextScore = contextScore
        self.depthScaling = depthScaling
    }
}

public struct WordBoostingSuggestion: Sendable, Equatable {
    public enum Difficulty: String, Sendable {
        case easy
        case moderate
        case hard
    }

    public let phrase: String
    public let suggestedBoost: Float
    public let difficulty: Difficulty
    public let tokenCount: Int
    public let tokenizations: [[Int]]
    public let reason: String
}

public struct WordBoostingTokenizerStatus: Sendable, Equatable {
    public enum Mode: String, Sendable {
        case sentencePieceModel
        case vocabFallback
    }

    public let mode: Mode
    public let path: String?
}

struct WordBoostingState: Sendable, Equatable {
    var node: Int = 0
}

struct WordBoostingContext: Sendable {
    struct Selection: Sendable, Equatable {
        let tokenId: Int
        let unboostedTokenId: Int
    }

    let tree: WordBoostingTree
    static let maxTokenizationsPerPhrase = 5

    init?(
        config: WordBoostingConfig?,
        vocabulary: NemotronVocabulary,
        tokenizer: NemotronSentencePieceUnigramTokenizer? = nil
    ) {
        guard let config else { return nil }

        var tokenizedPhrases: [(tokens: [Int], boost: Float)] = []
        var seen: [ [Int]: Float ] = [:]
        for phrase in config.phrases {
            guard phrase.boost > 0 else { continue }

            let encodings: [[Int]]
            if let tokenizer, let tokens = tokenizer.encodeForWordBoosting(phrase.text) {
                encodings = [tokens]
            } else {
                encodings = vocabulary.encodingsForWordBoosting(phrase.text, maxCount: 1)
            }

            for tokens in encodings {
                let weightedScore = phrase.boost * config.contextScore
                if let existing = seen[tokens] {
                    seen[tokens] = max(existing, weightedScore)
                } else {
                    seen[tokens] = weightedScore
                }
            }
        }
        tokenizedPhrases = seen.map { (tokens: $0.key, boost: $0.value) }

        guard !tokenizedPhrases.isEmpty else { return nil }
        self.tree = WordBoostingTree(
            phrases: tokenizedPhrases,
            depthScaling: max(1, config.depthScaling)
        )
    }

    static func suggestions(
        for phrases: [String],
        vocabulary: NemotronVocabulary,
        tokenizer: NemotronSentencePieceUnigramTokenizer? = nil
    ) -> [WordBoostingSuggestion] {
        phrases.map { phrase in
            let tokenizations: [[Int]]
            if let tokenizer, let tokens = tokenizer.encodeForWordBoosting(phrase) {
                tokenizations = [tokens]
            } else {
                tokenizations = vocabulary.encodingsForWordBoosting(phrase, maxCount: 1)
            }

            var unique: [[Int]] = []
            var seen = Set<[Int]>()
            for tokens in tokenizations where !tokens.isEmpty {
                if seen.insert(tokens).inserted {
                    unique.append(tokens)
                }
            }

            return suggestion(for: phrase, tokenizations: unique)
        }
    }

    private static func suggestion(for phrase: String, tokenizations: [[Int]]) -> WordBoostingSuggestion {
        guard !tokenizations.isEmpty else {
            return WordBoostingSuggestion(
                phrase: phrase,
                suggestedBoost: 1.25,
                difficulty: .hard,
                tokenCount: 0,
                tokenizations: [],
                reason: "Phrase could not be encoded with the Nemotron vocabulary"
            )
        }

        let bestTokenCount = tokenizations.map(\.count).min() ?? 0
        let words = phrase.split(whereSeparator: { $0.isWhitespace })
        let scalarCount = phrase.unicodeScalars.filter { !$0.properties.isWhitespace }.count
        let tokensPerWord = words.isEmpty ? Float(bestTokenCount) : Float(bestTokenCount) / Float(words.count)
        let hasShortShape = scalarCount <= 4
        let hasManyAlternatives = tokenizations.count >= 4

        let difficulty: WordBoostingSuggestion.Difficulty
        let boost: Float
        let reason: String

        if bestTokenCount >= 6 || tokensPerWord >= 5 {
            difficulty = .hard
            boost = 1.25
            reason = "Phrase is highly fragmented by the Nemotron vocabulary"
        } else if bestTokenCount >= 4 || tokensPerWord >= 3 || hasShortShape {
            difficulty = .hard
            boost = 1.25
            reason = hasShortShape
                ? "Phrase is short and acoustically ambiguous"
                : "Phrase is split into several BPE pieces"
        } else if bestTokenCount >= 3 || hasManyAlternatives {
            difficulty = .moderate
            boost = 0.95
            reason = "Phrase has moderate tokenization complexity"
        } else {
            difficulty = .easy
            boost = 0.75
            reason = "Phrase has a compact tokenization"
        }

        return WordBoostingSuggestion(
            phrase: phrase,
            suggestedBoost: boost,
            difficulty: difficulty,
            tokenCount: bestTokenCount,
            tokenizations: tokenizations,
            reason: reason
        )
    }

    func initialState() -> WordBoostingState {
        WordBoostingState()
    }

    func selectToken(
        from logits: UnsafeMutablePointer<Float>,
        count: Int,
        blankTokenId: Int,
        state: inout WordBoostingState
    ) -> Int {
        selectTokenWithDetails(
            from: logits,
            count: count,
            blankTokenId: blankTokenId,
            state: &state
        ).tokenId
    }

    func selectTokenWithDetails(
        from logits: UnsafeMutablePointer<Float>,
        count: Int,
        blankTokenId: Int,
        state: inout WordBoostingState
    ) -> Selection {
        let unboostedToken = argmax(logits, count: count)
        if unboostedToken == blankTokenId {
            return Selection(tokenId: blankTokenId, unboostedTokenId: unboostedToken)
        }

        var bestToken = -1
        var bestScore = -Float.infinity
        var bestNode = 0
        let currentNode = state.node

        for tokenId in 0..<count where tokenId != blankTokenId {
            let transition = tree.advance(from: currentNode, token: tokenId)
            let score = logits[tokenId] + transition.score
            if score > bestScore {
                bestScore = score
                bestToken = tokenId
                bestNode = transition.node
            }
        }

        guard bestToken >= 0 else {
            return Selection(tokenId: blankTokenId, unboostedTokenId: unboostedToken)
        }

        state.node = bestNode
        return Selection(tokenId: bestToken, unboostedTokenId: unboostedToken)
    }

    private func argmax(_ values: UnsafeMutablePointer<Float>, count: Int) -> Int {
        var bestIndex = 0
        var bestValue = values[0]
        for i in 1..<count where values[i] > bestValue {
            bestValue = values[i]
            bestIndex = i
        }
        return bestIndex
    }
}

struct WordBoostingTree: Sendable {
    struct Transition: Sendable, Equatable {
        let node: Int
        let score: Float
    }

    private struct Node: Sendable {
        var next: [Int: Int] = [:]
        var fail: Int = 0
        var tokenScore: Float = 0
        var nodeScore: Float = 0
        var backoffScore: Float = 0
        var isEnd: Bool = false
        var tokens: [Int] = []
    }

    private var nodes: [Node] = [Node()]
    private let unkScore: Float = 0

    init(phrases: [(tokens: [Int], boost: Float)], depthScaling: Float) {
        for phrase in phrases where !phrase.tokens.isEmpty && phrase.boost > 0 {
            add(phrase: phrase.tokens, contextScore: phrase.boost, depthScaling: depthScaling)
        }
        recomputeNodeScores()
        fillFailureLinks()
    }

    func advance(from node: Int, token: Int) -> Transition {
        var current = min(max(0, node), nodes.count - 1)
        var score: Float = 0

        while true {
            if let next = nodes[current].next[token] {
                return Transition(node: next, score: score + nodes[next].tokenScore)
            }
            if current == 0 {
                return Transition(node: 0, score: score + unkScore)
            }

            score += nodes[current].backoffScore
            current = nodes[current].fail
        }
    }

    func tokens(at node: Int) -> [Int] {
        nodes[min(max(0, node), nodes.count - 1)].tokens
    }

    private mutating func add(phrase: [Int], contextScore: Float, depthScaling: Float) {
        var current = 0
        for (index, token) in phrase.enumerated() {
            let candidateTokenScore: Float
            if index > 0 {
                candidateTokenScore = contextScore * depthScaling + log(Float(index + 1))
            } else {
                candidateTokenScore = contextScore
            }

            let next: Int
            if let existing = nodes[current].next[token] {
                next = existing
                nodes[next].tokenScore = max(nodes[next].tokenScore, candidateTokenScore)
            } else {
                next = nodes.count
                nodes[current].next[token] = next
                var node = Node(tokenScore: candidateTokenScore)
                node.tokens = nodes[current].tokens + [token]
                nodes.append(node)
            }

            nodes[next].nodeScore = nodes[current].nodeScore + nodes[next].tokenScore
            if index == phrase.count - 1 {
                nodes[next].isEnd = true
            }
            current = next
        }
    }

    private mutating func fillFailureLinks() {
        var queue: [Int] = []
        var head = 0

        for child in nodes[0].next.values {
            nodes[child].fail = 0
            queue.append(child)
        }

        while head < queue.count {
            let current = queue[head]
            head += 1

            for (token, child) in nodes[current].next {
                var fail = nodes[current].fail
                while fail != 0 && nodes[fail].next[token] == nil {
                    fail = nodes[fail].fail
                }
                nodes[child].fail = nodes[fail].next[token] ?? 0
                queue.append(child)
            }
        }

        for index in nodes.indices where index != 0 {
            let fail = nodes[index].fail
            nodes[index].backoffScore = nodes[index].isEnd ? 0 : nodes[fail].nodeScore - nodes[index].nodeScore
        }
    }

    private mutating func recomputeNodeScores() {
        var queue: [Int] = [0]
        var head = 0
        nodes[0].nodeScore = 0

        while head < queue.count {
            let current = queue[head]
            head += 1
            for child in nodes[current].next.values {
                nodes[child].nodeScore = nodes[current].nodeScore + nodes[child].tokenScore
                queue.append(child)
            }
        }
    }
}
