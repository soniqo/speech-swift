import Foundation

public enum ClefError: Error, LocalizedError {
    case invalidRequest(String)
    case invalidModel(String)
    public var errorDescription: String? {
        switch self {
        case .invalidRequest(let message), .invalidModel(let message): return message
        }
    }
}

/// One field in a joint decision. Array order determines the schema field order.
public struct ClefQuestion: Sendable {
    public enum Kind: Sendable {
        case choice([String: String])
        case noul(trueDescription: String = "The proposition is true or the answer is yes.",
                  falseDescription: String = "The proposition is false or the answer is no.")
        case score([String])
    }
    public let id: String
    public let instructions: String
    public let kind: Kind
    public init(id: String, instructions: String = "", kind: Kind) {
        self.id = id; self.instructions = instructions; self.kind = kind
    }
    var typeName: String {
        switch kind { case .choice: return "choice"; case .noul: return "noul"; case .score: return "score" }
    }
    var typeID: Int {
        switch kind { case .noul: return 0; case .choice: return 1; case .score: return 2 }
    }
    var options: [(String, String)] {
        switch kind {
        case .choice(let values): return values.keys.sorted().map { ($0, values[$0]!) }
        case .noul(let yes, let no): return [("true", yes), ("false", no)]
        case .score(let values): return values.enumerated().map { (String($0.offset), $0.element) }
        }
    }
}

/// Model probabilities are scores, not calibrated guarantees. No action is executed.
public struct ClefDecision: Sendable {
    public let id: String
    public let probabilities: [String: Float]
    public let selectedOption: String
    public let confidence: Float
    /// Expected zero-based criterion index, only for score questions.
    public let score: Float?
    /// Probability of true, only for noul questions.
    public let noul: Float?
}

public struct ClefResult: Sendable {
    public let decisions: [ClefDecision]
    public let inputTokens: Int
}

struct ClefEncodedQuestion {
    let question: ClefQuestion
    let span: Range<Int>
    let optionSpans: [Range<Int>]
}
struct ClefEncoding {
    let tokens: [Int]
    let questions: [ClefEncodedQuestion]

    static func encode(state: String, questions: [ClefQuestion], maxTokens: Int,
                       tokenize: (String) -> [Int]) throws -> ClefEncoding {
        guard !questions.isEmpty, Set(questions.map(\.id)).count == questions.count,
              questions.allSatisfy({ !$0.id.isEmpty && !$0.options.isEmpty }) else {
            throw ClefError.invalidRequest("Provide unique nonempty field IDs and at least one option per field.")
        }
        var schema = tokenize("\n\nSCHEMA FIELDS:\n")
        var metadata: [ClefEncodedQuestion] = []
        for (index, question) in questions.enumerated() {
            schema += tokenize("\nFIELD \(index + 1)\nID: \(question.id)\nTYPE: \(question.typeName)\nINSTRUCTION: ")
            let start = schema.count
            schema += tokenize(question.instructions.isEmpty ? question.id : question.instructions)
            let span = start..<schema.count
            schema += tokenize("\nALLOWED OPTIONS:\n")
            var options: [Range<Int>] = []
            for (i, option) in question.options.enumerated() {
                schema += tokenize("OPTION \(i + 1): ")
                let start = schema.count
                let data = try JSONSerialization.data(withJSONObject: ["description": option.1, "option_id": option.0],
                                                      options: [.sortedKeys, .withoutEscapingSlashes])
                schema += tokenize(String(decoding: data, as: UTF8.self))
                options.append(start..<schema.count)
                schema += tokenize("\n")
            }
            schema += tokenize("END FIELD\n")
            metadata.append(.init(question: question, span: span, optionSpans: options))
        }
        let prefix = tokenize("<|im_start|>system\nRead the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.<|im_end|>\n<|im_start|>user\nSTATE:\n")
        let suffix = tokenize("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")
        let stateTokens = tokenize(state)
        let count = prefix.count + stateTokens.count + schema.count + suffix.count
        guard count <= maxTokens else {
            throw ClefError.invalidRequest("Request requires \(count) tokens; maximum is \(maxTokens). State is never silently truncated.")
        }
        let offset = prefix.count + stateTokens.count
        func shift(_ range: Range<Int>) -> Range<Int> { (range.lowerBound + offset)..<(range.upperBound + offset) }
        return .init(tokens: prefix + stateTokens + schema + suffix,
                     questions: metadata.map { .init(question: $0.question, span: shift($0.span), optionSpans: $0.optionSpans.map(shift)) })
    }
}
