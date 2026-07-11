import Foundation

// MARK: - Public types

/// A function the LLM can invoke. Mirror of the C++ ``ToolDefinition``.
/// Not ``Sendable`` because ``parameters`` is `[String: Any]`; construct on
/// the worker thread or use ``@unchecked Sendable`` at the call site.
public struct FunctionDeclaration {
    public let name: String
    public let description: String
    /// JSON-Schema-ish parameter specification. Pass an `[String: Any]` that
    /// serialises to the same shape FunctionGemma expects on the wire — i.e.
    /// ``{"type":"object","properties":{...}}``.
    public let parameters: [String: Any]

    public init(name: String, description: String, parameters: [String: Any]) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }
}

/// A single parsed `<start_function_call>...<end_function_call>` block.
public struct FunctionCall: Sendable, Equatable {
    public let name: String
    public let arguments: [String: ArgumentValue]

    public init(name: String, arguments: [String: ArgumentValue]) {
        self.name = name
        self.arguments = arguments
    }
}

/// Values inside a FunctionGemma call. The grammar admits ints, doubles,
/// bools, escaped strings, and nested objects / arrays. We type-tag instead
/// of stringifying so callers can dispatch without re-parsing.
public enum ArgumentValue: Sendable, Equatable {
    case string(String)
    case int(Int64)
    case double(Double)
    case bool(Bool)
    case array([ArgumentValue])
    case object([String: ArgumentValue])
    case null
}

public enum FunctionGemmaError: Error, LocalizedError {
    case modelLoadFailed(String)
    case tokenizerLoadFailed(String)
    case inferenceFailed(String)
    case promptTooLong(actual: Int, max: Int)

    public var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let s):     return "FunctionGemma model load failed: \(s)"
        case .tokenizerLoadFailed(let s): return "FunctionGemma tokenizer load failed: \(s)"
        case .inferenceFailed(let s):     return "FunctionGemma inference failed: \(s)"
        case .promptTooLong(let a, let m):
            return "FunctionGemma prompt is \(a) tokens, cache holds \(m)"
        }
    }
}
