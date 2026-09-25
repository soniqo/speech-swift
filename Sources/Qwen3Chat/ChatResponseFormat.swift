import Foundation

/// Shape the reply must take, enforced token by token during decoding.
///
/// Set on ``ChatSamplingConfig/responseFormat``. `nil` (the default) leaves decoding exactly as it
/// was. A backend that cannot enforce a format fails the request with
/// ``ChatResponseFormatError/unsupportedBackend(_:)`` rather than decoding unconstrained text.
public enum ChatResponseFormat: Sendable, Equatable {
    /// Exactly one JSON document that validates against this JSON Schema (the schema's JSON text).
    ///
    /// At every step only tokens that keep the output a prefix of such a document are sampled;
    /// the turn ends as soon as the document is complete. Nothing is emitted before or after it —
    /// no prose, no code fence, no reasoning channel. Properties are written in a fixed order: the
    /// schema's `required` list first, then its remaining `properties` in the order they appear.
    /// Whitespace between tokens is limited to one space or one newline followed by at most 20
    /// spaces or tabs.
    ///
    /// Supported keywords: `type` (a name or a list of names), `enum`, `const`, `properties`,
    /// `required`, `additionalProperties` (`false`, or `true`/absent — only declared properties are
    /// written — or a schema on an object without `properties`), `minProperties`,
    /// `maxProperties`, `items`, `minItems`, `maxItems`, `uniqueItems: false`, `minLength`,
    /// `maxLength`, `pattern`, integer `minimum`,
    /// `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `anyOf`, and OpenAPI `nullable`.
    /// Annotations (`title`, `description`, `default`, `examples`, `$schema`, `$id`, `$comment`,
    /// `deprecated`, `readOnly`, `writeOnly`) are ignored. Every other keyword — `$ref`, `oneOf`,
    /// `allOf`, `not`, `format`, `multipleOf`, number bounds, tuple `items`, … — is rejected with
    /// ``ChatResponseFormatError/unsupported(keyword:path:)``, so a schema is never silently
    /// enforced only in part.
    ///
    /// `pattern` accepts literal characters and escapes, `.`, `\d \D \w \W \s \S`, bracket
    /// classes, groups, `|`, `* + ? {n} {n,} {n,m}` and `^`/`$` at the ends; backreferences,
    /// lookaround, word boundaries and lazy quantifiers are rejected.
    ///
    /// The document can be cut short only by ``ChatSamplingConfig/maxTokens``.
    case jsonSchema(String)

    /// Compile the format without decoding anything, so a caller can reject a schema up front.
    public func validate() throws {
        switch self {
        case .jsonSchema(let schema): _ = try JSONSchemaGrammar(schema: schema)
        }
    }
}

extension ChatSamplingConfig {
    /// Fails for a backend that decodes without constraints when a format was requested.
    func requireNoResponseFormat(backend: String) throws {
        if responseFormat != nil { throw ChatResponseFormatError.unsupportedBackend(backend) }
    }
}

/// Why a ``ChatResponseFormat`` could not be applied.
public enum ChatResponseFormatError: LocalizedError, Equatable {
    /// The schema text is not JSON, or is JSON that is not a usable schema.
    case invalidSchema(String)
    /// The schema uses a keyword or keyword value the constraint cannot enforce.
    case unsupported(keyword: String, path: String)
    /// This chat backend does not implement constrained decoding.
    case unsupportedBackend(String)
    /// Decoding reached a state with no admissible token. Indicates a matcher defect; the partial
    /// output has been emitted.
    case noAdmissibleToken

    public var errorDescription: String? {
        switch self {
        case .invalidSchema(let reason): "Invalid JSON schema: \(reason)"
        case .unsupported(let keyword, let path):
            "JSON schema keyword '\(keyword)' at \(path) is not supported by constrained decoding"
        case .unsupportedBackend(let backend):
            "\(backend) does not support a constrained response format"
        case .noAdmissibleToken: "Constrained decoding reached a state with no admissible token"
        }
    }
}
