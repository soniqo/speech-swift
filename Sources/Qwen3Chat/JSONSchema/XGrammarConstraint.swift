import CXGrammarBridge
import Foundation
import MLX

/// One constrained decode, whichever engine enforces it.
///
/// A decode step asks for the mask of admissible next tokens, samples under it, and hands the
/// sampled id back. `isDone` ends the turn as soon as the document is complete.
protocol TokenDecodeConstraint {
    /// Admissible next tokens, in the form the device sampler applies.
    mutating func nextMask() -> DeviceTokenMask
    /// Consume a sampled token. False when it was not admissible.
    mutating func accept(_ id: Int) -> Bool
    /// The document is complete; nothing may follow it.
    var isDone: Bool { get }
}

extension JSONTokenConstraint: TokenDecodeConstraint {
    mutating func nextMask() -> DeviceTokenMask {
        .allowance(JSONAllowanceMask(cleanChars: vocabulary.cleanChars, allowance: allowance()))
    }
}

/// Which engine enforces ``ChatResponseFormat/jsonSchema(_:)``.
///
/// XGrammar (the engine Ollama's MLX runner uses for structured outputs) precomputes
/// context-independent token masks when a grammar is compiled, so a step costs one bitmask fill.
/// The Swift matcher walks the vocabulary trie per step. Both accept exactly the keywords
/// ``JSONSchemaGrammar`` accepts; every other keyword is rejected before either compiles.
enum JSONConstraintEngine: String {
    case xgrammar
    case swift

    /// `SPEECH_SWIFT_JSON_CONSTRAINT=swift` selects the Swift matcher, for comparison runs.
    static var current: JSONConstraintEngine {
        ProcessInfo.processInfo.environment["SPEECH_SWIFT_JSON_CONSTRAINT"]
            .flatMap(JSONConstraintEngine.init(rawValue:)) ?? .xgrammar
    }
}

/// XGrammar's compiler over one model's vocabulary. Built once per model; compiled grammars are
/// cached inside it, so a schema repeated across requests compiles once.
final class XGrammarVocabulary: @unchecked Sendable {
    let size: Int
    private let compiler: OpaquePointer
    private let lock = NSLock()

    /// Whitespace allowed between JSON tokens: the Swift matcher's limit (one newline and up to
    /// twenty spaces or tabs). Unbounded whitespace is how a constrained decode runs to its
    /// token budget emitting nothing but indentation.
    static let maxWhitespace: Int32 = 21

    /// - Parameters:
    ///   - size: the logits width.
    ///   - tokens: decoded bytes per id; `nil` or empty for ids that must never be admitted.
    ///   - stopTokens: end-of-turn ids, admitted once the document is complete.
    init(size: Int, tokens: [[UInt8]?], stopTokens: [Int]) throws {
        self.size = size
        let count = min(size, tokens.count)
        var buffers: [UnsafeMutablePointer<CChar>?] = []
        var lengths: [Int32] = []
        buffers.reserveCapacity(count)
        lengths.reserveCapacity(count)
        defer { buffers.forEach { $0?.deallocate() } }
        for id in 0 ..< count {
            guard let bytes = tokens[id], !bytes.isEmpty else {
                buffers.append(nil)
                lengths.append(0)
                continue
            }
            let buffer = UnsafeMutablePointer<CChar>.allocate(capacity: bytes.count)
            bytes.withUnsafeBufferPointer { source in
                source.withMemoryRebound(to: CChar.self) {
                    buffer.initialize(from: $0.baseAddress!, count: bytes.count)
                }
            }
            buffers.append(buffer)
            lengths.append(Int32(bytes.count))
        }
        let stops = stopTokens.map(Int32.init)
        var error = [CChar](repeating: 0, count: 512)
        let created: OpaquePointer? = buffers.withUnsafeBufferPointer { pointers in
            pointers.withMemoryRebound(to: UnsafePointer<CChar>?.self) { tokens in
                xgb_compiler_create(
                    tokens.baseAddress, lengths, Int32(count), Int32(size),
                    stops, Int32(stops.count), 8, &error, error.count)
            }
        }
        guard let created else {
            throw ChatResponseFormatError.invalidSchema(
                "XGrammar vocabulary: \(String(cString: error))")
        }
        compiler = created
    }

    convenience init(gemma tokenizer: Gemma4Tokenizer, size: Int) throws {
        var tokens = [[UInt8]?](repeating: nil, count: size)
        for id in 0 ..< size where !tokenizer.isSpecialToken(id) {
            let bytes = tokenizer.tokenBytes(id)
            if !bytes.isEmpty { tokens[id] = bytes }
        }
        try self.init(size: size, tokens: tokens, stopTokens: tokenizer.eosTokenIds.sorted())
    }

    deinit { xgb_compiler_free(compiler) }

    func constraint(schema: String) throws -> XGrammarConstraint {
        // XGrammar writes properties in the order `properties` lists them; the Swift matcher, and
        // the prompts written for it, put the `required` list first. Reorder so both engines
        // produce the same field order.
        let ordered = Self.requiredPropertiesFirst(try OrderedJSON.parse(schema))
        let text = String(decoding: ordered.canonicalBytes, as: UTF8.self)
        var error = [CChar](repeating: 0, count: 512)
        lock.lock()
        let grammar = xgb_compile_json_schema(compiler, text, Self.maxWhitespace, &error, error.count)
        lock.unlock()
        guard let grammar else {
            throw ChatResponseFormatError.invalidSchema(String(cString: error))
        }
        return try XGrammarConstraint(grammar: grammar, vocabularySize: size)
    }
}

extension XGrammarVocabulary {
    /// Every object schema's `properties`, reordered: the `required` names in their listed order,
    /// then the remaining properties in the order they were written.
    static func requiredPropertiesFirst(_ json: OrderedJSON) -> OrderedJSON {
        switch json {
        case .array(let items):
            return .array(items.map(requiredPropertiesFirst))
        case .object(let members):
            var result = members.map { (key: $0.key, value: requiredPropertiesFirst($0.value)) }
            guard let p = result.firstIndex(where: { $0.key == "properties" }),
                  case .object(let properties) = result[p].value,
                  case .array(let required)? = json["required"]
            else { return .object(result) }
            let names = required.compactMap { value -> String? in
                if case .string(let name) = value { return name }
                return nil
            }
            var reordered: [(key: String, value: OrderedJSON)] = names.compactMap { name in
                properties.first { $0.key == name }
            }
            reordered += properties.filter { !names.contains($0.key) }
            result[p].value = .object(reordered)
            return .object(result)
        default:
            return json
        }
    }
}

/// One XGrammar-constrained decode.
final class XGrammarConstraint: TokenDecodeConstraint {
    private let grammar: OpaquePointer
    private let matcher: OpaquePointer
    private var words: [Int32]
    let vocabularySize: Int

    fileprivate init(grammar: OpaquePointer, vocabularySize: Int) throws {
        guard let matcher = xgb_matcher_create(grammar) else {
            xgb_grammar_free(grammar)
            throw ChatResponseFormatError.invalidSchema("XGrammar could not create a matcher")
        }
        self.grammar = grammar
        self.matcher = matcher
        self.vocabularySize = vocabularySize
        self.words = [Int32](repeating: 0, count: Int(xgb_bitmask_words(Int32(vocabularySize))))
    }

    deinit {
        xgb_matcher_free(matcher)
        xgb_grammar_free(grammar)
    }

    var isDone: Bool { xgb_matcher_is_completed(matcher) }

    func accept(_ id: Int) -> Bool {
        guard id >= 0, id < vocabularySize else { return false }
        return xgb_matcher_accept(matcher, Int32(id))
    }

    /// The packed mask for the next step. An unqueryable matcher yields an empty mask, which the
    /// decode loop reports as ``ChatResponseFormatError/noAdmissibleToken``.
    func nextMask() -> DeviceTokenMask {
        let filled = words.withUnsafeMutableBufferPointer {
            xgb_matcher_fill_bitmask(matcher, $0.baseAddress, Int32($0.count))
        }
        if !filled { words = [Int32](repeating: 0, count: words.count) }
        return .packed(PackedTokenMask(words: words))
    }
}

/// XGrammar's packed bitmask: bit `i % 32` of word `i / 32` is set when token `i` is admissible.
struct PackedTokenMask {
    let words: [Int32]

    var isEmpty: Bool { !words.contains { $0 != 0 } }

    /// Admitted logits pass, every other one becomes `-greatestFiniteMagnitude` — the value
    /// end-token suppression uses, so penalty, top-K, top-P and argmax treat both alike. The
    /// words are expanded to one flag per token on the device.
    func apply(to logits: MLXArray) -> MLXArray {
        let vocab = logits.dim(0)
        let packed = MLXArray(words.map { UInt32(bitPattern: $0) })
        let shifts = MLXArray((0 ..< 32).map { UInt32($0) })
        let bits = (packed.reshaped([words.count, 1]) >> shifts) & MLXArray(UInt32(1))
        var flags = bits.reshaped([words.count * 32])
        if flags.dim(0) > vocab { flags = flags[0 ..< vocab] }
        let floor = MLXArray(-Float.greatestFiniteMagnitude)
        if flags.dim(0) < vocab {
            var masked = full([vocab], values: floor)
            let n = flags.dim(0)
            masked[0 ..< n] = which(flags .!= MLXArray(UInt32(0)), logits[0 ..< n], floor)
            return masked
        }
        return which(flags .!= MLXArray(UInt32(0)), logits, floor)
    }
}
