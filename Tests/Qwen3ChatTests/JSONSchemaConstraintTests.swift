import XCTest
import Foundation
@testable import Qwen3Chat

/// The JSON-schema matcher on its own: which byte strings are prefixes of a valid document,
/// which are complete documents, and which schemas are refused. No MLX, no model.
final class JSONSchemaMatcherTests: XCTestCase {

    /// A source-linked fact extraction schema, serialised with sorted keys the way a host that
    /// does not preserve member order sends it.
    static let factSchema = #"{"additionalProperties":false,"properties":{"facts":{"items":{"additionalProperties":false,"properties":{"candidate":{"maxLength":640,"minLength":1,"type":"string"},"property":{"additionalProperties":false,"properties":{"text":{"type":"string"},"turn_ids":{"items":{"pattern":"^T[1-9][0-9]*$","type":"string"},"maxItems":4,"type":"array"}},"required":["text","turn_ids"],"type":"object"},"status":{"additionalProperties":false,"properties":{"text":{"type":"string"},"turn_ids":{"items":{"pattern":"^T[1-9][0-9]*$","type":"string"},"maxItems":4,"type":"array"}},"required":["text","turn_ids"],"type":"object"},"subject":{"additionalProperties":false,"properties":{"text":{"type":"string"},"turn_ids":{"items":{"pattern":"^T[1-9][0-9]*$","type":"string"},"maxItems":4,"type":"array"}},"required":["text","turn_ids"],"type":"object"},"timeframe":{"additionalProperties":false,"properties":{"text":{"type":"string"},"turn_ids":{"items":{"pattern":"^T[1-9][0-9]*$","type":"string"},"maxItems":4,"type":"array"}},"required":["text","turn_ids"],"type":"object"},"unit":{"additionalProperties":false,"properties":{"text":{"type":"string"},"turn_ids":{"items":{"pattern":"^T[1-9][0-9]*$","type":"string"},"maxItems":4,"type":"array"}},"required":["text","turn_ids"],"type":"object"},"value":{"additionalProperties":false,"properties":{"text":{"type":"string"},"turn_ids":{"items":{"pattern":"^T[1-9][0-9]*$","type":"string"},"maxItems":4,"type":"array"}},"required":["text","turn_ids"],"type":"object"}},"required":["candidate","subject","property","value","unit","status","timeframe"],"type":"object"},"maxItems":8,"type":"array"}},"required":["facts"],"type":"object"}"#

    static let validFactDocument = #"{"facts":[{"candidate":"forty thousand euros","subject":{"text":"the pilot","turn_ids":["T2"]},"property":{"text":"budget","turn_ids":["T2"]},"value":{"text":"forty thousand euros","turn_ids":["T2","T3"]},"unit":{"text":"","turn_ids":[]},"status":{"text":"accepted","turn_ids":["T3"]},"timeframe":{"text":"","turn_ids":[]}}]}"#

    private func matcher(_ schema: String, file: StaticString = #filePath, line: UInt = #line) throws -> JSONSchemaMatcher {
        JSONSchemaMatcher(grammar: try JSONSchemaGrammar(schema: schema))
    }

    /// nil when some byte is refused; otherwise whether the whole text is a complete document.
    private func run(_ m: JSONSchemaMatcher, _ text: String) -> Bool? {
        let states = m.advance(m.initial, bytes: Array(text.utf8))
        return states.isEmpty ? nil : m.canEnd(states)
    }

    private func assertComplete(_ schema: String, _ text: String, file: StaticString = #filePath, line: UInt = #line) throws {
        XCTAssertEqual(run(try matcher(schema), text), true, "should be a complete document: \(text)", file: file, line: line)
    }

    private func assertPrefixOnly(_ schema: String, _ text: String, file: StaticString = #filePath, line: UInt = #line) throws {
        XCTAssertEqual(run(try matcher(schema), text), false, "should be an incomplete prefix: \(text)", file: file, line: line)
    }

    private func assertRefused(_ schema: String, _ text: String, file: StaticString = #filePath, line: UInt = #line) throws {
        XCTAssertNil(run(try matcher(schema), text), "should be refused: \(text)", file: file, line: line)
    }

    // MARK: Objects

    func testFactDocumentIsAccepted() throws {
        try assertComplete(Self.factSchema, Self.validFactDocument)
        try assertComplete(Self.factSchema, #"{"facts":[]}"#)
        try assertPrefixOnly(Self.factSchema, #"{"facts":[{"candidate":"x","subject":"#)
    }

    func testTheFailureShapesSeenUnconstrainedAreRefused() throws {
        // A bare array where the object is required.
        try assertRefused(Self.factSchema, #"[{"candidate":"x"}]"#)
        // A string where an object is required.
        try assertRefused(Self.factSchema, #"{"facts":[{"candidate":"x","subject":"the pilot""#)
        // A field nested inside another field.
        try assertRefused(Self.factSchema, #"{"facts":[{"candidate":"x","subject":{"text":"a","property":"#)
        // Prose before the document.
        try assertRefused(Self.factSchema, #"Here is the JSON: {"#)
        // A code fence.
        try assertRefused(Self.factSchema, "```json\n{")
    }

    func testPropertiesFollowRequiredOrderAndAreAllPresent() throws {
        let schema = #"{"type":"object","additionalProperties":false,"required":["b","a"],"properties":{"a":{"type":"integer"},"b":{"type":"integer"},"c":{"type":"integer"}}}"#
        try assertComplete(schema, #"{"b":1,"a":2}"#)
        try assertComplete(schema, #"{"b":1,"a":2,"c":3}"#)
        try assertRefused(schema, #"{"a":"#)            // wrong order
        try assertRefused(schema, #"{"b":1}"#)           // required a missing
        try assertRefused(schema, #"{"b":1,"a":2,"d""#)  // undeclared key
        try assertRefused(schema, #"{"b":1,"b""#)        // repeated key
    }

    func testOptionalPropertiesMayBeSkippedInOrder() throws {
        let schema = #"{"type":"object","properties":{"x":{"type":"string"},"y":{"type":"string"},"z":{"type":"string"}}}"#
        try assertComplete(schema, #"{}"#)
        try assertComplete(schema, #"{"y":"b"}"#)
        try assertComplete(schema, #"{"x":"a","z":"c"}"#)
        try assertRefused(schema, #"{"z":"c","x""#)
    }

    func testPropertyCountBounds() throws {
        let schema = #"{"type":"object","maxProperties":2,"minProperties":1,"properties":{"x":{"type":"boolean"},"y":{"type":"boolean"},"z":{"type":"boolean"}}}"#
        try assertRefused(schema, "{}")
        try assertComplete(schema, #"{"z":true}"#)
        try assertComplete(schema, #"{"x":true,"y":false}"#)
        try assertRefused(schema, #"{"x":true,"y":false,"#)
    }

    func testFreeFormObjectAndValue() throws {
        try assertComplete(#"{"type":"object"}"#, #"{"any":[1,{"k":null}],"b":"s"}"#)
        try assertComplete("{}", #"[true,-1.5e3,"x"]"#)
        try assertComplete(#"{"type":"object","additionalProperties":{"type":"integer"}}"#, #"{"a":1,"b":2}"#)
        try assertRefused(#"{"type":"object","additionalProperties":{"type":"integer"}}"#, #"{"a":"#
            + #""x""#)
    }

    // MARK: Arrays

    func testArrayBounds() throws {
        let schema = #"{"type":"array","minItems":1,"maxItems":2,"items":{"type":"integer"}}"#
        try assertRefused(schema, "[]")
        try assertComplete(schema, "[1]")
        try assertComplete(schema, "[1,2]")
        try assertRefused(schema, "[1,2,")
        try assertComplete(#"{"type":"array","maxItems":0}"#, "[]")
        try assertRefused(#"{"type":"array","maxItems":0}"#, "[1")
    }

    // MARK: Strings

    func testEscapesAndUnicode() throws {
        let schema = #"{"type":"string"}"#
        try assertComplete(schema, #""a\"b\\c\/d\n\té""#)
        try assertComplete(schema, "\"Привет, 世界 🙂\"")
        try assertRefused(schema, "\"a\nb\"")           // raw control character
        try assertRefused(schema, #""\x""#)              // invalid escape
        try assertRefused(schema, #""\u12G"#)            // invalid hex
        try assertRefused(schema, #""\ud83d"#)           // surrogate half
        let m = try matcher(schema)
        XCTAssertTrue(m.advance(m.initial, bytes: [0x22, 0x80]).isEmpty, "lone continuation byte")
        XCTAssertTrue(m.advance(m.initial, bytes: [0x22, 0xC0, 0xAF]).isEmpty, "overlong encoding")
        XCTAssertTrue(m.advance(m.initial, bytes: [0x22, 0xED, 0xA0]).isEmpty, "encoded surrogate")
        XCTAssertFalse(m.advance(m.initial, bytes: [0x22, 0xD0]).isEmpty, "a lead byte waits for its continuation")
        XCTAssertTrue(m.advance(m.initial, bytes: [0x22, 0xD0, 0x22]).isEmpty, "a character cut short")
    }

    func testStringLengthCountsCharacters() throws {
        let schema = #"{"type":"string","minLength":2,"maxLength":3}"#
        try assertRefused(schema, #""a""#)
        try assertComplete(schema, #""ab""#)
        try assertComplete(schema, "\"жжж\"")
        try assertComplete(schema, #""a\nb""#)
        try assertRefused(schema, #""abcd"#)
    }

    func testEnums() throws {
        let schema = #"{"type":"string","enum":["support","refute","unresolved"]}"#
        try assertComplete(schema, #""refute""#)
        try assertPrefixOnly(schema, #""re"#)
        try assertRefused(schema, #""rex"#)
        try assertRefused(schema, #""support "#)
        try assertComplete(#"{"enum":[1,10,"a",null]}"#, "1")
        try assertComplete(#"{"enum":[1,10,"a",null]}"#, "10")
        try assertComplete(#"{"type":"array","items":{"enum":[1,10]}}"#, "[1,10]")
        try assertComplete(#"{"const":{"k":[true]}}"#, #"{"k":[true]}"#)
        XCTAssertThrowsError(try JSONSchemaGrammar(schema: #"{"type":"string","enum":[1]}"#))
    }

    func testPatterns() throws {
        let schema = #"{"type":"string","pattern":"^T[1-9][0-9]*$"}"#
        try assertComplete(schema, #""T1""#)
        try assertComplete(schema, #""T204""#)
        try assertRefused(schema, #""T0"#)
        try assertRefused(schema, #""T""#)
        try assertRefused(schema, #""t1"#)
        try assertRefused(schema, #""T1 "#)
        let word = #"{"type":"string","minLength":1,"pattern":"^\\S+$"}"#
        try assertComplete(word, #""budget""#)
        try assertRefused(word, #""two words"#)
        // Unanchored patterns match anywhere, as JSON Schema specifies.
        try assertComplete(#"{"type":"string","pattern":"ab"}"#, #""xxabyy""#)
        try assertRefused(#"{"type":"string","pattern":"ab"}"#, #""xxa""#)
        try assertComplete(#"{"type":"string","pattern":"^(?:[a-c]{2}|x+)-\\d?$"}"#, #""bc-7""#)
        try assertComplete(#"{"type":"string","pattern":"^(?:[a-c]{2}|x+)-\\d?$"}"#, #""xxx-""#)
        try assertRefused(#"{"type":"string","pattern":"^(?:[a-c]{2}|x+)-\\d?$"}"#, #""abc"#)
        try assertComplete(#"{"type":"string","pattern":"^[^\"\\\\]*$"}"#, #""plain""#)
        // A match must stay reachable inside maxLength.
        let short = #"{"type":"string","maxLength":3,"pattern":"^T[1-9][0-9]*$"}"#
        try assertComplete(short, #""T12""#)
        try assertRefused(short, #""T123"#)
        try assertRefused(#"{"type":"string","maxLength":1,"pattern":"^T[1-9]$"}"#, "\"")
    }

    // MARK: Numbers

    func testIntegerBounds() throws {
        let schema = #"{"type":"integer","minimum":0,"maximum":256}"#
        try assertComplete(schema, "0")
        try assertComplete(schema, "256")
        try assertComplete(schema, "25")
        try assertRefused(schema, "257")
        try assertRefused(schema, "01")
        try assertRefused(schema, "-1")
        try assertRefused(schema, "2567")
        let m = try matcher(#"{"type":"array","items":{"type":"integer","minimum":1,"maximum":16}}"#)
        XCTAssertEqual(run(m, "[1,16]"), true)
        XCTAssertNil(run(m, "[0"))
        XCTAssertNil(run(m, "[17"))
        try assertComplete(#"{"type":"integer","exclusiveMinimum":-3,"exclusiveMaximum":3}"#, "-2")
        try assertRefused(#"{"type":"integer","exclusiveMinimum":-3,"exclusiveMaximum":3}"#, "-3")
        try assertRefused(#"{"type":"integer"}"#, "1.5")
        try assertRefused(#"{"type":"integer"}"#, "1234567890123456789")
    }

    func testNumbers() throws {
        let schema = #"{"type":"number"}"#
        for good in ["0", "-0.5", "12.25e-3", "7E+2", "3"] { try assertComplete(schema, good) }
        for bad in ["01", "1.", ".5", "1e", "--1", "+1"] {
            XCTAssertNotEqual(run(try matcher(schema), bad), true, bad)
        }
    }

    // MARK: Unions, null, whitespace

    func testUnionsAndNullable() throws {
        let witness = #"{"type":"object","additionalProperties":false,"required":["turn_id","quote"],"properties":{"turn_id":{"type":"string","pattern":"^T[1-9][0-9]*$"},"quote":{"type":"string","minLength":1,"maxLength":1200}}}"#
        let schema = #"{"anyOf":["# + witness + #",{"type":"null"}]}"#
        try assertComplete(schema, "null")
        try assertComplete(schema, #"{"turn_id":"T3","quote":"we agreed"}"#)
        try assertRefused(schema, "nul1")
        try assertComplete(#"{"type":["string","null"]}"#, "null")
        try assertComplete(#"{"type":["string","null"]}"#, #""x""#)
        try assertComplete(#"{"type":"integer","nullable":true}"#, "null")
        try assertComplete(#"{"type":"boolean"}"#, "false")
    }

    func testWhitespaceIsBounded() throws {
        let schema = #"{"type":"object","properties":{"a":{"type":"array","items":{"type":"integer"}}}}"#
        try assertComplete(schema, "{ \"a\": [ 1 , 2 ] }")
        try assertComplete(schema, "{\n  \"a\":\n    [1,\n    2]\n}")
        try assertRefused(schema, "{  ")                 // two spaces
        try assertRefused(schema, "{\n\n")               // two newlines
        try assertRefused(schema, " {")                  // before the document
        try assertRefused(schema, "{}\n")                // after the document
        try assertRefused(schema, "{\n" + String(repeating: " ", count: 21))
    }

    func testCompleteDocumentTakesNothingMore() throws {
        let m = try matcher(#"{"type":"object"}"#)
        let done = m.advance(m.initial, bytes: Array("{}".utf8))
        XCTAssertTrue(m.isDone(done))
        XCTAssertTrue(m.advance(done, byte: 0x20).isEmpty)
        // A root integer may end, but could also continue.
        let n = try matcher(#"{"type":"integer"}"#)
        let twelve = n.advance(n.initial, bytes: Array("12".utf8))
        XCTAssertTrue(n.canEnd(twelve))
        XCTAssertFalse(n.isDone(twelve))
    }

    // MARK: Schema compilation

    func testUnsupportedKeywordsAreRefusedExplicitly() {
        let refused = [
            #"{"oneOf":[{"type":"string"}]}"#,
            #"{"allOf":[{"type":"string"}]}"#,
            ##"{"$ref":"#/$defs/x"}"##,
            #"{"type":"string","format":"date"}"#,
            #"{"type":"number","minimum":0}"#,
            #"{"type":"integer","multipleOf":2}"#,
            #"{"type":"array","uniqueItems":true}"#,
            #"{"type":"array","items":[{"type":"string"}]}"#,
            #"{"type":"object","patternProperties":{}}"#,
            #"{"type":"string","pattern":"(?=a)"}"#,
            #"{"type":"string","pattern":"(a)\\1"}"#,
            #"{"type":"string","pattern":"\\bword"}"#,
            #"{"type":"string","pattern":"a+?"}"#,
            #"{"type":"string","anyOf":[{"minLength":1}]}"#,
            "false",
        ]
        for schema in refused {
            XCTAssertThrowsError(try JSONSchemaGrammar(schema: schema), schema) { error in
                guard case ChatResponseFormatError.unsupported = error else {
                    return XCTFail("\(schema): expected .unsupported, got \(error)")
                }
            }
        }
        for invalid in ["{", #"{"type":"strin"}"#, #"{"type":"integer","minimum":5,"maximum":1}"#,
                        #"{"required":["a"],"properties":{},"additionalProperties":false,"maxProperties":0}"#] {
            XCTAssertThrowsError(try JSONSchemaGrammar(schema: invalid), invalid)
        }
        XCTAssertNoThrow(try ChatResponseFormat.jsonSchema(Self.factSchema).validate())
        XCTAssertNoThrow(try JSONSchemaGrammar(schema: #"{"type":"string","title":"t","description":"d","default":"x"}"#))
    }
}

/// Token-level masks over a small vocabulary whose tokens straddle JSON structure the way real
/// BPE tokens do (`"},{"`, `":"`, a multi-byte character split into byte tokens).
final class JSONTokenConstraintTests: XCTestCase {

    static let tokens: [String?] = [
        nil,                               // 0: a special token — never admissible
        "{", "}", "[", "]", ",", ":", "\"", "{\"", "\":", "\":\"", "\",\"", "\"},{\"", "\"}]}",
        "\"]", "\"}", "\"]}", "\",", "\":[", "\":[\"", "\"],\"", "]}", "}]}", "},{", "[]", "[\"",
        "facts", "candidate", "subject", "property", "value", "unit", "status", "timeframe",
        "text", "turn", "_ids", "turn_ids", "T", "T1", "1", "12", "0", "2",
        " ", "\n", "\n  ", "  ", "a", "ab", "the pilot", " budget", "forty", " thousand", "é",
        "\\", "\\\"", "\\n", "\\u", "00e9", "e9", "x\"", "abc\"}", "ok\",\"",
        "<0xC3>", "<0xA9>", "\u{1F642}", "true", "false", "null", "-", "ok",
    ]
    static let eos = 599
    /// Like a byte-fallback vocabulary, every byte also has a one-byte token, at `byteBase + b`.
    static let byteBase = 300

    static func vocabulary() -> JSONTokenVocabulary {
        var bytes: [[UInt8]?] = tokens.map { t in
            guard let t else { return nil }
            if t.hasPrefix("<0x"), let b = UInt8(t.dropFirst(3).dropLast(), radix: 16) { return [b] }
            return Array(t.utf8)
        }
        bytes += Array(repeating: nil, count: byteBase - bytes.count)
        bytes += (0 ... 255).map { [UInt8($0)] }
        return JSONTokenVocabulary(size: 600, tokens: bytes)
    }

    private func constraint(_ schema: String) throws -> JSONTokenConstraint {
        JSONTokenConstraint(grammar: try JSONSchemaGrammar(schema: schema),
                            vocabulary: Self.vocabulary(), endTokens: [Self.eos])
    }

    /// Every id checked one at a time against the matcher: the reference for the trie walk and
    /// the clean-token shortcut.
    private func bruteForce(_ c: JSONTokenConstraint) -> Set<Int> {
        var out = Set<Int>()
        for (id, b) in c.vocabulary.tokenBytes.enumerated() where !b.isEmpty {
            if !c.matcher.advance(c.states, bytes: b).isEmpty { out.insert(id) }
        }
        if c.matcher.canEnd(c.states) { out.insert(Self.eos) }
        return out
    }

    /// Walks a valid document byte by byte and checks the admissible set at every prefix.
    private func assertMasksMatchBruteForce(_ schema: String, _ document: String,
                                            file: StaticString = #filePath, line: UInt = #line) throws {
        var c = try constraint(schema)
        let m = c.matcher
        var states = m.initial
        for b in Array(document.utf8) {
            XCTAssertEqual(c.allowedIDs(), bruteForce(c), "at \(String(decoding: [b], as: UTF8.self))",
                           file: file, line: line)
            states = m.advance(states, byte: b)
            XCTAssertFalse(states.isEmpty, file: file, line: line)
            c = JSONTokenConstraint(copying: c, states: states)
        }
        XCTAssertEqual(c.allowedIDs(), bruteForce(c), file: file, line: line)
    }

    func testMasksMatchBruteForceAcrossADocument() throws {
        try assertMasksMatchBruteForce(JSONSchemaMatcherTests.factSchema, JSONSchemaMatcherTests.validFactDocument)
        try assertMasksMatchBruteForce(#"{"type":"object","properties":{"s":{"type":"string","maxLength":4}}}"#,
                                       #"{"s":"a\nbé"}"#)
        try assertMasksMatchBruteForce(#"{"anyOf":[{"type":"string","maxLength":2},{"type":"string","maxLength":9}]}"#,
                                       #""forty""#)
    }

    func testStructuralTokensSpanningBoundariesAreAdmitted() throws {
        var c = try constraint(JSONSchemaMatcherTests.factSchema)
        let id = { (s: String) in Self.tokens.firstIndex(of: s)! }
        let allowed0 = c.allowedIDs()
        XCTAssertTrue(allowed0.contains(id("{\"")))
        XCTAssertFalse(allowed0.contains(id("[")), "the root must be an object")
        XCTAssertFalse(allowed0.contains(0), "special tokens are never admissible")
        XCTAssertFalse(allowed0.contains(Self.eos))
        for t in ["{\"", "facts", "\":[", "{\"", "candidate", "\":\"", "the pilot"] {
            XCTAssertTrue(c.allowedIDs().contains(id(t)), t)
            XCTAssertTrue(c.accept(id(t)), t)
        }
        let inCandidate = c.allowedIDs()
        XCTAssertTrue(inCandidate.contains(id("\",\"")), "closing the string straight into the next key")
        XCTAssertFalse(inCandidate.contains(id("\"}]}")), "the object still needs its required fields")
        XCTAssertFalse(inCandidate.contains(id("\n")), "raw newline inside a string")
        XCTAssertTrue(inCandidate.contains(id("\\n")))
        XCTAssertTrue(inCandidate.contains(id("<0xC3>")))
        XCTAssertTrue(c.accept(id("<0xC3>")))
        let continuation = c.allowedIDs()
        XCTAssertTrue(continuation.contains(id("<0xA9>")))
        XCTAssertTrue(continuation.allSatisfy {
            let b = c.vocabulary.tokenBytes[$0]
            return b.count == 1 && (0x80 ... 0xBF).contains(b[0])
        }, "only a continuation byte may follow a lead byte")
        XCTAssertFalse(c.accept(id("a")))
    }

    func testDocumentEndIsForced() throws {
        var c = try constraint(#"{"type":"object","properties":{"ok":{"type":"boolean"}}}"#)
        let id = { (s: String) in Self.tokens.firstIndex(of: s)! }
        for t in ["{\"", "ok", "\":", "true"] { XCTAssertTrue(c.accept(id(t)), t) }
        XCTAssertFalse(c.isDone)
        XCTAssertFalse(c.allowedIDs().contains(Self.eos))
        XCTAssertTrue(c.accept(id("}")))
        XCTAssertTrue(c.isDone)
        XCTAssertEqual(c.allowedIDs(), [Self.eos])
    }

    /// Random walks through the admissible sets never strand the decoder and always end in a
    /// document that parses.
    func testRandomWalksAlwaysFinishWithValidJSON() throws {
        var rng = SeededGenerator(seed: 7)
        let schemas = [
            JSONSchemaMatcherTests.factSchema,
            #"{"type":"object","additionalProperties":false,"required":["verdict","reason"],"properties":{"verdict":{"type":"string","enum":["support","refute"]},"reason":{"type":"string","maxLength":12},"n":{"type":"integer","minimum":1,"maximum":16}}}"#,
            #"{"type":"array","minItems":1,"maxItems":3,"items":{"anyOf":[{"type":"null"},{"type":"string","pattern":"^T[1-9][0-9]*$"}]}}"#,
        ]
        for schema in schemas {
            for _ in 0 ..< 40 {
                var c = try constraint(schema)
                var output: [UInt8] = []
                var steps = 0
                while !c.isDone, steps < 400 {
                    steps += 1
                    let allowed = c.allowedIDs()
                    XCTAssertFalse(allowed.isEmpty, "stranded after \(String(decoding: output, as: UTF8.self))")
                    if allowed.isEmpty { break }
                    // Favour structural tokens so free strings end.
                    let structural = allowed.filter { $0 == Self.eos || c.vocabulary.cleanCharsHost[$0] == .max }
                    let pool = (!structural.isEmpty && Bool.random(using: &rng)) ? structural : allowed
                    let pick = pool.sorted()[Int.random(in: 0 ..< pool.count, using: &rng)]
                    if pick == Self.eos { break }
                    XCTAssertTrue(c.accept(pick))
                    output += c.vocabulary.tokenBytes[pick]
                }
                guard c.isDone || c.matcher.canEnd(c.states) else { continue }   // step cap reached
                let text = String(decoding: output, as: UTF8.self)
                XCTAssertNoThrow(try JSONSerialization.jsonObject(with: Data(output), options: [.fragmentsAllowed]), text)
            }
        }
    }
}

extension JSONTokenConstraint {
    /// The same constraint at other matcher states — lets tests step byte by byte.
    init(copying other: JSONTokenConstraint, states: [JSONSchemaMatcher.Stack]) {
        self = other
        self.replaceStates(states)
    }
}

struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }
}
