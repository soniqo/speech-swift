import Foundation
import MLX
import XCTest
@testable import Qwen3Chat

/// XGrammar over a small synthetic vocabulary: no model, no tokenizer file.
final class XGrammarConstraintTests: XCTestCase {
    /// Printable ASCII one byte per token, a few multi-byte JSON pieces, then an end token.
    private static let pieces: [String] = {
        var p = (0x20 ..< 0x7F).map { String(UnicodeScalar(UInt8($0))) }
        p += ["{\"", "\":", "\",\"", "true", "false", "null", "\n  ", "Ω"]
        return p
    }()
    private static var endToken: Int { pieces.count }
    private static var size: Int { pieces.count + 1 }

    private func vocabulary() throws -> XGrammarVocabulary {
        var tokens: [[UInt8]?] = Self.pieces.map { Array($0.utf8) }
        tokens.append(nil)
        return try XGrammarVocabulary(size: Self.size, tokens: tokens, stopTokens: [Self.endToken])
    }

    private func admitted(_ mask: DeviceTokenMask) -> [Int] {
        guard case .packed(let packed) = mask else { XCTFail("expected a packed mask"); return [] }
        return (0 ..< Self.size).filter { packed.words[$0 / 32] & Int32(bitPattern: 1 << UInt32($0 % 32)) != 0 }
    }

    /// Always taking the admissible token that sorts last (so strings get content and the walk
    /// is not the trivial shortest document) must end in a parseable, schema-valid document.
    private func greedyWalk(_ schema: String, prefer: (Int, Int) -> Bool) throws -> String {
        let c = try vocabulary().constraint(schema: schema)
        var text = ""
        for _ in 0 ..< 400 {
            let ids = admitted(c.nextMask())
            XCTAssertFalse(ids.isEmpty, "no admissible token after \(text.debugDescription)")
            guard let id = ids.sorted(by: prefer).first else { break }
            if id == Self.endToken { break }
            XCTAssertTrue(c.accept(id))
            text += Self.pieces[id]
            if c.isDone { break }
        }
        XCTAssertTrue(c.isDone, "walk ended before the document was complete: \(text)")
        return text
    }

    func testWalksProduceSchemaValidDocuments() throws {
        let schema = #"{"type":"object","additionalProperties":false,"required":["flag","tags","n"],"properties":{"flag":{"type":"boolean"},"tags":{"type":"array","minItems":1,"maxItems":2,"items":{"type":"string","enum":["a","bc"]}},"n":{"type":"integer","minimum":1,"maximum":9}}}"#
        for prefer in [{ (a: Int, b: Int) in a < b }, { (a: Int, b: Int) in a > b }] {
            let text = try greedyWalk(schema, prefer: prefer)
            let object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any], text)
            XCTAssertEqual(Set(object.keys), ["flag", "tags", "n"], text)
            XCTAssertNotNil(object["flag"] as? Bool, text)
            let tags = try XCTUnwrap(object["tags"] as? [String], text)
            XCTAssertTrue((1 ... 2).contains(tags.count), text)
            XCTAssertTrue(tags.allSatisfy { ["a", "bc"].contains($0) }, text)
            let n = try XCTUnwrap(object["n"] as? Int, text)
            XCTAssertTrue((1 ... 9).contains(n), text)
        }
    }

    /// A non-empty array requirement is what the Discover planner's `queries` field needs.
    func testMinItemsForbidsClosingAnEmptyArray() throws {
        let schema = #"{"type":"object","required":["q"],"additionalProperties":false,"properties":{"q":{"type":"array","minItems":1,"items":{"type":"string"}}}}"#
        let c = try vocabulary().constraint(schema: schema)
        for piece in ["{\"", "q", "\":", "["] {
            XCTAssertTrue(c.accept(Self.pieces.firstIndex(of: piece)!), piece)
        }
        let ids = Set(admitted(c.nextMask()))
        XCTAssertFalse(ids.contains(Self.pieces.firstIndex(of: "]")!), "an empty array must not close")
        XCTAssertTrue(ids.contains(Self.pieces.firstIndex(of: "\"")!))
    }

    func testRejectedTokenLeavesTheMatcherUsable() throws {
        let c = try vocabulary().constraint(schema: #"{"type":"boolean"}"#)
        XCTAssertFalse(c.accept(Self.pieces.firstIndex(of: "{\"")!))
        XCTAssertTrue(c.accept(Self.pieces.firstIndex(of: "true")!))
        XCTAssertTrue(c.isDone)
        XCTAssertEqual(admitted(c.nextMask()), [Self.endToken], "only the end token may follow")
    }

    func testInvalidSchemaIsReported() throws {
        XCTAssertThrowsError(try vocabulary().constraint(schema: "{not json"))
    }

    /// The packed mask keeps exactly the set bits and floors the rest, including logits past the
    /// last whole word.
    func testPackedMaskAppliesBitForBit() {
        let vocab = 70
        var words = [Int32](repeating: 0, count: 3)
        let allowed = [0, 5, 31, 32, 63, 64, 69]
        for id in allowed { words[id / 32] |= Int32(bitPattern: 1 << UInt32(id % 32)) }
        let logits = MLXArray((0 ..< vocab).map { Float($0) })
        let masked = PackedTokenMask(words: words).apply(to: logits).asArray(Float.self)
        for id in 0 ..< vocab {
            if allowed.contains(id) {
                XCTAssertEqual(masked[id], Float(id), "id \(id)")
            } else {
                XCTAssertEqual(masked[id], -Float.greatestFiniteMagnitude, "id \(id)")
            }
        }
        XCTAssertTrue(PackedTokenMask(words: [0, 0]).isEmpty)
    }

    /// Both engines share one keyword gate, so a keyword the Swift grammar rejects never reaches
    /// XGrammar to be enforced in part.
    func testUnsupportedKeywordIsRejectedBeforeEitherEngineCompiles() throws {
        let schema = ##"{"type":"object","properties":{"a":{"$ref":"#/definitions/x"}},"definitions":{"x":{"type":"string"}}}"##
        XCTAssertThrowsError(try ChatResponseFormat.jsonSchema(schema).validate())
    }

    /// Fields come out in the `required` order, as the Swift matcher writes them, whatever order
    /// `properties` lists them in.
    func testRequiredOrderDecidesFieldOrder() throws {
        let schema = #"{"type":"object","additionalProperties":false,"required":["b","a"],"properties":{"a":{"type":"boolean"},"b":{"type":"boolean"}}}"#
        let text = try greedyWalk(schema, prefer: { $0 < $1 })
        let b = try XCTUnwrap(text.range(of: "\"b\""), text)
        let a = try XCTUnwrap(text.range(of: "\"a\""), text)
        XCTAssertLessThan(b.lowerBound, a.lowerBound, text)
    }
}
