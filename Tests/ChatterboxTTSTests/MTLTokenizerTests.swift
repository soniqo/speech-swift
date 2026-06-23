import Hub
import XCTest

@testable import ChatterboxTTS

/// Numeric gate for the Swift `MTLTokenizer`: token ids must match the
/// reference golden exactly for the frontend-free languages we support first
/// (en / ar / hi).
final class MTLTokenizerTests: XCTestCase {
    private func loadTokenizer() async throws -> MTLTokenizer {
        let folder = try await HubApi().snapshot(
            from: "aufklarer/Chatterbox-Multilingual-MLX-fp16",
            matching: ["tokenizer.json"]
        )
        return try MTLTokenizer(modelFolder: folder)
    }

    func testEncodeMatchesReferenceGolden() async throws {
        let tok = try await loadTokenizer()

        // Golden ids from the reference tokenizer over aufklarer/…-fp16.
        XCTAssertEqual(
            tok.encode("Hello there, friend.", languageId: "en"),
            [708, 62, 84, 28, 2, 172, 7, 2, 19, 101, 204, 9])
        XCTAssertEqual(
            tok.encode("مرحبا بك", languageId: "ar"),
            [721, 1491, 1471, 1467, 1462, 1456, 2, 1462, 1489])
        XCTAssertEqual(
            tok.encode("नमस्ते दोस्त", languageId: "hi"),
            [722, 1706, 1712, 1720, 1740, 1702, 1734, 2, 1704, 1738, 1720, 1740, 1702])
    }

    func testLanguageTokenPrepended() async throws {
        let tok = try await loadTokenizer()
        // The first id is the [lang] control token: [en]=708, [ar]=721, [hi]=722.
        XCTAssertEqual(tok.encode("a", languageId: "en").first, 708)
        XCTAssertEqual(tok.encode("a", languageId: "ar").first, 721)
        XCTAssertEqual(tok.encode("a", languageId: "hi").first, 722)
    }
}
