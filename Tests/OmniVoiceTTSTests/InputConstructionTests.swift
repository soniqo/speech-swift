import Foundation
import MLX
import XCTest

@testable import OmniVoiceTTS

/// Gates the OmniVoice text front-end: `OmniVoiceInputBuilder.buildInputs(...)`
/// must reproduce the captured golden conditioning tensor + audio mask
/// (`det_input_ids[0]` / `det_audio_mask[0]`) EXACTLY (integer-equal).
///
/// Golden capture params (see the task brief):
///   text     = "The quick brown fox jumps over the lazy dog."
///   ref_text = "Hello there, this is a cloned voice."
///   lang     = "en", instruct = None, target_len = 63, denoise = default (True)
///
/// Fixtures under `/tmp/omnivoice_golden/`:
///   det_input_ids.i32  Int32 [2,8,143]  (row 0 = conditional branch)
///   det_audio_mask.i32 Int32 [2,143]
///   ref_tokens.i32     Int32 [8,52]     (reference codec tokens fixture)
/// Tokenizer at `/tmp/omnivoice-mlx/` (tokenizer.json, vocab 151676).
final class InputConstructionTests: XCTestCase {
    private static let goldenDir = "/tmp/omnivoice_golden"
    private static let modelDir = "/tmp/omnivoice-mlx"

    private func i32(_ path: String) throws -> [Int32] {
        let d = try Data(contentsOf: URL(fileURLWithPath: path))
        return d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
    }

    /// Resolve the seven OmniVoice control tokens through the tokenizer's
    /// added-tokens table — confirms each encodes to a single id.
    func testSpecialTokenIds() async throws {
        let dir = Self.modelDir
        guard FileManager.default.fileExists(atPath: "\(dir)/tokenizer.json") else {
            throw XCTSkip("missing \(dir)/tokenizer.json")
        }
        let tok = try await OmniVoiceTokenizer.load(from: URL(fileURLWithPath: dir))
        let expected: [(String, Int)] = [
            ("<|denoise|>", 151669),
            ("<|lang_start|>", 151670),
            ("<|lang_end|>", 151671),
            ("<|instruct_start|>", 151672),
            ("<|instruct_end|>", 151673),
            ("<|text_start|>", 151674),
            ("<|text_end|>", 151675),
        ]
        for (s, id) in expected {
            XCTAssertEqual(tok.tokenId(s), id, "special token \(s)")
            XCTAssertEqual(tok.encode(s).count, 1, "\(s) must encode to a single id")
            XCTAssertEqual(tok.encode(s), [id], "\(s) encode")
        }
        // Plain-text fragments used by the golden.
        XCTAssertEqual(tok.encode("en"), [268], "lang id")
        XCTAssertEqual(tok.encode("None"), [4064], "instruct None")
    }

    /// The hard gate: byte-exact reproduction of the golden conditioning ids/mask.
    func testBuildInputsMatchesGolden() async throws {
        let dir = Self.goldenDir, mdir = Self.modelDir
        for p in [
            "\(mdir)/tokenizer.json",
            "\(dir)/det_input_ids.i32",
            "\(dir)/det_audio_mask.i32",
            "\(dir)/ref_tokens.i32",
        ] where !FileManager.default.fileExists(atPath: p) {
            throw XCTSkip("missing \(p)")
        }

        let tok = try await OmniVoiceTokenizer.load(from: URL(fileURLWithPath: mdir))
        let builder = OmniVoiceInputBuilder(tokenizer: tok)

        let C = 8, L = 143
        let goldIds = try i32("\(dir)/det_input_ids.i32")  // [2,8,143]
        let goldMask = try i32("\(dir)/det_audio_mask.i32")  // [2,143]
        // Row 0 (conditional branch).
        let condIds = Array(goldIds[0 ..< (C * L)])
        let condMask = Array(goldMask[0 ..< L])

        let refFlat = try i32("\(dir)/ref_tokens.i32")  // [8,52]
        let refLen = refFlat.count / C  // 52
        let refTokens = MLXArray(refFlat, [C, refLen])

        let (ids, mask) = builder.buildInputs(
            text: "The quick brown fox jumps over the lazy dog.",
            refText: "Hello there, this is a cloned voice.",
            lang: "en",
            refAudioTokens: refTokens,
            targetLen: 63,
            denoise: true,
            instruct: nil
        )
        MLX.eval(ids, mask)

        XCTAssertEqual(ids.shape, [1, C, L], "input_ids shape")
        XCTAssertEqual(mask.shape, [1, L], "audio_mask shape")

        let gotIds = ids.reshaped([C * L]).asArray(Int32.self)
        let gotMask = mask.reshaped([L]).asArray(Int32.self)

        // Element-wise integer equality.
        if gotIds != condIds {
            let firstDiff = zip(gotIds, condIds).enumerated().first { $0.element.0 != $0.element.1 }
            XCTFail("input_ids differ; first mismatch at flat idx \(firstDiff?.offset ?? -1): "
                + "got \(firstDiff?.element.0 ?? -1) vs gold \(firstDiff?.element.1 ?? -1)")
        }
        XCTAssertEqual(gotMask, condMask, "audio_mask must be integer-equal")
    }

    /// Sanity-check the duration estimator returns a positive, monotonic count.
    func testDurationEstimatorBasic() {
        let est = RuleDurationEstimator.shared
        XCTAssertEqual(est.totalWeight("abc"), 3.0, accuracy: 1e-9)
        // "Hello, world." = 10 letters (10.0) + comma (0.5) + space (0.2) + period (0.5)
        XCTAssertEqual(est.totalWeight("Hello, world."), 11.2, accuracy: 1e-9)
        // CJK char weighs 3.0.
        XCTAssertEqual(est.totalWeight("\u{4F60}"), 3.0, accuracy: 1e-9)
    }
}
