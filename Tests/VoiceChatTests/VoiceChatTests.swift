import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import VoiceChat

/// End-to-end tests for the VoiceChat perception and language bundles.
///
/// The weights are 10–19 GB and are not fetched automatically, so every test
/// that needs them skips unless `VOICECHAT_BUNDLE` points at an exported
/// directory containing `encoder/` and `llm/`. The tests that need no weights
/// run everywhere and cover the parts most likely to regress silently.
final class VoiceChatTests: XCTestCase {

    private var bundle: URL? {
        guard let path = ProcessInfo.processInfo.environment["VOICECHAT_BUNDLE"] else { return nil }
        return URL(fileURLWithPath: path)
    }

    private func requireBundle() throws -> URL {
        guard let bundle else {
            throw XCTSkip("set VOICECHAT_BUNDLE to a directory containing encoder/ and llm/")
        }
        return bundle
    }

    // MARK: - No weights required

    /// The streaming mask is the single most dangerous thing to get wrong: an
    /// encoder with no mask attends to future audio, which a duplex model never
    /// has, and produces plausible-looking output with no error anywhere.
    func testChunkedLimitedMaskIsCausalAndBounded() {
        let mask = VoiceChatEncoder.chunkedLimitedMask(length: 8, leftContext: 3, rightContext: 0)
        XCTAssertEqual(mask.shape, [8, 8])

        // true means masked. Row i may see columns (i-3)...i and nothing else.
        let values = mask.asArray(Bool.self)
        for query in 0 ..< 8 {
            for key in 0 ..< 8 {
                let masked = values[query * 8 + key]
                let visible = key <= query && query - key <= 3
                XCTAssertEqual(masked, !visible,
                               "query \(query) key \(key): expected visible=\(visible)")
            }
        }
    }

    func testMaskNeverLetsAnyPositionSeeTheFuture() {
        let mask = VoiceChatEncoder.chunkedLimitedMask(length: 16, leftContext: 70, rightContext: 0)
        let values = mask.asArray(Bool.self)
        for query in 0 ..< 16 {
            for key in (query + 1) ..< 16 {
                XCTAssertTrue(values[query * 16 + key],
                              "position \(query) can see future position \(key)")
            }
        }
    }

    /// Each causal subsampling stage is floor(n/2)+1, so the encoder emits one
    /// frame more than a plain stride-2 stack would. Nominal rate is 80 ms.
    func testSubsamplingFrameCount() {
        XCTAssertEqual(CausalSubsampling.outputFrames(melFrames: 200), 26)
        XCTAssertEqual(CausalSubsampling.outputFrames(melFrames: 1600), 201)
        // Monotonic, and never more than mel/8 + 3.
        for melFrames in stride(from: 8, through: 4000, by: 137) {
            let out = CausalSubsampling.outputFrames(melFrames: melFrames)
            XCTAssertGreaterThan(out, 0)
            XCTAssertLessThanOrEqual(out, melFrames / 8 + 3)
        }
    }

    // MARK: - Weights required

    func testPerceptionEncodesIntoLanguageModelSpace() throws {
        let root = try requireBundle()
        let perception = try VoiceChatPerception.load(from: root.appendingPathComponent("encoder"))
        let config = perception.config

        // Three load-bearing config values; a stock Conformer gets all three
        // wrong and still loads.
        XCTAssertFalse(config.encoder.useBias)
        XCTAssertEqual(config.encoder.preEncodeFreqOut, 17)
        XCTAssertEqual(config.encoder.convNormType, "layer_norm")
        XCTAssertEqual(config.encoder.attContextSize, [70, 0])

        let melFrames = 200
        let mel = MLXArray.zeros([1, melFrames, config.encoder.featIn]) + 0.1
        let embeddings = perception(mel)
        eval(embeddings)

        XCTAssertEqual(embeddings.shape,
                       [1, CausalSubsampling.outputFrames(melFrames: melFrames),
                        config.modalityProj.outFeatures])
        XCTAssertTrue(all(isFinite(embeddings)).item(Bool.self), "encoder produced non-finite output")
    }

    func testLanguageModelProducesFiniteLogits() throws {
        let root = try requireBundle()
        let llm = try VoiceChatLanguageModel.load(from: root.appendingPathComponent("llm"))

        XCTAssertEqual(llm.configuration.numHiddenLayers, 56)
        XCTAssertEqual(llm.configuration.hiddenSize, 4480)
        let pattern = Array(llm.configuration.hybridOverridePattern)
        XCTAssertEqual(pattern.filter { $0 == "M" }.count, 27)
        XCTAssertEqual(pattern.filter { $0 == "-" }.count, 25)
        XCTAssertEqual(pattern.filter { $0 == "*" }.count, 4)

        let tokens = MLXArray([1, 2, 3, 4]).reshaped(1, 4)
        let logits = llm(tokens)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 4, llm.configuration.vocabSize])
        XCTAssertTrue(all(isFinite(logits)).item(Bool.self), "language model produced non-finite logits")
    }

    /// Regression: `NemotronHBackbone` builds its causal mask from the cache and
    /// falls back to no mask at all when there isn't one, which silently makes
    /// attention bidirectional. `VoiceChatLanguageModel` therefore always
    /// supplies a cache. If that is ever removed, later positions change while
    /// early ones do not — so compare a prefix against the full sequence.
    func testAttentionIsCausalAcrossSequenceLength() throws {
        let root = try requireBundle()
        let llm = try VoiceChatLanguageModel.load(from: root.appendingPathComponent("llm"))

        let full = MLXArray([5, 9, 13, 21, 34]).reshaped(1, 5)
        let prefix = full[0..., ..<3]

        let logitsFull = llm(full)
        let logitsPrefix = llm(prefix)
        eval(logitsFull, logitsPrefix)

        // Under causal attention the first three positions cannot depend on
        // tokens 4 and 5, so their logits must be unchanged.
        let a = logitsFull[0..., ..<3, 0...].asType(.float32)
        let b = logitsPrefix.asType(.float32)
        let deviation = mean(abs(a - b)).item(Float.self)
        let scale = mean(abs(a)).item(Float.self)
        XCTAssertLessThan(deviation / scale, 0.01,
                          "prefix logits changed when later tokens were added — attention is not causal")
    }

    func testFunctionHeadIsCarriedButNotPartOfTheModel() throws {
        let root = try requireBundle()
        let llm = try VoiceChatLanguageModel.load(from: root.appendingPathComponent("llm"))
        // The tool-call channel is a separate head the stock model has no slot
        // for; it must survive the load rather than be dropped.
        let head = try XCTUnwrap(llm.functionHead, "function_head missing from the bundle")
        XCTAssertEqual(head.shape, [llm.configuration.vocabSize, llm.configuration.hiddenSize])
    }

    /// Only 4 of 56 layers keep a growing KV cache; the 27 Mamba layers hold a
    /// fixed-size recurrent state instead. That split is why long conversations
    /// stay affordable, so assert the cache is built that way.
    func testCacheIsMostlyRecurrent() throws {
        let root = try requireBundle()
        let llm = try VoiceChatLanguageModel.load(from: root.appendingPathComponent("llm"))
        let cache = llm.newCache()
        XCTAssertEqual(cache.count, 31, "expected one cache per Mamba and attention layer")
        XCTAssertEqual(cache.filter { $0 is MambaCache }.count, 27)
    }
}
