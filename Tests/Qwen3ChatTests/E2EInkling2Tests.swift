import XCTest
import MLX
@testable import Qwen3Chat

/// Small, deterministic reference fixture exercises every decoder component,
/// including mixed INT5/INT8 weights, loop messages and sliding-cache eviction.
final class E2EInkling2ParityTests: XCTestCase {
    private var fixture: URL {
        Bundle.module.resourceURL!.appendingPathComponent("Inkling2")
    }
    func testFullAndCachedLogitsMatchReference() throws {
        try Device.withDefaultDevice(.cpu) { try checkReference(tolerance: 1e-5) }
    }
    func testGPUFullAndCachedLogitsMatchReference() throws {
        // CPU and Metal quantized reductions differ slightly, particularly in
        // batched prefill. Keep a separate bound from the CPU equation check.
        try Device.withDefaultDevice(.gpu) { try checkReference(tolerance: 0.004) }
    }
    private func checkReference(tolerance: Float) throws {
        let model = try Inkling2Model.fromDirectory(fixture)
        let ref = try JSONSerialization.jsonObject(with: Data(contentsOf: fixture.appendingPathComponent("reference.json"))) as! [String: Any]
        let tokens = ref["tokens"] as! [Int]
        func assertClose(_ output: MLXArray, _ key: String, file: StaticString = #filePath, line: UInt = #line) {
            let actual = output.asType(.float32).asArray(Float.self)
            let expected = (ref[key] as! [NSNumber]).map { $0.floatValue }
            XCTAssertEqual(actual.count, expected.count, file: file, line: line)
            let error = zip(actual, expected).map { abs($0 - $1) }.max() ?? .infinity
            print("[inkling2-parity] \(Device.defaultDevice()) \(key) max_error=\(error)")
            XCTAssertLessThan(error, tolerance, "\(key) maximum absolute error \(error)", file: file, line: line)
            let width = model.config.unpaddedVocabSize
            for start in stride(from: 0, to: actual.count, by: width) {
                let a = actual[start..<(start + width)], e = expected[start..<(start + width)]
                XCTAssertEqual(a.indices.max(by: { a[$0] < a[$1] }), e.indices.max(by: { e[$0] < e[$1] }))
            }
        }
        let state = Inkling2Model.State.initial(config: model.config)
        let (full, next) = try model.forward(inputIDs: MLXArray(tokens.map { Int32($0) }).reshaped(1, -1), state: state)
        assertClose(full, "full_logits")
        let (continued, _) = try model.forward(inputIDs: MLXArray([Int32(8)]).reshaped(1, 1), state: next)
        assertClose(continued, "continued_logits")
        var incremental = state
        var last: MLXArray?
        for token in tokens {
            let (step, cache) = try model.forward(inputIDs: MLXArray([Int32(token)]).reshaped(1, 1), state: incremental)
            eval([step] + cache.arrays); incremental = cache; last = step
        }
        assertClose(last!, "sequential_logits")
        XCTAssertEqual(incremental.position, tokens.count)
        XCTAssertEqual(incremental.layers.count, 16)
        for (visit, layer) in model.config.layerVisits.enumerated() {
            let expected = model.config.localLayers.contains(layer) ? 4 : tokens.count
            XCTAssertEqual(incremental.layers[visit].keys?.dim(2), expected)
        }
        let groupedTokens = ref["grouped_tokens"] as! [Int]
        let (grouped, _) = try model.forward(inputIDs: MLXArray(groupedTokens.map { Int32($0) }).reshaped(1, -1), state: state)
        assertClose(grouped, "grouped_logits")
    }
    func testRejectsMissingScalesAndInvalidInputs() throws {
        let model = try Inkling2Model.fromDirectory(fixture)
        var weights = model.weights
        weights.removeValue(forKey: "language_model.model.layers.2.mlp.switch_mlp.gate_proj.scales")
        XCTAssertThrowsError(try Inkling2Model(config: model.config, weights: weights))
        XCTAssertThrowsError(try model.forward(inputIDs: MLXArray([Int32(1)]), state: .initial(config: model.config)))
    }
}

/// Full-weight smoke checks are isolated from the reference fixture and CI.
final class E2EInkling2GenerationTests: XCTestCase {
    private static var cached: Inkling2Chat?
    override class func tearDown() {
        cached = nil
        MLX.Memory.clearCache()
        super.tearDown()
    }
    private func model() throws -> Inkling2Chat {
        if let cached = Self.cached { return cached }
        MLX.Memory.cacheLimit = 128 * 1024 * 1024
        let path = ProcessInfo.processInfo.environment["INKLING2_MODEL_DIR"] ?? "/tmp/Inkling-2-Mini-MLX-4bit"
        guard FileManager.default.fileExists(atPath: path + "/config.json") else {
            throw XCTSkip("Set INKLING2_MODEL_DIR to a complete Inkling 2 Mini export")
        }
        let loaded = try Inkling2Chat.fromDirectory(URL(fileURLWithPath: path))
        Self.cached = loaded
        return loaded
    }
    private func reference() throws -> [String: Any] {
        let path = ProcessInfo.processInfo.environment["INKLING2_REFERENCE"] ?? "/tmp/inkling2-full-reference.json"
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("Run scripts/tests/validate_inkling_model.py to record full-weight reference outputs")
        }
        return try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path))) as! [String: Any]
    }
    func testRealModelTextAndTokenizerMatchReference() throws {
        let ref = try reference(), model = try model()
        for item in ref["tokenization"] as! [[String: Any]] {
            let text = item["text"] as! String
            let expected = item["tokens"] as! [Int]
            XCTAssertEqual(model.encode(text), expected, "Tokenizer mismatch: \(text)")
            XCTAssertEqual(model.decode(expected), text)
        }
        let logits = try model.nextTokenLogits(promptTokens: ref["tokens"] as! [Int])
        let rendered = try Inkling2Template.render([.init(role: .user, content: "Reply with only the result of 2 + 2.")])
            + "<|message_model|><|content_text|>"
        XCTAssertEqual(model.encode(rendered), ref["tokens"] as! [Int], "Complete chat-template tokenization")
        let ids = ref["text_top_ids"] as! [Int]
        XCTAssertEqual(logits.indices.max(by: { logits[$0] < logits[$1] }), ids.first)
        let expected = (ref["text_top_logits"] as! [NSNumber]).map { $0.floatValue }
        let error = zip(ids, expected).map { abs(logits[$0.0] - $0.1) }.max() ?? .infinity
        let squaredError = zip(ids, expected).reduce(Float(0)) { $0 + pow(logits[$1.0] - $1.1, 2) }
        let normalizedError = sqrt(squaredError / expected.reduce(Float(0)) { $0 + $1 * $1 })
        // Full BF16 routing amplifies reduction differences between kernels and
        // graph compilations. Normalize the maximum error by the checkpoint's
        // score cap so this check applies to both INT4 and INT5 expert exports.
        // Keep aggregate drift and top-token identity as separate guards; the
        // small FP32 fixture checks the equations much more tightly.
        let scale = model.denseConfig.logitCap > 0 ? model.denseConfig.logitCap
            : max(expected.map { abs($0) }.max() ?? 0, 1)
        print("[inkling2-full] top_logits max_error=\(error) cap_relative_error=\(error / scale) nrmse=\(normalizedError)")
        XCTAssertLessThan(error / scale, 0.03)
        XCTAssertLessThan(normalizedError, 0.025)
    }
    func testRealModelVisionMatchesReference() throws {
        let ref = try reference(), model = try model()
        let rgb = (0..<(80 * 40 * 3)).map { UInt8($0 % 256) }
        let image = try Inkling2Image(width: 80, height: 40, rgb: rgb)
        let embeddings = try model.model.imageEmbeddings(image)
        let actual = embeddings.asType(.float32).asArray(Float.self)
        let expected = (ref["vision_embeddings"] as! [NSNumber]).map { $0.floatValue }
        XCTAssertEqual(embeddings.shape, ref["vision_shape"] as! [Int])
        let numerator = zip(actual, expected).reduce(Float(0)) { $0 + ($1.0 - $1.1) * ($1.0 - $1.1) }
        let denominator = expected.reduce(Float(0)) { $0 + $1 * $1 }
        XCTAssertLessThan(sqrt(numerator / denominator), 0.04)
    }
    func testRealModelAnswersArithmetic() async throws {
        let model = try model()
        var output = ""
        let sampling = ChatSamplingConfig(temperature: 0, topK: 0, topP: 1, maxTokens: 512, repetitionPenalty: 1)
        for try await part in model.generateStream(messages: [.init(role: .user, content: "Reply with only the result of 2 + 2.")], sampling: sampling) { output += part }
        XCTAssertTrue(output.contains("4"), "Expected a correct arithmetic answer, got: \(output)")
        XCTAssertFalse(output.contains("<|"))
        XCTAssertFalse(output.isEmpty)
    }
    func testRealModelRecognizesImageColors() async throws {
        let model = try model()
        let sampling = ChatSamplingConfig(temperature: 0, topK: 0, topP: 1, maxTokens: 512, repetitionPenalty: 1)
        for (color, pixel) in [("red", [UInt8(255), 0, 0]), ("blue", [UInt8(0), 0, 255])] {
            let image = try Inkling2Image(width: 40, height: 40,
                rgb: Array(repeating: pixel, count: 1600).flatMap { $0 })
            var output = ""
            for try await part in model.generateStream(
                messages: [.init(role: .user, content: "What is the main color of this image? Reply with just one color.")],
                reasoningEffort: 0, sampling: sampling, images: [image]) { output += part }
            XCTAssertTrue(output.lowercased().contains(color), "Expected \(color), got: \(output)")
        }
    }
}
