import XCTest
@testable import Clef

final class ClefEncodingTests: XCTestCase {
    private func bytes(_ value: String) -> [Int] { value.utf8.map(Int.init) }
    func testSchemaOrderAndSpans() throws {
        let fields: [ClefQuestion] = [
            .init(id: "route", instructions: "Choose a destination", kind: .choice(["z": "Last", "a": "First"])),
            .init(id: "safe", kind: .noul()),
            .init(id: "rating", kind: .score(["bad", "good"]))]
        let e = try ClefEncoding.encode(state: "Move left", questions: fields, maxTokens: 8192, tokenize: bytes)
        func string(_ span: Range<Int>) -> String { String(decoding: e.tokens[span].map(UInt8.init), as: UTF8.self) }
        XCTAssertEqual(string(e.questions[0].span), "Choose a destination")
        XCTAssertEqual(string(e.questions[1].span), "safe")
        XCTAssertEqual(string(e.questions[0].optionSpans[0]), "{\"description\":\"First\",\"option_id\":\"a\"}")
        XCTAssertTrue(string(e.questions[1].optionSpans[0]).contains("\"option_id\":\"true\""))
        XCTAssertTrue(string(e.questions[2].optionSpans[1]).contains("\"option_id\":\"1\""))
        XCTAssertTrue(String(decoding: e.tokens.map(UInt8.init), as: UTF8.self).hasSuffix("JOINT SCHEMA DECISIONS:"))
    }
    func testRejectsInvalidAndOversizedRequests() {
        let q = ClefQuestion(id: "route", kind: .choice(["left": "left"]))
        for questions in [[], [q, q], [.init(id: "empty", kind: .score([]))]] as [[ClefQuestion]] {
            XCTAssertThrowsError(try ClefEncoding.encode(state: "", questions: questions, maxTokens: 4096, tokenize: bytes))
        }
        XCTAssertThrowsError(try ClefEncoding.encode(state: String(repeating: "x", count: 500), questions: [q], maxTokens: 100, tokenize: bytes))
    }
    func testJSONRequestPreservesFieldOrderAndTypes() throws {
        let data = Data(#"{"state":"Lights on","questions":[{"id":"action","type":"choice","choices":{"on":"On","off":"Off"}},{"id":"urgent","type":"noul"},{"id":"priority","type":"score","levels":["normal","urgent"]}]}"#.utf8)
        let request = try JSONDecoder().decode(ClefRequest.self, from: data)
        let questions = try request.resolvedQuestions()
        XCTAssertEqual(request.state, "Lights on")
        XCTAssertEqual(questions.map(\.id), ["action", "urgent", "priority"])
        XCTAssertEqual(questions[0].options.map(\.0), ["off", "on"])
        XCTAssertEqual(questions[1].typeID, 0)
        XCTAssertEqual(questions[2].options.map(\.1), ["normal", "urgent"])
    }
    func testJSONRequestRejectsUnsupportedAndIncompleteFields() throws {
        for field in [#"{"id":"x","type":"unknown"}"#,
                      #"{"id":"x","type":"choice"}"#,
                      #"{"id":"x","type":"score"}"#] {
            let data = Data(("{\"state\":\"test\",\"questions\":[" + field + "]}").utf8)
            let request = try JSONDecoder().decode(ClefRequest.self, from: data)
            XCTAssertThrowsError(try request.resolvedQuestions())
        }
    }
    func testUnicodeAndEscaping() throws {
        let question = ClefQuestion(id: "行く", instructions: "café", kind: .choice(["a": "道/\"左\"\n"]))
        let encoding = try ClefEncoding.encode(state: "東京", questions: [question], maxTokens: 4096, tokenize: bytes)
        let span = encoding.questions[0].optionSpans[0]
        let data = Data(encoding.tokens[span].map(UInt8.init))
        let json = try JSONSerialization.jsonObject(with: data) as! [String: String]
        XCTAssertEqual(json["description"], "道/\"左\"\n")
    }
}

final class E2EClefTests: XCTestCase {
    func testLocalDecision() throws {
        guard let path = ProcessInfo.processInfo.environment["CLEF_MODEL_DIR"] else {
            throw XCTSkip("Set CLEF_MODEL_DIR to a local 4-bit Clef-flash export.")
        }
        let model = try Clef.load(from: URL(fileURLWithPath: path))
        let fields: [ClefQuestion] = [
            .init(id: "action", instructions: "Which action was requested?", kind: .choice([
                "lights_on": "Turn the lights on", "lights_off": "Turn the lights off", "other": "Another request"])),
            .init(id: "question", instructions: "Is the user asking a factual question?", kind: .noul()),
            .init(id: "urgency", instructions: "How urgent is the request?", kind: .score(["normal", "urgent"]))]
        let state = "Please turn the kitchen lights on."
        let result = try model.decide(state: state, questions: fields)
        let fixtureURL = Bundle.module.url(forResource: "full-model-reference", withExtension: "json", subdirectory: "Fixtures")!
        struct Reference: Decodable { let tokens: [Int]; let decisions: [String: [String: Float]] }
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: fixtureURL))
        XCTAssertEqual(try model.encode(state: state, questions: fields, maxTokens: 4096).tokens, reference.tokens)
        var maxDifference: Float = 0
        for decision in result.decisions {
            for (key, expected) in reference.decisions[decision.id]! {
                maxDifference = max(maxDifference, abs(decision.probabilities[key]! - expected))
                XCTAssertEqual(decision.probabilities[key]!, expected, accuracy: 0.006)
            }
        }
        print("Clef maximum reference probability difference: \(maxDifference)")
        var elapsed: [Double] = []
        let repeats = Int(ProcessInfo.processInfo.environment["CLEF_BENCH_REPEATS"] ?? "5") ?? 5
        for _ in 0..<repeats {
            let start = ContinuousClock.now
            let repeated = try model.decide(state: state, questions: fields)
            let dt = start.duration(to: .now).components
            elapsed.append(Double(dt.seconds) + Double(dt.attoseconds) / 1e18)
            XCTAssertEqual(repeated.decisions.map(\.selectedOption), result.decisions.map(\.selectedOption))
        }
        print("Clef repeated decision seconds: \(elapsed)")
        XCTAssertGreaterThan(result.inputTokens, 0)
        XCTAssertEqual(result.decisions.count, 3)
        XCTAssertEqual(result.decisions[0].selectedOption, "lights_on")
        for decision in result.decisions {
            XCTAssertEqual(decision.probabilities.values.reduce(0, +), 1, accuracy: 1e-5)
        }
        XCTAssertNotNil(result.decisions[1].noul)
        XCTAssertNotNil(result.decisions[2].score)
    }
}

import MLX

final class E2EClefHeadParityTests: XCTestCase {
    struct Tensor: Decodable {
        let shape: [Int]
        let values: [Float]
        var array: MLXArray { MLXArray(values).reshaped(shape) }
    }
    struct Fixture: Decodable {
        let config: ClefHeadConfig
        let weights: [String: Tensor]
        let hidden: Tensor
        let lexical: Tensor
        let tokens: [Int]
        let logits: [[Float]]
    }
    func testReferenceHeadCPU() throws {
        try Device.withDefaultDevice(.cpu) { try checkReferenceHead(tolerance: 2e-5) }
    }
    func testReferenceHeadMetal() throws {
        // Metal float32 attention differs slightly from the CPU reference.
        try Device.withDefaultDevice(.gpu) { try checkReferenceHead(tolerance: 5e-4) }
    }
    private func checkReferenceHead(tolerance: Float) throws {
        let url = Bundle.module.url(forResource: "head-reference", withExtension: "json", subdirectory: "Fixtures")!
        let f = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        let encoding = try ClefEncoding.encode(state: "Turn on the light", questions: [
            .init(id: "action", instructions: "Choose", kind: .choice(["off": "Off", "on": "On"])),
            .init(id: "safe", instructions: "Safe?", kind: .noul())], maxTokens: 8192, tokenize: { $0.utf8.map(Int.init) })
        XCTAssertEqual(encoding.tokens, f.tokens)
        let head = try ClefHead(config: f.config, weights: f.weights.mapValues(\.array))
        let lex = f.lexical.array
        let logits = head.logits(hidden: f.hidden.array, encoding: encoding, lexical: { lex[MLXArray($0.map(Int32.init))] })
        for (actual, expected) in zip(logits, f.logits) {
            for (a, b) in zip(actual.asArray(Float.self), expected) { XCTAssertEqual(a, b, accuracy: tolerance) }
        }
    }
    func testMissingHeadTensorFails() throws {
        let config = try JSONDecoder().decode(ClefHeadConfig.self, from: Data(#"{"hidden_size":8,"width":8,"routing_layers":1,"layers":1,"heads":2,"feedforward":12}"#.utf8))
        XCTAssertThrowsError(try ClefHead(config: config, weights: [:]))
    }
}

@testable import Qwen3Chat
import MLXNN

final class E2EClefBackboneTests: XCTestCase {
    func testModelOptimizationRequiresExplicitOptIn() {
        let config = Qwen3ChatConfig(hiddenSize: 64, numHiddenLayers: 2,
            numAttentionHeads: 1, numKeyValueHeads: 1, headDim: 64,
            intermediateSize: 128, vocabSize: 64, maxSeqLen: 16,
            ropeTheta: 10000, rmsNormEps: 1e-6, eosTokenId: 0, padTokenId: 0,
            quantization: "int4", quantizationBits: 4, quantizationGroupSize: 64,
            modelType: .qwen35, layerTypes: ["linear_attention", "full_attention"],
            linearNumKeyHeads: 2, linearKeyHeadDim: 64,
            linearNumValueHeads: 2, linearValueHeadDim: 64)
        let chat = Qwen35MLXModel(config: config)
        let optimized = Qwen35MLXModel(config: config, optimizedDeltaNet: true)
        XCTAssertFalse(chat.layers[0].deltaNet!.useFusedRecurrence)
        XCTAssertFalse(chat.layers[0].deltaNet!.useNativeConvolution)
        XCTAssertTrue(optimized.layers[0].deltaNet!.useFusedRecurrence)
        XCTAssertTrue(optimized.layers[0].deltaNet!.useNativeConvolution)
        XCTAssertNil(chat.layers[1].deltaNet)
        XCTAssertNil(optimized.layers[1].deltaNet)
    }

    func testGroupedDeltaHeadsMatchIncrementalInference() {
        checkIncremental(keyHeads: 1, valueHeads: 2, keyDim: 16, valueDim: 32)
    }
    func testFusedDeltaMatchesReference() {
        checkIncremental(keyHeads: 1, valueHeads: 2, keyDim: 128, valueDim: 128, length: 67)
    }
    func testEqualDeltaHeadsMatchIncrementalInference() {
        checkIncremental(keyHeads: 2, valueHeads: 2, keyDim: 64, valueDim: 64)
    }
    private func checkIncremental(keyHeads: Int, valueHeads: Int, keyDim: Int, valueDim: Int, length: Int = 3) {
        let config = Qwen3ChatConfig(hiddenSize: 64, numHiddenLayers: 1, numAttentionHeads: 1,
            numKeyValueHeads: 1, headDim: 64, intermediateSize: 128, vocabSize: 64,
            maxSeqLen: 16, ropeTheta: 10000, rmsNormEps: 1e-6, eosTokenId: 0, padTokenId: 0,
            quantization: "int4", quantizationBits: 4, quantizationGroupSize: 64,
            modelType: .qwen35, layerTypes: ["linear_attention"], linearNumKeyHeads: keyHeads,
            linearKeyHeadDim: keyDim, linearNumValueHeads: valueHeads, linearValueHeadDim: valueDim)
        MLXRandom.seed(0)
        let layer = DeltaNetLayer(config: config)
        XCTAssertFalse(layer.useFusedRecurrence)
        XCTAssertFalse(layer.useNativeConvolution)
        layer.useFusedRecurrence = true
        layer.useNativeConvolution = true
        layer.update(parameters: ModuleParameters(values: ["convWeight": .value(MLXArray.ones([2 * keyHeads * keyDim + valueHeads * valueDim, 4, 1]) / 4)]))
        let x = MLXArray((0..<(length * 64)).map { Float($0 % 17) / 17 }).reshaped(1, length, 64)
        let (whole, _) = layer(x)
        let (first, state) = layer(x[0..., ..<1])
        let (rest, _) = layer(x[0..., 1...], state: state)
        let streamed = concatenated([first, rest], axis: 1)
        layer.useFusedRecurrence = false
        layer.useNativeConvolution = false
        let (reference, _) = layer(x)
        XCTAssertLessThan(abs(whole - reference).max().item(Float.self), 1e-5)
        XCTAssertEqual(whole.shape, [1, length, 64])
        XCTAssertGreaterThan(abs(whole).max().item(Float.self), 1e-6)
        // Splitting changes the quantized matrix multiplication batch shape.
        // Compare both execution modes against the unfused implementation,
        // including its existing whole-versus-split rounding difference.
        let (referenceFirst, referenceState) = layer(x[0..., ..<1])
        let (referenceRest, _) = layer(x[0..., 1...], state: referenceState)
        let referenceStreamed = concatenated([referenceFirst, referenceRest], axis: 1)
        let baselineSplitError = abs(reference - referenceStreamed).max().item(Float.self)
        XCTAssertLessThan(abs(streamed - referenceStreamed).max().item(Float.self), 1e-5)
        XCTAssertLessThan(abs(whole - streamed).max().item(Float.self), baselineSplitError + 1e-5)

    }
}
