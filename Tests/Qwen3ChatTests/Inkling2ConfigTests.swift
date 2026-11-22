import XCTest
@testable import Qwen3Chat

final class Inkling2ConfigTests: XCTestCase {
    static func fixture() -> [String: Any] {
        ["model_type": "inkling2_loop", "text_config": [
            "hidden_size": 64, "num_hidden_layers": 6, "vocab_size": 64,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "n_routed_experts": 4, "num_experts_per_tok": 2, "intermediate_size": 64,
            "n_layers_per_attn_res_block": 2, "local_layer_ids": [0, 2, 4],
            "loop_config": ["core_start": 2, "core_end": 4, "num_passes": 3, "gate_rank": 4, "is_varloop": false]]]
    }
    func testLoopVisitsShareIndicesAndRequireIndependentCaches() throws {
        let c = try Inkling2Config(data: JSONSerialization.data(withJSONObject: Self.fixture()))
        XCTAssertEqual(c.layerVisits, [0, 1, 2, 3, 2, 3, 2, 3, 4, 5])
    }
    func testInvalidContextRejectsBeforeOpeningWeights() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let data = try JSONSerialization.data(withJSONObject: Self.fixture())
        try data.write(to: directory.appendingPathComponent("config.json"))
        let config = try Inkling2Config(data: data)
        for limit in [-1, 0, config.maxContext + 1] {
            XCTAssertThrowsError(try Inkling2Chat.fromDirectory(directory, contextLimit: limit)) { error in
                XCTAssertTrue(error is Inkling2Error)
                XCTAssertTrue(String(describing: error).contains("context"))
            }
        }
    }
    func testNonpositiveContextRejectsBeforeDownload() async {
        do {
            _ = try await Inkling2Chat.fromPretrained(modelId: "must-not-download/missing", contextLimit: 0)
            XCTFail("Expected invalid context")
        } catch {
            XCTAssertTrue(error is Inkling2Error)
            XCTAssertTrue(String(describing: error).contains("context"))
        }
    }
    func testMixedPrecisionOverrides() throws {
        var data = Self.fixture()
        data["quantization"] = ["bits": 5, "group_size": 32, "mode": "affine",
            "language_model.lm_head": ["bits": 8, "group_size": 64, "mode": "affine"]] as [String: Any]
        let c = try Inkling2Config(data: JSONSerialization.data(withJSONObject: data))
        XCTAssertEqual(c.quantization(for: "expert").bits, 5)
        XCTAssertEqual(c.quantization(for: "language_model.lm_head").bits, 8)
    }
    func testRejectsSourceQuantizationAndInvalidLoops() throws {
        var data = Self.fixture()
        data["quantization_config"] = ["quant_method": "modelopt"]
        XCTAssertThrowsError(try Inkling2Config(data: JSONSerialization.data(withJSONObject: data)))
        data = Self.fixture()
        var text = data["text_config"] as! [String: Any]
        text["loop_config"] = ["core_start": 1, "core_end": 4, "num_passes": 3, "gate_rank": 4]
        data["text_config"] = text
        XCTAssertThrowsError(try Inkling2Config(data: JSONSerialization.data(withJSONObject: data)))
    }
    func testRejectsInvalidLayerCountsBeforeBuildingDefaultLayerList() throws {
        for count in [-1, 0, Int.max] {
            var data = Self.fixture()
            var text = data["text_config"] as! [String: Any]
            text.removeValue(forKey: "local_layer_ids")
            text["num_hidden_layers"] = count
            data["text_config"] = text
            XCTAssertThrowsError(try Inkling2Config(data: JSONSerialization.data(withJSONObject: data)))
        }
    }
    func testNativeChatTemplateAndEffortPosition() throws {
        let rendered = try Inkling2Template.render([.init(role: .system, content: "Brief"),
            .init(role: .user, content: "Hi"), .init(role: .assistant, content: "Hello")])
        XCTAssertEqual(rendered,
            "<|message_system|><|content_text|>Brief<|end_message|>"
            + "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
            + "<|message_user|><|content_text|>Hi<|end_message|>"
            + "<|message_model|><|content_text|>Hello<|end_message|><|content_model_end_sampling|>")
        XCTAssertThrowsError(try Inkling2Template.render([], reasoningEffort: .nan))
        XCTAssertThrowsError(try Inkling2Template.render([], reasoningEffort: 1))
    }
    func testImagePatchOrderPaddingAndBudget() throws {
        let rgb = [UInt8](repeating: 255, count: 41 * 3)
        let image = try Inkling2Image(width: 41, height: 1, rgb: rgb)
        let pixels = try image.patchPixels()
        XCTAssertEqual(image.rows, 1); XCTAssertEqual(image.columns, 2)
        XCTAssertEqual(pixels[0], 1)
        XCTAssertEqual(pixels[40 * 40 * 3], 1)
        XCTAssertEqual(pixels[40 * 40 * 3 + 3], -1 / 127.5 - 1, accuracy: 1e-6)
        XCTAssertThrowsError(try image.patchPixels(maxTokens: 1))
        XCTAssertThrowsError(try Inkling2Image(width: 5000, height: 1, rgb: []))
    }
    func testImageFramingRepeatsPlaceholderPerPatch() throws {
        let result = try Inkling2Template.render([.init(role: .user, content: "Describe")], imageTokenCounts: [2])
        XCTAssertTrue(result.hasSuffix("<|message_user|><|content_image|><|unused_200054|><|unused_200054|><|end_message|>"))
        XCTAssertThrowsError(try Inkling2Template.render([], imageTokenCounts: [1]))
    }
}
