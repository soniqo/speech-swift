import XCTest
@testable import Qwen3Chat

final class ChatModelConfigTests: XCTestCase {
    private func writeConfig(_ json: String, named name: String = "config.json") throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3-chat-config-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent(name)
        try Data(json.utf8).write(to: url)
        return url
    }

    func testQwen3DenseConfigParsesStandardMLXConfig() throws {
        let url = try writeConfig("""
        {
          "hidden_size": 2560,
          "num_hidden_layers": 36,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "intermediate_size": 9728,
          "vocab_size": 151936,
          "rope_theta": 5000000,
          "rms_norm_eps": 0.000001,
          "tie_word_embeddings": true,
          "eos_token_id": [151645, 151643],
          "quantization": {"group_size": 64, "bits": 5}
        }
        """)

        let config = try Qwen3DenseConfig.load(from: url)

        XCTAssertEqual(config.hiddenSize, 2560)
        XCTAssertEqual(config.numHiddenLayers, 36)
        XCTAssertEqual(config.headDim, 80)
        XCTAssertEqual(config.eosTokenId, 151645)
        XCTAssertEqual(config.quantBits, 5)
        XCTAssertTrue(config.tieWordEmbeddings)
    }

    func testGemma4ConfigParsesNestedTextConfigAndDerivedLayerTypes() throws {
        let url = try writeConfig("""
        {
          "eos_token_id": [1, 106],
          "quantization": {"group_size": 64, "bits": 4},
          "text_config": {
            "hidden_size": 1536,
            "num_hidden_layers": 6,
            "intermediate_size": 6144,
            "num_attention_heads": 8,
            "head_dim": 256,
            "global_head_dim": 512,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 2,
            "hidden_size_per_layer_input": 256,
            "vocab_size": 262144,
            "vocab_size_per_layer_input": 262144,
            "sliding_window_pattern": 3,
            "rope_parameters": {
              "full_attention": {"rope_theta": 1000000, "partial_rotary_factor": 0.25},
              "sliding_attention": {"rope_theta": 10000}
            }
          }
        }
        """)

        let config = try Gemma4DenseConfig.load(from: url)

        XCTAssertEqual(config.hiddenSize, 1536)
        XCTAssertEqual(config.numHiddenLayers, 6)
        XCTAssertEqual(config.numKVSharedLayers, 2)
        XCTAssertEqual(config.eosTokenId, 1)
        XCTAssertEqual(config.quantBits, 4)
        XCTAssertEqual(config.headDim(forLayer: 0), 256)
        XCTAssertEqual(config.headDim(forLayer: 2), 512)
        XCTAssertEqual(config.layerTypes, [
            "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "full_attention",
        ])
    }
}

/// `Qwen3DenseConfig` and `Gemma4DenseConfig` used to read quantization only from the
/// nested `"quantization": {...}` object, so a checkpoint stating the flat
/// `quantization_bits` / `quantization_group_size` fields instead fell back to INT4/64
/// without an error — and `Module.update(parameters:)` then installed the checkpoint's
/// wider rows into those narrow layers, giving garbage output rather than a failure.
final class DenseChatQuantizationConfigTests: XCTestCase {
    private func writeConfig(_ json: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("chat-quantization-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("config.json")
        try Data(json.utf8).write(to: url)
        return url
    }

    /// A minimal Qwen3 dense config with the quantization statement under test spliced in.
    private func denseConfig(quantizationFields: String) throws -> Qwen3DenseConfig {
        try Qwen3DenseConfig.load(from: writeConfig("""
        {
          "hidden_size": 2560,
          "num_hidden_layers": 36,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "intermediate_size": 9728,
          "vocab_size": 151936,
          "eos_token_id": 151645\(quantizationFields.isEmpty ? "" : ",")
          \(quantizationFields)
        }
        """))
    }

    /// A minimal Gemma 4 config; `rootFields` and `textFields` place the quantization
    /// statement at the root or beside the text fields respectively.
    private func gemmaConfig(rootFields: String = "", textFields: String = "") throws
        -> Gemma4DenseConfig
    {
        try Gemma4DenseConfig.load(from: writeConfig("""
        {
          "eos_token_id": [1, 106]\(rootFields.isEmpty ? "" : ",")
          \(rootFields)
          ,"text_config": {
            "hidden_size": 1536,
            "num_hidden_layers": 6,
            "intermediate_size": 6144,
            "num_attention_heads": 8,
            "num_kv_shared_layers": 2,
            "sliding_window_pattern": 3\(textFields.isEmpty ? "" : ",")
            \(textFields)
          }
        }
        """))
    }

    // MARK: - Qwen3 dense

    func testDenseReadsFlatQuantizationFields() throws {
        let config = try denseConfig(quantizationFields: """
            "quantization_bits": 8,
            "quantization_group_size": 32
            """)

        XCTAssertEqual(config.quantBits, 8)
        XCTAssertEqual(config.quantGroupSize, 32)
    }

    func testDenseFlatFieldsWinOverAStaleNestedObject() throws {
        let config = try denseConfig(quantizationFields: """
            "quantization": {"group_size": 64, "bits": 4},
            "quantization_bits": 8,
            "quantization_group_size": 32
            """)

        XCTAssertEqual(config.quantBits, 8)
        XCTAssertEqual(config.quantGroupSize, 32)
    }

    func testDenseStillReadsTheNestedObject() throws {
        let config = try denseConfig(
            quantizationFields: #""quantization": {"group_size": 32, "bits": 5}"#)

        XCTAssertEqual(config.quantBits, 5)
        XCTAssertEqual(config.quantGroupSize, 32)
    }

    func testDenseFallsBackToTheQuantizationLabel() throws {
        let config = try denseConfig(quantizationFields: #""quantization": "int8""#)

        XCTAssertEqual(config.quantBits, 8)
        // The label carries no group size, so MLX's default stands.
        XCTAssertEqual(config.quantGroupSize, 64)
    }

    func testDenseCheckpointStatingNothingLoadsAsInt4() throws {
        let config = try denseConfig(quantizationFields: "")

        XCTAssertEqual(config.quantBits, 4)
        XCTAssertEqual(config.quantGroupSize, 64)
    }

    // MARK: - Gemma 4

    func testGemmaReadsFlatQuantizationFieldsAtTheRoot() throws {
        let config = try gemmaConfig(rootFields: """
            "quantization_bits": 8,
            "quantization_group_size": 32
            """)

        XCTAssertEqual(config.quantBits, 8)
        XCTAssertEqual(config.quantGroupSize, 32)
    }

    func testGemmaReadsFlatQuantizationFieldsUnderTextConfig() throws {
        let config = try gemmaConfig(textFields: """
            "quantization_bits": 8,
            "quantization_group_size": 32
            """)

        XCTAssertEqual(config.quantBits, 8)
        XCTAssertEqual(config.quantGroupSize, 32)
    }

    func testGemmaFlatFieldsWinOverAStaleNestedObject() throws {
        let config = try gemmaConfig(rootFields: """
            "quantization": {"group_size": 64, "bits": 4},
            "quantization_bits": 5,
            "quantization_group_size": 32
            """)

        XCTAssertEqual(config.quantBits, 5)
        XCTAssertEqual(config.quantGroupSize, 32)
    }

    func testGemmaStillReadsTheNestedObject() throws {
        let config = try gemmaConfig(
            rootFields: #""quantization": {"group_size": 32, "bits": 8}"#)

        XCTAssertEqual(config.quantBits, 8)
        XCTAssertEqual(config.quantGroupSize, 32)
    }

    func testGemmaFallsBackToTheQuantizationLabel() throws {
        let config = try gemmaConfig(rootFields: #""quantization": "int8""#)

        XCTAssertEqual(config.quantBits, 8)
        XCTAssertEqual(config.quantGroupSize, 64)
    }

    func testGemmaCheckpointStatingNothingLoadsAsInt4() throws {
        let config = try gemmaConfig()

        XCTAssertEqual(config.quantBits, 4)
        XCTAssertEqual(config.quantGroupSize, 64)
    }

    // MARK: - Shared precedence rule

    func testLabelThatNamesNoUsableWidthIsIgnored() throws {
        // "bf16" parses to 16, which MLX cannot pack — ignore it rather than trust it.
        XCTAssertNil(ChatQuantization.bits(fromLabel: "bf16"))
        XCTAssertEqual(try denseConfig(quantizationFields: #""quantization": "bf16""#).quantBits, 4)
        XCTAssertEqual(try gemmaConfig(rootFields: #""quantization": "bf16""#).quantBits, 4)
    }

    /// The three configs must agree, so they resolve through one helper rather than three
    /// copies of the precedence rule.
    func testEveryChatConfigResolvesTheSameStatementIdentically() throws {
        let fields = """
            "quantization": "int4",
            "quantization_bits": 8,
            "quantization_group_size": 32
            """
        let qwen35 = try JSONDecoder().decode(Qwen3ChatConfig.self, from: Data("""
        {
          "hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 8,
          "num_key_value_heads": 2, "head_dim": 256, "intermediate_size": 3584,
          "vocab_size": 248320, "max_seq_len": 2048, "rope_theta": 10000000,
          "rms_norm_eps": 1e-6, "eos_token_id": 248046, "pad_token_id": 248044,
          \(fields)
        }
        """.utf8))

        let expected = ChatQuantization(bits: 8, groupSize: 32)
        XCTAssertEqual(qwen35.resolvedQuantization, expected)
        let dense = try denseConfig(quantizationFields: fields)
        XCTAssertEqual(ChatQuantization(bits: dense.quantBits, groupSize: dense.quantGroupSize),
                       expected)
        let gemma = try gemmaConfig(rootFields: fields)
        XCTAssertEqual(ChatQuantization(bits: gemma.quantBits, groupSize: gemma.quantGroupSize),
                       expected)
    }
}
