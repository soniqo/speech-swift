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
