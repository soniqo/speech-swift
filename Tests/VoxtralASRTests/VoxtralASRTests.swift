import Foundation
import MLX
import MLXNN
import XCTest
@testable import VoxtralASR

final class VoxtralASRTests: XCTestCase {
    func testDecodesPublishedConfiguration() throws {
        let json = """
        {
          "model_type": "voxtral",
          "audio_token_id": 24,
          "projector_hidden_act": "gelu",
          "vocab_size": 131072,
          "hidden_size": 3072,
          "audio_config": {
            "hidden_size": 1280, "num_hidden_layers": 32,
            "intermediate_size": 5120, "num_attention_heads": 20,
            "num_key_value_heads": 20, "rms_norm_eps": 0.00001,
            "head_dim": 64, "rope_theta": 1000000,
            "vocab_size": 51866, "num_mel_bins": 128,
            "max_source_positions": 1500
          },
          "text_config": {
            "vocab_size": 131072, "max_position_embeddings": 131072,
            "hidden_size": 3072, "intermediate_size": 8192,
            "num_hidden_layers": 30, "num_attention_heads": 32,
            "num_key_value_heads": 8, "rms_norm_eps": 0.00001,
            "head_dim": 128, "tie_word_embeddings": false,
            "bos_token_id": 1, "eos_token_id": 2, "rope_theta": 100000000
          },
          "quantization": {"group_size": 64, "bits": 5, "mode": "affine"}
        }
        """.data(using: .utf8)!

        let config = try JSONDecoder().decode(VoxtralConfig.self, from: json)

        XCTAssertEqual(config.modelType, "voxtral")
        XCTAssertEqual(config.audioTokenID, 24)
        XCTAssertEqual(config.audioConfig.numHiddenLayers, 32)
        XCTAssertEqual(config.audioConfig.maxSourcePositions, 1_500)
        XCTAssertEqual(config.textConfig.numHiddenLayers, 30)
        XCTAssertEqual(config.textConfig.numKeyValueHeads, 8)
        XCTAssertEqual(config.quantization, VoxtralMLXQuantization(bits: 5, groupSize: 64))
    }

    func testQuantizationContractRejectsInt7() throws {
        XCTAssertNoThrow(try VoxtralModel.validateQuantization(
            VoxtralMLXQuantization(bits: 5, groupSize: 64)))
        XCTAssertNoThrow(try VoxtralModel.validateQuantization(
            VoxtralMLXQuantization(bits: 8, groupSize: 64)))
        XCTAssertThrowsError(try VoxtralModel.validateQuantization(
            VoxtralMLXQuantization(bits: 7, groupSize: 64)))
        XCTAssertThrowsError(try VoxtralModel.validateQuantization(
            VoxtralMLXQuantization(bits: 5, groupSize: 0)))
    }

    func testThirtySecondPackingGeometry() {
        XCTAssertEqual(VoxtralAudioFrontend.paddedChunkCount(forSampleCount: 0), 1)
        XCTAssertEqual(VoxtralAudioFrontend.paddedChunkCount(forSampleCount: 480_000), 1)
        XCTAssertEqual(VoxtralAudioFrontend.paddedChunkCount(forSampleCount: 480_001), 2)
        XCTAssertEqual(VoxtralAudioFrontend.audioTokenCount(forSampleCount: 1), 375)
        XCTAssertEqual(VoxtralAudioFrontend.audioTokenCount(forSampleCount: 960_000), 750)
    }

    func testOfficialLanguagePromptTokenIDs() {
        XCTAssertEqual(VoxtralTokenizer.languagePromptTokens("en"), [9_909, 1_058, 1_262])
        XCTAssertEqual(VoxtralTokenizer.languagePromptTokens("French"), [9_909, 1_058, 7_064])
        XCTAssertEqual(VoxtralTokenizer.languagePromptTokens("nl-NL"), [9_909, 24_082, 1_108])
        XCTAssertEqual(VoxtralTokenizer.languagePromptTokens("unsupported"), [9_909, 1_058, 1_262])
    }

    func testTekkenPromptAndByteDecode() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let tekken = """
        {
          "config": {"default_num_special_tokens": 1000},
          "special_tokens": [
            {"rank": 1, "token_str": "<s>"},
            {"rank": 3, "token_str": "[INST]"},
            {"rank": 4, "token_str": "[/INST]"},
            {"rank": 24, "token_str": "[AUDIO]"},
            {"rank": 25, "token_str": "[BEGIN_AUDIO]"},
            {"rank": 34, "token_str": "[TRANSCRIBE]"}
          ],
          "vocab": [
            {"token_bytes": "SGVsbG8="},
            {"token_bytes": "IHdvcmxk"}
          ]
        }
        """
        let url = directory.appendingPathComponent("tekken.json")
        try Data(tekken.utf8).write(to: url)
        let tokenizer = try VoxtralTokenizer(tekkenURL: url)

        XCTAssertEqual(
            tokenizer.transcriptionPrompt(audioTokenCount: 2, language: "en"),
            [1, 3, 25, 24, 24, 4, 9_909, 1_058, 1_262, 34])
        XCTAssertEqual(tokenizer.decode([1_000, 1_001, 2]), "Hello world")
    }

    func testPublishedVariantsUseSupportedFormats() {
        XCTAssertEqual(VoxtralVariant.allCases, [.fp16, .int5, .int8])
        XCTAssertEqual(VoxtralModel.defaultModelId, VoxtralVariant.int5.modelId)
        XCTAssertTrue(VoxtralVariant.int5.modelId.hasSuffix("5bit"))
        XCTAssertTrue(VoxtralVariant.int8.modelId.hasSuffix("8bit"))
    }

    func testLastStateProjectionMatchesFullSequenceLastRow() throws {
        let json = """
        {
          "vocab_size": 64, "max_position_embeddings": 64,
          "hidden_size": 8, "intermediate_size": 16,
          "num_hidden_layers": 1, "num_attention_heads": 2,
          "num_key_value_heads": 1, "rms_norm_eps": 0.00001,
          "head_dim": 4, "tie_word_embeddings": false,
          "bos_token_id": 1, "eos_token_id": 2, "rope_theta": 100000000
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(VoxtralTextConfig.self, from: json)
        let languageModel = VoxtralLanguageModel(config)
        let hidden = MLXArray((0..<24).map { Float($0) / 24 })
            .reshaped(1, 3, 8)

        let fullLogits = languageModel.logits(hidden)
        let lastLogits = languageModel.logits(hidden[0, -1])
        eval(fullLogits, lastLogits)

        XCTAssertEqual(fullLogits.shape, [1, 3, 64])
        XCTAssertEqual(lastLogits.shape, [64])
        let largestDifference = MLX.max(abs(fullLogits[0, -1] - lastLogits))
            .item(Float.self)
        XCTAssertLessThan(largestDifference, 1e-3)
        XCTAssertEqual(
            fullLogits[0, -1].argMax().item(Int.self),
            lastLogits.argMax().item(Int.self))
    }

    func testTinyModelParameterTreeMatchesConvertedKeys() throws {
        let configJSON = """
        {
          "model_type": "voxtral", "audio_token_id": 24,
          "projector_hidden_act": "gelu", "vocab_size": 64, "hidden_size": 8,
          "audio_config": {
            "hidden_size": 8, "num_hidden_layers": 1, "intermediate_size": 32,
            "num_attention_heads": 2, "num_key_value_heads": 2,
            "rms_norm_eps": 0.00001, "head_dim": 4, "rope_theta": 1000000,
            "vocab_size": 64, "num_mel_bins": 4, "max_source_positions": 4
          },
          "text_config": {
            "vocab_size": 64, "max_position_embeddings": 64,
            "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 1, "num_attention_heads": 2,
            "num_key_value_heads": 1, "rms_norm_eps": 0.00001,
            "head_dim": 4, "tie_word_embeddings": false,
            "bos_token_id": 1, "eos_token_id": 2, "rope_theta": 100000000
          }
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(VoxtralConfig.self, from: configJSON)
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let tekken = """
        {"config":{"default_num_special_tokens":1},"special_tokens":[],
         "vocab":[{"token_bytes":"eA=="}]}
        """
        let tekkenURL = directory.appendingPathComponent("tekken.json")
        try Data(tekken.utf8).write(to: tekkenURL)
        let tokenizer = try VoxtralTokenizer(tekkenURL: tekkenURL)
        let model = try VoxtralModel(config: config, tokenizer: tokenizer)
        let keys = Set(model.parameters().flattened().map { $0.0 })

        XCTAssertTrue(keys.contains("audio_tower.conv1.weight"))
        XCTAssertTrue(keys.contains("audio_tower.layers.0.self_attn.q_proj.weight"))
        XCTAssertTrue(keys.contains("multi_modal_projector.linear_1.weight"))
        XCTAssertTrue(keys.contains("language_model.model.layers.0.self_attn.q_proj.weight"))
        XCTAssertTrue(keys.contains("language_model.lm_head.weight"))
    }

    func testInt5BundleRoundTripsThroughPublishedLoader() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let configJSON = """
        {
          "model_type": "voxtral", "audio_token_id": 24,
          "projector_hidden_act": "gelu", "vocab_size": 64, "hidden_size": 64,
          "audio_config": {
            "hidden_size": 64, "num_hidden_layers": 1, "intermediate_size": 256,
            "num_attention_heads": 4, "num_key_value_heads": 4,
            "rms_norm_eps": 0.00001, "head_dim": 16, "rope_theta": 1000000,
            "vocab_size": 64, "num_mel_bins": 4, "max_source_positions": 4
          },
          "text_config": {
            "vocab_size": 64, "max_position_embeddings": 64,
            "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 1, "num_attention_heads": 4,
            "num_key_value_heads": 2, "rms_norm_eps": 0.00001,
            "head_dim": 16, "tie_word_embeddings": false,
            "bos_token_id": 1, "eos_token_id": 2, "rope_theta": 100000000
          },
          "quantization": {"group_size": 64, "bits": 5, "mode": "affine"}
        }
        """
        try Data(configJSON.utf8).write(
            to: directory.appendingPathComponent("config.json"))
        let tekken = """
        {"config":{"default_num_special_tokens":64},"special_tokens":[
          {"rank":1,"token_str":"<s>"},
          {"rank":3,"token_str":"[INST]"},
          {"rank":4,"token_str":"[/INST]"},
          {"rank":24,"token_str":"[AUDIO]"},
          {"rank":25,"token_str":"[BEGIN_AUDIO]"},
          {"rank":34,"token_str":"[TRANSCRIBE]"}
        ],"vocab":[{"token_bytes":"eA=="}]}
        """
        let tekkenURL = directory.appendingPathComponent("tekken.json")
        try Data(tekken.utf8).write(to: tekkenURL)

        let config = try JSONDecoder().decode(
            VoxtralConfig.self, from: Data(configJSON.utf8))
        let tokenizer = try VoxtralTokenizer(tekkenURL: tekkenURL)
        let source = try VoxtralModel(config: config, tokenizer: tokenizer)
        MLXNN.quantize(model: source) { path, module in
            guard !path.hasPrefix("audio_tower"),
                  module is Linear || module is Embedding else { return nil }
            return (64, 5, .affine)
        }
        let parameters = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        try MLX.save(
            arrays: parameters,
            url: directory.appendingPathComponent("model.safetensors"))

        let loaded = try VoxtralModel.fromDirectory(directory)
        let loadedKeys = Set(loaded.parameters().flattened().map { $0.0 })
        XCTAssertEqual(loaded.config.quantization?.bits, 5)
        XCTAssertTrue(loadedKeys.contains(where: {
            $0.hasPrefix("language_model.") && $0.hasSuffix(".scales")
        }))
        XCTAssertFalse(loadedKeys.contains(where: {
            $0.hasPrefix("audio_tower.") && $0.hasSuffix(".scales")
        }))
    }
}
