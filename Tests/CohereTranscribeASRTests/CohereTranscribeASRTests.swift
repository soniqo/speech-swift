import XCTest
import AudioCommon
import MLX
import MLXNN
@testable import CohereTranscribeASR

final class CohereTranscribeASRTests: XCTestCase {
    func testDecodesExportedConfiguration() throws {
        let json = """
        {
          "model_type": "cohere_asr",
          "vocab_size": 16384,
          "sample_rate": 16000,
          "max_audio_clip_s": 35,
          "encoder": {
            "d_model": 1280,
            "ff_expansion_factor": 4,
            "n_heads": 8,
            "conv_kernel_size": 9,
            "n_layers": 48,
            "pos_emb_max_len": 5000,
            "subsampling_conv_channels": 256,
            "subsampling_factor": 8,
            "feat_in": 128
          },
          "transf_decoder": {
            "config_dict": {
              "hidden_size": 1024,
              "inner_size": 4096,
              "num_attention_heads": 8,
              "num_layers": 8,
              "max_sequence_length": 1024
            }
          },
          "quantization": {"group_size": 64, "bits": 5, "mode": "affine"}
        }
        """.data(using: .utf8)!

        let config = try JSONDecoder().decode(CohereTranscribeConfig.self, from: json)

        XCTAssertEqual(config.modelType, "cohere_asr")
        XCTAssertEqual(config.vocabSize, 16_384)
        XCTAssertEqual(config.sampleRate, 16_000)
        XCTAssertEqual(config.overlapChunkSecond, 5)
        XCTAssertEqual(config.minEnergyWindowSamples, 1_600)
        XCTAssertEqual(config.encoder.nLayers, 48)
        XCTAssertEqual(config.encoder.dModel, 1_280)
        XCTAssertEqual(config.decoder.hiddenSize, 1_024)
        XCTAssertEqual(config.quantization, CohereMLXQuantization(bits: 5, groupSize: 64))
    }

    func testVocabFallsBackToHeadShape() throws {
        let json = """
        {
          "model_type": "cohere_asr",
          "sample_rate": 16000,
          "max_audio_clip_s": 35,
          "head": {"num_classes": 16384},
          "encoder": {
            "d_model": 1280, "ff_expansion_factor": 4, "n_heads": 8,
            "conv_kernel_size": 9, "n_layers": 48, "pos_emb_max_len": 5000,
            "subsampling_conv_channels": 256, "subsampling_factor": 8, "feat_in": 128
          },
          "transf_decoder": {"config_dict": {
            "hidden_size": 1024, "inner_size": 4096, "num_attention_heads": 8,
            "num_layers": 8, "max_sequence_length": 1024
          }}
        }
        """.data(using: .utf8)!

        let config = try JSONDecoder().decode(CohereTranscribeConfig.self, from: json)
        XCTAssertEqual(config.vocabSize, 16_384)
        XCTAssertNil(config.quantization)
    }

    func testQuantizationContractRejectsInt7() throws {
        XCTAssertNoThrow(try CohereTranscribeModel.validateQuantization(
            CohereMLXQuantization(bits: 5, groupSize: 64)))
        XCTAssertNoThrow(try CohereTranscribeModel.validateQuantization(
            CohereMLXQuantization(bits: 8, groupSize: 64)))
        XCTAssertThrowsError(try CohereTranscribeModel.validateQuantization(
            CohereMLXQuantization(bits: 7, groupSize: 64)))
        XCTAssertThrowsError(try CohereTranscribeModel.validateQuantization(
            CohereMLXQuantization(bits: 5, groupSize: 0)))
    }

    func testLanguageAliasesAndFallback() {
        XCTAssertEqual(CohereTranscribeTokenizer.languageToken(for: "English"), "<|en|>")
        XCTAssertEqual(CohereTranscribeTokenizer.languageToken(for: "zh-CN"), "<|zh|>")
        XCTAssertEqual(CohereTranscribeTokenizer.languageToken(for: "Japanese"), "<|ja|>")
        XCTAssertEqual(CohereTranscribeTokenizer.languageToken(for: "unsupported"), "<|en|>")
    }

    func testPreEmphasisKnownValues() {
        let output = CohereAudioFrontend.preEmphasized([1, 2, -1])
        XCTAssertEqual(output[0], 1, accuracy: 1e-6)
        XCTAssertEqual(output[1], 1.03, accuracy: 1e-6)
        XCTAssertEqual(output[2], -2.94, accuracy: 1e-6)
        XCTAssertEqual(CohereAudioFrontend.preEmphasized([]), [0])
    }

    func testNormalizationUsesOnlyCompleteHopFrames() {
        XCTAssertEqual(CohereAudioFrontend.normalizationFrameCount(sampleCount: 0), 0)
        XCTAssertEqual(CohereAudioFrontend.normalizationFrameCount(sampleCount: 159), 0)
        XCTAssertEqual(CohereAudioFrontend.normalizationFrameCount(sampleCount: 160), 1)
        XCTAssertEqual(CohereAudioFrontend.normalizationFrameCount(sampleCount: 16_000), 100)
    }

    func testLongAudioSplitsAtQuietBoundary() {
        var audio = [Float](repeating: 1, count: 30)
        audio[14] = 0
        audio[15] = 0
        let ranges = CohereTranscribeModel.energyChunkRanges(
            audio: audio,
            sampleRate: 10,
            maximumDuration: 2,
            boundaryContext: 1,
            energyWindowSamples: 2)

        XCTAssertEqual(ranges, [0..<14, 14..<30])
    }

    func testEmptyAudioProducesOneSafeRange() {
        XCTAssertEqual(CohereTranscribeModel.energyChunkRanges(
            audio: [], sampleRate: 16_000, maximumDuration: 35,
            boundaryContext: 5, energyWindowSamples: 1_600), [0..<0])
    }

    func testPublishedVariantsUseSupportedFormats() {
        XCTAssertEqual(CohereTranscribeVariant.allCases, [.fp16, .int5, .int8])
        XCTAssertEqual(CohereTranscribeModel.defaultModelId, CohereTranscribeVariant.int5.modelId)
        XCTAssertTrue(CohereTranscribeVariant.int5.modelId.hasSuffix("5bit"))
        XCTAssertTrue(CohereTranscribeVariant.int8.modelId.hasSuffix("8bit"))
    }

    func testPublishedDownloadFilesContainOnlyRuntimeInputs() {
        XCTAssertEqual(
            CohereTranscribeModel.downloadAdditionalFiles,
            ["tokenizer.model", "tokenizer_config.json"])
    }

    func testTinyModelParameterTreeMatchesCheckpointAliases() throws {
        let json = """
        {
          "model_type": "cohere_asr", "vocab_size": 16,
          "sample_rate": 16000, "max_audio_clip_s": 35,
          "encoder": {
            "d_model": 8, "ff_expansion_factor": 2, "n_heads": 2,
            "conv_kernel_size": 3, "n_layers": 1, "pos_emb_max_len": 32,
            "subsampling_conv_channels": 4, "subsampling_factor": 8, "feat_in": 16
          },
          "transf_decoder": {"config_dict": {
            "hidden_size": 4, "inner_size": 8, "num_attention_heads": 2,
            "num_layers": 1, "max_sequence_length": 32
          }}
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(CohereTranscribeConfig.self, from: json)
        let piece = SentencePieceModel.Piece(
            text: "x", score: 0, type: SentencePieceModel.PieceType.normal.rawValue)
        let tokenizer = CohereTranscribeTokenizer(
            pieces: Array(repeating: piece, count: 16),
            specialTokenToID: ["<|endoftext|>": 0])
        let model = try CohereTranscribeModel(config: config, tokenizer: tokenizer)
        let keys = Set(model.parameters().flattened().map { $0.0 })

        XCTAssertTrue(keys.contains("encoder.subsampling.conv0.weight"))
        XCTAssertTrue(keys.contains("encoder.layers.0.self_attn.qkv_proj.weight"))
        XCTAssertTrue(keys.contains("decoder.core.layers.0.first_sub_layer.qkv_proj.weight"))
        XCTAssertTrue(keys.contains("bridge_proj.weight"))
        XCTAssertTrue(keys.contains("lm_head.weight"))
    }

    func testInt5BundleRoundTripsThroughPublishedLoader() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let configJSON = """
        {
          "model_type": "cohere_asr", "vocab_size": 64,
          "sample_rate": 16000, "max_audio_clip_s": 35,
          "encoder": {
            "d_model": 64, "ff_expansion_factor": 2, "n_heads": 4,
            "conv_kernel_size": 3, "n_layers": 1, "pos_emb_max_len": 32,
            "subsampling_conv_channels": 8, "subsampling_factor": 8, "feat_in": 64
          },
          "transf_decoder": {"config_dict": {
            "hidden_size": 64, "inner_size": 128, "num_attention_heads": 4,
            "num_layers": 1, "max_sequence_length": 32
          }},
          "quantization": {"group_size": 64, "bits": 5, "mode": "affine"}
        }
        """
        try Data(configJSON.utf8).write(
            to: directory.appendingPathComponent("config.json"))
        try Self.minimalSentencePieceModel(pieceCount: 64).write(
            to: directory.appendingPathComponent("tokenizer.model"))
        let tokenizerJSON = """
        {"added_tokens_decoder": {
          "0":{"content":"<|endoftext|>"},
          "1":{"content":"<|startofcontext|>"},
          "2":{"content":"<|startoftranscript|>"},
          "3":{"content":"<|emo:undefined|>"},
          "4":{"content":"<|en|>"},
          "5":{"content":"<|pnc|>"},
          "6":{"content":"<|noitn|>"},
          "7":{"content":"<|notimestamp|>"},
          "8":{"content":"<|nodiarize|>"}
        }}
        """
        try Data(tokenizerJSON.utf8).write(
            to: directory.appendingPathComponent("tokenizer_config.json"))

        let config = try JSONDecoder().decode(
            CohereTranscribeConfig.self, from: Data(configJSON.utf8))
        let piece = SentencePieceModel.Piece(
            text: "x", score: 0,
            type: SentencePieceModel.PieceType.normal.rawValue)
        let tokenizer = CohereTranscribeTokenizer(
            pieces: Array(repeating: piece, count: 64),
            specialTokenToID: ["<|endoftext|>": 0])
        let source = try CohereTranscribeModel(config: config, tokenizer: tokenizer)
        MLXNN.quantize(model: source) { _, module in
            (module is Linear || module is Embedding) ? (64, 5, .affine) : nil
        }
        let parameters = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        try MLX.save(
            arrays: parameters,
            url: directory.appendingPathComponent("model.safetensors"))

        let loaded = try CohereTranscribeModel.fromDirectory(directory)
        let loadedKeys = Set(loaded.parameters().flattened().map { $0.0 })
        XCTAssertEqual(loaded.config.quantization?.bits, 5)
        XCTAssertTrue(loadedKeys.contains(where: { $0.hasSuffix(".scales") }))
    }

    private static func minimalSentencePieceModel(pieceCount: Int) -> Data {
        // ModelProto.pieces (field 1) containing a SentencePiece whose only
        // explicit field is piece="x". All lengths fit in one-byte varints.
        let encodedPiece: [UInt8] = [0x0A, 0x03, 0x0A, 0x01, 0x78]
        return Data((0..<pieceCount).flatMap { _ in encodedPiece })
    }
}
