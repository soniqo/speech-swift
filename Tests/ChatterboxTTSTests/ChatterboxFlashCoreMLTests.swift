#if canImport(CoreML)
import CoreML
import Foundation
import XCTest

@testable import ChatterboxTTS

final class ChatterboxFlashCoreMLTests: XCTestCase {
    func testBundleConfigDecodesExportedShapeMetadata() throws {
        let directory = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let json = """
        {
          "model_type": "chatterbox_flash_coreml",
          "base_model": "ResembleAI/chatterbox-flash",
          "format": "coreml-mlmodelc",
          "components": {
            "t3": {
              "path": "t3",
              "text_len": 256,
              "block_size": 16,
              "max_seq": 1024
            },
            "audio": {
              "path": "audio",
              "token_len": 192,
              "mel_len": 384
            }
          }
        }
        """
        try json.write(to: directory.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

        let config = try ChatterboxFlashCoreMLBundleConfig.load(from: directory)

        XCTAssertEqual(config.bundle.sourceModel, "ResembleAI/chatterbox-flash")
        XCTAssertEqual(config.components.t3.speechLen, 1024)
        XCTAssertEqual(config.components.t3.blockSize, 16)
        XCTAssertEqual(config.components.t3.speechVocabSize, 8194)
        XCTAssertEqual(config.components.audio.sampleRate, 24_000)
        XCTAssertEqual(config.components.audio.melLen, 384)
    }

    func testT3ConfigDecodesStandaloneExportConfig() throws {
        let directory = try makeTempDirectory()
        let t3Directory = directory.appendingPathComponent("t3", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: t3Directory, withIntermediateDirectories: true)

        let json = """
        {
          "text_len": 256,
          "block_size": 16,
          "max_seq": 1024,
          "cond_len": 34,
          "prompt_speech_len": 150,
          "hidden_size": 1024,
          "num_layers": 30,
          "num_attention_heads": 16,
          "num_key_value_heads": 16,
          "head_dim": 64,
          "kv_cache_shape": [1, 30720, 1, 1024],
          "text_vocab_size": 704,
          "speech_vocab_size": 8194,
          "mask_token_id": 8194,
          "start_speech_token": 6561,
          "stop_speech_token": 6562,
          "start_text_token": 255,
          "stop_text_token": 0
        }
        """
        try json.write(to: t3Directory.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

        let config = try ChatterboxFlashT3Config.load(from: directory)

        XCTAssertEqual(config.maxSeq, 1024)
        XCTAssertEqual(config.speechLen, 1024)
        XCTAssertEqual(config.condLen, 34)
        XCTAssertEqual(config.promptSpeechLen, 150)
        XCTAssertEqual(config.prefixLen, 291)
        XCTAssertEqual(config.speechVocabSize, 8194)
        XCTAssertEqual(config.maskTokenId, 8194)
        XCTAssertEqual(config.startSpeechToken, 6561)
        XCTAssertEqual(config.stopSpeechToken, 6562)
    }

    func testAudioConfigDecodesNestedAudioConfig() throws {
        let directory = try makeTempDirectory()
        let audioDirectory = directory.appendingPathComponent("audio", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: audioDirectory, withIntermediateDirectories: true)

        let json = """
        {
          "sample_rate": 24000,
          "token_len": 192,
          "mel_len": 384,
          "token_mel_ratio": 2,
          "samples_per_mel_frame": 480,
          "ref_embedding_dim": 192,
          "speaker_projection_dim": 80
        }
        """
        try json.write(to: audioDirectory.appendingPathComponent("audio_config.json"), atomically: true, encoding: .utf8)

        let config = try ChatterboxFlashAudioConfig.load(from: directory)

        XCTAssertEqual(config.tokenLen, 192)
        XCTAssertEqual(config.tokenMelRatio, 2)
        XCTAssertEqual(config.samplesPerMelFrame, 480)
        XCTAssertEqual(config.refEmbeddingDim, 192)
    }

    func testReferenceStoresPromptConditioning() {
        let reference = ChatterboxFlashS3GenReference(
            embedding: Array(repeating: 0.25, count: 192),
            promptToken: [10, 11, 12],
            promptFeature: Array(0..<160).map(Float.init),
            promptFeatureFrames: 2
        )

        XCTAssertEqual(reference.embedding.count, 192)
        XCTAssertEqual(reference.promptToken, [10, 11, 12])
        XCTAssertEqual(reference.promptFeatureFrames, 2)
        XCTAssertEqual(reference.promptFeature.count, 160)
        XCTAssertEqual(reference.promptFeature[80], 80)
    }

    func testGaussianRNGIsDeterministic() {
        var lhs = ChatterboxFlashGaussianRNG(seed: 42)
        var rhs = ChatterboxFlashGaussianRNG(seed: 42)

        let a = lhs.normal(count: 8)
        let b = rhs.normal(count: 8)

        XCTAssertEqual(a.count, 8)
        XCTAssertEqual(a, b)
        XCTAssertFalse(a.allSatisfy { $0 == 0 })
    }

    func testGenerationOptionsDefaultKeepsNullPrefillOptional() {
        XCTAssertEqual(ChatterboxFlashGenerationOptions().cfgScale, 0)
    }

    func testFlashTokenizerUsesWhitespaceBPEAndSpecialTokens() throws {
        let directory = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("tokenizer.json")
        let json = """
        {
          "model": {
            "unk_token": "[UNK]",
            "vocab": {
              "[STOP]": 0,
              "[UNK]": 1,
              "[SPACE]": 2,
              "[START]": 255,
              "C": 10,
              "o": 11,
              "Co": 12,
              "r": 13,
              "e": 14,
              "re": 15,
              "M": 16,
              "L": 17,
              ".": 18
            },
            "merges": ["C o", "r e"]
          },
          "added_tokens": [
            {"id": 0, "content": "[STOP]"},
            {"id": 1, "content": "[UNK]"},
            {"id": 2, "content": "[SPACE]"},
            {"id": 255, "content": "[START]"}
          ]
        }
        """
        try json.write(to: url, atomically: true, encoding: .utf8)
        let tokenizer = try ChatterboxFlashTokenizer(tokenizerURL: url)

        XCTAssertEqual(tokenizer.encode("Core ML."), [255, 12, 15, 2, 16, 17, 18, 0])
        let padded = try tokenizer.encodePadded(
            "Core",
            config: ChatterboxFlashT3Config(
                textLen: 8,
                maxSeq: 32,
                blockSize: 4,
                condLen: 2,
                promptSpeechLen: 4,
                hiddenSize: 8,
                numLayers: 1,
                numAttentionHeads: 1,
                numKeyValueHeads: 1,
                headDim: 8,
                kvCacheShape: [1, 8, 1, 32],
                textVocabSize: 256,
                speechVocabSize: 16,
                maskTokenId: 16,
                startSpeechToken: 10,
                stopSpeechToken: 11,
                startTextToken: 255,
                stopTextToken: 0
            )
        )
        XCTAssertEqual(padded, [255, 12, 15, 0, 0, 0, 0, 0])
    }

    func testT3PrefixPlanCompactsPaddedText() throws {
        let config = ChatterboxFlashT3Config(
            textLen: 8,
            maxSeq: 32,
            blockSize: 4,
            condLen: 2,
            promptSpeechLen: 4,
            hiddenSize: 8,
            numLayers: 1,
            numAttentionHeads: 1,
            numKeyValueHeads: 1,
            headDim: 8,
            kvCacheShape: [1, 8, 1, 32],
            textVocabSize: 256,
            speechVocabSize: 16,
            maskTokenId: 16,
            startSpeechToken: 10,
            stopSpeechToken: 11,
            startTextToken: 255,
            stopTextToken: 0
        )

        let plan = try ChatterboxFlashT3Graphs.makePrefixPlan(config: config, encodedTextCount: 3)

        XCTAssertEqual(plan.compactPrefixLen, 6)
        XCTAssertEqual(plan.positionIds.count, config.prefixLen)
        XCTAssertEqual(plan.positionIds.last, 5)
        XCTAssertEqual(plan.keyPaddingMask[0..<5], Array(repeating: Float(0), count: 5)[...])
        XCTAssertEqual(plan.keyPaddingMask[5..<10], Array(repeating: Float(-1.0e4), count: 5)[...])
        XCTAssertEqual(plan.keyPaddingMask[10], 0)
        XCTAssertEqual(plan.prefixCacheMap[0 * config.maxSeq + 0], 1)
        XCTAssertEqual(plan.prefixCacheMap[4 * config.maxSeq + 4], 1)
        XCTAssertEqual(plan.prefixCacheMap[10 * config.maxSeq + 5], 1)
        XCTAssertEqual(plan.prefixCacheMap[(5 * config.maxSeq)..<(6 * config.maxSeq)].reduce(0, +), 0)
    }

    func testT3BlockPositionsUseCacheOffset() {
        let ids = ChatterboxFlashT3Graphs.makeBlockPositionIds(prefixLen: 39, blockStart: 16, blockSize: 4)
        XCTAssertEqual(ids, [55, 56, 57, 58])
    }

    func testToFloat32ReadsInt8MultiArrayWithoutSDKCaseReference() throws {
        guard let int8DataType = MLMultiArrayDataType(rawValue: 0x20000 | 8) else {
            throw XCTSkip("CoreML SDK does not expose the int8 data type raw value")
        }

        let array: MLMultiArray
        do {
            array = try MLMultiArray(shape: [4], dataType: int8DataType)
        } catch {
            throw XCTSkip("CoreML runtime does not support int8 MLMultiArray allocation")
        }
        guard array.dataType.rawValue == int8DataType.rawValue else {
            throw XCTSkip("CoreML runtime does not preserve int8 MLMultiArray allocation")
        }

        let values: [Int8] = [-8, -1, 0, 127]
        let pointer = array.dataPointer.bindMemory(to: Int8.self, capacity: values.count)
        pointer.update(from: values, count: values.count)

        XCTAssertEqual(try ChatterboxFlashCoreMLBridge.toFloat32(array), [-8, -1, 0, 127])
    }

    func testNumpyFloat32VectorReader() throws {
        let values: [Float] = [0.25, -1.5, 3.0]
        let parsed = try ChatterboxFlashNumpy.parseFloat32Vector(makeNpy(values), label: "fixture.npy")

        XCTAssertEqual(parsed.count, values.count)
        for (lhs, rhs) in zip(parsed, values) {
            XCTAssertEqual(lhs, rhs, accuracy: 1e-6)
        }
    }

    private func makeTempDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ChatterboxFlashCoreMLTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func makeNpy(_ values: [Float]) -> Data {
        var data = Data([0x93])
        data.append("NUMPY".data(using: .ascii)!)
        data.append(contentsOf: [1, 0])
        var header = "{'descr': '<f4', 'fortran_order': False, 'shape': (\(values.count),), }"
        let baseLength = 10 + header.utf8.count + 1
        let padding = (16 - (baseLength % 16)) % 16
        header += String(repeating: " ", count: padding) + "\n"
        let headerLength = UInt16(header.utf8.count)
        data.append(UInt8(headerLength & 0xff))
        data.append(UInt8((headerLength >> 8) & 0xff))
        data.append(header.data(using: .ascii)!)
        for value in values {
            let bits = value.bitPattern
            data.append(UInt8(bits & 0xff))
            data.append(UInt8((bits >> 8) & 0xff))
            data.append(UInt8((bits >> 16) & 0xff))
            data.append(UInt8((bits >> 24) & 0xff))
        }
        return data
    }
}
#endif
