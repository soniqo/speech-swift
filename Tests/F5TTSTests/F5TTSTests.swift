import AudioCommon
@testable import F5TTS
import XCTest

final class F5TTSTests: XCTestCase {
    func testConfigParsesExporterShape() throws {
        let dir = try makeBundle()
        defer { try? FileManager.default.removeItem(at: dir) }

        let config = try F5TTSConfig.load(from: dir.appendingPathComponent("config.json"))

        XCTAssertEqual(config.modelType, "f5-tts")
        XCTAssertEqual(config.modelName, "F5TTS_v1_Base")
        XCTAssertEqual(config.architecture.dim, 1024)
        XCTAssertEqual(config.architecture.depth, 22)
        XCTAssertEqual(config.architecture.heads, 16)
        XCTAssertEqual(config.architecture.textDim, 512)
        XCTAssertEqual(config.melSpec.targetSampleRate, 24_000)
        XCTAssertEqual(config.melSpec.nMelChannels, 100)
        XCTAssertEqual(config.files.model, "model.safetensors")
        XCTAssertFalse(config.commercialUse)
    }

    func testBundleLoadsAndReportsMemory() async throws {
        let dir = try makeBundle()
        defer { try? FileManager.default.removeItem(at: dir) }

        let model = try await F5TTSModel.fromBundle(dir)

        XCTAssertEqual(model.sampleRate, 24_000)
        XCTAssertEqual(model.config.license, "cc-by-nc-4.0")
        XCTAssertEqual(model.memoryFootprint, 32)
        XCTAssertTrue(model.isLoaded)
        XCTAssertEqual(try model.tokenizer.encode("Hi!"), [39, 0, 1])
    }

    func testBundleLoaderValidatesRequiredFiles() throws {
        let dir = try makeBundle(writeVocoder: false)
        defer { try? FileManager.default.removeItem(at: dir) }

        XCTAssertThrowsError(try F5TTSBundleLoader.load(from: dir)) { error in
            XCTAssertEqual(error as? F5TTSError, .missingRequiredFile("vocos.safetensors"))
        }
    }

    func testTokenizerNormalizesAsciiAndRejectsCJK() throws {
        let tokenizer = F5TTSTokenizer(vocab: [
            " ": 0, "!": 1, "\"": 2, "'": 7, ",": 12,
            "H": 39, "a": 62, "e": 304, "i": 1121, "r": 1671,
        ])

        let tokenized = try tokenizer.tokenize("Hi; “a”")

        XCTAssertEqual(tokenized.symbols, ["H", "i", ",", " ", "\"", "a", "\""])
        XCTAssertEqual(tokenized.ids, [39, 1121, 12, 0, 2, 62, 2])
        XCTAssertThrowsError(try tokenizer.encode("你好")) { error in
            XCTAssertTrue((error as? F5TTSError)?.localizedDescription.contains("pinyin") == true)
        }
    }

    func testReferenceTextNormalizationMatchesF5Inference() {
        XCTAssertEqual(F5TTSFlow.normalizedReferenceText("hello"), "hello.  ")
        XCTAssertEqual(F5TTSFlow.normalizedReferenceText("hello."), "hello.  ")
        XCTAssertEqual(F5TTSFlow.normalizedReferenceText("你好。"), "你好。")
    }

    func testTextOnlyGenerateRequiresReferenceAudio() async throws {
        let dir = try makeBundle()
        defer { try? FileManager.default.removeItem(at: dir) }
        let model = try await F5TTSModel.fromBundle(dir)

        do {
            _ = try await model.generate(text: "Hello", language: "english")
            XCTFail("Expected reference-required error")
        } catch let error as F5TTSError {
            XCTAssertTrue(error.localizedDescription.contains("reference audio"))
        }
    }

    private func makeBundle(writeVocoder: Bool = true) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("f5tts-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let config = """
        {
          "model_type": "f5-tts",
          "model_name": "F5TTS_v1_Base",
          "source_repo": "SWivid/F5-TTS",
          "source_checkpoint": "F5TTS_v1_Base/model_1250000.safetensors",
          "vocoder_repo": "charactr/vocos-mel-24khz",
          "license": "cc-by-nc-4.0",
          "commercial_use": false,
          "precision": "fp16",
          "architecture": {
            "backbone": "DiT",
            "dim": 1024,
            "depth": 22,
            "heads": 16,
            "ff_mult": 2,
            "text_dim": 512,
            "text_mask_padding": true,
            "qk_norm": null,
            "conv_layers": 4,
            "pe_attn_head": null,
            "attn_backend": "torch",
            "attn_mask_enabled": false,
            "checkpoint_activations": false
          },
          "mel_spec": {
            "target_sample_rate": 24000,
            "n_mel_channels": 100,
            "hop_length": 256,
            "win_length": 1024,
            "n_fft": 1024,
            "mel_spec_type": "vocos"
          },
          "files": {
            "model": "model.safetensors",
            "vocoder": "vocos.safetensors",
            "vocoder_config": "vocos_config.yaml",
            "vocab": "vocab.txt"
          },
          "conversion": {
            "f5": {
              "source_tensors": 366,
              "saved_tensors": 364,
              "skipped": ["initted", "step"],
              "source_dtype_counts": {"torch.float32": 364},
              "size_mb": 643.0
            },
            "vocos": {
              "source_tensors": 83,
              "saved_tensors": 83,
              "size_mb": 25.9
            }
          }
        }
        """
        try config.write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try "F5-TTS test bundle\n".write(to: dir.appendingPathComponent("README.md"), atomically: true, encoding: .utf8)
        let vocab = [
            " ", "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "=", ">", "?", "@",
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "a",
        ].joined(separator: "\r\n") + "\r\n"
        try vocab.write(
            to: dir.appendingPathComponent("vocab.txt"),
            atomically: true,
            encoding: .utf8)
        try Data(repeating: 0, count: 16).write(to: dir.appendingPathComponent("model.safetensors"))
        if writeVocoder {
            try Data(repeating: 0, count: 16).write(to: dir.appendingPathComponent("vocos.safetensors"))
        }
        try "sample_rate: 24000\n".write(to: dir.appendingPathComponent("vocos_config.yaml"), atomically: true, encoding: .utf8)
        return dir
    }
}
