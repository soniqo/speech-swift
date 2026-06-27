import MLX
@testable import IndicMioTTS
import XCTest

final class IndicMioRuntimeTests: XCTestCase {
    func testPromptUsesIndicMioChatTemplate() throws {
        let text = "नमस्ते, मैं आज बहुत खुश हूँ। <happy>"
        let prompt = try IndicMioPrompt.chatPrompt(for: text)

        XCTAssertEqual(
            prompt,
            "<|im_start|>user\nनमस्ते, मैं आज बहुत खुश हूँ। <happy><|im_end|>\n<|im_start|>assistant\n"
        )
    }

    func testPromptRejectsUnsupportedAngleMarker() {
        XCTAssertThrowsError(try IndicMioPrompt.chatPrompt(for: "नमस्ते <whisper>")) { error in
            XCTAssertEqual(error as? IndicMioError, .unsupportedMarker("<whisper>"))
        }
    }

    func testSpeechTokenMappingUsesIndicMioOffset() {
        XCTAssertEqual(IndicMioSpeechTokens.speechCode(from: 151_668), nil)
        XCTAssertEqual(IndicMioSpeechTokens.speechCode(from: 151_669), 0)
        XCTAssertEqual(IndicMioSpeechTokens.speechCode(from: 151_670), 1)
        XCTAssertEqual(IndicMioSpeechTokens.speechCode(from: 164_468), 12_799)
        XCTAssertEqual(IndicMioSpeechTokens.speechCode(from: 164_469), nil)
    }

    func testMioCodecFSQDecode() {
        let ids = MLXArray([Int32(0), 1, 8, 64, 512, 2_560, 12_799])
            .expandedDimensions(axis: 0)
        let decoded = MioCodecFSQ.decode(ids)
        eval(decoded)
        let values = decoded.asArray(Float.self)

        XCTAssertEqual(decoded.shape, [1, 7, 5])
        XCTAssertEqual(values[0], -1.0, accuracy: 1e-6)
        XCTAssertEqual(values[1], -1.0, accuracy: 1e-6)
        XCTAssertEqual(values[2], -1.0, accuracy: 1e-6)
        XCTAssertEqual(values[3], -1.0, accuracy: 1e-6)
        XCTAssertEqual(values[4], -1.0, accuracy: 1e-6)

        XCTAssertEqual(values[5], -0.75, accuracy: 1e-6)
        XCTAssertEqual(values[6], -1.0, accuracy: 1e-6)

        let last = Array(values.suffix(5))
        XCTAssertEqual(last[0], 0.75, accuracy: 1e-6)
        XCTAssertEqual(last[1], 0.75, accuracy: 1e-6)
        XCTAssertEqual(last[2], 0.75, accuracy: 1e-6)
        XCTAssertEqual(last[3], 1.0, accuracy: 1e-6)
        XCTAssertEqual(last[4], 1.0, accuracy: 1e-6)
    }

    func testDecodePlanUsesTwentyFiveHzCodecRate() {
        let plan = MioCodecDecodePlan(tokenCount: 69)
        XCTAssertEqual(plan.sampleRate, 24_000)
        XCTAssertEqual(plan.estimatedSamples, 66_240)
        XCTAssertEqual(plan.stftFrames, 138)
    }

    func testModelConfigParsesIndicMioShape() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("indic-mio-config-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let configURL = dir.appendingPathComponent("config.json")
        let json = """
        {
          "hidden_size": 1024,
          "num_hidden_layers": 28,
          "num_attention_heads": 16,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "intermediate_size": 3072,
          "vocab_size": 164480,
          "rope_theta": 1000000,
          "rms_norm_eps": 0.000001,
          "tie_word_embeddings": true,
          "eos_token_id": 151645,
          "pad_token_id": 151643
        }
        """
        try json.write(to: configURL, atomically: true, encoding: .utf8)

        let config = try IndicMioModelConfig.load(from: configURL)
        XCTAssertEqual(config.hiddenSize, 1024)
        XCTAssertEqual(config.numHiddenLayers, 28)
        XCTAssertEqual(config.vocabSize, 164_480)
        XCTAssertTrue(config.tieWordEmbeddings)
        XCTAssertEqual(config.eosTokenId, 151_645)
        XCTAssertEqual(config.padTokenId, 151_643)
    }
}
