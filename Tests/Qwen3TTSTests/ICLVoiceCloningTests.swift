import XCTest
@testable import Qwen3TTS
import MLX
import AudioCommon

// MARK: - Unit Tests

final class ICLVoiceCloningTests: XCTestCase {

    // MARK: - VectorQuantizer encode()

    func testVectorQuantizerCodebookEncode() {
        let codebook = VectorQuantizerCodebook(codebookSize: 16, codebookDim: 4)
        // Input: [1, 3, 4] — 3 vectors of dim 4
        let input = MLXArray.zeros([1, 3, 4])
        let codes = codebook.encode(input)
        XCTAssertEqual(codes.shape, [1, 3], "Should produce one index per vector")
        // All zero vectors should map to the same codebook entry
        let c0 = codes[0, 0].item(Int32.self)
        let c1 = codes[0, 1].item(Int32.self)
        XCTAssertEqual(c0, c1, "Identical inputs should map to same codebook entry")
    }

    func testVectorQuantizerRoundTrip() {
        let codebook = VectorQuantizerCodebook(codebookSize: 256, codebookDim: 8)
        // Encode then decode should produce vectors close to the original codebook entries
        let indices = MLXArray([Int32(0), Int32(5), Int32(100)]).expandedDimensions(axis: 0)  // [1, 3]
        let decoded = codebook.decode(indices)  // [1, 3, 8]
        let reEncoded = codebook.encode(decoded)  // [1, 3]
        eval(reEncoded)
        // Round-trip should recover the same indices
        XCTAssertEqual(reEncoded[0, 0].item(Int32.self), 0)
        XCTAssertEqual(reEncoded[0, 1].item(Int32.self), 5)
        XCTAssertEqual(reEncoded[0, 2].item(Int32.self), 100)
    }

    // MARK: - SpeechTokenizerEncoder construction

    func testEncoderConstruction() {
        let config = SpeechTokenizerDecoderConfig()
        let encoder = SpeechTokenizerEncoder(config: config)
        // Verify expected structure
        XCTAssertEqual(encoder.encoderBlocks.count, 4, "Should have 4 downsample blocks")
        XCTAssertEqual(encoder.config.numQuantizers, 16, "Should have 16 codebooks")
    }

    func testEncoderBlockDimensions() {
        let config = SpeechTokenizerDecoderConfig()
        let encoder = SpeechTokenizerEncoder(config: config)
        // Input conv: 1 → 96 channels
        XCTAssertNotNil(encoder.inputConv)
        // Post conv: latentDim → hiddenSize
        XCTAssertNotNil(encoder.postConv)
    }

    // MARK: - ResidualVectorQuantizer encode

    func testRVQEncode() {
        let rvq = ResidualVectorQuantizer(
            numQuantizers: 2, codebookSize: 64,
            codebookDim: 8, outputDim: 16)
        let input = MLXArray.zeros([1, 5, 16])  // [B, T, outputDim]
        let codes = rvq.encode(input)
        eval(codes)
        XCTAssertEqual(codes.shape, [1, 2, 5], "Should produce [B, numQuantizers, T]")
    }

    func testSplitRVQEncode() {
        let config = SpeechTokenizerDecoderConfig()
        let splitRVQ = SplitResidualVectorQuantizer(config: config)
        let input = MLXArray.zeros([1, 3, config.hiddenSize])  // [B, T, 512]
        let codes = splitRVQ.encode(input)
        eval(codes)
        XCTAssertEqual(codes.shape, [1, 16, 3], "Should produce [B, 16, T]")
    }

    // MARK: - Config

    func testICLMethodExists() {
        // Verify the ICL method signature compiles
        // (actual E2E test requires model weights)
        let method = Qwen3TTSModel.fromPretrainedWithEncoder
        XCTAssertNotNil(method)
    }
}

// MARK: - E2E Tests

final class E2EICLVoiceCloningTests: XCTestCase {

    func testE2EEncoderWeightLoading() async throws {
        // Load model with encoder
        let (tts, encoder) = try await Qwen3TTSModel.fromPretrainedWithEncoder()

        // Verify encoder loaded
        XCTAssertEqual(encoder.encoderBlocks.count, 4)

        // Quick forward pass with silence
        let silence = [Float](repeating: 0, count: 24000)  // 1s at 24kHz
        let codes = encoder.encode(samples: silence)
        eval(codes)
        XCTAssertEqual(codes.dim(0), 1, "Batch size should be 1")
        XCTAssertEqual(codes.dim(1), 16, "Should have 16 codebooks")
        XCTAssertGreaterThan(codes.dim(2), 0, "Should produce at least 1 frame")
        print("Encoder: 1s silence → \(codes.dim(2)) codec frames")
    }

    func testE2EICLSynthesis() async throws {
        let (tts, encoder) = try await Qwen3TTSModel.fromPretrainedWithEncoder()

        // Load test audio as reference
        let refAudio = try AudioFileLoader.load(
            url: URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav"),
            targetSampleRate: 24000)
        XCTAssertGreaterThan(refAudio.count, 0)

        // Extract speech region (5.17s - 8.37s)
        let startSample = Int(5.17 * 24000)
        let endSample = min(Int(8.37 * 24000), refAudio.count)
        let refSpeech = Array(refAudio[startSample..<endSample])

        let waveform = tts.synthesizeWithVoiceCloneICL(
            text: "Hello, this is a test.",
            referenceAudio: refSpeech,
            referenceSampleRate: 24000,
            referenceText: "Can you guarantee that the replacement part will be shipped tomorrow?",
            language: "english",
            sampling: {
                var s = SamplingConfig()
                s.maxTokens = 50
                return s
            }(),
            codecEncoder: encoder)

        XCTAssertGreaterThan(waveform.count, 0, "Should produce audio output")
        let duration = Double(waveform.count) / 24000.0
        print("ICL synthesis: \(waveform.count) samples (\(String(format: "%.2f", duration))s)")
    }
}
