import MLX
import XCTest

@testable import VoiceChat

final class VoiceChatSpeechTests: XCTestCase {
    func testSpeechConfigurationDecodesExportContract() throws {
        let json = #"""
        {
          "model_type": "nemotron_voicechat_tts",
          "sample_rate": 22050,
          "frame_samples": 1764,
          "frame_seconds": 0.08,
          "speaker": "Aria",
          "prompt_frames": 37,
          "hidden_size": 1152,
          "num_hidden_layers": 28,
          "num_attention_heads": 16,
          "head_dim": 72,
          "latent_size": 512,
          "num_quantizers": 31,
          "codebook_size": 1024,
          "num_iter": 8,
          "guidance_scale": 0.2,
          "top_p": 0.95,
          "noise_scale": 0.001,
          "codec_dense_fp16": true,
          "weight_layout": "nemo",
          "quantization": {"group_size": 64, "bits": 5},
          "quantization_per_tensor": {
            "tts_model.tts_model.mog_head.proj_mus.weight": {
              "group_size": 64, "bits": 8
            }
          }
        }
        """#
        let config = try JSONDecoder().decode(
            VoiceChatSpeechConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.sampleRate, 22_050)
        XCTAssertEqual(config.frameSamples, 1_764)
        XCTAssertEqual(config.promptFrames, 37)
        XCTAssertEqual(config.numIter, 8)
        XCTAssertTrue(config.codecDenseFP16)
        XCTAssertEqual(config.quantization?.bits, 5)
        XCTAssertEqual(
            config.quantizationPerTensor?["tts_model.tts_model.mog_head.proj_mus.weight"]?.bits,
            8)
    }

    func testCodecFrameGeometryIsExactlyEightyMilliseconds() {
        XCTAssertEqual(
            Double(VoiceChatCodec.samplesPerFrame) / Double(VoiceChatCodec.sampleRate),
            0.08, accuracy: 1e-12)
    }

    func testPublishedMaskGITScheduleUsesEightProgressiveIterations() {
        XCTAssertEqual(
            VoiceChatSpeechDecoder.maskGITAssignmentCounts(),
            [0, 0, 0, 1, 1, 3, 4, 22])
        XCTAssertEqual(
            VoiceChatSpeechDecoder.maskGITAssignmentCounts().reduce(0, +),
            31)
    }

    func testDefaultPromptDoesNotBiasTurnTimingWithGreetingInstruction() {
        XCTAssertFalse(
            VoiceChatSession.defaultSystemPrompt.localizedCaseInsensitiveContains("greet"))
        XCTAssertTrue(
            VoiceChatSession.greetingSystemPrompt.localizedCaseInsensitiveContains("greet"))
    }

    func testSessionRejectsInvalidSamplingParameters() {
        XCTAssertNoThrow(try VoiceChatSession.validate(
            sampling: .init(), speech: .init()))
        XCTAssertThrowsError(try VoiceChatSession.validate(
            sampling: .init(temperature: .nan), speech: .init()))
        XCTAssertThrowsError(try VoiceChatSession.validate(
            sampling: .init(topP: 0), speech: .init()))
        XCTAssertThrowsError(try VoiceChatSession.validate(
            sampling: .init(repetitionPenalty: 0), speech: .init()))
        XCTAssertThrowsError(try VoiceChatSession.validate(
            sampling: .init(), speech: .init(topP: 1.1)))
        XCTAssertThrowsError(try VoiceChatSession.validate(
            sampling: .init(), speech: .init(noise: -.infinity)))
    }

    func testSilenceDurationValidationCannotTrapOnNonFiniteInput() {
        XCTAssertEqual(
            try VoiceChatSession.silenceSampleCount(seconds: 0.08),
            VoiceChatSession.inputSamplesPerFrame)
        for invalid in [-1.0, Double.infinity, Double.nan] {
            XCTAssertThrowsError(
                try VoiceChatSession.silenceSampleCount(seconds: invalid))
        }
        XCTAssertThrowsError(try VoiceChatSession.silenceSampleCount(
            seconds: VoiceChatSession.maximumSilenceSeconds + 0.01))
    }

    func testPeriodicHannMatchesTorchConvention() {
        let window = VoiceChatCodec.periodicHann(16).asArray(Float.self)
        XCTAssertEqual(window.count, 16)
        XCTAssertEqual(window[0], 0, accuracy: 1e-7)
        XCTAssertEqual(window[8], 1, accuracy: 1e-7)
        XCTAssertGreaterThan(window[15], 0, "periodic Hann must not duplicate the zero endpoint")
    }

    func testInverseSTFTProducesOneFrameOfFiniteNearSilence() {
        // Very negative magnitude logits squash essentially to zero. This tests
        // the magnitude/phase contract and exact 1764-independent 4x geometry
        // without requiring model weights.
        let magnitude = MLXArray.zeros([1, 1, 9]) - 100
        let phase = MLXArray.zeros([1, 1, 9])
        let audio = VoiceChatCodec.inverseSTFT(
            MLX.concatenated([magnitude, phase], axis: -1))
        eval(audio)

        XCTAssertEqual(audio.shape, [1, 4])
        XCTAssertTrue(MLX.all(MLX.isFinite(audio)).item(Bool.self))
        XCTAssertLessThan(MLX.max(MLX.abs(audio)).item(Float.self), 1e-6)
    }
}
