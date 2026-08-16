import MLX
import XCTest

@testable import VoiceChat

final class VoiceChatSpeechTests: XCTestCase {
    func testMatrixOutputRowsSelectOnlyRequestedVocabularyEntries() throws {
        let matrix = try VoiceChatMatrix(
            weights: [
                "head.weight": MLXArray([
                    Float(1), 2,
                    3, 4,
                    5, 6,
                ]).reshaped([3, 2]),
            ],
            name: "head.weight",
            quantization: nil)

        let selected = matrix.outputRows(MLXArray([2, 0]))
        eval(selected)

        XCTAssertEqual(selected.shape, [2, 2])
        XCTAssertEqual(selected.asArray(Float.self), [5, 6, 1, 2])
    }

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

    func testBoundedSpeechCachePreservesPromptAndRecentContext() {
        let cache = VoiceChatSpeechAttentionCache(
            retainedPrefixFrames: 2,
            recentContextFrames: 3)

        func frames(_ values: [Float]) -> MLXArray {
            MLXArray(values).reshaped([1, 1, values.count, 1])
        }

        _ = cache.update(keys: frames([0, 1]), values: frames([10, 11]))
        for value in 2 ... 7 {
            _ = cache.update(
                keys: frames([Float(value)]),
                values: frames([Float(value + 10)]))
        }
        eval(cache.keys!, cache.values!)

        XCTAssertEqual(cache.offset, 8, "RoPE offset must remain absolute")
        XCTAssertEqual(cache.keys!.shape, [1, 1, 5, 1])
        XCTAssertEqual(
            cache.keys!.reshaped([-1]).asArray(Float.self),
            [0, 1, 5, 6, 7])
        XCTAssertEqual(
            cache.values!.reshaped([-1]).asArray(Float.self),
            [10, 11, 15, 16, 17])
    }

    func testPublishedMaskGITScheduleUsesEightProgressiveIterations() {
        XCTAssertEqual(
            VoiceChatSpeechDecoder.maskGITAssignmentCounts(),
            [0, 0, 0, 1, 1, 3, 4, 22])
        XCTAssertEqual(
            VoiceChatSpeechDecoder.maskGITAssignmentCounts().reduce(0, +),
            31)
    }

    func testIncrementalRVQLatentMatchesSelectedCodeReconstruction() {
        let codebooks = MLXArray([
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 0.5, 0.5, 1.0, -1.0,
            0.0, 0.0, -0.5, 1.0, 1.0, 0.5,
            0.0, 0.0, -1.0, -0.5, 0.25, 1.0,
        ] as [Float]).reshaped([4, 3, 2])
        let latent = MLXArray([0.8, 0.2] as [Float]).reshaped([1, 1, 2])
        var codes = MLXArray.full(
            [1, 1, 4], values: MLXArray(Int32(3)), dtype: .int32)
        var incremental = MLXArray.zeros([1, 1, 2], dtype: .float32)
        var filled = 0

        for count in [1, 2] {
            let assignment = VoiceChatSpeechDecoder.assignRVQCodes(
                residualCodebooks: codebooks,
                latent: latent,
                to: codes,
                startingAt: filled,
                count: count,
                retainEmbeddings: true)
            codes = assignment.codes
            for selected in assignment.embeddings {
                incremental = incremental + selected
            }
            filled += count

            var reconstructed = MLXArray.zeros([1, 1, 2], dtype: .float32)
            for index in 0 ..< filled {
                reconstructed = reconstructed
                    + codebooks[index][codes[0..., 0..., index]]
            }
            eval(incremental, reconstructed)
            XCTAssertLessThan(
                MLX.max(MLX.abs(incremental - reconstructed)).item(Float.self),
                1e-7)
        }

        let finalAssignment = VoiceChatSpeechDecoder.assignRVQCodes(
            residualCodebooks: codebooks,
            latent: latent,
            to: codes,
            startingAt: filled,
            count: 1,
            retainEmbeddings: false)
        eval(finalAssignment.codes)
        XCTAssertTrue(finalAssignment.embeddings.isEmpty)
        XCTAssertLessThan(MLX.max(finalAssignment.codes).item(Int.self), 3)
    }

    func testDefaultPromptDoesNotBiasTurnTimingWithGreetingInstruction() {
        XCTAssertTrue(
            VoiceChatSession.defaultSystemPrompt.contains("Your name is Soniqo"))
        XCTAssertFalse(
            VoiceChatSession.defaultSystemPrompt.contains("NVIDIA Voice Chat"))
        XCTAssertFalse(
            VoiceChatSession.defaultSystemPrompt.localizedCaseInsensitiveContains("greet"))
        XCTAssertTrue(
            VoiceChatSession.greetingSystemPrompt.localizedCaseInsensitiveContains("greet"))
    }

    func testDefaultPromptDoesNotClaimUnavailableExternalActions() {
        let prompt = VoiceChatSession.defaultSystemPrompt

        XCTAssertTrue(prompt.contains("cannot access apps"))
        XCTAssertTrue(prompt.contains("calendars, reminders"))
        XCTAssertTrue(prompt.contains("Never claim to schedule"))
        XCTAssertTrue(prompt.contains("do not ask for confirmation"))
        XCTAssertTrue(VoiceChatSession.greetingSystemPrompt.hasPrefix(prompt))
    }

    func testSystemPromptPredictionsDoNotBecomeChannelFeedback() {
        let prompt = VoiceChatSession.channelFeedbackAfterStep(
            record: false,
            textToken: 101,
            functionToken: 202,
            padID: 0)
        XCTAssertEqual(prompt.text, 0)
        XCTAssertEqual(prompt.function, 0)

        let generated = VoiceChatSession.channelFeedbackAfterStep(
            record: true,
            textToken: 101,
            functionToken: 202,
            padID: 0)
        XCTAssertEqual(generated.text, 101)
        XCTAssertEqual(generated.function, 202)
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
        XCTAssertThrowsError(try VoiceChatSession.validate(
            sampling: .init(), speech: .init(recentContextFrames: 0)))
        XCTAssertThrowsError(try VoiceChatSession.validate(
            sampling: .init(),
            speech: .init(),
            turnTaking: .init(endOfUtteranceFrames: 0)))
        XCTAssertNoThrow(try VoiceChatSession.validate(
            sampling: .init(),
            speech: .init(),
            turnTaking: .modelNative))
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
