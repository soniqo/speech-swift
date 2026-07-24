import Foundation
import XCTest
@testable import AudioCommon
@testable import NemotronStreamingASR

private func nemotronMLXConfiguration(bits: Int = 5) throws
    -> NemotronMLXConfiguration
{
    let json = """
    {
      "model_type": "nemotron_streaming_rnnt_mlx",
      "sample_rate": 16000,
      "vocab_size": 13087,
      "preprocessor": {
        "sample_rate": 16000,
        "features": 128,
        "n_fft": 512,
        "window_size": 0.025,
        "window_stride": 0.01,
        "preemph": 0.97
      },
      "encoder": {
        "feat_in": 128,
        "n_layers": 24,
        "d_model": 1024,
        "n_heads": 8,
        "ff_expansion_factor": 4,
        "subsampling_factor": 8,
        "conv_kernel_size": 9,
        "subsampling_conv_channels": 256,
        "use_bias": false,
        "conv_norm_type": "layer_norm"
      },
      "decoder": {
        "blank_as_pad": true,
        "vocab_size": 13087,
        "prednet": {
          "pred_hidden": 640,
          "pred_rnn_layers": 2
        }
      },
      "joint": {
        "num_classes": 13087,
        "jointnet": {
          "joint_hidden": 640,
          "activation": "relu",
          "encoder_hidden": 1024,
          "pred_hidden": 640
        }
      },
      "prompt_kernel": {
        "num_prompts": 128,
        "hidden": 2048,
        "d_model": 1024
      },
      "streaming": {
        "chunk_ms": 320,
        "mel_frames": 32,
        "pre_cache_size": 9,
        "output_frames": 4,
        "attention_left_context": 56,
        "conv_cache_size": 8
      },
      "quantization": {
        "mode": "affine",
        "bits": \(bits),
        "group_size": 64
      }
    }
    """
    return try JSONDecoder().decode(
        NemotronMLXConfiguration.self,
        from: Data(json.utf8)
    )
}

final class NemotronMLXTests: XCTestCase {
    func testDefaultPretrainedLoaderIsUnambiguous() {
        let loader = {
            try await NemotronStreamingASRMLXModel.fromPretrained(
                offlineMode: true
            )
        }
        _ = loader
    }

    func testPublishedVariantIdentifiersAndPrecisions() {
        XCTAssertEqual(
            NemotronMLXVariant.int5.modelId,
            "aufklarer/Nemotron-3.5-ASR-Streaming-0.6B-MLX-5bit"
        )
        XCTAssertEqual(NemotronMLXVariant.int5.quantizationBits, 5)
        XCTAssertEqual(
            NemotronMLXVariant.int8.modelId,
            "aufklarer/Nemotron-3.5-ASR-Streaming-0.6B-MLX-8bit"
        )
        XCTAssertEqual(NemotronMLXVariant.int8.quantizationBits, 8)
        XCTAssertEqual(
            NemotronStreamingASRMLXModel.defaultModelId,
            NemotronMLXVariant.int5.modelId
        )
    }

    func testPublishedInt5AndInt8ConfigurationsValidate() throws {
        try nemotronMLXConfiguration(bits: 5).validate()
        try nemotronMLXConfiguration(bits: 8).validate()
    }

    func testUnsupportedQuantizationFailsClosed() throws {
        let configuration = try nemotronMLXConfiguration(bits: 4)
        XCTAssertThrowsError(try configuration.validate()) { error in
            XCTAssertTrue(
                error.localizedDescription.contains(
                    "only affine group-64 INT5 and INT8"
                )
            )
        }
    }

    func testMLXOrderedVocabularyLoadsWithoutRenumbering() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("nemotron-mlx-vocab-\(UUID()).json")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("[\"▁zero\",\"one\",\"▁two\"]".utf8).write(to: url)

        let vocabulary = try NemotronVocabulary.load(from: url)

        XCTAssertEqual(vocabulary.count, 3)
        XCTAssertEqual(vocabulary.decode([0, 1, 2]), "zeroone two")
    }
}

final class E2ENemotronMLXTests: XCTestCase {
    func testLocalINT5BundleStreamsBundledSpeech() async throws {
        try await assertLocalBundle(
            environmentVariable: "NEMOTRON_MLX_LOCAL_BUNDLE",
            expectedBits: 5
        )
    }

    func testLocalINT8BundleStreamsBundledSpeech() async throws {
        try await assertLocalBundle(
            environmentVariable: "NEMOTRON_MLX_INT8_LOCAL_BUNDLE",
            expectedBits: 8
        )
    }

    private func assertLocalBundle(
        environmentVariable: String,
        expectedBits: Int
    ) async throws {
        guard
            let path = ProcessInfo.processInfo.environment[
                environmentVariable
            ],
            !path.isEmpty
        else {
            throw XCTSkip(
                "Set \(environmentVariable) to an exported INT\(expectedBits) bundle"
            )
        }

        let model = try await NemotronStreamingASRMLXModel.fromDirectory(
            URL(fileURLWithPath: path)
        )
        XCTAssertEqual(model.quantizationBits, expectedBits)
        let audioURL = try XCTUnwrap(
            Bundle.module.url(
                forResource: "test_audio",
                withExtension: "wav"
            )
        )
        let audio = try AudioFileLoader.load(
            url: audioURL,
            targetSampleRate: 16_000
        )

        var partials: [NemotronStreamingASRModel.PartialTranscript] = []
        for await partial in model.transcribeStream(
            audio: audio,
            sampleRate: 16_000,
            language: "en-US"
        ) {
            partials.append(partial)
        }

        XCTAssertGreaterThan(partials.count, 1)
        XCTAssertEqual(partials.last?.isFinal, true)
        let finalText = try XCTUnwrap(partials.last?.text)
        XCTAssertFalse(finalText.isEmpty)
        XCTAssertTrue(
            finalText.localizedCaseInsensitiveContains("replacement")
        )
        XCTAssertTrue(
            finalText.localizedCaseInsensitiveContains("tomorrow")
        )
        XCTAssertFalse(finalText.contains("<en-US>"))
        print("Nemotron MLX INT\(expectedBits) final: \(finalText)")
    }
}
