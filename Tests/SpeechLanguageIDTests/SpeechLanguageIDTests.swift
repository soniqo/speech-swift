import AudioCommon
import Foundation
import MLX
import XCTest
@testable import SpeechLanguageID

final class SpeechLanguageIDTests: XCTestCase {
    func testSpeechBrainFrontendMatchesPinnedPythonReference() throws {
        let sampleCount = 1_600
        let audio = (0..<sampleCount).map { index -> Float in
            let position = Float(index)
            return 0.1 * sin(2 * Float.pi * 440 * position / 16_000)
                + 0.03 * cos(2 * Float.pi * 997 * position / 16_000)
        }

        let features = try SpeechBrainFbank(melBinCount: 60).extract(audio)
        XCTAssertEqual(features.frameCount, 11)
        XCTAssertEqual(features.melBinCount, 60)
        XCTAssertEqual(features.values.count, 660)

        let firstFrameReference: [Float] = [
            -7.054196358, -5.820843697, -4.971723080, -5.359392643,
            -2.733277798, -3.522770643, -1.269868374, -0.520141542,
            2.369039297, 6.836628437, 13.425397873, 14.531822205,
        ]
        for (actual, expected) in zip(
            features.values.prefix(firstFrameReference.count),
            firstFrameReference
        ) {
            XCTAssertEqual(actual, expected, accuracy: 0.01)
        }
        XCTAssertEqual(features.values[1 * 60 + 17], -26.001571655, accuracy: 0.01)
        XCTAssertEqual(features.values[5 * 60 + 31], -50.083141327, accuracy: 0.01)
        XCTAssertEqual(features.values[10 * 60 + 59], -29.250514984, accuracy: 0.01)
        XCTAssertEqual(try XCTUnwrap(features.values.min()), -59.967133, accuracy: 0.01)
        XCTAssertEqual(try XCTUnwrap(features.values.max()), 20.032866, accuracy: 0.01)
    }

    func testFrontendRejectsEmptyAudio() {
        XCTAssertThrowsError(try SpeechBrainFbank(melBinCount: 60).extract([]))
    }

    func testSpeechBrainConvolutionUsesReflectionPadding() {
        let values = MLXArray([Float(1), 2, 3, 4, 5], [1, 5, 1])
        let actual = speechBrainReflectPad(values, padding: 2)
        eval(actual)
        XCTAssertEqual(actual.asArray(Float.self), [3, 2, 1, 2, 3, 4, 5, 4, 3])
    }

    func testConfigurationDecodesAndValidates() throws {
        let configuration = try JSONDecoder().decode(
            LanguageIDModelConfiguration.self,
            from: Data(Self.configurationJSON.utf8)
        )
        XCTAssertEqual(configuration.nMels, 60)
        XCTAssertEqual(configuration.classCount, 107)
        XCTAssertNoThrow(try configuration.validate(for: .mlx))
        XCTAssertThrowsError(try configuration.validate(for: .coreML))
    }

    func testConfigurationRejectsMissingSourceRevision() throws {
        let invalidJSON = Self.configurationJSON.replacingOccurrences(
            of: #""source_revision": "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9""#,
            with: #""source_revision": """#
        )
        let configuration = try JSONDecoder().decode(
            LanguageIDModelConfiguration.self,
            from: Data(invalidJSON.utf8)
        )
        XCTAssertThrowsError(try configuration.validate(for: .mlx))
    }

    func testConfigurationRejectsUnsafeFrameLimits() throws {
        let invalidJSON = Self.configurationJSON.replacingOccurrences(
            of: #""maximum_mel_frames": 3001"#,
            with: #""maximum_mel_frames": 9223372036854775807"#
        )
        let configuration = try JSONDecoder().decode(
            LanguageIDModelConfiguration.self,
            from: Data(invalidJSON.utf8)
        )
        XCTAssertThrowsError(try configuration.validate(for: .mlx))
    }

    func testLabelLoaderRequiresStableContiguousIndexes() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString,
            isDirectory: true
        )
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: directory) }

        let labels = [
            SpokenLanguageLabel(
                id: 0,
                code: "iw",
                name: "Hebrew",
                upstreamLabel: "iw: Hebrew"
            ),
            SpokenLanguageLabel(
                id: 1,
                code: "jw",
                name: "Javanese",
                upstreamLabel: "jw: Javanese"
            ),
        ]
        try JSONEncoder().encode(labels).write(
            to: directory.appendingPathComponent("labels.json")
        )
        XCTAssertEqual(
            try SpokenLanguageLabelLoader.load(from: directory, expectedCount: 2),
            labels
        )

        let reordered = [labels[1], labels[0]]
        try JSONEncoder().encode(reordered).write(
            to: directory.appendingPathComponent("labels.json"),
            options: .atomic
        )
        XCTAssertThrowsError(
            try SpokenLanguageLabelLoader.load(from: directory, expectedCount: 2)
        )
    }

    func testLabelLoaderRejectsEmptyPublicCode() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString,
            isDirectory: true
        )
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: directory) }

        let labels = [
            SpokenLanguageLabel(id: 0, code: "", name: "Hebrew", upstreamLabel: "iw: Hebrew"),
        ]
        try JSONEncoder().encode(labels).write(
            to: directory.appendingPathComponent("labels.json")
        )
        XCTAssertThrowsError(
            try SpokenLanguageLabelLoader.load(from: directory, expectedCount: 1)
        )
    }

    func testLocalLoaderRejectsArtifactPathTraversal() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString,
            isDirectory: true
        )
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: directory) }

        let invalidJSON = Self.configurationJSON.replacingOccurrences(
            of: #""artifact": "model.safetensors""#,
            with: #""artifact": "../model.safetensors""#
        )
        try Data(invalidJSON.utf8).write(
            to: directory.appendingPathComponent("config.json")
        )
        let labels = (0..<107).map { index in
            SpokenLanguageLabel(
                id: index,
                code: "language-\(index)",
                name: "Language \(index)",
                upstreamLabel: "language-\(index): Language \(index)"
            )
        }
        try JSONEncoder().encode(labels).write(
            to: directory.appendingPathComponent("labels.json")
        )

        XCTAssertThrowsError(
            try SpeechLanguageIdentifier.fromLocal(directory: directory, engine: .mlx)
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("safe top-level"))
        }
    }

    func testResultBestIsFirstRankedPrediction() {
        let prediction = LanguageIDPrediction(
            label: SpokenLanguageLabel(
                id: 94,
                code: "th",
                name: "Thai",
                upstreamLabel: "th: Thai"
            ),
            probability: 0.9,
            logProbability: log(0.9)
        )
        let result = LanguageIDResult(
            predictions: [prediction],
            analyzedDuration: 3,
            windowCount: 1
        )
        XCTAssertEqual(result.best, prediction)
    }

    private static let configurationJSON = #"""
    {
      "model_type": "speechbrain-ecapa-voxlingua107-language-id",
      "task": "audio-classification",
      "format": "mlx",
      "sample_rate": 16000,
      "n_fft": 400,
      "win_length": 400,
      "hop_length": 160,
      "n_mels": 60,
      "minimum_mel_frames": 10,
      "maximum_mel_frames": 3001,
      "embedding_dimension": 256,
      "class_count": 107,
      "output_name": "log_probabilities",
      "artifact": "model.safetensors",
      "source_model": "speechbrain/lang-id-voxlingua107-ecapa",
      "source_revision": "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9"
    }
    """#
}

final class E2ELanguageIDTests: XCTestCase {
    func testThaiUpstreamExampleMatchesOfficialClassifierIndex() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["SPEECH_LANGUAGE_ID_MODEL_DIR"],
              let audioPath = environment["SPEECH_LANGUAGE_ID_TEST_AUDIO"]
        else {
            throw XCTSkip(
                "Set SPEECH_LANGUAGE_ID_MODEL_DIR and SPEECH_LANGUAGE_ID_TEST_AUDIO"
            )
        }

        let model = try SpeechLanguageIdentifier.fromLocal(
            directory: URL(fileURLWithPath: modelPath),
            engine: .mlx
        )
        let audio = try AudioFileLoader.load(
            url: URL(fileURLWithPath: audioPath),
            targetSampleRate: 16_000
        )
        let result = try model.identify(audio: audio, sampleRate: 16_000, topK: 3)
        XCTAssertEqual(result.best?.label.id, 94)
        XCTAssertEqual(result.best?.label.code, "th")
        XCTAssertGreaterThan(result.best?.probability ?? 0, 0.95)
    }

    func testThaiUpstreamExampleMatchesOfficialClassifierIndexCoreML() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["SPEECH_LANGUAGE_ID_COREML_MODEL_DIR"],
              let audioPath = environment["SPEECH_LANGUAGE_ID_TEST_AUDIO"]
        else {
            throw XCTSkip(
                "Set SPEECH_LANGUAGE_ID_COREML_MODEL_DIR and SPEECH_LANGUAGE_ID_TEST_AUDIO"
            )
        }

        let model = try SpeechLanguageIdentifier.fromLocal(
            directory: URL(fileURLWithPath: modelPath),
            engine: .coreML
        )
        let audio = try AudioFileLoader.load(
            url: URL(fileURLWithPath: audioPath),
            targetSampleRate: 16_000
        )
        let result = try model.identify(audio: audio, sampleRate: 16_000, topK: 3)
        XCTAssertEqual(result.best?.label.id, 94)
        XCTAssertEqual(result.best?.label.code, "th")
        XCTAssertGreaterThan(result.best?.probability ?? 0, 0.95)
    }

    func testLongRecordingUsesMultipleDurationWeightedWindows() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["SPEECH_LANGUAGE_ID_MODEL_DIR"],
              let audioPath = environment["SPEECH_LANGUAGE_ID_TEST_AUDIO"]
        else {
            throw XCTSkip(
                "Set SPEECH_LANGUAGE_ID_MODEL_DIR and SPEECH_LANGUAGE_ID_TEST_AUDIO"
            )
        }

        let model = try SpeechLanguageIdentifier.fromLocal(
            directory: URL(fileURLWithPath: modelPath),
            engine: .mlx
        )
        let clip = try AudioFileLoader.load(
            url: URL(fileURLWithPath: audioPath),
            targetSampleRate: 16_000
        )
        let audio = clip + clip + clip
        let result = try model.identify(audio: audio, sampleRate: 16_000, topK: 1)

        XCTAssertEqual(result.windowCount, 2)
        XCTAssertEqual(result.analyzedDuration, Double(audio.count) / 16_000, accuracy: 0.001)
        XCTAssertEqual(result.best?.label.code, "th")
    }
}
