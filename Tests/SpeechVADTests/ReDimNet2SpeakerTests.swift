import Foundation
import XCTest
@testable import SpeechVAD
import AudioCommon

#if canImport(CoreML)
final class ReDimNet2SpeakerTests: XCTestCase {
    func testPublishedConfiguration() {
        XCTAssertEqual(
            ReDimNet2SpeakerModel.defaultModelId,
            "aufklarer/ReDimNet2-B6-CoreML")
        XCTAssertEqual(ReDimNet2SpeakerModel.inputSampleRate, 16_000)
        XCTAssertEqual(ReDimNet2SpeakerModel.inputSampleCount, 96_000)
        XCTAssertEqual(ReDimNet2SpeakerModel.minimumSampleCount, 32_000)
        XCTAssertEqual(
            ReDimNet2SpeakerModel.minimumShortUtteranceSampleCount,
            9_600)
        XCTAssertEqual(ReDimNet2SpeakerModel.embeddingDimension, 192)
        XCTAssertEqual(ReDimNet2SpeakerModel.defaultArtifactRevision, "frontend-fp32-v1")
    }

    func testDecodesPublishedModelConfiguration() throws {
        let data = Data("""
        {
          "model_type": "redimnet2-b6-speaker-coreml",
          "sample_rate": 16000,
          "input_samples": 96000,
          "embedding_dimension": 192,
          "input_name": "audio",
          "output_name": "embedding",
          "compiled_model": "ReDimNet2B6.mlmodelc"
        }
        """.utf8)

        let configuration = try ReDimNet2SpeakerModel.decodeConfiguration(data)
        XCTAssertEqual(configuration.inputSamples, 96_000)
        XCTAssertEqual(configuration.embeddingDimension, 192)
    }

    func testRejectsIncompatibleModelConfiguration() {
        let data = Data("""
        {
          "model_type": "redimnet2-b6-speaker-coreml",
          "sample_rate": 16000,
          "input_samples": 160000,
          "embedding_dimension": 192,
          "input_name": "audio",
          "output_name": "embedding",
          "compiled_model": "ReDimNet2B6.mlmodelc"
        }
        """.utf8)

        XCTAssertThrowsError(try ReDimNet2SpeakerModel.decodeConfiguration(data))
    }

    func testPreparedAudioRepeatsShortCleanSpeech() throws {
        let samples = (0..<32_000).map(Float.init)
        let prepared = try ReDimNet2SpeakerModel.preparedAudio(samples)

        XCTAssertEqual(prepared.count, 96_000)
        XCTAssertEqual(Array(prepared[0..<32_000]), samples)
        XCTAssertEqual(Array(prepared[32_000..<64_000]), samples)
        XCTAssertEqual(Array(prepared[64_000..<96_000]), samples)
    }

    func testPreparedAudioCenterCropsLongSpeech() throws {
        let samples = (0..<128_000).map(Float.init)
        let prepared = try ReDimNet2SpeakerModel.preparedAudio(samples)

        XCTAssertEqual(prepared.count, 96_000)
        XCTAssertEqual(prepared.first, 16_000)
        XCTAssertEqual(prepared.last, 111_999)
    }

    func testPreparedAudioKeepsExactWindow() throws {
        let samples = [Float](repeating: 0.25, count: 96_000)
        XCTAssertEqual(try ReDimNet2SpeakerModel.preparedAudio(samples), samples)
    }

    func testPreparedAudioRejectsLessThanTwoSeconds() {
        XCTAssertThrowsError(
            try ReDimNet2SpeakerModel.preparedAudio(
                [Float](repeating: 0, count: 31_999))) { error in
            XCTAssertTrue(error.localizedDescription.contains("at least 2.0 seconds"))
        }
    }

    func testPreparedShortUtteranceRepeatsSixTenthsOfASecond() throws {
        let samples = (0..<9_600).map(Float.init)
        let prepared = try ReDimNet2SpeakerModel.preparedShortUtteranceAudio(samples)

        XCTAssertEqual(prepared.count, 96_000)
        for repetition in 0..<10 {
            let start = repetition * samples.count
            XCTAssertEqual(Array(prepared[start..<(start + samples.count)]), samples)
        }
    }

    func testPreparedShortUtteranceRejectsLessThanSixTenthsOfASecond() {
        XCTAssertThrowsError(
            try ReDimNet2SpeakerModel.preparedShortUtteranceAudio(
                [Float](repeating: 0, count: 9_599))) { error in
            XCTAssertTrue(error.localizedDescription.contains("at least 0.6 seconds"))
        }
    }

    func testPreparedAudioRejectsNonFiniteSamples() {
        var samples = [Float](repeating: 0, count: 32_000)
        samples[100] = .nan
        XCTAssertThrowsError(try ReDimNet2SpeakerModel.preparedAudio(samples))
    }

    func testCosineSimilarity() {
        XCTAssertEqual(
            ReDimNet2SpeakerModel.cosineSimilarity([1, 0], [1, 0]),
            1,
            accuracy: 1e-6)
        XCTAssertEqual(
            ReDimNet2SpeakerModel.cosineSimilarity([1, 0], [0, 1]),
            0,
            accuracy: 1e-6)
        XCTAssertEqual(
            ReDimNet2SpeakerModel.cosineSimilarity([1], [1, 0]),
            0)
    }

    func testDefaultCacheDoesNotTreatTheLegacyExportAsCurrent() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let legacy = root.appendingPathComponent("config.json")
        let original = Data("legacy export".utf8)
        try original.write(to: legacy)
        let current = ReDimNet2SpeakerModel.modelCacheDirectory(in: root)
        XCTAssertNotEqual(current, root)
        XCTAssertTrue(current.path.hasSuffix("revisions/frontend-fp32-v1"))
        XCTAssertFalse(ReDimNet2SpeakerModel.isCached(at: root))
        XCTAssertEqual(try Data(contentsOf: legacy), original)
        XCTAssertEqual(
            ReDimNet2SpeakerModel.modelCacheDirectory(in: root, modelId: "example/custom-identity"),
            root)
    }

    func testDefaultArtifactRejectsLegacyMetadataAndCorruptedCompiledFiles() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let directory = ReDimNet2SpeakerModel.modelCacheDirectory(in: root)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        var metadata: [String: Any] = [
            "model_type": "redimnet2-b6-speaker-coreml", "sample_rate": 16000,
            "input_samples": 96000, "embedding_dimension": 192,
            "input_name": "audio", "output_name": "embedding",
            "compiled_model": "ReDimNet2B6.mlmodelc",
        ]
        let legacy = try ReDimNet2SpeakerModel.decodeConfiguration(
            JSONSerialization.data(withJSONObject: metadata))
        XCTAssertThrowsError(try ReDimNet2SpeakerModel.validateDefaultArtifact(legacy, at: directory))
        metadata["artifact_revision"] = ReDimNet2SpeakerModel.defaultArtifactRevision
        metadata["compute_precision"] = ReDimNet2SpeakerModel.defaultComputePrecision
        metadata["compiled_files_sha256"] = ReDimNet2SpeakerModel.defaultArtifactChecksums
        let encoded = try JSONSerialization.data(withJSONObject: metadata)
        try encoded.write(to: directory.appendingPathComponent("config.json"))
        for name in ReDimNet2SpeakerModel.defaultArtifactChecksums.keys {
            let file = directory.appendingPathComponent("ReDimNet2B6.mlmodelc").appendingPathComponent(name)
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data("corrupt".utf8).write(to: file)
        }
        // Presence routes the load; bytes must still pass validation before inference.
        XCTAssertTrue(ReDimNet2SpeakerModel.isCached(at: root))
        let configuration = try ReDimNet2SpeakerModel.decodeConfiguration(encoded)
        XCTAssertThrowsError(try ReDimNet2SpeakerModel.validateDefaultArtifact(configuration, at: directory))
        metadata["artifact_revision"] = "retired"
        try JSONSerialization.data(withJSONObject: metadata).write(
            to: directory.appendingPathComponent("config.json"))
        XCTAssertFalse(ReDimNet2SpeakerModel.isCached(at: root))
    }
}

final class E2EReDimNet2SpeakerTests: XCTestCase {
    private func loadModel() async throws -> ReDimNet2SpeakerModel {
        if let directory = ProcessInfo.processInfo.environment[
            "REDIMNET2_COREML_MODEL_DIR"]
        {
            let source = URL(fileURLWithPath: directory, isDirectory: true)
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            let target = ReDimNet2SpeakerModel.modelCacheDirectory(in: root)
            try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
            addTeardownBlock { try? FileManager.default.removeItem(at: root) }
            for name in ["config.json", "ReDimNet2B6.mlmodelc"] {
                try FileManager.default.copyItem(at: source.appendingPathComponent(name),
                                                to: target.appendingPathComponent(name))
            }
            return try await ReDimNet2SpeakerModel.fromPretrained(
                cacheDir: root,
                offlineMode: true)
        }
        return try await ReDimNet2SpeakerModel.fromPretrained()
    }

    func testE2EEmbeddingIsNormalizedAndDeterministic() async throws {
        let model = try await loadModel()
        let audioURL = URL(
            fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav")
        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: audioURL)

        try model.prewarm()
        let first = try model.embed(audio: samples, sampleRate: sampleRate)
        let second = try model.embed(audio: samples, sampleRate: sampleRate)

        XCTAssertEqual(first.count, 192)
        let norm = sqrt(first.reduce(Float(0)) { $0 + $1 * $1 })
        XCTAssertEqual(norm, 1, accuracy: 0.002)
        XCTAssertEqual(
            ReDimNet2SpeakerModel.cosineSimilarity(first, second),
            1,
            accuracy: 0.0001)
    }

    func testE2EShortUtteranceEmbeddingIsNormalizedAndDeterministic() async throws {
        let model = try await loadModel()
        let audioURL = URL(
            fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav")
        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: audioURL)
        let shortSamples = Array(samples.prefix(Int(Double(sampleRate) * 0.75)))

        let first = try model.embedShortUtterance(
            audio: shortSamples, sampleRate: sampleRate)
        let second = try model.embedShortUtterance(
            audio: shortSamples, sampleRate: sampleRate)

        XCTAssertEqual(first.count, 192)
        XCTAssertEqual(
            sqrt(first.reduce(Float(0)) { $0 + $1 * $1 }),
            1,
            accuracy: 0.002)
        XCTAssertEqual(
            ReDimNet2SpeakerModel.cosineSimilarity(first, second),
            1,
            accuracy: 0.0001)
    }

    func testE2EDeterministicWaveformProducesValidEmbedding() async throws {
        let model = try await loadModel()
        var waveform = [Float](repeating: 0, count: 96_000)
        for index in waveform.indices {
            let time = Float(index) / 16_000
            waveform[index] = 0.12 * sin(2 * .pi * 173 * time)
                + 0.06 * sin(2 * .pi * 271 * time)
        }

        let embedding = try model.embed(audio: waveform, sampleRate: 16_000)
        XCTAssertEqual(embedding.count, 192)
        XCTAssertTrue(embedding.allSatisfy(\.isFinite))
        XCTAssertEqual(
            sqrt(embedding.reduce(Float(0)) { $0 + $1 * $1 }),
            1,
            accuracy: 0.002)
    }

    func testE2ESparseAndQuietWaveformsRemainFinite() async throws {
        let model = try await loadModel()
        var sparse = [Float](repeating: 0, count: 96_000)
        for index in 40_000..<40_128 {
            sparse[index] = 0.12 * sin(2 * .pi * 173 * Float(index - 40_000) / 16_000)
        }
        for audio in [sparse, sparse.map { $0 * 0.0001 }] {
            let embedding = try model.embed(audio: audio, sampleRate: 16_000)
            XCTAssertEqual(embedding.count, 192)
            XCTAssertTrue(embedding.allSatisfy(\.isFinite))
            XCTAssertEqual(sqrt(embedding.reduce(Float(0)) { $0 + $1 * $1 }), 1, accuracy: 0.002)
        }
    }

    func testE2ELocalNumericalRegressionAudioRemainsFinite() async throws {
        guard let path = ProcessInfo.processInfo.environment["REDIMNET2_REGRESSION_AUDIO_F32"] else {
            throw XCTSkip("Set a local numerical-regression PCM path; recordings remain external")
        }
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        XCTAssertEqual(data.count % 4, 0)
        var audio = [Float](repeating: 0, count: data.count / 4)
        _ = audio.withUnsafeMutableBytes { data.copyBytes(to: $0) }
        let model = try await loadModel()
        let embedding = try model.embed(audio: audio, sampleRate: 16_000)
        XCTAssertEqual(embedding.count, 192)
        XCTAssertTrue(embedding.allSatisfy(\.isFinite))
    }
}
#endif
