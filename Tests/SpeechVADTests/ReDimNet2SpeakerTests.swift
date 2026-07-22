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
}

final class E2EReDimNet2SpeakerTests: XCTestCase {
    private func loadModel() async throws -> ReDimNet2SpeakerModel {
        if let directory = ProcessInfo.processInfo.environment[
            "REDIMNET2_COREML_MODEL_DIR"]
        {
            return try await ReDimNet2SpeakerModel.fromPretrained(
                cacheDir: URL(fileURLWithPath: directory, isDirectory: true),
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
}
#endif
