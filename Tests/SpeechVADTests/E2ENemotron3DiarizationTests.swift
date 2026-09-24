import CoreML
import Foundation
import XCTest
import AudioCommon
@testable import SpeechVAD

final class E2ENemotron3DiarizationTests: XCTestCase {
    func testIncrementalCoreMLMatchesShortCadenceReplay() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["NEMOTRON3_DIARIZATION_COREML_DIR"],
              let audioPath = environment["NEMOTRON3_DIARIZATION_E2E_AUDIO"] else {
            throw XCTSkip("Set Core ML model and E2E audio paths")
        }
        let model = try Nemotron3Diarizer.fromCoreMLDirectory(
            URL(fileURLWithPath: modelPath, isDirectory: true),
            computeUnits: .all)
        let audio = try AudioFileLoader.load(
            url: URL(fileURLWithPath: audioPath), targetSampleRate: 16_000)
        let reference = try model.diarize(
            audio: audio, sampleRate: 16_000,
            coreEncoderFrames: 6, rightContextEncoderFrames: 7)
        let session = try model.makeStreamingSession()
        var cursor = 0
        while cursor < audio.count {
            let end = min(audio.count, cursor + 8_000)
            _ = try session.push(audio: Array(audio[cursor..<end]))
            cursor = end
        }
        let streamed = try session.finish()
        XCTAssertEqual(streamed.numSpeakers, reference.numSpeakers)
        XCTAssertEqual(streamed.segments.count, reference.segments.count)
        for (actual, expected) in zip(streamed.segments, reference.segments) {
            XCTAssertEqual(actual.speakerId, expected.speakerId)
            XCTAssertEqual(actual.startTime, expected.startTime, accuracy: 0.011)
            XCTAssertEqual(actual.endTime, expected.endTime, accuracy: 0.011)
        }
    }

    func testLocalCoreMLAndMLXArtifactsAgree() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let coreMLPath = environment["NEMOTRON3_DIARIZATION_COREML_DIR"],
              let mlxPath = environment["NEMOTRON3_DIARIZATION_MLX_DIR"] else {
            throw XCTSkip("Set both local Nemotron 3 artifact directories")
        }

        var chunk = [Float](repeating: 0, count: 3_040 * 128)
        for frame in 0..<3_040 {
            for mel in 0..<128 {
                chunk[frame * 128 + mel] =
                    sin(Float(frame) * 0.013 + Float(mel) * 0.031) * 0.25 - 7
            }
        }

        var coreML: Nemotron3CoreMLBackend? = try Nemotron3CoreMLBackend(
            directory: URL(fileURLWithPath: coreMLPath, isDirectory: true),
            computeUnits: .cpuOnly)
        let coreEmbeddings = try XCTUnwrap(coreML).preencode(chunk: chunk)
        var packed = [Float](repeating: 0, count: 684 * 512)
        packed.replaceSubrange(0..<coreEmbeddings.count, with: coreEmbeddings)
        let coreOutput = try XCTUnwrap(coreML).predictHead(
            packedEmbeddings: packed, validLength: 380)

        // Full audio → mel → streaming cache → 10 ms binarization smoke test.
        var diarizer: Nemotron3Diarizer? = Nemotron3Diarizer(
            backend: try XCTUnwrap(coreML))
        var audio = [Float](repeating: 0, count: 2 * 16_000)
        for index in audio.indices {
            let frequency: Float = index < 16_000 ? 190 : 330
            audio[index] = 0.2 * sin(2 * .pi * frequency * Float(index) / 16_000)
        }
        let result = try XCTUnwrap(diarizer).diarize(
            audio: audio, sampleRate: 16_000)
        XCTAssertLessThanOrEqual(result.numSpeakers, 8)
        for segment in result.segments {
            XCTAssertGreaterThanOrEqual(segment.startTime, 0)
            XCTAssertGreaterThan(segment.endTime, segment.startTime)
            XCTAssertLessThanOrEqual(segment.endTime, 2.001)
            XCTAssertTrue((0..<8).contains(segment.speakerId))
        }
        diarizer = nil
        coreML = nil

        let mlx = try Nemotron3MLXBackend(
            directory: URL(fileURLWithPath: mlxPath, isDirectory: true))
        let mlxEmbeddings = try mlx.preencode(chunk: chunk)
        var mlxPacked = [Float](repeating: 0, count: 684 * 512)
        mlxPacked.replaceSubrange(0..<mlxEmbeddings.count, with: mlxEmbeddings)
        let mlxOutput = try mlx.predictHead(
            packedEmbeddings: mlxPacked, validLength: 380)

        XCTAssertEqual(coreEmbeddings.count, 380 * 512)
        XCTAssertEqual(mlxEmbeddings.count, 380 * 512)
        XCTAssertGreaterThan(cosine(coreEmbeddings, mlxEmbeddings), 0.99)
        XCTAssertEqual(coreOutput.probabilities10ms.count, 684 * 8 * 8)
        XCTAssertEqual(mlxOutput.probabilities10ms.count, 684 * 8 * 8)
        XCTAssertLessThan(
            meanAbsoluteError(
                coreOutput.probabilities10ms,
                mlxOutput.probabilities10ms),
            0.03)
    }

    private func cosine(_ left: [Float], _ right: [Float]) -> Float {
        XCTAssertEqual(left.count, right.count)
        var dot: Double = 0
        var leftNorm: Double = 0
        var rightNorm: Double = 0
        for index in left.indices {
            let l = Double(left[index])
            let r = Double(right[index])
            dot += l * r
            leftNorm += l * l
            rightNorm += r * r
        }
        return Float(dot / max(sqrt(leftNorm * rightNorm), 1.0e-12))
    }

    private func meanAbsoluteError(_ left: [Float], _ right: [Float]) -> Float {
        XCTAssertEqual(left.count, right.count)
        var total: Double = 0
        for index in left.indices {
            total += abs(Double(left[index] - right[index]))
        }
        return Float(total / Double(max(left.count, 1)))
    }
}
