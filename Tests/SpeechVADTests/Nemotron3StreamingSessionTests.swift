import XCTest
@testable import SpeechVAD

final class Nemotron3StreamingSessionTests: XCTestCase {
    func testEmptyStreamFinishesWithoutInference() throws {
        let backend = FakeNemotron3Backend()
        let diarizer = Nemotron3Diarizer(backend: backend)
        let session = try diarizer.makeStreamingSession()

        XCTAssertTrue(try session.finish().segments.isEmpty)
        XCTAssertEqual(session.confirmedThroughSample, 0)
        XCTAssertEqual(backend.headCalls, 0)
        XCTAssertTrue(try session.finish().segments.isEmpty)
    }

    func testFirstConfirmationAndFinalTailAreBounded() throws {
        let backend = FakeNemotron3Backend()
        let diarizer = Nemotron3Diarizer(backend: backend)
        let session = try diarizer.makeStreamingSession()
        let first = Nemotron3StreamingSession.firstConfirmationSampleCount()
        XCTAssertEqual(first, 16_960)

        _ = try session.push(audio: [Float](repeating: 0, count: first - 1))
        XCTAssertEqual(session.confirmedThroughSample, 0)
        XCTAssertEqual(backend.headCalls, 0)

        _ = try session.push(audio: [0])
        XCTAssertEqual(session.confirmedThroughSample, 7_680)
        XCTAssertEqual(backend.headCalls, 1)

        let final = try session.finish()
        XCTAssertEqual(session.confirmedThroughSample, first)
        XCTAssertEqual(final.numSpeakers, 1)
        XCTAssertFalse(final.segments.isEmpty)
        XCTAssertLessThanOrEqual(final.segments.last!.endTime, Float(first) / 16_000 + 0.011)

        session.reset()
        XCTAssertEqual(session.confirmedThroughSample, 0)
        XCTAssertTrue(session.currentResult().segments.isEmpty)
    }

    func testInvalidGeometryFailsClosed() throws {
        let diarizer = Nemotron3Diarizer(backend: FakeNemotron3Backend())
        XCTAssertThrowsError(try diarizer.makeStreamingSession(coreEncoderFrames: 0))
        XCTAssertThrowsError(try diarizer.makeStreamingSession(
            coreEncoderFrames: 340, rightContextEncoderFrames: 41))
    }
}

private final class FakeNemotron3Backend: Nemotron3InferenceBackend {
    let learnedSilenceEmbedding = [Float](repeating: 0, count: 512)
    var headCalls = 0

    func preencode(chunk: [Float]) throws -> [Float] {
        XCTAssertEqual(chunk.count, 3_040 * 128)
        return [Float](repeating: 0, count: 380 * 512)
    }

    func predictHead(
        packedEmbeddings: [Float], validLength: Int
    ) throws -> Nemotron3HeadOutput {
        XCTAssertEqual(packedEmbeddings.count, 684 * 512)
        XCTAssertTrue((0...684).contains(validLength))
        headCalls += 1
        var high = [Float](repeating: 0, count: 684 * 8 * 8)
        var low = [Float](repeating: 0, count: 684 * 8)
        for frame in 0..<(684 * 8) { high[frame * 8] = 0.95 }
        for frame in 0..<684 { low[frame * 8] = 0.95 }
        return Nemotron3HeadOutput(
            probabilities10ms: high,
            probabilities80ms: low)
    }
}
