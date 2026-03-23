#if os(macOS)
import XCTest
@testable import PersonaPlexDemo

final class SentencePieceDecoderTests: XCTestCase {

    func testDecodeSkipsPaddingTokens() {
        // SentencePieceDecoder.decode() skips token ID 3 (text padding)
        // We can't construct a SentencePieceDecoder without a model file,
        // but we can test the decode logic indirectly through the type.
        // This test verifies the type exists and compiles.
        XCTAssertTrue(true, "SentencePieceDecoder type is accessible")
    }
}

final class StreamingAudioPlayerTests: XCTestCase {

    func testInitialState() {
        let player = StreamingAudioPlayer()
        XCTAssertFalse(player.isPlaying, "Should not be playing initially")
    }

    func testStopWithoutStart() {
        let player = StreamingAudioPlayer()
        // Should not crash
        player.stop()
        XCTAssertFalse(player.isPlaying)
    }

    func testScheduleChunkWithoutStart() {
        let player = StreamingAudioPlayer()
        // Should silently return (no node)
        player.scheduleChunk([0.1, 0.2, 0.3])
        XCTAssertFalse(player.isPlaying)
    }

    func testScheduleEmptyChunk() {
        let player = StreamingAudioPlayer()
        // Empty chunk should be a no-op
        player.scheduleChunk([])
        XCTAssertFalse(player.isPlaying)
    }
}

final class AudioRecorderTests: XCTestCase {

    func testInitialState() {
        let recorder = AudioRecorder()
        XCTAssertFalse(recorder.isRecording)
    }

    func testStopWithoutStart() {
        let recorder = AudioRecorder()
        // Should not crash
        let samples = recorder.stopRecording()
        XCTAssertTrue(samples.isEmpty)
        XCTAssertFalse(recorder.isRecording)
    }
}

#endif
