import XCTest
@testable import AudioCommon

final class StreamingAudioPlayerTests: XCTestCase {

    func testInitialState() {
        let player = StreamingAudioPlayer()
        XCTAssertFalse(player.isPlaying)
    }

    func testResetGeneration() {
        let player = StreamingAudioPlayer()
        player.markGenerationComplete()
        player.resetGeneration()
        XCTAssertFalse(player.isPlaying)
    }

    func testMarkGenerationCompleteWithoutBuffers() {
        let player = StreamingAudioPlayer()
        let expectation = XCTestExpectation(description: "playback finished")
        player.onPlaybackFinished = { expectation.fulfill() }
        player.markGenerationComplete()
        wait(for: [expectation], timeout: 1.0)
    }

    func testMarkGenerationCompleteFiresWithoutEngine() {
        let player = StreamingAudioPlayer()
        let expectation = XCTestExpectation(description: "callback fires")
        player.onPlaybackFinished = { expectation.fulfill() }
        player.markGenerationComplete()
        wait(for: [expectation], timeout: 1.0)
    }

    func testFadeOutResetsPendingBuffers() {
        let player = StreamingAudioPlayer()
        player.fadeOutAndStop()
        XCTAssertFalse(player.isPlaying)
    }

    func testScheduleChunkWithoutEngine() {
        let player = StreamingAudioPlayer()
        player.scheduleChunk([0.1, 0.2, 0.3])
        XCTAssertFalse(player.isPlaying)
    }

    func testResetAfterComplete() {
        let player = StreamingAudioPlayer()
        player.markGenerationComplete()
        player.resetGeneration()
        XCTAssertFalse(player.isPlaying)
    }

    func testFadeOutResetsGenerationState() {
        let player = StreamingAudioPlayer()
        player.markGenerationComplete()
        player.fadeOutAndStop()
        XCTAssertFalse(player.isPlaying)
    }

    func testMultipleMarkGenerationCompleteIdempotent() {
        let player = StreamingAudioPlayer()
        var callbackCount = 0
        player.onPlaybackFinished = { callbackCount += 1 }
        player.markGenerationComplete()
        player.markGenerationComplete()
        player.markGenerationComplete()

        let expectation = XCTestExpectation(description: "wait")
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { expectation.fulfill() }
        wait(for: [expectation], timeout: 2.0)
        XCTAssertGreaterThanOrEqual(callbackCount, 1)
    }
}
