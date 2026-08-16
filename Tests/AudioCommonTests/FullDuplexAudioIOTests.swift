#if canImport(AVFoundation)
import XCTest
@testable import AudioCommon

final class FullDuplexAudioIOTests: XCTestCase {
    func testDefaultConfigurationMatchesVoiceChatAudioContract() throws {
        let configuration = FullDuplexAudioIO.Configuration()
        XCTAssertEqual(configuration.inputSampleRate, 16_000)
        XCTAssertEqual(configuration.outputSampleRate, 22_050)
        XCTAssertEqual(configuration.inputBufferFrames, 1_024)
        XCTAssertEqual(configuration.playbackPrebufferFrames, 3)
        XCTAssertTrue(configuration.enableAEC)
        XCTAssertNoThrow(try configuration.validate())
    }

    func testConfigurationRejectsInvalidRatesAndBufferSizes() {
        assertInvalid(.init(inputSampleRate: 0))
        assertInvalid(.init(outputSampleRate: 0))
        assertInvalid(.init(inputBufferFrames: 0))
        assertInvalid(.init(playbackPrebufferFrames: 0))
        assertInvalid(.init(playbackPrebufferFrames: 33))
    }

    func testInitialStateAndStatisticsAreEmptyWithoutOpeningHardware() {
        let audio = FullDuplexAudioIO()
        XCTAssertEqual(audio.state, .stopped)
        XCTAssertEqual(audio.statistics(), .init(
            scheduledBuffers: 0,
            completedBuffers: 0,
            underruns: 0,
            microphoneLevel: 0))
        audio.stop()
        XCTAssertEqual(audio.state, .stopped)
    }

    func testPlaybackRecoveryUsesLargerBoundedCushion() {
        XCTAssertEqual(
            FullDuplexAudioIO.recoveryPrebufferFrames(initial: 3), 8)
        XCTAssertEqual(
            FullDuplexAudioIO.recoveryPrebufferFrames(initial: 8), 16)
        XCTAssertEqual(
            FullDuplexAudioIO.recoveryPrebufferFrames(initial: 24), 32)
    }

    private func assertInvalid(
        _ configuration: FullDuplexAudioIO.Configuration,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertThrowsError(
            try configuration.validate(), file: file, line: line)
    }
}

final class E2EFullDuplexAudioIOTests: XCTestCase {
    func testStartsVoiceProcessingGraphOnCurrentAudioDevices() throws {
        guard ProcessInfo.processInfo.environment["FULL_DUPLEX_AUDIO_HARDWARE_TEST"] == "1" else {
            throw XCTSkip("Set FULL_DUPLEX_AUDIO_HARDWARE_TEST=1 to open the audio devices")
        }

        let sampleCounter = LockedSampleCounter()
        let audio = FullDuplexAudioIO()
        try audio.start { sampleCounter.add($0.count) }
        XCTAssertEqual(audio.state, .running)

        let silence = [Float](repeating: 0, count: 2_205)
        for _ in 0..<3 { audio.schedulePlayback(silence) }
        let deadline = Date().addingTimeInterval(1)
        while (audio.statistics().scheduledBuffers < 3
               || sampleCounter.value == 0),
              Date() < deadline {
            Thread.sleep(forTimeInterval: 0.01)
        }
        XCTAssertEqual(audio.statistics().scheduledBuffers, 3)
        XCTAssertGreaterThan(sampleCounter.value, 0)
        XCTAssertEqual(audio.state, .running)

        // Let the initial three-buffer queue drain, then prove that playback
        // recovers with the larger bounded cushion used after an underrun.
        let underrunDeadline = Date().addingTimeInterval(2)
        while audio.statistics().underruns == 0,
              Date() < underrunDeadline {
            Thread.sleep(forTimeInterval: 0.01)
        }
        XCTAssertGreaterThanOrEqual(audio.statistics().underruns, 1)

        let recoveryFrames = FullDuplexAudioIO.recoveryPrebufferFrames(
            initial: audio.configuration.playbackPrebufferFrames)
        for _ in 0..<recoveryFrames {
            audio.schedulePlayback(silence)
        }
        let recoveryDeadline = Date().addingTimeInterval(2)
        while audio.statistics().scheduledBuffers < 3 + recoveryFrames,
              Date() < recoveryDeadline {
            Thread.sleep(forTimeInterval: 0.01)
        }
        XCTAssertEqual(
            audio.statistics().scheduledBuffers,
            3 + recoveryFrames)
        XCTAssertEqual(audio.state, .running)

        audio.stop()
        XCTAssertEqual(audio.state, .stopped)

        // Regression for AVFAudio -10875 after a previous graph had already
        // initialized and stopped. Reusing the same wrapper must construct a
        // fresh Voice Processing graph and resume microphone delivery.
        let samplesBeforeRestart = sampleCounter.value
        try audio.start { sampleCounter.add($0.count) }
        let restartDeadline = Date().addingTimeInterval(1)
        while sampleCounter.value == samplesBeforeRestart,
              Date() < restartDeadline {
            Thread.sleep(forTimeInterval: 0.01)
        }
        XCTAssertEqual(audio.state, .running)
        XCTAssertGreaterThan(sampleCounter.value, samplesBeforeRestart)
        audio.stop()
        XCTAssertEqual(audio.state, .stopped)
    }
}

private final class LockedSampleCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0

    var value: Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }

    func add(_ amount: Int) {
        lock.lock()
        count += amount
        lock.unlock()
    }
}
#endif
