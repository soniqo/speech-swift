#if os(macOS)
import XCTest
@testable import SpeechDemo

final class EchoMicrophoneGateTests: XCTestCase {
    func testMutedInputPreservesTimingWithSilence() {
        let samples: [Float] = [0.2, -0.4, 0.6]

        XCTAssertEqual(
            EchoMicrophoneGate.samplesToPush(samples, muted: true),
            [0, 0, 0])
    }

    func testUnmutedInputPassesThrough() {
        let samples: [Float] = [0.2, -0.4, 0.6]

        XCTAssertEqual(
            EchoMicrophoneGate.samplesToPush(samples, muted: false),
            samples)
    }
}
#endif
