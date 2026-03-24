#if canImport(AVFoundation)
import XCTest
@testable import AudioCommon

/// E2E tests: generate audio, play through StreamingAudioPlayer, verify completion.
/// These use real AVAudioEngine — require audio hardware (Mac, not CI).
final class E2EStreamingPlaybackTests: XCTestCase {

    // MARK: - Short utterance (< pre-buffer)

    func testShortUtteranceCompletes() {
        // Simulate short TTS: 0.5s of audio, pre-buffer 2s.
        // Audio doesn't fill pre-buffer → markGenerationComplete flushes it.
        let player = StreamingAudioPlayer()
        player.preBufferDuration = 2.0
        player.ensureStandaloneEngine()

        let expectation = XCTestExpectation(description: "short utterance plays")
        player.onPlaybackFinished = { expectation.fulfill() }

        player.resetGeneration()
        // 0.5s of 440Hz sine at 48kHz
        let samples = generateTone(frequency: 440, duration: 0.5, sampleRate: 48000)
        player.scheduleChunk(samples)
        player.markGenerationComplete()

        wait(for: [expectation], timeout: 5.0)
        XCTAssertFalse(player.isPlaying)
        player.stop()
    }

    // MARK: - Long utterance with streaming chunks

    func testLongUtteranceStreamingCompletes() {
        // Simulate streaming TTS: 10 chunks of 0.5s each = 5s total.
        // Pre-buffer 1s. Chunks arrive every 250ms (faster than real-time).
        let player = StreamingAudioPlayer()
        player.preBufferDuration = 1.0
        player.ensureStandaloneEngine()

        let expectation = XCTestExpectation(description: "long streaming completes")
        player.onPlaybackFinished = { expectation.fulfill() }
        player.resetGeneration()

        // Schedule 10 chunks with simulated generation delay
        DispatchQueue.global(qos: .userInitiated).async {
            for i in 0..<10 {
                let chunk = self.generateTone(
                    frequency: Float(440 + i * 50), duration: 0.5, sampleRate: 48000)
                player.scheduleChunk(chunk)
                usleep(250_000) // 250ms between chunks
            }
            player.markGenerationComplete()
        }

        // 5s audio + 1s pre-buffer + margin
        wait(for: [expectation], timeout: 10.0)
        XCTAssertFalse(player.isPlaying)
        player.stop()
    }

    // MARK: - Long utterance with slow generation (RTF > 0.5)

    func testSlowGenerationNoUnderflow() {
        // Simulate slow TTS: 5 chunks of 1s, each takes 600ms to generate.
        // Pre-buffer 2s. Total: 5s audio, 3s generation after pre-buffer.
        let player = StreamingAudioPlayer()
        player.preBufferDuration = 2.0
        player.ensureStandaloneEngine()

        let expectation = XCTestExpectation(description: "slow generation completes")
        player.onPlaybackFinished = { expectation.fulfill() }
        player.resetGeneration()

        DispatchQueue.global(qos: .userInitiated).async {
            for i in 0..<5 {
                let chunk = self.generateTone(
                    frequency: Float(300 + i * 100), duration: 1.0, sampleRate: 48000)
                player.scheduleChunk(chunk)
                usleep(600_000) // 600ms generation time per 1s chunk
            }
            player.markGenerationComplete()
        }

        wait(for: [expectation], timeout: 12.0)
        XCTAssertFalse(player.isPlaying)
        player.stop()
    }

    // MARK: - Interrupt mid-playback

    func testInterruptMidPlayback() {
        let player = StreamingAudioPlayer()
        player.preBufferDuration = 0.5
        player.ensureStandaloneEngine()
        player.resetGeneration()

        // Schedule 3s of audio
        let samples = generateTone(frequency: 440, duration: 3.0, sampleRate: 48000)
        player.scheduleChunk(samples)

        // Wait 0.5s then interrupt
        let expectation = XCTestExpectation(description: "interrupted")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            player.fadeOutAndStop()
            XCTAssertFalse(player.isPlaying)
            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 3.0)
        player.stop()
    }

    // MARK: - Multiple cycles

    func testMultipleCycles() {
        let player = StreamingAudioPlayer()
        player.preBufferDuration = 0.5
        player.ensureStandaloneEngine()

        for cycle in 0..<3 {
            let expectation = XCTestExpectation(description: "cycle \(cycle)")
            player.onPlaybackFinished = { expectation.fulfill() }
            player.resetGeneration()

            let chunk = generateTone(frequency: Float(440 + cycle * 200), duration: 0.3, sampleRate: 48000)
            player.scheduleChunk(chunk)
            player.markGenerationComplete()

            wait(for: [expectation], timeout: 3.0)
        }
        player.stop()
    }

    // MARK: - Zero pre-buffer (Kokoro-style single pass)

    func testSinglePassNoPreBuffer() {
        let player = StreamingAudioPlayer()
        player.preBufferDuration = 0
        player.ensureStandaloneEngine()

        let expectation = XCTestExpectation(description: "single pass completes")
        player.onPlaybackFinished = { expectation.fulfill() }
        player.resetGeneration()

        // All audio at once (like Kokoro)
        let samples = generateTone(frequency: 440, duration: 2.0, sampleRate: 48000)
        player.scheduleChunk(samples)
        player.markGenerationComplete()

        wait(for: [expectation], timeout: 5.0)
        player.stop()
    }

    // MARK: - Helpers

    private func generateTone(frequency: Float, duration: Double, sampleRate: Int) -> [Float] {
        let count = Int(Double(sampleRate) * duration)
        return (0..<count).map { i in
            sin(2.0 * .pi * frequency * Float(i) / Float(sampleRate)) * 0.3
        }
    }
}
#endif
