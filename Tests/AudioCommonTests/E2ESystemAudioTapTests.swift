#if os(macOS)
import AVFoundation
import XCTest
@testable import AudioCommon

/// Real-hardware capture test: plays a tone through the default output and
/// asserts the system tap hears it, then that excluding the current process
/// silences the capture again.
///
/// Requirements: audio output hardware and the macOS audio-recording permission
/// for the test host (macOS prompts on the first run). Run without other audio
/// playing — the tap captures the whole system mix, so background music would
/// leak into the exclusion phase and skew the comparison.
final class E2ESystemAudioTapTests: XCTestCase {

    private final class CaptureSink {
        private let lock = NSLock()
        private var samples: [Float] = []

        func append(_ chunk: [Float]) {
            lock.lock()
            samples.append(contentsOf: chunk)
            lock.unlock()
        }

        var count: Int {
            lock.lock()
            defer { lock.unlock() }
            return samples.count
        }

        var rms: Float {
            lock.lock()
            defer { lock.unlock() }
            guard !samples.isEmpty else { return 0 }
            var sum: Float = 0
            for sample in samples { sum += sample * sample }
            return (sum / Float(samples.count)).squareRoot()
        }
    }

    private func makeToneEngine(frequency: Double = 440) throws -> AVAudioEngine {
        let engine = AVAudioEngine()
        let outputFormat = engine.outputNode.outputFormat(forBus: 0)
        let sampleRate = outputFormat.sampleRate
        var phase = 0.0
        let increment = 2.0 * Double.pi * frequency / sampleRate

        let source = AVAudioSourceNode { _, _, frameCount, audioBufferList -> OSStatus in
            let buffers = UnsafeMutableAudioBufferListPointer(audioBufferList)
            for frame in 0..<Int(frameCount) {
                let value = Float(sin(phase)) * 0.4
                phase += increment
                if phase > 2.0 * Double.pi { phase -= 2.0 * Double.pi }
                for buffer in buffers {
                    guard let data = buffer.mData else { continue }
                    data.assumingMemoryBound(to: Float.self)[frame] = value
                }
            }
            return noErr
        }

        engine.attach(source)
        let monoFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: sampleRate,
            channels: 1, interleaved: false)!
        engine.connect(source, to: engine.mainMixerNode, format: monoFormat)
        engine.mainMixerNode.outputVolume = 0.6
        try engine.start()
        return engine
    }

    private func capture(excludeCurrentProcess: Bool, seconds: TimeInterval) throws -> (
        sink: CaptureSink, framesCaptured: UInt64, nonSilentFrames: UInt64, tapRate: Double
    ) {
        let tap = SystemAudioTap(excludeCurrentProcess: excludeCurrentProcess)
        let sink = CaptureSink()
        try tap.start(targetSampleRate: 16000) { sink.append($0) }
        defer { tap.stop() }
        Thread.sleep(forTimeInterval: seconds)
        return (sink, tap.framesCaptured, tap.nonSilentFrames, tap.tapSampleRate)
    }

    func testCapturesOwnToneAndExcludingOwnProcessSilencesIt() throws {
        let engine = try makeToneEngine()
        defer { engine.stop() }

        // Give the output engine a moment to actually start rendering.
        Thread.sleep(forTimeInterval: 0.3)

        // Phase 1: the tap includes this process, so it must hear the tone.
        let included = try capture(excludeCurrentProcess: false, seconds: 1.5)

        XCTAssertGreaterThan(included.framesCaptured, 0, "the tap delivered no buffers at all")
        XCTAssertGreaterThan(included.tapRate, 0, "tap sample rate was never published")
        XCTAssertGreaterThan(
            included.nonSilentFrames, 0,
            "tap delivered only silence — if this persists, the audio-recording permission is "
                + "likely missing for the test host (macOS Privacy & Security → Screen & System "
                + "Audio Recording)")
        XCTAssertGreaterThan(
            included.sink.rms, 0.02,
            "captured level too low for the played tone (rms \(included.sink.rms))")
        // 1.5 s at 16 kHz minus startup latency should still exceed one second of audio.
        XCTAssertGreaterThan(included.sink.count, 16_000, "resampled capture is shorter than 1 s")

        // Phase 2: excluding the current process must remove our own tone. When
        // every audible process is excluded the HAL may deliver no callbacks at
        // all, so zero buffers is a valid pass — what matters is that no
        // meaningful energy from our tone reaches the capture.
        let excluded = try capture(excludeCurrentProcess: true, seconds: 1.5)
        XCTAssertLessThan(
            excluded.sink.rms, max(included.sink.rms * 0.2, 0.005),
            "self-exclusion still captured this process's tone (rms \(excluded.sink.rms) vs "
                + "included \(included.sink.rms)) — was other audio playing during the test?")
    }
}
#endif
