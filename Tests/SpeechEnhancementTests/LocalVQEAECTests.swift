import XCTest
@testable import SpeechEnhancement

final class LocalVQEAECTests: XCTestCase {
    private final class IdentityResidualMask: LocalVQEAECResidualMasking {
        private(set) var resetCount = 0

        func reset() {
            resetCount += 1
        }

        func warmUp() throws {}

        func process(
            microphoneSpectrum: [Float],
            referenceSpectrum: [Float]
        ) throws -> [Float] {
            XCTAssertEqual(microphoneSpectrum.count, 512)
            XCTAssertEqual(referenceSpectrum.count, 512)
            return microphoneSpectrum
        }
    }

    func testPublicConstantsPinNativeStreamingContract() {
        XCTAssertEqual(LocalVQEEchoCanceller.sampleRate, 16_000)
        XCTAssertEqual(LocalVQEEchoCanceller.fftSize, 512)
        XCTAssertEqual(LocalVQEEchoCanceller.frameSize, 256)
        XCTAssertEqual(LocalVQEEchoCanceller.algorithmicLatencySamples, 256)
        XCTAssertEqual(
            LocalVQEEchoCanceller.defaultModelId,
            "aufklarer/LocalVQE-v1.4-AEC-200K-CoreML")
    }

    func testProcessFrameRejectsMissingOrNonFiniteStreams() throws {
        let canceller = try makeCanceller()
        XCTAssertThrowsError(try canceller.processFrame(
            microphone: [Float](repeating: 0, count: 255),
            reference: [Float](repeating: 0, count: 256))) { error in
                XCTAssertEqual(
                    error as? LocalVQEEchoCancellationError,
                    .invalidFrameLength(
                        stream: "Microphone", expected: 256, actual: 255))
            }

        var microphone = [Float](repeating: 0, count: 256)
        microphone[14] = .nan
        XCTAssertThrowsError(try canceller.processFrame(
            microphone: microphone,
            reference: [Float](repeating: 0, count: 256))) { error in
                XCTAssertEqual(
                    error as? LocalVQEEchoCancellationError,
                    .nonFiniteSamples(stream: "Microphone"))
            }
    }

    func testBatchRejectsUnsynchronizedInputsBeforeResettingState() throws {
        let mask = IdentityResidualMask()
        let canceller = try makeCanceller(mask: mask)
        XCTAssertThrowsError(try canceller.process(
            microphone: [0, 1],
            reference: [0],
            sampleRate: 16_000)) { error in
                XCTAssertEqual(
                    error as? LocalVQEEchoCancellationError,
                    .mismatchedStreamLengths(microphone: 2, reference: 1))
            }
        XCTAssertEqual(mask.resetCount, 0)
    }

    /// Regression for the production failure where enabling echo handling
    /// caused the local microphone to disappear. With no playback reference,
    /// the adaptive stage must leave microphone-only speech intact.
    func testSilentReferenceDoesNotMuteMicrophone() throws {
        let canceller = try makeCanceller()
        let frame0 = sineFrame(frequency: 1_000, frameIndex: 0)
        let frame1 = sineFrame(frequency: 1_000, frameIndex: 1)
        let silence = [Float](repeating: 0, count: 256)

        let firstOutput = try canceller.processFrame(
            microphone: frame0, reference: silence)
        let secondOutput = try canceller.processFrame(
            microphone: frame1, reference: silence)

        XCTAssertLessThan(rms(firstOutput), 0.02, "First hop carries codec latency")
        XCTAssertGreaterThan(rms(secondOutput), 0.2, "Microphone speech was muted")
        XCTAssertLessThan(rmse(secondOutput, frame0), 0.015)
    }

    func testResetRestoresAllStreamingState() throws {
        let mask = IdentityResidualMask()
        let canceller = try makeCanceller(mask: mask)
        let microphone = sineFrame(frequency: 700, frameIndex: 0)
        let reference = [Float](repeating: 0, count: 256)

        let first = try canceller.processFrame(
            microphone: microphone, reference: reference)
        _ = try canceller.processFrame(
            microphone: microphone, reference: reference)
        canceller.reset()
        let afterReset = try canceller.processFrame(
            microphone: microphone, reference: reference)

        XCTAssertEqual(mask.resetCount, 1)
        XCTAssertEqual(first.count, afterReset.count)
        XCTAssertLessThan(rmse(first, afterReset), 1e-6)
        XCTAssertEqual(canceller.currentDelaySamples, 0)
    }

    func testOfflineDelayPrimingFindsDelayedReference() throws {
        let filter = try LocalVQEAECAdaptiveFilter(
            weights: [Float](repeating: 0, count: 2_742),
            enablePrealignment: true)
        let count = 48_000
        let delay = 3_000
        var generator = LCG(seed: 20_260_614)
        var reference = [Float](repeating: 0, count: count)
        for index in reference.indices {
            reference[index] = generator.nextFloat() * 0.3
        }
        var microphone = [Float](repeating: 0, count: count)
        for index in delay..<count {
            microphone[index] = 0.6 * reference[index - delay]
        }

        try filter.primeDelay(microphone: microphone, reference: reference)

        XCTAssertGreaterThan(filter.delayConfidence, 8)
        XCTAssertGreaterThanOrEqual(filter.currentDelaySamples, delay - 384)
        XCTAssertLessThanOrEqual(filter.currentDelaySamples, delay)
        XCTAssertEqual(filter.currentDelaySamples % 128, 0)
    }

    private func makeCanceller(
        mask: IdentityResidualMask = IdentityResidualMask()
    ) throws -> LocalVQEEchoCanceller {
        let adaptiveFilter = try LocalVQEAECAdaptiveFilter(
            weights: [Float](repeating: 0, count: 2_742),
            enablePrealignment: false)
        return LocalVQEEchoCanceller(
            adaptiveFilter: adaptiveFilter,
            residualMask: mask,
            codec: try LocalVQEAECCodec(window: sqrtHann()))
    }

    private func sqrtHann() -> [Float] {
        (0..<512).map { index in
            sqrt(0.5 - 0.5 * cos(
                2 * Float.pi * (Float(index) + 0.5) / 512))
        }
    }

    private func sineFrame(frequency: Float, frameIndex: Int) -> [Float] {
        (0..<256).map { index in
            let sample = frameIndex * 256 + index
            return 0.5 * sin(
                2 * Float.pi * frequency * Float(sample) / 16_000)
        }
    }

    private func rms(_ values: [Float]) -> Float {
        sqrt(values.reduce(0) { $0 + $1 * $1 } / Float(values.count))
    }

    private func rmse(_ lhs: [Float], _ rhs: [Float]) -> Float {
        XCTAssertEqual(lhs.count, rhs.count)
        let error = zip(lhs, rhs).reduce(Float.zero) {
            $0 + ($1.0 - $1.1) * ($1.0 - $1.1)
        }
        return sqrt(error / Float(lhs.count))
    }

    private struct LCG {
        private var state: UInt64

        init(seed: UInt64) {
            state = seed
        }

        mutating func nextFloat() -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1
            let unit = Float((state >> 40) & 0xFF_FFFF) / Float(0xFF_FFFF)
            return 2 * unit - 1
        }
    }
}
