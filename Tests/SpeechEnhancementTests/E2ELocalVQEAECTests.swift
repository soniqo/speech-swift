import XCTest
@testable import SpeechEnhancement

/// End-to-end coverage for the published LocalVQE Core ML bundle. The `E2E`
/// prefix keeps network/model tests out of the default unit suite.
@MainActor
final class E2ELocalVQEAECTests: XCTestCase {
    private static var sharedCanceller: LocalVQEEchoCanceller?

    private func canceller() async throws -> LocalVQEEchoCanceller {
        if let shared = Self.sharedCanceller {
            shared.reset()
            return shared
        }
        // Exercise the production resolver. CI can still force CPU execution
        // with SPEECH_COREML_COMPUTE_UNITS=cpuOnly when required.
        let loaded = try await LocalVQEEchoCanceller.fromPretrained()
        Self.sharedCanceller = loaded
        return loaded
    }

    func testPublishedBundleLoadsAndKeepsSilenceFinite() async throws {
        let canceller = try await canceller()
        let silence = [Float](repeating: 0, count: 513)
        let output = try canceller.process(
            microphone: silence,
            reference: silence,
            sampleRate: 16_000,
            primeDelay: false)

        XCTAssertEqual(output.count, silence.count)
        XCTAssertTrue(output.allSatisfy(\.isFinite))
        XCTAssertLessThan(output.map(abs).max() ?? 0, 0.001)
    }

    /// Production regression: a missing/silent playback reference must not
    /// make local microphone speech disappear in the learned residual mask.
    func testPublishedModelPreservesMicrophoneOnlySpeech() async throws {
        let canceller = try await canceller()
        let sampleRate = 16_000
        let count = sampleRate * 2
        let microphone = (0..<count).map { index in
            let time = Float(index) / Float(sampleRate)
            let envelope = 0.65 + 0.35 * sin(2 * Float.pi * 2.3 * time)
            return envelope * (
                0.28 * sin(2 * Float.pi * 190 * time)
                    + 0.12 * sin(2 * Float.pi * 430 * time))
        }
        let output = try canceller.process(
            microphone: microphone,
            reference: [Float](repeating: 0, count: count),
            sampleRate: sampleRate,
            primeDelay: false)

        let start = sampleRate
        let target = Array(microphone[start..<(count - 256)])
        let alignedOutput = Array(output[(start + 256)..<count])
        let gainDB = 20 * log10(
            max(rms(alignedOutput), 1e-9) / max(rms(target), 1e-9))
        let correlation = cosineSimilarity(alignedOutput, target)
        print(
            "LocalVQE microphone-only: gain \(gainDB) dB, "
                + "correlation \(correlation)")

        XCTAssertGreaterThan(gainDB, -6, "Microphone-only speech was suppressed")
        XCTAssertGreaterThan(correlation, 0.8, "Microphone-only speech was distorted")
    }

    func testFarEndEchoEnergyIsReduced() async throws {
        let canceller = try await canceller()
        let sampleRate = 16_000
        let count = sampleRate * 4
        let delay = 2_400
        var generator = LCG(seed: 20_260_722)
        var reference = [Float](repeating: 0, count: count)
        for index in reference.indices {
            reference[index] = generator.nextFloat() * 0.25
        }
        var microphone = [Float](repeating: 0, count: count)
        for index in delay..<count {
            microphone[index] = 0.65 * reference[index - delay]
            if index >= delay + 83 {
                microphone[index] += 0.2 * reference[index - delay - 83]
            }
        }

        let start = ContinuousClock.now
        let output = try canceller.process(
            microphone: microphone,
            reference: reference,
            sampleRate: sampleRate,
            primeDelay: true)
        let elapsed = start.duration(to: .now)
        let elapsedComponents = elapsed.components
        let elapsedSeconds = Double(elapsedComponents.seconds)
            + Double(elapsedComponents.attoseconds) / 1e18
        let rtf = elapsedSeconds / (Double(count) / Double(sampleRate))
        let throughput = 1.0 / rtf
        let inputRMS = rms(Array(microphone.suffix(sampleRate)))
        let outputRMS = rms(Array(output.suffix(sampleRate)))
        let attenuationDB = 20 * log10(max(outputRMS, 1e-9) / max(inputRMS, 1e-9))
        print(
            "LocalVQE AEC 4 s: \(elapsedSeconds) s, RTF \(rtf), "
                + "throughput \(throughput)xRT, tail attenuation "
                + "\(attenuationDB) dB")

        XCTAssertEqual(output.count, microphone.count)
        XCTAssertTrue(output.allSatisfy(\.isFinite))
        XCTAssertLessThan(attenuationDB, -6.0)
        XCTAssertGreaterThan(canceller.delayConfidence, 8)
        XCTAssertGreaterThan(canceller.currentDelaySamples, 0)
    }

    /// Echo attenuation alone is not sufficient: a canceller that outputs
    /// silence would pass that check. Verify near-end speech remains present
    /// while far-end playback is active.
    func testNearEndSpeechSurvivesDoubleTalk() async throws {
        let canceller = try await canceller()
        let sampleRate = 16_000
        let count = sampleRate * 4
        let delay = 2_400
        var generator = LCG(seed: 20_260_723)
        var reference = [Float](repeating: 0, count: count)
        var nearEnd = [Float](repeating: 0, count: count)
        for index in 0..<count {
            reference[index] = generator.nextFloat() * 0.22
            let time = Float(index) / Float(sampleRate)
            let envelope = 0.7 + 0.3 * sin(2 * Float.pi * 1.7 * time)
            nearEnd[index] = envelope * (
                0.2 * sin(2 * Float.pi * 170 * time)
                    + 0.08 * sin(2 * Float.pi * 510 * time))
        }
        var microphone = nearEnd
        for index in delay..<count {
            microphone[index] += 0.65 * reference[index - delay]
            if index >= delay + 83 {
                microphone[index] += 0.2 * reference[index - delay - 83]
            }
        }

        let output = try canceller.process(
            microphone: microphone,
            reference: reference,
            sampleRate: sampleRate,
            primeDelay: true)
        let start = sampleRate * 2
        let target = Array(nearEnd[start..<(count - 256)])
        let mixture = Array(microphone[start..<(count - 256)])
        let alignedOutput = Array(output[(start + 256)..<count])
        let nearEndGainDB = 20 * log10(
            max(rms(alignedOutput), 1e-9) / max(rms(target), 1e-9))
        let correlation = cosineSimilarity(alignedOutput, target)
        let mixtureError = rmse(mixture, target)
        let enhancedError = rmse(alignedOutput, target)
        print(
            "LocalVQE double-talk: near-end gain \(nearEndGainDB) dB, "
                + "correlation \(correlation), error \(mixtureError) -> "
                + "\(enhancedError)")

        XCTAssertGreaterThan(
            nearEndGainDB, -12, "Near-end speech disappeared during double-talk")
        XCTAssertGreaterThan(
            correlation, 0.5, "Near-end speech identity was not preserved")
        XCTAssertLessThan(
            enhancedError, mixtureError, "Echo error did not improve")
    }

    private func rms(_ values: [Float]) -> Float {
        sqrt(values.reduce(0) { $0 + $1 * $1 } / Float(values.count))
    }

    private func rmse(_ lhs: [Float], _ rhs: [Float]) -> Float {
        XCTAssertEqual(lhs.count, rhs.count)
        let squaredError = zip(lhs, rhs).reduce(Float.zero) {
            let difference = $1.0 - $1.1
            return $0 + difference * difference
        }
        return sqrt(squaredError / Float(lhs.count))
    }

    private func cosineSimilarity(_ lhs: [Float], _ rhs: [Float]) -> Float {
        XCTAssertEqual(lhs.count, rhs.count)
        var dot = Float.zero
        var lhsEnergy = Float.zero
        var rhsEnergy = Float.zero
        for (left, right) in zip(lhs, rhs) {
            dot += left * right
            lhsEnergy += left * left
            rhsEnergy += right * right
        }
        return dot / sqrt(max(lhsEnergy * rhsEnergy, 1e-18))
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
