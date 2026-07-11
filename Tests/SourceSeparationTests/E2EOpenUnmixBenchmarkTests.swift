import XCTest
@testable import SourceSeparation
import Foundation

/// RTF baseline for `SourceSeparator.separate(...)`.
///
/// Reports end-to-end + per-stage timing on a synthetic stereo signal so we
/// can target optimizations at the actual bottleneck. Synthetic input means
/// no MUSDB18-HQ download required; numbers are wall-clock RTF, not SDR.
///
/// Run with:
///   swift test --filter "SourceSeparationTests.E2EOpenUnmixBenchmarkTests"
final class E2EOpenUnmixBenchmarkTests: XCTestCase {

    private static var _separator: SourceSeparator?

    private var separator: SourceSeparator {
        get throws {
            guard let s = Self._separator else { throw XCTSkip("Model not loaded") }
            return s
        }
    }

    override func setUp() async throws {
        try await super.setUp()
        if Self._separator == nil {
            Self._separator = try await SourceSeparator.fromPretrained()
        }
    }

    /// Synthesize a stereo signal with broadband content so the model has
    /// something non-trivial to chew on. Not a real song, but stationary
    /// enough that timing is reproducible across runs.
    private func makeStereoTestSignal(seconds: Double, sampleRate: Int = 44100) -> [[Float]] {
        let n = Int(Double(sampleRate) * seconds)
        var left = [Float](repeating: 0, count: n)
        var right = [Float](repeating: 0, count: n)
        var rng = SystemRandomNumberGenerator()
        for i in 0..<n {
            let t = Float(i) / Float(sampleRate)
            let tone = 0.3 * sin(2 * .pi * 220 * t)
                     + 0.2 * sin(2 * .pi * 440 * t)
                     + 0.15 * sin(2 * .pi * 880 * t)
            left[i]  = tone + Float.random(in: -0.1...0.1, using: &rng)
            right[i] = tone * 0.9 + Float.random(in: -0.1...0.1, using: &rng)
        }
        return [left, right]
    }

    private func report(_ label: String, _ m: SourceSeparationMetrics, wiener: Bool) {
        let unpackLines = m.modelSecByTarget
            .sorted(by: { $0.key.rawValue < $1.key.rawValue })
            .map { "    unpack \($0.key.rawValue.padding(toLength: 7, withPad: " ", startingAt: 0)) \(fmt($0.value))s" }
            .joined(separator: "\n")
        let summary = """

        [BENCH \(label) wiener=\(wiener)]
          audio           = \(fmt(m.audioSeconds))s
          total           = \(fmt(m.totalSec))s
          RTF             = \(String(format: "%.3f", m.rtf))  (\(String(format: "%.1f", 1.0 / max(m.rtf, 1e-9)))x real-time)
          stft forward    = \(fmt(m.stftForwardSec))s
          model graph     = \(fmt(m.modelGraphBuildSec))s
          model eval (GPU)= \(fmt(m.modelEvalSec))s
        \(unpackLines)
          wiener          = \(fmt(m.wienerSec))s
          inverse stft    = \(fmt(m.inverseStftSec))s
          accounted       = \(fmt(m.stftForwardSec + m.modelTotalSec + m.wienerSec + m.inverseStftSec))s
        """
        print(summary)
    }

    private func fmt(_ x: Double) -> String { String(format: "%6.3f", x) }

    func testBenchmark10sWithWiener() throws {
        let s = try separator
        let secs = 10.0
        let audio = makeStereoTestSignal(seconds: secs)

        // Warm caches / lazy init
        _ = s.separate(audio: audio, sampleRate: 44100, wiener: false)

        var captured = SourceSeparationMetrics()
        _ = s.separate(audio: audio, sampleRate: 44100, wiener: true) { captured = $0 }
        report("10s", captured, wiener: true)
        XCTAssertEqual(captured.modelSecByTarget.count, 4, "expected 4 stems")
        XCTAssertGreaterThan(captured.totalSec, 0)
    }

    func testBenchmark10sNoWiener() throws {
        let s = try separator
        let secs = 10.0
        let audio = makeStereoTestSignal(seconds: secs)

        _ = s.separate(audio: audio, sampleRate: 44100, wiener: false)

        var captured = SourceSeparationMetrics()
        _ = s.separate(audio: audio, sampleRate: 44100, wiener: false) { captured = $0 }
        report("10s", captured, wiener: false)
    }

    func testBenchmark30sWithWiener() throws {
        let s = try separator
        let secs = 30.0
        let audio = makeStereoTestSignal(seconds: secs)

        _ = s.separate(audio: audio, sampleRate: 44100, wiener: false)

        var captured = SourceSeparationMetrics()
        _ = s.separate(audio: audio, sampleRate: 44100, wiener: true) { captured = $0 }
        report("30s", captured, wiener: true)
    }
}
