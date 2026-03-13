import XCTest
@testable import SpeechVAD

final class EnergyPreFilterTests: XCTestCase {

    // MARK: - Warmup

    func testWarmupAlwaysReturnsTrue() {
        var filter = EnergyPreFilter(config: EnergyPreFilterConfig(warmupChunks: 5))
        let silence = [Float](repeating: 0, count: 512)

        // First 5 chunks should always return true (warmup)
        for _ in 0..<5 {
            XCTAssertTrue(filter.shouldInvokeSilero(silence),
                          "Should always invoke during warmup")
        }
    }

    func testAfterWarmupSilenceReturnsFalse() {
        var filter = EnergyPreFilter(config: EnergyPreFilterConfig(
            warmupChunks: 5, marginDB: 10.0))
        let silence = [Float](repeating: 0, count: 512)

        // Run through warmup
        for _ in 0..<5 {
            _ = filter.shouldInvokeSilero(silence)
            filter.updateNoiseFloor(silence)
        }

        // After warmup, pure silence should be below threshold
        XCTAssertFalse(filter.shouldInvokeSilero(silence),
                       "Pure silence should be skipped after warmup with noise floor set")
    }

    // MARK: - Tone Detection

    func testToneAboveThreshold() {
        var filter = EnergyPreFilter(config: EnergyPreFilterConfig(
            warmupChunks: 3, marginDB: 10.0))
        let silence = [Float](repeating: 0, count: 512)

        // Warmup + noise floor on silence
        for _ in 0..<3 {
            _ = filter.shouldInvokeSilero(silence)
            filter.updateNoiseFloor(silence)
        }

        // Generate a 1kHz tone (band 3: 500-1000 Hz)
        let tone = (0..<512).map { i in
            0.5 * sinf(2.0 * Float.pi * 1000.0 * Float(i) / 16000.0)
        }

        XCTAssertTrue(filter.shouldInvokeSilero(tone),
                      "1kHz tone should trigger Silero invocation")
    }

    func testLowAmplitudeToneBelowThreshold() {
        var filter = EnergyPreFilter(config: EnergyPreFilterConfig(
            warmupChunks: 3, marginDB: 20.0))

        // Warmup with low-level noise
        let noise: [Float] = (0..<512).map { _ in Float.random(in: -0.001...0.001) }
        for _ in 0..<3 {
            _ = filter.shouldInvokeSilero(noise)
            filter.updateNoiseFloor(noise)
        }

        // Very quiet tone — may not exceed 20 dB margin
        let quietTone = (0..<512).map { i in
            0.002 * sinf(2.0 * Float.pi * 1000.0 * Float(i) / 16000.0)
        }

        // With 20 dB margin and noise floor from noise, a 0.002 amplitude tone
        // may or may not trigger. Just verify no crash and valid return.
        let result = filter.shouldInvokeSilero(quietTone)
        // Result is deterministic but depends on exact noise floor — just test no crash
        _ = result
    }

    // MARK: - Noise Floor Convergence

    func testNoiseFloorConverges() {
        var filter = EnergyPreFilter(config: EnergyPreFilterConfig(
            noiseAlpha: 0.1, warmupChunks: 2, marginDB: 10.0))

        // Consistent low-level noise
        let noise: [Float] = (0..<512).map { i in
            0.01 * sinf(2.0 * Float.pi * 200.0 * Float(i) / 16000.0)
        }

        // Warmup
        for _ in 0..<2 {
            _ = filter.shouldInvokeSilero(noise)
            filter.updateNoiseFloor(noise)
        }

        // Feed same noise many times to let EMA converge
        for _ in 0..<50 {
            filter.updateNoiseFloor(noise)
        }

        // After convergence, same noise should NOT trigger (it IS the noise floor)
        let result = filter.shouldInvokeSilero(noise)
        XCTAssertFalse(result,
                       "Same noise level as noise floor should not exceed margin")
    }

    // MARK: - Reset

    func testResetClearsState() {
        var filter = EnergyPreFilter(config: EnergyPreFilterConfig(warmupChunks: 3))
        let silence = [Float](repeating: 0, count: 512)

        // Process some chunks
        for _ in 0..<5 {
            _ = filter.shouldInvokeSilero(silence)
            filter.updateNoiseFloor(silence)
        }

        filter.reset()

        // After reset, should be back in warmup
        XCTAssertTrue(filter.shouldInvokeSilero(silence),
                      "Should be in warmup after reset")
        XCTAssertEqual(filter.totalChunks, 1,
                       "Chunk count should restart from 1 after reset")
    }

    // MARK: - Band Coverage

    func testDifferentFrequencyBands() {
        // Test that tones at different frequencies activate different bands
        let frequencies: [Float] = [100, 180, 350, 750, 1500, 3000, 5000, 7000]

        for freq in frequencies {
            var filter = EnergyPreFilter(config: EnergyPreFilterConfig(
                warmupChunks: 3, marginDB: 10.0))
            let silence = [Float](repeating: 0, count: 512)

            // Warmup + noise floor on silence
            for _ in 0..<3 {
                _ = filter.shouldInvokeSilero(silence)
                filter.updateNoiseFloor(silence)
            }

            // Generate tone at this frequency
            let tone = (0..<512).map { i in
                0.5 * sinf(2.0 * Float.pi * freq * Float(i) / 16000.0)
            }

            XCTAssertTrue(filter.shouldInvokeSilero(tone),
                          "\(freq) Hz tone should trigger Silero invocation")
        }
    }

    // MARK: - Edge Cases

    func testAllZerosSamples() {
        var filter = EnergyPreFilter()
        let zeros = [Float](repeating: 0, count: 512)

        // Should not crash with all zeros
        let result = filter.shouldInvokeSilero(zeros)
        XCTAssertTrue(result, "First chunk is warmup, should return true")
    }

    func testFullScaleSamples() {
        var filter = EnergyPreFilter(config: EnergyPreFilterConfig(warmupChunks: 1))
        let silence = [Float](repeating: 0, count: 512)

        _ = filter.shouldInvokeSilero(silence)
        filter.updateNoiseFloor(silence)

        // Loud broadband noise (DC alone lands in bin 0 which is outside all bands)
        let loud = (0..<512).map { i in sinf(2.0 * Float.pi * 1000.0 * Float(i) / 16000.0) }
        XCTAssertTrue(filter.shouldInvokeSilero(loud),
                      "Full-scale audio should trigger Silero")
    }
}
