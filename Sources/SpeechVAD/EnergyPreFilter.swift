import Foundation
import Accelerate

/// Fast DSP energy pre-filter for Silero VAD.
///
/// Computes 8-band log-energy via FFT and compares against an adaptive noise floor.
/// Chunks below the noise floor + margin are skipped (no neural inference needed).
/// Achieves 40-60% skip rate on typical meeting/podcast audio with significant silence.
///
/// Band layout (512-point FFT @ 16kHz, bin = 31.25 Hz):
/// ```
/// Band 0:  62-125 Hz   (bins 2-4)    — hum
/// Band 1: 125-250 Hz   (bins 4-8)    — rumble
/// Band 2: 250-500 Hz   (bins 8-16)   — fundamental F0
/// Band 3: 500-1000 Hz  (bins 16-32)  — vowels
/// Band 4: 1000-2000 Hz (bins 32-64)  — formants
/// Band 5: 2000-4000 Hz (bins 64-128) — consonants
/// Band 6: 4000-6000 Hz (bins 128-192)— sibilants
/// Band 7: 6000-8000 Hz (bins 192-256)— noise
/// ```
public struct EnergyPreFilter: Sendable {

    /// Number of frequency bands.
    static let bandCount = 8

    /// FFT size (matches Silero chunk size).
    private static let fftSize = 512
    private static let log2N: vDSP_Length = 9  // log2(512)

    /// Band boundaries as FFT bin ranges (start inclusive, end exclusive).
    private static let bandBins: [(start: Int, end: Int)] = [
        (2, 4),     // 62-125 Hz
        (4, 8),     // 125-250 Hz
        (8, 16),    // 250-500 Hz
        (16, 32),   // 500-1000 Hz
        (32, 64),   // 1000-2000 Hz
        (64, 128),  // 2000-4000 Hz
        (128, 192), // 4000-6000 Hz
        (192, 256), // 6000-8000 Hz
    ]

    private let config: EnergyPreFilterConfig

    /// Per-band noise floor in dB (EMA-smoothed).
    private var noiseFloor: [Float]

    /// Number of chunks processed (for warmup).
    private var chunksProcessed: Int = 0

    /// Hanning window (precomputed).
    private let window: [Float]

    /// vDSP FFT setup (wrapped for Sendable).
    private let fftSetupWrapper: FFTSetupWrapper

    /// Create a new energy pre-filter.
    public init(config: EnergyPreFilterConfig = .default) {
        self.config = config
        self.noiseFloor = [Float](repeating: -Float.infinity, count: Self.bandCount)
        self.chunksProcessed = 0

        // Precompute Hanning window
        var win = [Float](repeating: 0, count: Self.fftSize)
        vDSP_hann_window(&win, vDSP_Length(Self.fftSize), Int32(vDSP_HANN_NORM))
        self.window = win

        guard let setup = vDSP_create_fftsetup(Self.log2N, FFTRadix(kFFTRadix2)) else {
            fatalError("Failed to create vDSP FFT setup for EnergyPreFilter")
        }
        self.fftSetupWrapper = FFTSetupWrapper(setup)
    }

    /// Determine whether the given chunk needs neural inference.
    ///
    /// During warmup (first N chunks), always returns `true` to let Silero run
    /// and accumulate a reliable noise floor estimate.
    ///
    /// - Parameter samples: exactly 512 PCM Float32 samples at 16kHz
    /// - Returns: `true` if any band exceeds `noiseFloor + marginDB`
    public mutating func shouldInvokeSilero(_ samples: [Float]) -> Bool {
        precondition(samples.count == Self.fftSize,
                     "EnergyPreFilter expects \(Self.fftSize) samples, got \(samples.count)")

        chunksProcessed += 1

        // Always invoke during warmup
        if chunksProcessed <= config.warmupChunks {
            return true
        }

        let bandEnergies = computeBandEnergies(samples)

        // Check if any band exceeds noise floor + margin
        for i in 0..<Self.bandCount {
            if bandEnergies[i] > noiseFloor[i] + config.marginDB {
                return true
            }
        }

        return false
    }

    /// Update the adaptive noise floor estimate.
    ///
    /// Call this only during confirmed silence periods (state machine is in `.silence`).
    /// Uses exponential moving average to track the noise floor.
    ///
    /// - Parameter samples: exactly 512 PCM Float32 samples at 16kHz
    public mutating func updateNoiseFloor(_ samples: [Float]) {
        precondition(samples.count == Self.fftSize,
                     "EnergyPreFilter expects \(Self.fftSize) samples, got \(samples.count)")

        let bandEnergies = computeBandEnergies(samples)

        for i in 0..<Self.bandCount {
            if noiseFloor[i] == -Float.infinity {
                // First update — initialize directly
                noiseFloor[i] = bandEnergies[i]
            } else {
                // EMA update
                noiseFloor[i] = (1 - config.noiseAlpha) * noiseFloor[i] + config.noiseAlpha * bandEnergies[i]
            }
        }
    }

    /// Reset all state (noise floor, chunk counter).
    public mutating func reset() {
        noiseFloor = [Float](repeating: -Float.infinity, count: Self.bandCount)
        chunksProcessed = 0
    }

    /// Number of chunks processed so far.
    public var totalChunks: Int { chunksProcessed }

    // MARK: - DSP

    /// Compute per-band log energy (dB) via 512-point real FFT.
    private func computeBandEnergies(_ samples: [Float]) -> [Float] {
        // Apply Hanning window
        var windowed = [Float](repeating: 0, count: Self.fftSize)
        vDSP_vmul(samples, 1, window, 1, &windowed, 1, vDSP_Length(Self.fftSize))

        // Pack into split complex for vDSP_fft_zrip
        let halfN = Self.fftSize / 2
        var realp = [Float](repeating: 0, count: halfN)
        var imagp = [Float](repeating: 0, count: halfN)

        // Interleave even/odd samples into real/imag
        for i in 0..<halfN {
            realp[i] = windowed[2 * i]
            imagp[i] = windowed[2 * i + 1]
        }

        realp.withUnsafeMutableBufferPointer { rBuf in
            imagp.withUnsafeMutableBufferPointer { iBuf in
                var splitComplex = DSPSplitComplex(
                    realp: rBuf.baseAddress!,
                    imagp: iBuf.baseAddress!
                )
                vDSP_fft_zrip(fftSetupWrapper.setup, &splitComplex, 1, Self.log2N, FFTDirection(kFFTDirection_Forward))
            }
        }

        // Compute magnitude-squared: |X[k]|² = real² + imag²
        // For real FFT of N points, we get N/2+1 unique bins (0..N/2).
        // vDSP_fft_zrip packs: realp[0] = DC, imagp[0] = Nyquist, rest are normal.
        var magnitudeSq = [Float](repeating: 0, count: halfN + 1)
        magnitudeSq[0] = realp[0] * realp[0]           // DC (imag = 0)
        magnitudeSq[halfN] = imagp[0] * imagp[0]       // Nyquist (imag = 0)
        for k in 1..<halfN {
            magnitudeSq[k] = realp[k] * realp[k] + imagp[k] * imagp[k]
        }

        // Sum magnitude-squared per band, convert to dB
        var bandEnergies = [Float](repeating: 0, count: Self.bandCount)
        for (band, bins) in Self.bandBins.enumerated() {
            var sum: Float = 0
            let count = bins.end - bins.start
            guard count > 0, bins.end <= magnitudeSq.count else { continue }

            // vDSP sum of the slice
            magnitudeSq.withUnsafeBufferPointer { buf in
                vDSP_sve(buf.baseAddress! + bins.start, 1, &sum, vDSP_Length(count))
            }

            // Average energy per bin, then convert to dB
            let avgEnergy = sum / Float(count)
            bandEnergies[band] = 10.0 * log10f(max(avgEnergy, 1e-10))
        }

        return bandEnergies
    }
}

// MARK: - Sendable FFT Setup Wrapper

/// Wraps an `OpaquePointer` (FFTSetup) to satisfy Sendable.
/// vDSP FFT setups are read-only after creation, so this is safe.
private final class FFTSetupWrapper: @unchecked Sendable {
    let setup: FFTSetup

    init(_ setup: FFTSetup) {
        self.setup = setup
    }

    deinit {
        vDSP_destroy_fftsetup(setup)
    }
}
