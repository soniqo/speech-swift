import Accelerate
import Foundation

/// Time-major SpeechBrain log-mel features.
public struct SpeechBrainLogMelFeatures: Sendable, Equatable {
    public let values: [Float]
    public let frameCount: Int
    public let melBinCount: Int

    public init(values: [Float], frameCount: Int, melBinCount: Int) {
        self.values = values
        self.frameCount = frameCount
        self.melBinCount = melBinCount
    }
}

/// CPU frontend matching `speechbrain.lobes.features.Fbank` for the published
/// ECAPA VoxCeleb and VoxLingua107 checkpoints.
///
/// The 400-point DFT is evaluated as a pair of BLAS matrix multiplies. Using a
/// power-of-two FFT would change the frequency grid from 201 to 257 bins and
/// produce materially different embeddings.
public final class SpeechBrainFbank: @unchecked Sendable {
    public static let sampleRate = 16_000
    public static let fftSize = 400
    public static let windowLength = 400
    public static let hopLength = 160
    public static let frequencyBinCount = 201

    public let melBinCount: Int

    private let window: [Float]
    /// Row-major `[fftSize, frequencyBinCount]`.
    private let dftRealBasis: [Float]
    /// Row-major `[fftSize, frequencyBinCount]`.
    private let dftImaginaryBasis: [Float]
    /// Row-major `[frequencyBinCount, melBinCount]`.
    private let melFilterbank: [Float]

    public init(melBinCount: Int) {
        precondition(melBinCount > 0, "melBinCount must be positive")
        self.melBinCount = melBinCount

        window = (0..<Self.windowLength).map { index in
            0.54 - 0.46
                * cos(2 * Float.pi * Float(index) / Float(Self.windowLength))
        }

        var real = [Float](
            repeating: 0,
            count: Self.fftSize * Self.frequencyBinCount
        )
        var imaginary = real
        for sample in 0..<Self.fftSize {
            for bin in 0..<Self.frequencyBinCount {
                let angle = -2 * Double.pi * Double(sample * bin)
                    / Double(Self.fftSize)
                let offset = sample * Self.frequencyBinCount + bin
                real[offset] = Float(cos(angle))
                imaginary[offset] = Float(sin(angle))
            }
        }
        dftRealBasis = real
        dftImaginaryBasis = imaginary
        melFilterbank = Self.makeFilterbank(melBinCount: melBinCount)
    }

    /// Extract `[frames, melBinCount]` log-mel values from 16 kHz mono PCM.
    ///
    /// The centered STFT uses constant-zero padding, a periodic Hamming window,
    /// power spectrum, SpeechBrain's symmetric triangular mel filters, dB
    /// conversion, and an 80 dB global dynamic-range floor.
    public func extract(_ audio: [Float]) throws -> SpeechBrainLogMelFeatures {
        guard !audio.isEmpty else {
            throw SpeechLanguageIDError.invalidAudio("audio is empty")
        }

        let centerPadding = Self.fftSize / 2
        var padded = [Float](
            repeating: 0,
            count: audio.count + 2 * centerPadding
        )
        padded.replaceSubrange(
            centerPadding..<(centerPadding + audio.count),
            with: audio
        )

        let frameCount = (padded.count - Self.fftSize) / Self.hopLength + 1
        var frames = [Float](
            repeating: 0,
            count: frameCount * Self.fftSize
        )
        padded.withUnsafeBufferPointer { source in
            frames.withUnsafeMutableBufferPointer { destination in
                for frame in 0..<frameCount {
                    vDSP_vmul(
                        source.baseAddress! + frame * Self.hopLength,
                        1,
                        window,
                        1,
                        destination.baseAddress! + frame * Self.fftSize,
                        1,
                        vDSP_Length(Self.fftSize)
                    )
                }
            }
        }

        let spectrumCount = frameCount * Self.frequencyBinCount
        var real = [Float](repeating: 0, count: spectrumCount)
        var imaginary = real
        vDSP_mmul(
            frames,
            1,
            dftRealBasis,
            1,
            &real,
            1,
            vDSP_Length(frameCount),
            vDSP_Length(Self.frequencyBinCount),
            vDSP_Length(Self.fftSize)
        )
        vDSP_mmul(
            frames,
            1,
            dftImaginaryBasis,
            1,
            &imaginary,
            1,
            vDSP_Length(frameCount),
            vDSP_Length(Self.frequencyBinCount),
            vDSP_Length(Self.fftSize)
        )

        var power = [Float](repeating: 0, count: spectrumCount)
        real.withUnsafeMutableBufferPointer { realBuffer in
            imaginary.withUnsafeMutableBufferPointer { imaginaryBuffer in
                var split = DSPSplitComplex(
                    realp: realBuffer.baseAddress!,
                    imagp: imaginaryBuffer.baseAddress!
                )
                vDSP_zvmags(
                    &split,
                    1,
                    &power,
                    1,
                    vDSP_Length(spectrumCount)
                )
            }
        }

        var mel = [Float](
            repeating: 0,
            count: frameCount * melBinCount
        )
        vDSP_mmul(
            power,
            1,
            melFilterbank,
            1,
            &mel,
            1,
            vDSP_Length(frameCount),
            vDSP_Length(melBinCount),
            vDSP_Length(Self.frequencyBinCount)
        )

        var lowerBound: Float = 1e-10
        var upperBound = Float.greatestFiniteMagnitude
        vDSP_vclip(
            mel,
            1,
            &lowerBound,
            &upperBound,
            &mel,
            1,
            vDSP_Length(mel.count)
        )
        var count = Int32(mel.count)
        vvlog10f(&mel, mel, &count)
        var decibelScale: Float = 10
        vDSP_vsmul(
            mel,
            1,
            &decibelScale,
            &mel,
            1,
            vDSP_Length(mel.count)
        )

        var peak = -Float.infinity
        vDSP_maxv(mel, 1, &peak, vDSP_Length(mel.count))
        var dynamicRangeFloor = peak - 80
        vDSP_vclip(
            mel,
            1,
            &dynamicRangeFloor,
            &upperBound,
            &mel,
            1,
            vDSP_Length(mel.count)
        )

        return SpeechBrainLogMelFeatures(
            values: mel,
            frameCount: frameCount,
            melBinCount: melBinCount
        )
    }

    private static func makeFilterbank(melBinCount: Int) -> [Float] {
        func hertzToMel(_ hertz: Float) -> Float {
            2_595 * log10(1 + hertz / 700)
        }

        func melToHertz(_ mel: Float) -> Float {
            700 * (pow(10, mel / 2_595) - 1)
        }

        let melMinimum = hertzToMel(0)
        let melMaximum = hertzToMel(Float(sampleRate) / 2)
        let points = (0..<(melBinCount + 2)).map { index -> Float in
            let mel = melMinimum
                + Float(index) * (melMaximum - melMinimum)
                    / Float(melBinCount + 1)
            return melToHertz(mel)
        }
        let centers = Array(points[1...melBinCount])
        // SpeechBrain uses each filter's lower band on both sides. This is
        // intentionally different from the usual asymmetric HTK triangle.
        let bands = (0..<melBinCount).map { points[$0 + 1] - points[$0] }

        var filters = [Float](
            repeating: 0,
            count: frequencyBinCount * melBinCount
        )
        for bin in 0..<frequencyBinCount {
            let frequency = Float(bin) * Float(sampleRate / 2)
                / Float(frequencyBinCount - 1)
            for mel in 0..<melBinCount {
                let slope = (frequency - centers[mel]) / bands[mel]
                filters[bin * melBinCount + mel] = max(
                    0,
                    min(slope + 1, -slope + 1)
                )
            }
        }
        return filters
    }
}
