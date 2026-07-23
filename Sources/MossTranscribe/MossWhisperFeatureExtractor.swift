import Accelerate
import Foundation

/// A flat row-major `[melBins, timeFrames]` log-mel tensor.
public struct MossLogMelFeatures: Sendable, Equatable {
    public let data: [Float]
    public let melBins: Int
    public let timeFrames: Int

    public init(data: [Float], melBins: Int, timeFrames: Int) {
        self.data = data
        self.melBins = melBins
        self.timeFrames = timeFrames
    }
}

/// MLX-free Whisper frontend matching the published MOSS processor.
///
/// Unlike the older Qwen3-ASR frontend, this uses the model's actual
/// 400-point DFT rather than zero-padding to 512 bins. That distinction is
/// required because the published MOSS encoder was validated against
/// Hugging Face's 201-bin `WhisperFeatureExtractor` output.
public final class MossWhisperFeatureExtractor: @unchecked Sendable {
    public static let sampleRate = 16_000
    public static let fftSize = 400
    public static let hopLength = 160
    public static let melBins = 80
    public static let chunkSamples = 480_000
    public static let timeFrames = 3_000
    public static let encoderStrideSamples = 1_280

    private let hannWindow: [Float]
    /// Row-major `[fftSize, frequencyBins]`.
    private let dftRealBasis: [Float]
    /// Row-major `[fftSize, frequencyBins]`.
    private let dftImaginaryBasis: [Float]
    /// Row-major `[frequencyBins, melBins]`.
    private let melFilterbank: [Float]

    public init() {
        hannWindow = (0..<Self.fftSize).map { index in
            0.5 * (1 - cos(2 * Float.pi * Float(index) / Float(Self.fftSize)))
        }
        let frequencyBins = Self.fftSize / 2 + 1
        var realBasis = [Float](
            repeating: 0,
            count: Self.fftSize * frequencyBins
        )
        var imaginaryBasis = realBasis
        for sample in 0..<Self.fftSize {
            for bin in 0..<frequencyBins {
                let angle =
                    -2 * Double.pi * Double(sample * bin)
                    / Double(Self.fftSize)
                realBasis[sample * frequencyBins + bin] =
                    Float(cos(angle))
                imaginaryBasis[sample * frequencyBins + bin] =
                    Float(sin(angle))
            }
        }
        dftRealBasis = realBasis
        dftImaginaryBasis = imaginaryBasis
        melFilterbank = Self.makeMelFilterbank()
    }

    /// Number of decoder audio tokens produced for an unpadded waveform.
    public static func audioTokenCount(sampleCount: Int) -> Int {
        guard sampleCount > 0 else { return 0 }
        return (sampleCount - 1) / encoderStrideSamples + 1
    }

    /// Extract exactly `[80, 3000]` features from one at-most-30-second chunk.
    ///
    /// The waveform is right-padded to 30 seconds before the centered STFT,
    /// matching `MossTranscribeDiarizeProcessor._pad_or_trim_audio`.
    public func extractPaddedChunk(_ audio: [Float]) throws -> MossLogMelFeatures {
        guard !audio.isEmpty else {
            throw MossTranscribeError.invalidAudio("the waveform is empty")
        }
        guard audio.count <= Self.chunkSamples else {
            throw MossTranscribeError.invalidAudio(
                "one encoder chunk may contain at most \(Self.chunkSamples) samples"
            )
        }

        var fixedAudio = [Float](repeating: 0, count: Self.chunkSamples)
        fixedAudio.withUnsafeMutableBufferPointer { destination in
            audio.withUnsafeBufferPointer { source in
                destination.baseAddress?.update(
                    from: source.baseAddress!,
                    count: source.count
                )
            }
        }

        let pad = Self.fftSize / 2
        var centered = [Float](
            repeating: 0,
            count: fixedAudio.count + 2 * pad
        )
        for index in 0..<pad {
            centered[index] = fixedAudio[pad - index]
        }
        centered.replaceSubrange(
            pad..<(pad + fixedAudio.count),
            with: fixedAudio
        )
        for index in 0..<pad {
            centered[pad + fixedAudio.count + index] =
                fixedAudio[fixedAudio.count - 2 - index]
        }

        // Center padding produces 3001 frames; Whisper drops the last before
        // dynamic-range clipping and returns 3000.
        let stftFrames =
            (centered.count - Self.fftSize) / Self.hopLength + 1
        let frequencyBins = Self.fftSize / 2 + 1

        // Accelerate's DFT setup rejects length 400. A batched matrix-form
        // DFT keeps the exact frequency grid while BLAS evaluates every
        // frame together efficiently.
        var windowedFrames = [Float](
            repeating: 0,
            count: stftFrames * Self.fftSize
        )
        centered.withUnsafeBufferPointer { source in
            windowedFrames.withUnsafeMutableBufferPointer { destination in
                for frame in 0..<stftFrames {
                    vDSP_vmul(
                        source.baseAddress! + frame * Self.hopLength,
                        1,
                        hannWindow,
                        1,
                        destination.baseAddress! + frame * Self.fftSize,
                        1,
                        vDSP_Length(Self.fftSize)
                    )
                }
            }
        }

        let spectrumCount = stftFrames * frequencyBins
        var outputReal = [Float](repeating: 0, count: spectrumCount)
        var outputImaginary = [Float](repeating: 0, count: spectrumCount)
        vDSP_mmul(
            windowedFrames,
            1,
            dftRealBasis,
            1,
            &outputReal,
            1,
            vDSP_Length(stftFrames),
            vDSP_Length(frequencyBins),
            vDSP_Length(Self.fftSize)
        )
        vDSP_mmul(
            windowedFrames,
            1,
            dftImaginaryBasis,
            1,
            &outputImaginary,
            1,
            vDSP_Length(stftFrames),
            vDSP_Length(frequencyBins),
            vDSP_Length(Self.fftSize)
        )
        var power = [Float](repeating: 0, count: spectrumCount)
        outputReal.withUnsafeMutableBufferPointer { real in
            outputImaginary.withUnsafeMutableBufferPointer { imaginary in
                var split = DSPSplitComplex(
                    realp: real.baseAddress!,
                    imagp: imaginary.baseAddress!
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

        var melByFrame = [Float](
            repeating: 0,
            count: stftFrames * Self.melBins
        )
        vDSP_mmul(
            power,
            1,
            melFilterbank,
            1,
            &melByFrame,
            1,
            vDSP_Length(stftFrames),
            vDSP_Length(Self.melBins),
            vDSP_Length(frequencyBins)
        )

        let trimmedCount = Self.timeFrames * Self.melBins
        var minimum: Float = 1e-10
        var maximum = Float.greatestFiniteMagnitude
        vDSP_vclip(
            melByFrame,
            1,
            &minimum,
            &maximum,
            &melByFrame,
            1,
            vDSP_Length(trimmedCount)
        )
        var logCount = Int32(trimmedCount)
        vvlog10f(&melByFrame, melByFrame, &logCount)

        var peak = -Float.infinity
        vDSP_maxv(
            melByFrame,
            1,
            &peak,
            vDSP_Length(trimmedCount)
        )
        var floor = peak - 8
        vDSP_vclip(
            melByFrame,
            1,
            &floor,
            &maximum,
            &melByFrame,
            1,
            vDSP_Length(trimmedCount)
        )
        var scale: Float = 0.25
        var offset: Float = 1
        vDSP_vsmsa(
            melByFrame,
            1,
            &scale,
            &offset,
            &melByFrame,
            1,
            vDSP_Length(trimmedCount)
        )

        var melMajor = [Float](repeating: 0, count: trimmedCount)
        for frame in 0..<Self.timeFrames {
            let sourceBase = frame * Self.melBins
            for mel in 0..<Self.melBins {
                melMajor[mel * Self.timeFrames + frame] =
                    melByFrame[sourceBase + mel]
            }
        }
        return MossLogMelFeatures(
            data: melMajor,
            melBins: Self.melBins,
            timeFrames: Self.timeFrames
        )
    }

    private static func makeMelFilterbank() -> [Float] {
        let frequencyBins = fftSize / 2 + 1
        let minimumLogHertz = 1_000.0
        let minimumLogMel = 15.0
        let logStep = 27.0 / log(6.4)

        func hertzToMel(_ hertz: Double) -> Double {
            if hertz < minimumLogHertz {
                return 3 * hertz / 200
            }
            return minimumLogMel
                + log(hertz / minimumLogHertz) * logStep
        }

        func melToHertz(_ mel: Double) -> Double {
            if mel < minimumLogMel {
                return 200 * mel / 3
            }
            return minimumLogHertz
                * exp((mel - minimumLogMel) / logStep)
        }

        let melMinimum = hertzToMel(0)
        let melMaximum = hertzToMel(Double(sampleRate) / 2)
        let points = (0..<(melBins + 2)).map { index -> Double in
            let mel = melMinimum
                + Double(index) * (melMaximum - melMinimum)
                / Double(melBins + 1)
            return melToHertz(mel)
        }
        let differences = zip(points, points.dropFirst()).map { $1 - $0 }

        var filters = [Float](
            repeating: 0,
            count: frequencyBins * melBins
        )
        for bin in 0..<frequencyBins {
            let hertz = Double(bin) * Double(sampleRate) / Double(fftSize)
            for mel in 0..<melBins {
                let rising = (hertz - points[mel]) / differences[mel]
                let falling = (points[mel + 2] - hertz) / differences[mel + 1]
                let triangle = max(0, min(rising, falling))
                let slaney = 2 / (points[mel + 2] - points[mel])
                filters[bin * melBins + mel] = Float(triangle * slaney)
            }
        }
        return filters
    }
}
