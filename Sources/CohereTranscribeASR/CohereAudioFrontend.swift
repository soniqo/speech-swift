import AudioCommon
import Foundation
import MLX
import MLXFFT

/// Cohere's exact 16 kHz frontend: pre-emphasis, constant-centred 512-point
/// STFT, 128-bin Slaney filterbank, natural log, and per-utterance MVN.
public final class CohereAudioFrontend: @unchecked Sendable {
    public let sampleRate: Int
    public let nFFT: Int
    public let winLength: Int
    public let hopLength: Int
    public let melBins: Int

    private let window: MLXArray
    private let filters: MLXArray

    public init(
        sampleRate: Int = 16_000,
        nFFT: Int = 512,
        winLength: Int = 400,
        hopLength: Int = 160,
        melBins: Int = 128
    ) {
        self.sampleRate = sampleRate
        self.nFFT = nFFT
        self.winLength = winLength
        self.hopLength = hopLength
        self.melBins = melBins
        self.window = Self.centeredHann(nFFT: nFFT, winLength: winLength)
        self.filters = Self.slaneyFilters(
            sampleRate: sampleRate, nFFT: nFFT, melBins: melBins)
    }

    /// Returns `[1, melBins, frames]` float32 features.
    public func process(_ audio: [Float], inputSampleRate: Int) -> MLXArray {
        let samples = inputSampleRate == sampleRate
            ? audio
            : AudioFileLoader.resample(audio, from: inputSampleRate, to: sampleRate)
        return extract(samples)
    }

    public func extract(_ samples: [Float]) -> MLXArray {
        let signal = MLXArray(Self.preEmphasized(samples)).asType(.float32)
        let pad = nFFT / 2
        let padded = MLX.padded(signal, widths: [IntOrPair((pad, pad))])
        let frameCount = max(1, 1 + (padded.shape[0] - nFFT) / hopLength)
        let frames = asStrided(
            padded,
            [frameCount, nFFT],
            strides: [hopLength, 1],
            offset: 0)
        let spectrum = rfft(frames * window.expandedDimensions(axis: 0), axis: -1)
        let power = MLX.abs(spectrum).square()
        var mel = MLX.matmul(power, filters)
        mel = MLX.log(mel + MLXArray(pow(Float(2), -24)))
        mel = mel.transposed(1, 0).expandedDimensions(axis: 0)
        let validFrameCount = min(
            Self.normalizationFrameCount(sampleCount: samples.count, hopLength: hopLength),
            mel.dim(2))
        guard validFrameCount > 0 else {
            return MLXArray.zeros(mel.shape, dtype: mel.dtype)
        }
        let valid = mel[0..., 0..., 0..<validFrameCount]
        let mean = valid.mean(axes: [2], keepDims: true)
        let variance: MLXArray
        if validFrameCount > 1 {
            let centered = valid - mean
            variance = (centered * centered).sum(axes: [2], keepDims: true)
                / Float(validFrameCount - 1)
        } else {
            variance = MLXArray.zeros(mean.shape, dtype: mean.dtype)
        }
        mel = (mel - mean) / (MLX.sqrt(variance) + 1e-5)
        if validFrameCount < mel.dim(2) {
            let mask = (MLXArray(0..<mel.dim(2)) .< validFrameCount)
                .reshaped(1, 1, mel.dim(2))
            mel = MLX.where(mask, mel, MLXArray.zeros(mel.shape, dtype: mel.dtype))
        }
        return mel
    }

    static func normalizationFrameCount(sampleCount: Int, hopLength: Int = 160) -> Int {
        max(0, sampleCount) / max(1, hopLength)
    }

    static func preEmphasized(_ samples: [Float], factor: Float = 0.97) -> [Float] {
        guard let first = samples.first else { return [0] }
        var output = [Float](repeating: 0, count: samples.count)
        output[0] = first
        for index in 1..<samples.count {
            output[index] = samples[index] - factor * samples[index - 1]
        }
        return output
    }

    private static func centeredHann(nFFT: Int, winLength: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: nFFT)
        let offset = max(0, (nFFT - winLength) / 2)
        let denominator = Float(max(1, winLength - 1))
        for index in 0..<min(winLength, nFFT) {
            values[offset + index] = 0.5 * (1 - cos(2 * .pi * Float(index) / denominator))
        }
        return MLXArray(values)
    }

    private static func slaneyFilters(sampleRate: Int, nFFT: Int, melBins: Int) -> MLXArray {
        let frequencyBins = nFFT / 2 + 1
        let minMel = hzToMel(0)
        let maxMel = hzToMel(Float(sampleRate) / 2)
        let melPoints = (0..<(melBins + 2)).map {
            minMel + Float($0) * (maxMel - minMel) / Float(melBins + 1)
        }
        let hzPoints = melPoints.map(melToHz)
        var result = [Float](repeating: 0, count: frequencyBins * melBins)
        for bin in 0..<frequencyBins {
            let frequency = Float(bin * sampleRate) / Float(nFFT)
            for mel in 0..<melBins {
                let lower = hzPoints[mel]
                let center = hzPoints[mel + 1]
                let upper = hzPoints[mel + 2]
                let rising = (frequency - lower) / max(center - lower, 1e-12)
                let falling = (upper - frequency) / max(upper - center, 1e-12)
                let triangle = max(0, min(rising, falling))
                result[bin * melBins + mel] = triangle * 2 / max(upper - lower, 1e-12)
            }
        }
        return MLXArray(result, [frequencyBins, melBins])
    }

    private static func hzToMel(_ hz: Float) -> Float {
        let spacing: Float = 200 / 3
        let minLogHz: Float = 1_000
        let minLogMel = minLogHz / spacing
        let logStep = log(Float(6.4)) / 27
        return hz >= minLogHz ? minLogMel + log(hz / minLogHz) / logStep : hz / spacing
    }

    private static func melToHz(_ mel: Float) -> Float {
        let spacing: Float = 200 / 3
        let minLogHz: Float = 1_000
        let minLogMel = minLogHz / spacing
        let logStep = log(Float(6.4)) / 27
        return mel >= minLogMel ? minLogHz * exp(logStep * (mel - minLogMel)) : spacing * mel
    }
}
