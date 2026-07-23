import AudioCommon
import Foundation
import MLX
import MLXFFT

public struct VoxtralAudioFeatures {
    public let values: MLXArray
    public let chunkCount: Int
    public let audioTokenCount: Int
}

/// Whisper-compatible Voxtral preprocessing with mandatory 30-second padding.
public final class VoxtralAudioFrontend: @unchecked Sendable {
    public static let sampleRate = 16_000
    public static let chunkSamples = 480_000
    public static let framesPerChunk = 3_000
    public static let audioTokensPerChunk = 375

    public let melBins: Int
    private let nFFT = 400
    private let hopLength = 160
    private let window: MLXArray
    private let filters: MLXArray

    public init(melBins: Int = 128) {
        self.melBins = melBins
        self.window = Self.periodicHann(length: nFFT)
        self.filters = Self.slaneyFilters(
            sampleRate: Self.sampleRate,
            nFFT: nFFT,
            melBins: melBins)
    }

    public static func paddedChunkCount(forSampleCount sampleCount: Int) -> Int {
        max(1, Int(ceil(Double(max(0, sampleCount)) / Double(chunkSamples))))
    }

    public static func audioTokenCount(forSampleCount sampleCount: Int) -> Int {
        paddedChunkCount(forSampleCount: sampleCount) * audioTokensPerChunk
    }

    public func process(
        _ audio: [Float],
        inputSampleRate: Int
    ) -> VoxtralAudioFeatures {
        var samples = inputSampleRate == Self.sampleRate
            ? audio
            : AudioFileLoader.resample(audio, from: inputSampleRate, to: Self.sampleRate)
        if samples.isEmpty { samples = [0] }
        let chunks = Self.paddedChunkCount(forSampleCount: samples.count)
        samples.append(contentsOf: repeatElement(
            Float(0), count: chunks * Self.chunkSamples - samples.count))

        let padded = Self.reflectPadded(samples, by: nFFT / 2)
        let paddedArray = MLXArray(padded).asType(.float32)
        let frameCount = 1 + (padded.count - nFFT) / hopLength
        let frames = asStrided(
            paddedArray,
            [frameCount, nFFT],
            strides: [hopLength, 1],
            offset: 0)
        let spectrum = rfft(frames * window.expandedDimensions(axis: 0), axis: -1)
        var power = MLX.abs(spectrum).square()
        // WhisperFeatureExtractor discards the final centred frame.
        power = power[0..<(power.shape[0] - 1), 0...]
        var mel = MLX.matmul(power, filters)
        mel = MLX.maximum(mel, MLXArray(Float(1e-10)))
        var logMel = MLX.log10(mel)
        logMel = MLX.maximum(logMel, logMel.max() - 8)
        logMel = (logMel + 4) / 4
        let features = logMel.reshaped(chunks, Self.framesPerChunk, melBins)
        return VoxtralAudioFeatures(
            values: features,
            chunkCount: chunks,
            audioTokenCount: chunks * Self.audioTokensPerChunk)
    }

    private static func periodicHann(length: Int) -> MLXArray {
        MLXArray((0..<length).map {
            0.5 * (1 - cos(2 * Float.pi * Float($0) / Float(length)))
        })
    }

    private static func reflectPadded(_ samples: [Float], by pad: Int) -> [Float] {
        guard pad > 0 else { return samples }
        func reflectedIndex(_ index: Int) -> Int {
            guard samples.count > 1 else { return 0 }
            var value = index
            while value < 0 || value >= samples.count {
                value = value < 0 ? -value : 2 * samples.count - value - 2
            }
            return value
        }
        return (-pad..<(samples.count + pad)).map { samples[reflectedIndex($0)] }
    }

    private static func slaneyFilters(sampleRate: Int, nFFT: Int, melBins: Int) -> MLXArray {
        let frequencyBins = nFFT / 2 + 1
        let minMel = hzToMel(0)
        let maxMel = hzToMel(Float(sampleRate) / 2)
        let points = (0..<(melBins + 2)).map {
            melToHz(minMel + Float($0) * (maxMel - minMel) / Float(melBins + 1))
        }
        var result = [Float](repeating: 0, count: frequencyBins * melBins)
        for bin in 0..<frequencyBins {
            let frequency = Float(bin * sampleRate) / Float(nFFT)
            for mel in 0..<melBins {
                let lower = points[mel]
                let center = points[mel + 1]
                let upper = points[mel + 2]
                let rising = (frequency - lower) / max(center - lower, 1e-12)
                let falling = (upper - frequency) / max(upper - center, 1e-12)
                result[bin * melBins + mel] = max(0, min(rising, falling))
                    * 2 / max(upper - lower, 1e-12)
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
