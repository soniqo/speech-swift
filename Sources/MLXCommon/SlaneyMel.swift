import Foundation
import MLX
import MLXFFT

/// Parameters for a librosa-style slaney mel front-end.
public struct SlaneyMelConfig: Sendable {
    public var sampleRate: Int
    public var nFft: Int
    public var hop: Int
    public var win: Int
    public var nMels: Int
    public var fmin: Float
    public var fmax: Float
    /// Exponent applied to the STFT magnitude before the mel projection
    /// (1.0 = magnitude, 2.0 = power spectrogram).
    public var power: Float
    /// If true, apply `log(max(mel, logFloor))` after the mel projection.
    public var logMel: Bool
    public var logFloor: Float
    /// If true, reflect-pad the signal by `nFft/2` each side (librosa `center=True`).
    public var centerPad: Bool
    /// If true, use PyTorch's default periodic Hann window. The default remains
    /// symmetric for existing mel front-ends that were validated that way.
    public var periodicHann: Bool

    public init(
        sampleRate: Int, nFft: Int, hop: Int, win: Int, nMels: Int,
        fmin: Float, fmax: Float, power: Float = 1.0,
        logMel: Bool = false, logFloor: Float = 1e-5, centerPad: Bool = true,
        periodicHann: Bool = false
    ) {
        self.sampleRate = sampleRate
        self.nFft = nFft
        self.hop = hop
        self.win = win
        self.nMels = nMels
        self.fmin = fmin
        self.fmax = fmax
        self.power = power
        self.logMel = logMel
        self.logFloor = logFloor
        self.centerPad = centerPad
        self.periodicHann = periodicHann
    }
}

/// Shared librosa-`norm='slaney'` mel spectrogram. The windowed frames are built
/// on the host and the STFT runs on the MLX backend via `rfft`. The reflect
/// padding excludes the boundary sample and the filterbank matches
/// `librosa.filters.mel(norm='slaney', htk=False)`, so it reproduces a
/// `stft(center=True, pad_mode='reflect')` + slaney `mel_filters` exactly.
///
/// Lifted from the validated FlashSR mel and parameterized so the VoiceEncoder
/// (40-mel, power, no-log) and other models share one implementation.
public enum SlaneyMel {

    /// Returns the mel spectrogram as `(frames, nMels)` MLX float32.
    public static func melSpec(samples: [Float], config c: SlaneyMelConfig) -> MLXArray {
        melSpecMLX(samples: samples, config: c)
    }

    /// Returns the mel spectrogram as `(frames, nMels)` MLX float32 (preferred —
    /// the STFT runs on the MLX backend via `rfft`, matching the reference implementation exactly).
    public static func melSpecMLX(samples: [Float], config c: SlaneyMelConfig) -> MLXArray {
        let nBins = c.nFft / 2 + 1
        let sig = c.centerPad ? reflectPad1D(samples, pad: c.nFft / 2) : samples
        let frames = max(0, (sig.count - c.nFft) / c.hop + 1)

        // Windowed frames [frames, nFft] (built on host; one rfft on the backend).
        let window = hannWindow(length: c.win, nFft: c.nFft, periodic: c.periodicHann)
        var framed = [Float](repeating: 0, count: frames * c.nFft)
        for t in 0 ..< frames {
            let start = t * c.hop
            for i in 0 ..< c.nFft { framed[t * c.nFft + i] = sig[start + i] * window[i] }
        }
        let framedMx = MLXArray(framed, [frames, c.nFft])

        // STFT magnitude via MLX rfft (arbitrary nFft, matches mx.fft.rfft).
        let spec = rfft(framedMx, axis: -1)              // [frames, nBins] complex
        var mag = abs(spec)                              // [frames, nBins] real
        if c.power != 1.0 { mag = MLX.pow(mag, MLXArray(c.power)) }

        // Slaney mel projection: [frames, nBins] @ [nBins, nMels].
        let fbFloats = mlFilterbankSlaney(
            sr: c.sampleRate, nFft: c.nFft, nMels: c.nMels, fmin: c.fmin, fmax: c.fmax)
        let fb = MLXArray(fbFloats, [c.nMels, nBins])
        var mel = matmul(mag, fb.transposed())           // [frames, nMels]
        if c.logMel { mel = MLX.log(maximum(mel, MLXArray(c.logFloor))) }
        return mel
    }

    // MARK: - Internals (lifted from FlashSR/Mel.swift)

    private static func hannWindow(length: Int, nFft: Int, periodic: Bool) -> [Float] {
        var w = [Float](repeating: 0, count: nFft)
        let denom = Float(max(periodic ? length : length - 1, 1))
        for i in 0 ..< length {
            w[i] = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / denom))
        }
        // If win < nFft the window is zero-padded to nFft.
        return w
    }

    /// numpy `mode='reflect'` padding (excludes the boundary sample).
    private static func reflectPad1D(_ row: [Float], pad: Int) -> [Float] {
        var out = [Float](repeating: 0, count: pad + row.count + pad)
        for i in 0 ..< row.count { out[pad + i] = row[i] }
        for i in 0 ..< pad { out[i] = row[pad - i] }
        let last = row.count - 1
        for i in 0 ..< pad { out[pad + row.count + i] = row[last - 1 - i] }
        return out
    }

    /// Slaney-normalised mel filterbank, row-major `(n_mels, n_fft/2 + 1)`.
    private static func mlFilterbankSlaney(
        sr: Int, nFft: Int, nMels: Int, fmin: Float, fmax: Float
    ) -> [Float] {
        let nBins = nFft / 2 + 1
        let f = Float(sr)
        var fftFreqs = [Float](repeating: 0, count: nBins)
        for i in 0 ..< nBins { fftFreqs[i] = Float(i) * f / Float(nFft) }
        let melMin = hzToMelSlaney(fmin)
        let melMax = hzToMelSlaney(fmax)
        var melPoints = [Float](repeating: 0, count: nMels + 2)
        for i in 0 ..< (nMels + 2) {
            let frac = Float(i) / Float(nMels + 1)
            melPoints[i] = melMin + frac * (melMax - melMin)
        }
        let hzPoints = melPoints.map { melToHzSlaney($0) }
        var fb = [Float](repeating: 0, count: nMels * nBins)
        for m in 0 ..< nMels {
            let lower = hzPoints[m]
            let center = hzPoints[m + 1]
            let upper = hzPoints[m + 2]
            for k in 0 ..< nBins {
                let freq = fftFreqs[k]
                if freq < lower || freq > upper { continue }
                if freq <= center {
                    fb[m * nBins + k] = (freq - lower) / max(center - lower, 1e-12)
                } else {
                    fb[m * nBins + k] = (upper - freq) / max(upper - center, 1e-12)
                }
            }
            let enorm: Float = 2.0 / max(upper - lower, 1e-12)
            for k in 0 ..< nBins { fb[m * nBins + k] *= enorm }
        }
        return fb
    }

    private static func hzToMelSlaney(_ hz: Float) -> Float {
        let fSpacing: Float = 200.0 / 3.0
        let minLogHz: Float = 1000.0
        let minLogMel: Float = minLogHz / fSpacing
        let logstep: Float = log(Float(6.4)) / Float(27.0)
        return hz >= minLogHz ? minLogMel + log(hz / minLogHz) / logstep : hz / fSpacing
    }

    private static func melToHzSlaney(_ mel: Float) -> Float {
        let fSpacing: Float = 200.0 / 3.0
        let minLogHz: Float = 1000.0
        let minLogMel: Float = minLogHz / fSpacing
        let logstep: Float = log(Float(6.4)) / Float(27.0)
        return mel >= minLogMel ? minLogHz * exp(logstep * (mel - minLogMel)) : fSpacing * mel
    }
}
