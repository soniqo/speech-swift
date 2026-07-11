import Accelerate
import Foundation
import MLX

/// HiFi-GAN-style log-mel preprocessing for FlashSR.
///
/// Matches the upstream `mel.py` (`get_hifigan_mel_spec`):
///   - Reflect-pad audio by `(n_fft - hop) // 2` on each side (HiFi-GAN convention,
///     equivalent to librosa.stft with `center=False, pad_mode='reflect'`).
///   - STFT with Hann window, magnitude (|S|).
///   - Slaney-normalised mel filterbank applied to magnitude.
///   - `log(clamp(mel, 1e-5))`.
///
/// Output shape: `(B, n_mels, T_frames)` as an MLX float32 array. The caller
/// transposes to NHWC for the VAE.
public enum FlashSRMelPreprocessor {

    /// Convert `audio` (B, samples) to log-mel `(B, n_mels, frames)`.
    public static func melSpec(
        audio: MLXArray,
        config: FlashSRMelConfig
    ) -> MLXArray {
        // Pull samples to host. Numerically lighter to do STFT + filterbank
        // on CPU via Accelerate than to compose it from MLX rfft + matmul
        // for an offline pre-processing step (the model is the hot path).
        let arr = audio.asArray(Float.self)
        let B = audio.dim(0)
        let T = audio.dim(1)

        // Reflect pad: (n_fft - hop) // 2 on each side
        let pad = (config.nFft - config.hop) / 2
        let paddedLen = T + 2 * pad

        // Hann window (length n_fft)
        let window = hannWindow(length: config.nFft)
        // Slaney mel filterbank (n_mels × (n_fft/2 + 1))
        let filterbank = mlFilterbankSlaney(
            sr: config.sr, nFft: config.nFft, nMels: config.nMels,
            fmin: config.fmin, fmax: config.fmax)
        let nBins = config.nFft / 2 + 1

        // For each batch row: pad → STFT → magnitude → mel-project → log
        var allMels: [Float] = []
        var frameCount = 0
        for b in 0..<B {
            var row = Array(arr[(b * T)..<((b + 1) * T)])
            // Reflect pad
            let padded = reflectPad1D(row: &row, padLeft: pad, padRight: pad)
            let stftMag = stftMagnitude(
                samples: padded, nFft: config.nFft, hop: config.hop, window: window)
            // stftMag: [frames * nBins] row-major (frames, nBins)
            let frames = stftMag.count / nBins
            frameCount = frames
            // mel[m, t] = Σ_f filterbank[m, f] * stftMag[t, f]
            // Resulting mel laid out as [frames, n_mels] then we'll transpose later.
            var melRow = [Float](repeating: 0, count: frames * config.nMels)
            for t in 0..<frames {
                for m in 0..<config.nMels {
                    var s: Float = 0
                    for f in 0..<nBins {
                        s += filterbank[m * nBins + f] * stftMag[t * nBins + f]
                    }
                    // Clamp + log
                    melRow[t * config.nMels + m] = log(max(s, 1e-5))
                }
            }
            allMels.append(contentsOf: melRow)
        }

        // Result laid out as B × frames × n_mels. Caller wants (B, n_mels, frames).
        // Transpose now.
        var out = [Float](repeating: 0, count: B * config.nMels * frameCount)
        for b in 0..<B {
            for t in 0..<frameCount {
                for m in 0..<config.nMels {
                    out[(b * config.nMels + m) * frameCount + t] =
                        allMels[(b * frameCount + t) * config.nMels + m]
                }
            }
        }
        let _ = paddedLen   // silence unused
        return MLXArray(out, [B, config.nMels, frameCount])
    }

    // MARK: - Internals

    private static func hannWindow(length: Int) -> [Float] {
        var w = [Float](repeating: 0, count: length)
        for i in 0..<length {
            // Periodic Hann (matches torch.hann_window default).
            w[i] = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(length)))
        }
        return w
    }

    private static func reflectPad1D(row: inout [Float], padLeft: Int, padRight: Int) -> [Float] {
        var out = [Float](repeating: 0, count: padLeft + row.count + padRight)
        for i in 0..<row.count { out[padLeft + i] = row[i] }
        // numpy mode='reflect' excludes the boundary sample.
        for i in 0..<padLeft { out[i] = row[padLeft - i] }
        let last = row.count - 1
        for i in 0..<padRight { out[padLeft + row.count + i] = row[last - 1 - i] }
        return out
    }

    /// vDSP-based STFT magnitude. Returns row-major `(frames, n_bins)`.
    private static func stftMagnitude(
        samples: [Float], nFft: Int, hop: Int, window: [Float]
    ) -> [Float] {
        let nBins = nFft / 2 + 1
        let frames = max(0, (samples.count - nFft) / hop + 1)
        let fftSetup = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(nFft), .FORWARD)!
        defer { vDSP_DFT_DestroySetup(fftSetup) }

        var output = [Float](repeating: 0, count: frames * nBins)
        var realIn = [Float](repeating: 0, count: nFft)
        var imagIn = [Float](repeating: 0, count: nFft)
        var realOut = [Float](repeating: 0, count: nFft)
        var imagOut = [Float](repeating: 0, count: nFft)

        for t in 0..<frames {
            let start = t * hop
            for i in 0..<nFft {
                realIn[i] = samples[start + i] * window[i]
                imagIn[i] = 0
            }
            vDSP_DFT_Execute(
                fftSetup,
                realIn, imagIn,
                &realOut, &imagOut)
            for b in 0..<nBins {
                let re = realOut[b]
                let im = imagOut[b]
                output[t * nBins + b] = (re * re + im * im).squareRoot()
            }
        }
        return output
    }

    /// Slaney-normalised mel filterbank, matching `librosa.filters.mel(norm='slaney')`.
    /// Returns row-major `(n_mels, n_fft/2 + 1)`.
    private static func mlFilterbankSlaney(
        sr: Int, nFft: Int, nMels: Int, fmin: Float, fmax: Float
    ) -> [Float] {
        let nBins = nFft / 2 + 1
        let f = Float(sr)
        // FFT bin frequencies
        var fftFreqs = [Float](repeating: 0, count: nBins)
        for i in 0..<nBins { fftFreqs[i] = Float(i) * f / Float(nFft) }
        // Mel filterbank centers: convert min/max Hz → mel, evenly space, back to Hz
        let melMin = hzToMelSlaney(fmin)
        let melMax = hzToMelSlaney(fmax)
        var melPoints = [Float](repeating: 0, count: nMels + 2)
        for i in 0..<(nMels + 2) {
            let frac = Float(i) / Float(nMels + 1)
            melPoints[i] = melMin + frac * (melMax - melMin)
        }
        var hzPoints = melPoints.map { melToHzSlaney($0) }
        // Build triangular filters
        var fb = [Float](repeating: 0, count: nMels * nBins)
        for m in 0..<nMels {
            let lower = hzPoints[m]
            let center = hzPoints[m + 1]
            let upper = hzPoints[m + 2]
            for k in 0..<nBins {
                let freq = fftFreqs[k]
                if freq < lower || freq > upper { continue }
                if freq <= center {
                    fb[m * nBins + k] = (freq - lower) / max(center - lower, 1e-12)
                } else {
                    fb[m * nBins + k] = (upper - freq) / max(upper - center, 1e-12)
                }
            }
            // Slaney normalisation: 2 / (upper - lower)
            let enorm: Float = 2.0 / max(upper - lower, 1e-12)
            for k in 0..<nBins {
                fb[m * nBins + k] *= enorm
            }
        }
        _ = hzPoints
        return fb
    }

    private static func hzToMelSlaney(_ hz: Float) -> Float {
        // Slaney/librosa: linear below 1000 Hz, log above.
        let fSpacing: Float = 200.0 / 3.0
        let minLogHz: Float = 1000.0
        let minLogMel: Float = minLogHz / fSpacing
        let logstep: Float = log(Float(6.4)) / Float(27.0)
        if hz >= minLogHz {
            return minLogMel + log(hz / minLogHz) / logstep
        } else {
            return hz / fSpacing
        }
    }

    private static func melToHzSlaney(_ mel: Float) -> Float {
        let fSpacing: Float = 200.0 / 3.0
        let minLogHz: Float = 1000.0
        let minLogMel: Float = minLogHz / fSpacing
        let logstep: Float = log(Float(6.4)) / Float(27.0)
        if mel >= minLogMel {
            return minLogHz * exp(logstep * (mel - minLogMel))
        } else {
            return fSpacing * mel
        }
    }
}
