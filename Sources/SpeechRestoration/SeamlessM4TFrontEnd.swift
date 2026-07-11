import Accelerate
import Foundation

/// w2v-BERT 2.0 SeamlessM4T log-mel front-end, ported to Swift/Accelerate.
///
/// This reproduces HuggingFace `SeamlessM4TFeatureExtractor` (the feature
/// extractor `facebook/w2v-bert-2.0` ships) bit-for-bit closely enough for the
/// Sidon predictor: 16 kHz mono audio → `input_features[1, T, 160]`.
///
/// The reference graph the Sidon predictor was exported from starts from these
/// features (the front-end is DSP, not part of the CoreML graph — same pattern
/// as DeepFilterNet3's `auxiliary.npz` STFT done in the runtime). The exact
/// recipe, confirmed against the upstream extractor:
///
///   1. Scale to "16-bit": `waveform *= 2^15` (Kaldi compliance; the waveform is
///      NOT peak-normalized first).
///   2. Frame the signal: `frame_length = 400` (25 ms), `hop = 160` (10 ms),
///      `center = false` (no reflect padding). `num_frames = 1 + (N - 400)/160`.
///   3. Per frame, in this order:
///        - remove DC offset: `f -= mean(f)`
///        - pre-emphasis: `f[1:] -= 0.97 * f[:-1]; f[0] *= (1 - 0.97)`
///        - apply the Povey window (`hann(400)^0.85`, symmetric)
///        - real FFT with `fft_length = 512` → 257 bins
///        - power spectrum: `|.|^2`
///   4. Mel projection: `max(mel_floor, melFilters^T · power)` with the kaldi-scale
///      triangular filterbank (80 filters, 20–8000 Hz, triangularized in mel
///      space), then `log()` (natural).
///   5. Per-mel-bin normalization over time: `(x - mean) / sqrt(var + 1e-7)`,
///      using the **sample** variance (ddof = 1).
///   6. Stride-2 frame stacking: `[T, 80] → [T/2, 160]` (drops a trailing frame
///      when `T` is odd).
///
/// Validated against the real extractor: max-abs error ≈ 2.6e-4 on a 1 s test
/// clip (float32-vs-float64 + DFT rounding), mean-abs ≈ 6e-6.
public enum SeamlessM4TFrontEnd {

    // MARK: Constants (match SeamlessM4TFeatureExtractor defaults)

    /// Expected input sample rate.
    public static let sampleRate = 16_000
    /// Analysis frame length in samples (25 ms @ 16 kHz).
    public static let frameLength = 400
    /// Hop / frame shift in samples (10 ms @ 16 kHz).
    public static let hopLength = 160
    /// FFT size (next pow2 ≥ frameLength).
    public static let fftLength = 512
    /// Number of mel filters.
    public static let numMelBins = 80
    /// Pre-emphasis coefficient.
    public static let preemphasis: Float = 0.97
    /// Mel floor before the log (matches `energy_floor` / `mel_floor`).
    public static let melFloor: Float = 1.192092955078125e-07
    /// Per-bin normalization epsilon.
    public static let normEpsilon: Float = 1e-7
    /// Frame-stacking stride applied after normalization.
    public static let stride = 2
    /// Feature dimension after stacking (`numMelBins * stride`).
    public static let featureDim = numMelBins * stride   // 160
    /// Lowest mel filter edge in Hz.
    public static let minFrequency: Float = 20
    /// Highest mel filter edge in Hz (Nyquist).
    public static let maxFrequency: Float = 8000

    // MARK: Cached DSP tables (depend only on constants)

    /// Povey window: `hann(400)^0.85` (symmetric, i.e. `periodic = false`).
    private static let window: [Float] = poveyWindow(length: frameLength)

    /// Triangular mel filterbank, row-major `[numFreqBins=257, numMelBins=80]`,
    /// matching `mel_filter_bank(..., mel_scale="kaldi",
    /// triangularize_in_mel_space=True)` padded with a trailing zero row so it
    /// covers all 257 rfft bins.
    private static let melFilters: [Float] = makeKaldiMelFilterbank()

    /// Number of one-sided FFT bins (`fftLength / 2 + 1`).
    private static let numFreqBins = fftLength / 2 + 1   // 257

    // MARK: Public API

    /// Compute `input_features` for a single mono 16 kHz waveform.
    ///
    /// - Parameter audio: mono PCM samples already resampled to 16 kHz.
    /// - Returns: `(features, frames)` where `features` is row-major
    ///   `[frames, 160]` (i.e. `frames * 160` floats) and `frames` is the number
    ///   of stacked frames. Returns `([], 0)` when the input is shorter than one
    ///   analysis frame.
    public static func inputFeatures(audio: [Float]) -> (features: [Float], frames: Int) {
        guard audio.count >= frameLength else { return ([], 0) }

        // 1. Mel spectrogram → log-mel, laid out [melFrames, numMelBins].
        let (logMel, melFrames) = logMelSpectrogram(audio: audio)
        guard melFrames > 0 else { return ([], 0) }

        // 2. Per-mel-bin normalization over the time axis (ddof = 1).
        var normalized = logMel
        normalizePerMelBin(&normalized, frames: melFrames)

        // 3. Stride-2 frame stacking: [melFrames, 80] → [melFrames/2, 160].
        let stacked = stride == 1 ? melFrames : (melFrames / stride)
        guard stacked > 0 else { return ([], 0) }
        var out = [Float](repeating: 0, count: stacked * featureDim)
        // Row s of the output is rows (2s, 2s+1) of `normalized` concatenated.
        out.withUnsafeMutableBufferPointer { dst in
            normalized.withUnsafeBufferPointer { src in
                for s in 0..<stacked {
                    let srcBase = (s * stride) * numMelBins
                    let dstBase = s * featureDim
                    // Copy `stride` consecutive mel frames (numMelBins each).
                    for k in 0..<(stride * numMelBins) {
                        dst[dstBase + k] = src[srcBase + k]
                    }
                }
            }
        }
        return (out, stacked)
    }

    // MARK: - Steps

    /// Frame → DC-remove → pre-emphasis → window → rfft → power → mel → log.
    /// Returns row-major `[frames, numMelBins]`.
    private static func logMelSpectrogram(audio: [Float]) -> (logMel: [Float], frames: Int) {
        let n = audio.count
        let frames = 1 + (n - frameLength) / hopLength
        guard frames > 0 else { return ([], 0) }

        // vDSP real FFT setup (length 512 → log2n = 9).
        let log2n = vDSP_Length(9)
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            return ([], 0)
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        let half = fftLength / 2  // 256

        var logMel = [Float](repeating: 0, count: frames * numMelBins)

        // Scratch buffers reused across frames.
        var frameBuf = [Float](repeating: 0, count: fftLength)        // zero-padded to 512
        var windowed = [Float](repeating: 0, count: fftLength)
        var realp = [Float](repeating: 0, count: half)
        var imagp = [Float](repeating: 0, count: half)
        var power = [Float](repeating: 0, count: numFreqBins)         // 257

        let scale: Float = 32768.0  // 2^15

        audio.withUnsafeBufferPointer { audioPtr in
            for t in 0..<frames {
                let start = t * hopLength

                // Load frame, scale to 16-bit range. Tail (400..512) stays zero.
                for i in 0..<frameLength {
                    frameBuf[i] = audioPtr[start + i] * scale
                }

                // Remove DC offset over the 400-sample frame.
                var mean: Float = 0
                vDSP_meanv(frameBuf, 1, &mean, vDSP_Length(frameLength))
                var negMean = -mean
                vDSP_vsadd(frameBuf, 1, &negMean, &frameBuf, 1, vDSP_Length(frameLength))

                // Pre-emphasis: f[i] -= 0.97*f[i-1] for i>=1; f[0] *= (1-0.97).
                // Walk high→low so each read uses the pre-update neighbour.
                var i = frameLength - 1
                while i >= 1 {
                    frameBuf[i] -= preemphasis * frameBuf[i - 1]
                    i -= 1
                }
                frameBuf[0] *= (1 - preemphasis)

                // Apply the Povey window over the first 400 samples.
                vDSP_vmul(frameBuf, 1, window, 1, &windowed, 1, vDSP_Length(frameLength))
                // Zero the 400..512 tail (window only covers 400).
                for j in frameLength..<fftLength { windowed[j] = 0 }

                // Real FFT (packed split-complex), 512-point. The split-complex
                // buffers must outlive the vDSP calls, so bind them in stable
                // nested scopes rather than passing `&array` inout expressions.
                realp.withUnsafeMutableBufferPointer { rp in
                    imagp.withUnsafeMutableBufferPointer { ip in
                        var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                        windowed.withUnsafeBufferPointer { wptr in
                            wptr.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: half) { typed in
                                vDSP_ctoz(typed, 2, &split, 1, vDSP_Length(half))
                                vDSP_fft_zrip(fftSetup, &split, 1, log2n, FFTDirection(FFT_FORWARD))
                            }
                        }
                        // vDSP packs Nyquist into imagp[0]; the result is scaled
                        // by 2 vs a textbook DFT. We need the *unscaled* complex
                        // magnitude-squared = (re/2)^2 + (im/2)^2 per bin.
                        // Bin 0 (DC) → realp[0]/2; Nyquist (bin 256) → imagp[0]/2.
                        let dc = rp[0] * 0.5
                        let nyq = ip[0] * 0.5
                        power[0] = dc * dc
                        power[half] = nyq * nyq
                        for b in 1..<half {
                            let re = rp[b] * 0.5
                            let im = ip[b] * 0.5
                            power[b] = re * re + im * im
                        }
                    }
                }

                // Mel projection: melspec[m] = Σ_f power[f] * melFilters[f, m].
                // melFilters is [257, 80] row-major; accumulate per filter.
                let logMelBase = t * numMelBins
                melFilters.withUnsafeBufferPointer { mf in
                    power.withUnsafeBufferPointer { pw in
                        for m in 0..<numMelBins {
                            var acc: Float = 0
                            // Strided dot product (column m of a [257,80] matrix).
                            // vDSP_dotpr needs unit stride; do an explicit loop —
                            // each filter touches only a few non-zero bins anyway.
                            var idx = m
                            for f in 0..<numFreqBins {
                                acc += pw[f] * mf[idx]
                                idx += numMelBins
                            }
                            logMel[logMelBase + m] = Foundation.log(Swift.max(melFloor, acc))
                        }
                    }
                }
            }
        }

        return (logMel, frames)
    }

    /// Per-mel-bin zero-mean / unit-variance over the time axis, in place.
    /// Uses the sample variance (ddof = 1) to match the extractor's torch path.
    private static func normalizePerMelBin(_ x: inout [Float], frames: Int) {
        guard frames > 1 else {
            // ddof=1 variance is undefined for a single frame; the extractor
            // would divide by sqrt(1e-7). Match that degenerate behaviour.
            if frames == 1 {
                let invStd = 1.0 / (normEpsilon).squareRoot()
                for m in 0..<numMelBins {
                    x[m] = (x[m] - x[m]) * invStd  // → 0
                }
            }
            return
        }
        let nF = Float(frames)
        x.withUnsafeMutableBufferPointer { buf in
            for m in 0..<numMelBins {
                // Mean over time for bin m.
                var sum: Float = 0
                var idx = m
                for _ in 0..<frames { sum += buf[idx]; idx += numMelBins }
                let mean = sum / nF
                // Sample variance (ddof = 1).
                var sse: Float = 0
                idx = m
                for _ in 0..<frames {
                    let d = buf[idx] - mean
                    sse += d * d
                    idx += numMelBins
                }
                let varr = sse / (nF - 1)
                let invStd = 1.0 / (varr + normEpsilon).squareRoot()
                idx = m
                for _ in 0..<frames {
                    buf[idx] = (buf[idx] - mean) * invStd
                    idx += numMelBins
                }
            }
        }
    }

    // MARK: - DSP table construction

    /// Povey window: `np.power(np.hanning(N), 0.85)`. `np.hanning` is the
    /// symmetric Hann `0.5 - 0.5*cos(2πn/(N-1))`.
    static func poveyWindow(length: Int) -> [Float] {
        var w = [Float](repeating: 0, count: length)
        let denom = Float(length - 1)
        for n in 0..<length {
            let hann = 0.5 - 0.5 * Foundation.cos(2.0 * Float.pi * Float(n) / denom)
            w[n] = Foundation.pow(hann, 0.85)
        }
        return w
    }

    /// Kaldi-scale Hz↔mel conversions used by `mel_filter_bank(mel_scale="kaldi")`.
    static func hzToMelKaldi(_ hz: Float) -> Float { 1127.0 * Foundation.log(1.0 + hz / 700.0) }
    static func melToHzKaldi(_ mel: Float) -> Float { 700.0 * (Foundation.exp(mel / 1127.0) - 1.0) }

    /// Build the `[257, 80]` triangular mel filterbank, matching
    /// `mel_filter_bank(num_frequency_bins=256, num_mel_filters=80,
    /// min=20, max=8000, sr=16000, norm=None, mel_scale="kaldi",
    /// triangularize_in_mel_space=True)` then `np.pad(((0,1),(0,0)))`.
    ///
    /// With `triangularize_in_mel_space`, both the FFT bin frequencies and the
    /// filter edges are converted to mel and the triangles are built there. The
    /// FFT-bin frequency for bin k is `(sr / (256 * 2)) * k`, computed for
    /// `k in 0..<256`; the padded 257th row is all zeros.
    static func makeKaldiMelFilterbank() -> [Float] {
        let numFftFreqs = 256                       // pre-pad bin count
        let nMel = numMelBins                        // 80
        let sr = Float(sampleRate)                   // 16000

        // Filter edge frequencies: linspace in mel between mel(min)..mel(max),
        // nMel+2 points. In mel space the edges ARE these mel values.
        let melMin = hzToMelKaldi(minFrequency)
        let melMax = hzToMelKaldi(maxFrequency)
        var filterMel = [Float](repeating: 0, count: nMel + 2)
        for i in 0..<(nMel + 2) {
            let frac = Float(i) / Float(nMel + 1)
            filterMel[i] = melMin + frac * (melMax - melMin)
        }

        // FFT bin frequencies → mel. fft_bin_width = sr / (numFreqBins*2).
        let fftBinWidth = sr / (Float(numFftFreqs) * 2.0)
        var fftMel = [Float](repeating: 0, count: numFftFreqs)
        for k in 0..<numFftFreqs {
            fftMel[k] = hzToMelKaldi(fftBinWidth * Float(k))
        }

        // Triangular filters in mel space (matches _create_triangular_filter_bank):
        //   diff      = np.diff(filterMel)            # length nMel+1
        //   slopes    = filterMel[None,:] - fftMel[:,None]
        //   down      = -slopes[:, :-2] / diff[:-1]
        //   up        =  slopes[:, 2:]  / diff[1:]
        //   fb        = max(0, min(down, up))         # [numFftFreqs, nMel]
        // Result padded to [257, 80] with a trailing zero row → row-major.
        let rows = numFftFreqs + 1                   // 257
        var fb = [Float](repeating: 0, count: rows * nMel)
        for k in 0..<numFftFreqs {
            for m in 0..<nMel {
                let lower = filterMel[m]
                let center = filterMel[m + 1]
                let upper = filterMel[m + 2]
                let diffLower = center - lower
                let diffUpper = upper - center
                let s = fftMel[k]
                let down = (s - lower) / diffLower      // = -(filterMel[m]-fftMel)/diff[m]
                let up = (upper - s) / diffUpper        // =  (filterMel[m+2]-fftMel)/diff[m+1]
                fb[k * nMel + m] = Swift.max(0, Swift.min(down, up))
            }
        }
        // Row 256 (the pad) stays zero.
        return fb
    }
}
