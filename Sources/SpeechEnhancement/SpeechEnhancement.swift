import Foundation
import MLXCommon
import CoreML
import AudioCommon

/// Speech enhancement using DeepFilterNet3 on Core ML (FP16, Neural Engine).
///
/// Removes background noise from speech audio. The neural network runs on
/// Apple's Neural Engine via Core ML, while signal processing (STFT, ERB
/// filterbank, deep filtering) runs on CPU via Accelerate/vDSP.
///
/// ```swift
/// let enhancer = try await SpeechEnhancer.fromPretrained()
/// let cleanAudio = try enhancer.enhance(audio: noisyAudio, sampleRate: 48000)
/// ```
public final class SpeechEnhancer {

    /// Default HuggingFace model ID
    public static let defaultModelId = "aufklarer/DeepFilterNet3-CoreML"

    /// Native sample rate (48kHz)
    public static let sampleRate = 48000

    /// Whether the model is loaded and ready for inference.
    var _isLoaded = true

    let config: DeepFilterNet3Config
    var network: DeepFilterNet3Network?
    let stft: STFTProcessor

    // ERB filterbank matrices
    let erbFb: [Float]       // [freqBins, erbBands]
    let erbInvFb: [Float]    // [erbBands, freqBins]

    // Normalization state
    var meanNormState: [Float]
    var unitNormState: [Float]
    let meanNormStateInit: [Float]
    let unitNormStateInit: [Float]

    // STFT streaming state
    var analysisMem: [Float]
    var synthesisMem: [Float]

    init(
        network: DeepFilterNet3Network,
        config: DeepFilterNet3Config,
        auxData: DeepFilterNet3WeightLoader.AuxiliaryData
    ) {
        self.network = network
        self.config = config

        self.stft = STFTProcessor(
            fftSize: config.fftSize,
            hopSize: config.hopSize,
            window: auxData.window)

        self.erbFb = auxData.erbFb
        self.erbInvFb = auxData.erbInvFb

        self.meanNormState = auxData.meanNormState
        self.unitNormState = auxData.unitNormState
        self.meanNormStateInit = auxData.meanNormState
        self.unitNormStateInit = auxData.unitNormState

        self.analysisMem = [Float](repeating: 0, count: config.fftSize - config.hopSize)
        self.synthesisMem = [Float](repeating: 0, count: config.fftSize - config.hopSize)
    }

    /// Enhance audio by removing noise (single-shot batch mode).
    ///
    /// Processes the entire audio at once. Signal processing runs on CPU (vDSP),
    /// neural network inference runs via Core ML.
    ///
    /// **Length limit**: the CoreML model is exported with a `RangeDim(1, 6000)`
    /// on the time axis (~60 s at 48 kHz / 10 ms hop). Inputs exceeding this
    /// throw a CoreML prediction error. For longer audio use `enhanceChunked`,
    /// which slices into windows that fit the cap and stitches with a
    /// crossfade.
    ///
    /// - Parameters:
    ///   - audio: Input audio samples (mono, Float32)
    ///   - sampleRate: Sample rate of input audio (resampled to 48kHz if needed)
    /// - Returns: Enhanced audio at 48kHz
    public func enhance(audio: [Float], sampleRate: Int) throws -> [Float] {
        var samples = audio
        if sampleRate != Self.sampleRate {
            samples = AudioFileLoader.resample(audio, from: sampleRate, to: Self.sampleRate)
        }
        resetState()
        return try enhanceCore(audio48k: samples)
    }

    /// Run the enhance pipeline on already-48-kHz audio WITHOUT resetting
    /// streaming state. Used internally by `enhance(...)` (after one fresh
    /// `resetState()`) and by `enhanceChunked(...)` (which resets once at the
    /// start and carries STFT + normalization state across all chunks, so
    /// only the GRU state inside CoreML re-zeros at each boundary).
    private func enhanceCore(audio48k samples: [Float]) throws -> [Float] {
        let hop = config.hopSize
        let freqBins = config.freqBins  // 481 (960/2 + 1)

        // Pad audio so inverse STFT captures the full signal
        let paddedSamples = samples + [Float](repeating: 0, count: hop)

        // STFT — 960-point DFT producing 481 frequency bins
        let (specReal, specImag) = stft.forward(audio: paddedSamples, analysisMem: &analysisMem)
        let numFrames = specReal.count / freqBins
        guard numFrames > 0 else { return [] }

        // Compute ERB features
        var erbFeats = computeERBFeatures(
            real: specReal, imag: specImag,
            erbFb: erbFb,
            freqBins: freqBins, erbBands: config.erbBands, numFrames: numFrames)

        // Apply mean normalization
        applyMeanNormalization(
            &erbFeats, state: &meanNormState,
            alpha: config.normAlpha,
            erbBands: config.erbBands, numFrames: numFrames)

        // Extract first dfBins for spec features
        var specFeatReal = [Float](repeating: 0, count: numFrames * config.dfBins)
        var specFeatImag = [Float](repeating: 0, count: numFrames * config.dfBins)
        for t in 0..<numFrames {
            for f in 0..<config.dfBins {
                specFeatReal[t * config.dfBins + f] = specReal[t * freqBins + f]
                specFeatImag[t * config.dfBins + f] = specImag[t * freqBins + f]
            }
        }

        // Apply unit normalization
        applyUnitNormalization(
            real: &specFeatReal, imag: &specFeatImag,
            state: &unitNormState,
            alpha: config.normAlpha,
            dfBins: config.dfBins, numFrames: numFrames)

        // No lookahead padding here — the Core ML model applies it internally

        // Create Core ML input arrays
        // ERB: [1, 1, T, 32] (NCHW)
        let erbInput = try MLMultiArray(shape: [1, 1, numFrames as NSNumber, config.erbBands as NSNumber], dataType: .float32)
        let erbPtr = erbInput.dataPointer.assumingMemoryBound(to: Float.self)
        erbFeats.withUnsafeBufferPointer { src in
            erbPtr.update(from: src.baseAddress!, count: numFrames * config.erbBands)
        }

        // Spec: [1, 2, T, 96] (NCHW: channel 0=real, channel 1=imag)
        let specInput = try MLMultiArray(shape: [1, 2, numFrames as NSNumber, config.dfBins as NSNumber], dataType: .float32)
        let specPtr = specInput.dataPointer.assumingMemoryBound(to: Float.self)
        let specChannelStride = numFrames * config.dfBins
        specFeatReal.withUnsafeBufferPointer { src in
            specPtr.update(from: src.baseAddress!, count: specChannelStride)
        }
        specFeatImag.withUnsafeBufferPointer { src in
            (specPtr + specChannelStride).update(from: src.baseAddress!, count: specChannelStride)
        }

        // Run neural network (single pass — GRU state is sequential, can't be chunked)
        guard let network else { throw DeepFilterNet3Network.DeepFilterNet3Error.predictionFailed }
        let (erbMaskArray, coefsArray) = try network.predict(featErb: erbInput, featSpec: specInput)

        // Extract ERB mask [1, 1, T, 32] — handle float16 output from Core ML
        let erbMaskCount = numFrames * config.erbBands
        var erbMaskFlat = [Float](repeating: 0, count: erbMaskCount)
        extractMLMultiArrayFlat(erbMaskArray, into: &erbMaskFlat, count: erbMaskCount)

        // Extract DF coefficients [1, 5, T, 96, 2] → [T, F, O, C=2]
        let dfOrder = config.dfOrder
        let coefsCount = dfOrder * numFrames * config.dfBins * 2
        var coefsRaw = [Float](repeating: 0, count: coefsCount)
        extractMLMultiArrayFlat(coefsArray, into: &coefsRaw, count: coefsCount)

        // Reshape from Core ML layout [O, T, F, 2] to [T, F, O, 2]
        var coefsFlat = [Float](repeating: 0, count: coefsCount)
        for t in 0..<numFrames {
            for f in 0..<config.dfBins {
                for o in 0..<dfOrder {
                    let srcIdx = ((o * numFrames + t) * config.dfBins + f) * 2
                    let dstIdx = ((t * config.dfBins + f) * dfOrder + o) * 2
                    coefsFlat[dstIdx] = coefsRaw[srcIdx]
                    coefsFlat[dstIdx + 1] = coefsRaw[srcIdx + 1]
                }
            }
        }

        // Apply ERB mask to full spectrum
        var enhancedReal = specReal
        var enhancedImag = specImag
        applyERBMask(
            specReal: &enhancedReal, specImag: &enhancedImag,
            erbMask: erbMaskFlat,
            erbInvFb: erbInvFb,
            erbBands: config.erbBands, freqBins: freqBins, numFrames: numFrames)

        // Apply deep filtering to lowest dfBins
        let (dfReal, dfImag) = applyDeepFiltering(
            specReal: specReal, specImag: specImag,
            coefs: coefsFlat,
            dfBins: config.dfBins, dfOrder: config.dfOrder,
            dfLookahead: config.dfLookahead,
            numFrames: numFrames, freqBins: freqBins)

        // Combine: DF-enhanced for bins 0..<dfBins, ERB-masked for rest
        for t in 0..<numFrames {
            for f in 0..<config.dfBins {
                enhancedReal[t * freqBins + f] = dfReal[t * config.dfBins + f]
                enhancedImag[t * freqBins + f] = dfImag[t * config.dfBins + f]
            }
        }

        // Inverse STFT
        let rawOutput = stft.inverse(real: enhancedReal, imag: enhancedImag, synthesisMem: &synthesisMem)

        // Trim the hop-size latency from prepended analysis memory
        let trimStart = hop
        let trimEnd = min(trimStart + samples.count, rawOutput.count)
        guard trimEnd > trimStart else { return [] }
        return Array(rawOutput[trimStart..<trimEnd])
    }

    /// Enhance long-form audio that exceeds the single-shot 60 s cap by
    /// splitting into overlapping windows, running `enhance` on each, and
    /// stitching with an equal-power (sin/cos) crossfade.
    ///
    /// Inputs at or under `chunkSeconds` go through the existing single-shot
    /// path — `enhance(audio:sampleRate:)` — and are bit-identical to today.
    ///
    /// **State continuity**: between chunks the function preserves the STFT
    /// analysis/synthesis overlap buffers AND the EMA normalization state.
    /// Those are *exact* across the seam (no glitch). The only state that
    /// can't be carried is the GRU hidden state inside the CoreML graph —
    /// it re-initializes per `predict()` call. The first ~100–200 ms of each
    /// non-leading chunk are therefore mildly degraded while the GRUs
    /// re-converge; the crossfade region is sized to mask this for
    /// speech-dominated content. Stationary low-SNR noise may exhibit a
    /// brief noise-floor flicker at boundaries.
    ///
    /// - Parameters:
    ///   - audio: Input audio samples (mono, Float32) at any sample rate
    ///   - sampleRate: Sample rate of input audio (resampled to 48kHz if needed)
    ///   - chunkSeconds: Target window size in seconds. Default 45 s leaves
    ///     15 s of headroom under the model's 60 s cap. Must be > 1 s.
    ///   - overlapMs: Crossfade overlap between adjacent chunks in
    ///     milliseconds. Default 500 ms gives the GRU plenty of warm-up time
    ///     inside the fade region. Must be < chunkSeconds × 1000 / 2.
    /// - Returns: Enhanced audio at 48 kHz, same length as the resampled
    ///   input (±1 sample for rounding).
    public func enhanceChunked(
        audio: [Float],
        sampleRate: Int,
        chunkSeconds: Double = 45.0,
        overlapMs: Int = 500
    ) throws -> [Float] {
        precondition(chunkSeconds > 1.0, "chunkSeconds must be > 1 s")
        precondition(Double(overlapMs) < chunkSeconds * 1000 / 2,
                     "overlapMs must be < chunkSeconds*1000/2")

        var samples = audio
        if sampleRate != Self.sampleRate {
            samples = AudioFileLoader.resample(audio, from: sampleRate, to: Self.sampleRate)
        }
        if samples.isEmpty { return [] }

        let rate = Self.sampleRate
        let chunkSamples = Int(chunkSeconds * Double(rate))

        // Short input → single shot through the established path. Bit-identical
        // to enhance(...) on this same input.
        if samples.count <= chunkSamples {
            resetState()
            return try enhanceCore(audio48k: samples)
        }

        let overlapSamples = max(1, overlapMs * rate / 1000)
        let hop = chunkSamples - overlapSamples

        // Equal-division so the tail isn't a tiny fragment (mirrors the
        // OmnilingualMLXModel chunker fix — small tails have unreliable
        // per-window stats and produce worse seams).
        let numChunks = max(2, Int(ceil(Double(samples.count - overlapSamples) / Double(hop))))
        let evenChunkSamples = max(
            overlapSamples + 1,
            (samples.count + overlapSamples * (numChunks - 1) + numChunks - 1) / numChunks
        )
        let evenHop = evenChunkSamples - overlapSamples

        // Reset once; the loop below preserves STFT + norm state across chunks.
        resetState()

        var output = [Float]()
        output.reserveCapacity(samples.count)
        var prevTail: [Float] = []

        var offset = 0
        while offset < samples.count {
            let end = min(offset + evenChunkSamples, samples.count)
            let chunk = Array(samples[offset..<end])
            let enhanced = try enhanceCore(audio48k: chunk)
            // The enhance path returns ≈ chunk.count samples (length preserved).
            // Tail < overlapSamples can happen on the last chunk if rounding
            // shaved it; clamp to keep the math safe.
            let effectiveOverlap = min(overlapSamples, enhanced.count, prevTail.count)

            if prevTail.isEmpty {
                // First chunk — keep everything except the trailing overlap,
                // stash that overlap as the tail to crossfade against next.
                let keepEnd = max(0, enhanced.count - overlapSamples)
                output.append(contentsOf: enhanced[0..<keepEnd])
                prevTail = Array(enhanced[keepEnd..<enhanced.count])
            } else {
                // Equal-power crossfade over `effectiveOverlap` samples.
                // t in [0,1]; fadeOut = cos(t*π/2), fadeIn = sin(t*π/2);
                // |fadeOut|² + |fadeIn|² = 1 so the energy of stationary
                // content stays flat across the seam.
                for i in 0..<effectiveOverlap {
                    let t = Float(i) / Float(max(effectiveOverlap - 1, 1))
                    let fadeOut = cos(t * .pi / 2)
                    let fadeIn = sin(t * .pi / 2)
                    output.append(prevTail[i] * fadeOut + enhanced[i] * fadeIn)
                }
                // Any prevTail past the effectiveOverlap is older content
                // that this chunk doesn't cover — append it raw to avoid
                // dropping samples.
                if prevTail.count > effectiveOverlap {
                    output.append(contentsOf: prevTail[effectiveOverlap..<prevTail.count])
                }
                // Keep the middle of this chunk, stash its trailing overlap.
                let midStart = effectiveOverlap
                let midEnd = max(midStart, enhanced.count - overlapSamples)
                output.append(contentsOf: enhanced[midStart..<midEnd])
                prevTail = Array(enhanced[midEnd..<enhanced.count])
            }

            offset += evenHop
            // If the remaining input is too small for a usable chunk, stop —
            // the unbroken tail of `prevTail` covers the gap.
            if samples.count - offset < overlapSamples * 2 { break }
        }
        // Flush the final tail.
        output.append(contentsOf: prevTail)
        // Length should match samples.count ±1. Clamp if slightly over.
        if output.count > samples.count {
            output.removeLast(output.count - samples.count)
        }
        return output
    }

    /// Reset all streaming state (STFT buffers, normalization).
    public func resetState() {
        analysisMem = [Float](repeating: 0, count: config.fftSize - config.hopSize)
        synthesisMem = [Float](repeating: 0, count: config.fftSize - config.hopSize)
        meanNormState = meanNormStateInit
        unitNormState = unitNormStateInit
    }

    /// Load a pretrained DeepFilterNet3 model.
    ///
    /// Downloads Core ML model and auxiliary data on first use, then caches locally.
    ///
    /// - Parameters:
    ///   - modelId: HuggingFace model ID
    ///   - progressHandler: Callback for download progress
    /// - Returns: Ready-to-use speech enhancer
    public static func fromPretrained(
        modelId: String = defaultModelId,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> SpeechEnhancer {
        progressHandler?(0.0, "Downloading model...")

        let cacheDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)

        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: cacheDir,
            additionalFiles: [
                "no_weights.safetensors",  // suppress default *.safetensors glob
                // Pre-compiled CoreML — shipped by aufklarer/DeepFilterNet3-CoreML
                // alongside the legacy .mlpackage. On-device compileModel() is
                // known to drift per runtime (Mac vs simulator vs iPhone), so
                // the goal is to never reach that code path for new installs.
                "DeepFilterNet3.mlmodelc/**",
                "auxiliary.npz",
            ],
            offlineMode: offlineMode,
            progressHandler: { progress in
                progressHandler?(progress * 0.8, "Downloading model...")
            }
        )

        progressHandler?(0.8, "Loading model...")

        let config = DeepFilterNet3Config.default
        let (network, auxData) = try DeepFilterNet3WeightLoader.load(from: cacheDir)

        progressHandler?(1.0, "Ready")

        return SpeechEnhancer(network: network, config: config, auxData: auxData)
    }

    /// Extract float32 data from an MLMultiArray, handling float16 output from Core ML.
    private func extractMLMultiArrayFlat(_ array: MLMultiArray, into output: inout [Float], count: Int) {
        if array.dataType == .float16 {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<count {
                output[i] = Float(ptr[i])
            }
        } else {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
            output.withUnsafeMutableBufferPointer { dst in
                dst.baseAddress!.update(from: ptr, count: count)
            }
        }
    }

}
