import MLX
import MLXNN
import Foundation
import AudioCommon

/// Music source separation: split a stereo mix into stems (vocals, drums, bass, other).
///
/// Uses Open-Unmix (UMX-HQ) — 4 independent BiLSTM models, one per target.
///
/// ```swift
/// let separator = try await SourceSeparator.fromPretrained()
/// let stems = separator.separate(audio: stereoAudio, sampleRate: 44100)
/// // stems.vocals, stems.drums, stems.bass, stems.other — each [2, N]
/// ```
public final class SourceSeparator {

    public static let defaultModelId = "aufklarer/OpenUnmix-HQ-MLX"
    public static let largeModelId = "aufklarer/OpenUnmix-L-MLX"

    public let config: OpenUnmixConfig
    private let models: [SeparationTarget: OpenUnmixStemModel]
    private let stft: STFTProcessor

    public init(config: OpenUnmixConfig, models: [SeparationTarget: OpenUnmixStemModel]) {
        self.config = config
        self.models = models
        self.stft = STFTProcessor(nFFT: config.nFFT, nHop: config.nHop)
    }

    /// Separate a stereo audio mix into stems.
    ///
    /// - Parameters:
    ///   - audio: Stereo audio as `[[Float]]` — `audio[0]` = left, `audio[1]` = right
    ///   - sampleRate: Input sample rate. Resampled to the model rate
    ///     (`config.sampleRate`, 44.1 kHz) at mastering quality if different.
    ///   - targets: Which stems to extract (default: all 4)
    ///   - wiener: Apply Wiener post-filtering (improves SDR ~0.5 dB, requires all 4 stems)
    /// - Returns: Dictionary of target → stereo audio `[[Float]]` at the model rate
    public func separate(
        audio: [[Float]],
        sampleRate: Int = 44100,
        targets: [SeparationTarget] = SeparationTarget.allCases,
        wiener: Bool = true,
        metricsHandler: ((SourceSeparationMetrics) -> Void)? = nil
    ) -> [SeparationTarget: [[Float]]] {
        guard audio.count == 2 else {
            // Mono → duplicate to stereo
            let mono = audio.first ?? []
            return separate(
                audio: [mono, mono], sampleRate: sampleRate,
                targets: targets, wiener: wiener, metricsHandler: metricsHandler)
        }

        // UMX is trained at 44.1 kHz; feeding another rate would mis-map every
        // STFT bin (wrong pitch/tempo). Bring the mix to the model rate first.
        // Music → mastering quality for full-band fidelity.
        let stereo: [[Float]]
        if sampleRate != config.sampleRate {
            stereo = AudioFileLoader.resampleStereo(
                audio, from: sampleRate, to: config.sampleRate, quality: .mastering)
        } else {
            stereo = audio
        }

        let left = stereo[0]
        let right = stereo[1]
        let length = left.count

        var metrics = SourceSeparationMetrics()
        metrics.audioSeconds = Double(length) / Double(config.sampleRate)
        let totalStart = CFAbsoluteTimeGetCurrent()

        // STFT each channel
        let stftStart = CFAbsoluteTimeGetCurrent()
        let (leftReal, leftImag) = stft.forward(left)
        let (rightReal, rightImag) = stft.forward(right)

        let T = leftReal.count
        let leftMag = stft.magnitude(real: leftReal, imag: leftImag)
        let rightMag = stft.magnitude(real: rightReal, imag: rightImag)

        // Stack into [T, 2, 2049] for the model
        var inputMag = [[Float]](repeating: [Float](repeating: 0, count: 2 * stft.nBins), count: T)
        for t in 0..<T {
            for f in 0..<stft.nBins {
                inputMag[t][f] = leftMag[t][f]
                inputMag[t][stft.nBins + f] = rightMag[t][f]
            }
        }
        metrics.stftForwardSec = CFAbsoluteTimeGetCurrent() - stftStart

        // Build the MLX input once — identical across all stems. Avoids rebuilding
        // a flat `[Float]` and a fresh MLXArray inside the per-stem loop.
        let flatInput = inputMag.flatMap { $0 }
        let mlxInput = MLXArray(flatInput, [T, 2, stft.nBins])

        // Launch all target models lazily, then single batched eval. MLX is
        // lazy by default, so deferring `eval()` until all forward graphs are
        // built lets the scheduler interleave kernel dispatches across stems
        // instead of fully serializing on each `eval()`.
        let graphStart = CFAbsoluteTimeGetCurrent()
        let pending: [(target: SeparationTarget, output: MLXArray)] = targets.compactMap { target in
            guard let model = models[target] else { return nil }
            return (target, model(mlxInput))
        }
        metrics.modelGraphBuildSec = CFAbsoluteTimeGetCurrent() - graphStart

        let evalStart = CFAbsoluteTimeGetCurrent()
        eval(pending.map(\.output))
        metrics.modelEvalSec = CFAbsoluteTimeGetCurrent() - evalStart

        // Unpack each stem's output to nested [[Float]] for the existing
        // Wiener / iSTFT path. GPU work is already complete by this point.
        var targetMags: [(target: SeparationTarget, left: [[Float]], right: [[Float]])] = []
        for entry in pending {
            let unpackStart = CFAbsoluteTimeGetCurrent()
            let outputFlat = entry.output.asArray(Float.self)
            var leftOutMag = [[Float]](repeating: [Float](repeating: 0, count: stft.nBins), count: T)
            var rightOutMag = [[Float]](repeating: [Float](repeating: 0, count: stft.nBins), count: T)
            for t in 0..<T {
                for f in 0..<stft.nBins {
                    leftOutMag[t][f] = outputFlat[t * 2 * stft.nBins + f]
                    rightOutMag[t][f] = outputFlat[t * 2 * stft.nBins + stft.nBins + f]
                }
            }
            metrics.modelSecByTarget[entry.target] = CFAbsoluteTimeGetCurrent() - unpackStart
            targetMags.append((entry.target, leftOutMag, rightOutMag))
        }

        var results: [SeparationTarget: [[Float]]] = [:]

        if wiener && targetMags.count > 1 {
            // Multichannel Wiener EM + inverse STFT, both on the GPU. Wiener
            // returns MLXArrays of shape [J, T, F]; we stack L/R into a
            // channel axis, run a single batched inverseMLX over [J, 2, T, F],
            // and only convert to [[Float]] at the very end. This removes the
            // per-(t, f) scalar-write loop that used to bridge the two stages.
            let wienerStart = CFAbsoluteTimeGetCurrent()
            let allLeftMags = targetMags.map(\.left)
            let allRightMags = targetMags.map(\.right)
            let refined = WienerFilterMLX.applyMLX(
                targetMagsL: allLeftMags,
                targetMagsR: allRightMags,
                mixRealL: leftReal, mixImagL: leftImag,
                mixRealR: rightReal, mixImagR: rightImag,
                iterations: 1)
            metrics.wienerSec = CFAbsoluteTimeGetCurrent() - wienerStart

            let istftStart = CFAbsoluteTimeGetCurrent()
            // Stack [J, T, F] L and R along a new channel axis → [J, 2, T, F].
            let realJC = stacked([refined.realL, refined.realR], axis: 1)
            let imagJC = stacked([refined.imagL, refined.imagR], axis: 1)
            let audioJC = stft.inverseMLX(real: realJC, imag: imagJC, length: length)
            // Single sync + readback: [J, 2, length] → flat [J*2*length]
            eval(audioJC)
            let flat = audioJC.asArray(Float.self)
            for (i, entry) in targetMags.enumerated() {
                let base = i * 2 * length
                let leftStem = Array(flat[base ..< base + length])
                let rightStem = Array(flat[base + length ..< base + 2 * length])
                results[entry.target] = [leftStem, rightStem]
            }
            metrics.inverseStftSec = CFAbsoluteTimeGetCurrent() - istftStart
        } else {
            // No Wiener — direct phase reconstruction
            let istftStart = CFAbsoluteTimeGetCurrent()
            for entry in targetMags {
                let leftStem = stft.applyMaskAndInvert(
                    maskedMag: entry.left, origReal: leftReal, origImag: leftImag, length: length)
                let rightStem = stft.applyMaskAndInvert(
                    maskedMag: entry.right, origReal: rightReal, origImag: rightImag, length: length)
                results[entry.target] = [leftStem, rightStem]
            }
            metrics.inverseStftSec = CFAbsoluteTimeGetCurrent() - istftStart
        }

        metrics.totalSec = CFAbsoluteTimeGetCurrent() - totalStart
        metricsHandler?(metrics)
        return results
    }

    /// Load pretrained Open-Unmix model from HuggingFace.
    public static func fromPretrained(
        modelId: String = defaultModelId,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> SourceSeparator {
        let config = modelId.contains("L-MLX") ? OpenUnmixConfig.umxl : OpenUnmixConfig.umxhq

        let cacheDir: URL
        do {
            cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId, reason: "Cache directory error", underlying: error)
        }

        // Download weights
        progressHandler?(0.0, "Downloading model...")
        do {
            try await HuggingFaceDownloader.downloadWeights(
                modelId: modelId,
                to: cacheDir,
                additionalFiles: [
                    "vocals.safetensors",
                    "drums.safetensors",
                    "bass.safetensors",
                    "other.safetensors",
                    "config.json",
                ]
            ) { fraction in
                progressHandler?(fraction * 0.7, "Downloading model...")
            }
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId, reason: "Download failed", underlying: error)
        }

        // Load each target
        var models: [SeparationTarget: OpenUnmixStemModel] = [:]
        let targets: [SeparationTarget] = [.vocals, .drums, .bass, .other]

        for (i, target) in targets.enumerated() {
            let progress = 0.7 + Double(i) * 0.075
            progressHandler?(progress, "Loading \(target.rawValue)...")

            let weightsURL = cacheDir.appendingPathComponent("\(target.rawValue).safetensors")
            guard FileManager.default.fileExists(atPath: weightsURL.path) else {
                throw AudioModelError.modelLoadFailed(
                    modelId: modelId, reason: "\(target.rawValue).safetensors not found")
            }

            let model = OpenUnmixStemModel(hiddenSize: config.hiddenSize)
            let weights = try MLX.loadArrays(url: weightsURL)
            let params = ModuleParameters.unflattened(weights)
            try model.update(parameters: params)
            model.train(false)  // Use pretrained running_mean/running_var in BatchNorm
            // Concatenate LSTM input/hidden weight matrices so each timestep
            // runs one matmul instead of two. Must happen after update().
            model.prepareForInference()
            models[target] = model
        }

        progressHandler?(1.0, "Ready")
        return SourceSeparator(config: config, models: models)
    }
}
