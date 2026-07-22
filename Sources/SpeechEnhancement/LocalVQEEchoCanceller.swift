import AudioCommon
import CoreML
import Foundation

/// Errors surfaced by ``LocalVQEEchoCanceller`` before or during inference.
public enum LocalVQEEchoCancellationError: Error, LocalizedError, Equatable {
    case invalidSampleRate(Int)
    case mismatchedStreamLengths(microphone: Int, reference: Int)
    case invalidFrameLength(stream: String, expected: Int, actual: Int)
    case nonFiniteSamples(stream: String)
    case missingFrontendArray(String)
    case invalidFrontendArray(name: String, expected: Int, actual: Int)
    case nonFiniteFrontendArray(String)
    case invalidControllerWeightCount(expected: Int, actual: Int)
    case frontendInitializationFailed
    case frontendDelayEstimationFailed
    case frontendProcessingFailed
    case fftInitializationFailed
    case missingModelOutput(String)
    case invalidModelOutputCount(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidSampleRate(let sampleRate):
            return "Sample rate must be positive; received \(sampleRate) Hz."
        case .mismatchedStreamLengths(let microphone, let reference):
            return "Microphone and playback reference must be synchronized and equally long (\(microphone) vs \(reference) samples)."
        case .invalidFrameLength(let stream, let expected, let actual):
            return "\(stream) frame must contain exactly \(expected) samples; received \(actual)."
        case .nonFiniteSamples(let stream):
            return "\(stream) contains NaN or infinite samples."
        case .missingFrontendArray(let name):
            return "LocalVQE frontend archive is missing '\(name)'."
        case .invalidFrontendArray(let name, let expected, let actual):
            return "LocalVQE frontend array '\(name)' has \(actual) values; expected \(expected)."
        case .nonFiniteFrontendArray(let name):
            return "LocalVQE frontend array '\(name)' contains non-finite values."
        case .invalidControllerWeightCount(let expected, let actual):
            return "LocalVQE controller has \(actual) weights; expected \(expected)."
        case .frontendInitializationFailed:
            return "Could not initialize the LocalVQE adaptive echo-cancellation frontend."
        case .frontendDelayEstimationFailed:
            return "LocalVQE could not estimate playback-reference delay."
        case .frontendProcessingFailed:
            return "LocalVQE adaptive echo-cancellation frontend rejected the frame."
        case .fftInitializationFailed:
            return "Could not initialize the LocalVQE 512-point spectral transform."
        case .missingModelOutput(let name):
            return "LocalVQE Core ML output '\(name)' is missing."
        case .invalidModelOutputCount(let expected, let actual):
            return "LocalVQE Core ML output has \(actual) values; expected \(expected)."
        }
    }
}

/// Stateful, two-stream acoustic echo cancellation using LocalVQE v1.4-AEC.
///
/// The microphone and playback reference remain separate throughout capture.
/// Each synchronized 16 kHz frame passes through bulk-delay estimation, the
/// model's learned adaptive filter, and its stateful Core ML residual mask.
/// This is echo cancellation, not generic denoising: a playback reference is
/// required and the microphone must never be replaced with a mixed signal.
///
/// Create one instance per recording and call ``reset()`` between recordings.
/// Calls are stateful and must be serialized by the caller.
public final class LocalVQEEchoCanceller {
    public static let defaultModelId =
        "aufklarer/LocalVQE-v1.4-AEC-200K-CoreML"
    public static let sampleRate = 16_000
    public static let fftSize = 512
    public static let frameSize = 256
    public static let algorithmicLatencySamples = frameSize

    private let adaptiveFilter: LocalVQEAECAdaptiveFilter
    private let residualMask: LocalVQEAECResidualMasking
    private let codec: LocalVQEAECCodec

    init(
        adaptiveFilter: LocalVQEAECAdaptiveFilter,
        residualMask: LocalVQEAECResidualMasking,
        codec: LocalVQEAECCodec
    ) {
        self.adaptiveFilter = adaptiveFilter
        self.residualMask = residualMask
        self.codec = codec
    }

    /// Current playback-reference delay selected by the online GCC-PHAT
    /// estimator. It remains zero until the estimator has enough evidence.
    public var currentDelaySamples: Int {
        adaptiveFilter.currentDelaySamples
    }

    /// Confidence of the most recent playback-reference delay estimate.
    public var delayConfidence: Float {
        adaptiveFilter.delayConfidence
    }

    /// Load the compiled Core ML mask and adaptive-filter controller.
    public static func fromPretrained(
        modelId: String = defaultModelId,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        computeUnits: MLComputeUnits = CoreMLComputeUnitsResolver.resolved(
            default: .cpuAndNeuralEngine),
        enablePrealignment: Bool = true,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> LocalVQEEchoCanceller {
        let modelDirectory = try cacheDir
            ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        progressHandler?(0.0, "Downloading LocalVQE AEC...")
        try await HuggingFaceDownloader.downloadFiles(
            modelId: modelId,
            to: modelDirectory,
            files: [
                "config.json",
                "LocalVQEAECResidualMask.mlmodelc/**",
                "LocalVQEAECFrontend.npz",
            ],
            offlineMode: offlineMode,
            progressHandler: { fraction in
                progressHandler?(fraction * 0.75, "Downloading LocalVQE AEC...")
            })

        progressHandler?(0.78, "Loading adaptive echo canceller...")
        let artifacts = try LocalVQEAECArtifacts.load(from: modelDirectory)
        let adaptiveFilter = try LocalVQEAECAdaptiveFilter(
            weights: artifacts.controllerWeights,
            enablePrealignment: enablePrealignment)
        let residualMask = try LocalVQEAECResidualMask(
            modelURL: artifacts.modelURL,
            computeUnits: computeUnits)
        let codec = try LocalVQEAECCodec(window: artifacts.analysisWindow)
        let canceller = LocalVQEEchoCanceller(
            adaptiveFilter: adaptiveFilter,
            residualMask: residualMask,
            codec: codec)

        progressHandler?(0.9, "Prewarming LocalVQE AEC...")
        try residualMask.warmUp()
        residualMask.reset()
        progressHandler?(1.0, "Ready")
        return canceller
    }

    /// Reset delay estimation, adaptive-filter memory, spectral overlap, and
    /// every Core ML temporal state. Call once when a recording ends or when
    /// either input stream is discontinuous.
    public func reset() {
        adaptiveFilter.reset()
        residualMask.reset()
        codec.reset()
    }

    /// Cancel echo in one synchronized 16 ms frame.
    ///
    /// - Parameters:
    ///   - microphone: Exactly 256 mono Float32 samples at 16 kHz.
    ///   - reference: The exact playback/system-audio reference for the same
    ///     capture interval, also 256 mono Float32 samples at 16 kHz.
    /// - Returns: 256 echo-cancelled samples. The stream has one frame (16 ms)
    ///   of algorithmic spectral latency.
    public func processFrame(
        microphone: [Float],
        reference: [Float]
    ) throws -> [Float] {
        try validateFrame(microphone, named: "Microphone")
        try validateFrame(reference, named: "Playback reference")
        let frontend = try adaptiveFilter.process(
            microphone: microphone, reference: reference)
        let spectra = codec.analyze(
            residual: frontend.residual,
            echoEstimate: frontend.echoEstimate)
        let enhancedSpectrum = try residualMask.process(
            microphoneSpectrum: spectra.microphone,
            referenceSpectrum: spectra.reference)
        let output = codec.synthesize(spectrum: enhancedSpectrum)
        guard output.allSatisfy(\.isFinite) else {
            throw AudioModelError.inferenceFailed(
                operation: "LocalVQE AEC synthesis",
                reason: "Model produced NaN or infinite audio")
        }
        return output
    }

    /// Cancel echo in a complete synchronized clip.
    ///
    /// Inputs are resampled together to the model's native 16 kHz. By default,
    /// the complete clip is used once to lock bulk delay before filtering,
    /// avoiding the online estimator's acquisition period. The returned audio
    /// is mono Float32 at 16 kHz and has the same native-rate sample count as
    /// the resampled microphone input.
    public func process(
        microphone: [Float],
        reference: [Float],
        sampleRate: Int,
        primeDelay: Bool = true
    ) throws -> [Float] {
        guard sampleRate > 0 else {
            throw LocalVQEEchoCancellationError.invalidSampleRate(sampleRate)
        }
        guard microphone.count == reference.count else {
            throw LocalVQEEchoCancellationError.mismatchedStreamLengths(
                microphone: microphone.count, reference: reference.count)
        }
        guard microphone.allSatisfy(\.isFinite) else {
            throw LocalVQEEchoCancellationError.nonFiniteSamples(
                stream: "Microphone")
        }
        guard reference.allSatisfy(\.isFinite) else {
            throw LocalVQEEchoCancellationError.nonFiniteSamples(
                stream: "Playback reference")
        }
        if microphone.isEmpty {
            reset()
            return []
        }

        let nativeMicrophone: [Float]
        let nativeReference: [Float]
        if sampleRate == Self.sampleRate {
            nativeMicrophone = microphone
            nativeReference = reference
        } else {
            nativeMicrophone = AudioFileLoader.resample(
                microphone, from: sampleRate, to: Self.sampleRate)
            nativeReference = AudioFileLoader.resample(
                reference, from: sampleRate, to: Self.sampleRate)
        }
        guard nativeMicrophone.count == nativeReference.count else {
            throw LocalVQEEchoCancellationError.mismatchedStreamLengths(
                microphone: nativeMicrophone.count,
                reference: nativeReference.count)
        }

        reset()
        if primeDelay {
            try adaptiveFilter.primeDelay(
                microphone: nativeMicrophone, reference: nativeReference)
        }
        let targetCount = nativeMicrophone.count
        let paddedCount = ((targetCount + Self.frameSize - 1) / Self.frameSize)
            * Self.frameSize
        var paddedMicrophone = nativeMicrophone
        var paddedReference = nativeReference
        paddedMicrophone.append(contentsOf: repeatElement(
            0, count: paddedCount - targetCount))
        paddedReference.append(contentsOf: repeatElement(
            0, count: paddedCount - targetCount))

        var output = [Float]()
        output.reserveCapacity(paddedCount)
        for offset in stride(from: 0, to: paddedCount, by: Self.frameSize) {
            let range = offset..<(offset + Self.frameSize)
            output.append(contentsOf: try processFrame(
                microphone: Array(paddedMicrophone[range]),
                reference: Array(paddedReference[range])))
        }
        if output.count > targetCount {
            output.removeLast(output.count - targetCount)
        }
        return output
    }

    private func validateFrame(_ samples: [Float], named stream: String) throws {
        guard samples.count == Self.frameSize else {
            throw LocalVQEEchoCancellationError.invalidFrameLength(
                stream: stream,
                expected: Self.frameSize,
                actual: samples.count)
        }
        guard samples.allSatisfy(\.isFinite) else {
            throw LocalVQEEchoCancellationError.nonFiniteSamples(stream: stream)
        }
    }
}
