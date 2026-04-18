import CoreML
import Foundation
#if canImport(os)
import os
#endif

/// CoreML model loader that surfaces Neural Engine fallback.
///
/// `MLModel(contentsOf:configuration:)` silently succeeds when
/// ``MILCompilerForANE`` fails — the model just runs on CPU instead of
/// ANE. Users only see the performance cliff: RTF jumps from ~0.04 to
/// ~1.8 on wake-word, ASR slows 5–20×, etc. They have no way to
/// correlate this with the CoreML runtime's ``E5RT encountered an STL
/// exception. msg = MILCompilerForANE error`` stderr message.
///
/// This helper:
/// 1. Times the load.
/// 2. Logs a single structured line per model with name + compute
///    units + elapsed ms.
/// 3. When the requested compute units include `.cpuAndNeuralEngine`
///    (or `.all`) and the load completes faster than a typical ANE
///    compile, emits a one-time warning pointing users at the
///    fallback diagnostic.
///
/// Usage:
/// ```swift
/// let encoder = try CoreMLLoader.load(
///     url: cacheDir.appendingPathComponent("encoder.mlmodelc"),
///     computeUnits: .cpuAndNeuralEngine,
///     name: "parakeet-eou-encoder"
/// )
/// ```
public enum CoreMLLoader {

    /// Seconds under which an ANE-eligible load is considered suspicious
    /// (likely CPU fallback). Calibrated against observed behaviour:
    /// - Successful ANE compile: ~200–800 ms on cold cache, ~20–50 ms
    ///   cached.
    /// - CPU fallback after ANE compile failure: <10 ms regardless of
    ///   cache state.
    ///
    /// Picking 15 ms keeps false positives low on warm caches while
    /// still catching the silent-fallback case on cold systems.
    private static let aneCompileFloorSeconds: Double = 0.015

    /// Track which model names we've already warned about so we don't
    /// spam the log. Protected by ``warnedQueue``.
    private static var warnedNames = Set<String>()
    private static let warnedQueue = DispatchQueue(
        label: "com.qwen3speech.coreml-loader.warned"
    )

    /// Load a compiled CoreML model with instrumentation.
    public static func load(
        url: URL,
        computeUnits: MLComputeUnits,
        name: String? = nil
    ) throws -> MLModel {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        return try load(url: url, configuration: config, name: name)
    }

    /// Load with an explicit ``MLModelConfiguration``.
    public static func load(
        url: URL,
        configuration: MLModelConfiguration,
        name: String? = nil
    ) throws -> MLModel {
        let label = name ?? url.deletingPathExtension().lastPathComponent
        let unitsLabel = describe(units: configuration.computeUnits)
        let start = Date()
        let model = try MLModel(contentsOf: url, configuration: configuration)
        let elapsed = Date().timeIntervalSince(start)
        let ms = Int((elapsed * 1000).rounded())
        AudioLog.modelLoading.info("CoreML loaded \(label) in \(ms)ms (units=\(unitsLabel))")

        let aneEligible =
            configuration.computeUnits == .cpuAndNeuralEngine ||
            configuration.computeUnits == .all
        if aneEligible && elapsed < aneCompileFloorSeconds {
            maybeWarn(
                name: label,
                message: """
                CoreML model '\(label)' loaded in \(ms)ms with compute units \
                \(unitsLabel). This is faster than a typical Neural Engine \
                compile (~200–800 ms cold, ~20–50 ms cached). If console logs \
                show 'MILCompilerForANE error', the model has fallen back to \
                CPU and inference may be 5–20× slower than expected.
                """
            )
        }
        return model
    }

    // MARK: - Private

    private static func maybeWarn(name: String, message: String) {
        warnedQueue.sync {
            guard !warnedNames.contains(name) else { return }
            warnedNames.insert(name)
            AudioLog.modelLoading.warning("\(message)")
        }
    }

    private static func describe(units: MLComputeUnits) -> String {
        switch units {
        case .cpuOnly: return "cpuOnly"
        case .cpuAndGPU: return "cpuAndGPU"
        case .all: return "all"
        case .cpuAndNeuralEngine: return "cpuAndNeuralEngine"
        @unknown default: return "unknown(\(units.rawValue))"
        }
    }

    /// Reset the per-process warning set. Exposed for tests so a fresh
    /// run of the helper can emit a warning again.
    public static func resetWarningState() {
        warnedQueue.sync { warnedNames.removeAll() }
    }
}
