#if canImport(CoreML)
import CoreML
import Foundation

/// Resolves the `MLComputeUnits` a CoreML model should load with, honoring
/// the `SPEECH_COREML_COMPUTE_UNITS` environment override.
///
/// **Why this exists.** A `.mlmodelc` is compiled MIL, but the device-specific
/// program (ANE *or* GPU/Metal) is generated the *first time* the model loads.
/// On real M-series hardware that first-load codegen takes seconds; on a
/// virtualized GitHub `macos-15` runner it **hangs**: the runner has no usable
/// Neural Engine (so `.cpuAndNeuralEngine`/`.all` stall attempting the ANE
/// compile) AND its paravirtual GPU can't JIT CoreML's Metal program for a
/// stateful graph (so `.cpuAndGPU` stalls too). Both were observed as 17-27 min
/// silent hangs loading the Qwen3-ASR T=128 stateful decoder.
///
/// On-device we keep the normal default (callers pass it as `fallback`; the env
/// is unset so this is a no-op). In CI we set `SPEECH_COREML_COMPUTE_UNITS=cpuOnly`
/// so every loader skips ANE *and* GPU codegen — pure CPU MIL execution loads
/// instantly, is deterministic, and yields identical text for our roundtrip
/// assertions (~86 ms/step for the T=128 decoder, fine for correctness tests).
public enum CoreMLComputeUnitsResolver {
    public static let envKey = "SPEECH_COREML_COMPUTE_UNITS"

    /// Returns the env-overridden compute units, or `fallback` when unset/unrecognized.
    /// Accepted env values (case-insensitive): `ane`/`cpuAndNeuralEngine`,
    /// `gpu`/`cpuAndGPU`, `cpu`/`cpuOnly`, `all`.
    public static func resolved(default fallback: MLComputeUnits) -> MLComputeUnits {
        guard let raw = ProcessInfo.processInfo.environment[envKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased(), !raw.isEmpty
        else {
            return fallback
        }
        switch raw {
        case "ane", "cpuandneuralengine", "neuralengine":
            return .cpuAndNeuralEngine
        case "gpu", "cpuandgpu":
            return .cpuAndGPU
        case "cpu", "cpuonly":
            return .cpuOnly
        case "all":
            return .all
        default:
            return fallback
        }
    }
}
#endif
