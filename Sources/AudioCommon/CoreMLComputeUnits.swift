#if canImport(CoreML)
import CoreML
import Foundation

/// Resolves the `MLComputeUnits` a CoreML model should load with, honoring
/// the `SPEECH_COREML_COMPUTE_UNITS` environment override.
///
/// **Why this exists.** A `.mlmodelc` is compiled MIL, but the Apple Neural
/// Engine program is chip-and-OS-specific and is generated the *first time*
/// the model loads with `.cpuAndNeuralEngine` (or `.all`). On real M-series
/// hardware that first-load ANE compile takes seconds; on a virtualized
/// GitHub `macos-15` runner — which has no usable Neural Engine — the OS
/// still attempts the ANE compile of a large graph and **hangs** (observed:
/// a 27-minute silent stall loading the Qwen3-ASR T=128 decoder, vs ~5 min
/// for every other shard).
///
/// On-device we keep the ANE (callers pass their normal default as
/// `fallback`). In CI we set `SPEECH_COREML_COMPUTE_UNITS=cpuAndGPU` so the
/// runner skips the ANE compile entirely — GPU/CPU execution is fast and
/// deterministic (measured ~86 ms/step CPU for the T=128 decoder) and gives
/// the same numerical results for our text-matching assertions.
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
