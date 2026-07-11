import Foundation

/// Per-stage wall-clock timing for one `SourceSeparator.separate(...)` call.
///
/// Populated and delivered to the caller via the `metricsHandler` callback on
/// `separate(...)`. All times are in seconds. Use `rtf` to compare against
/// audio duration (lower = faster; <1.0 means faster than real-time).
public struct SourceSeparationMetrics: Sendable {
    /// Total audio duration in seconds (mono length / sample rate).
    public var audioSeconds: Double = 0

    /// Forward STFT on both channels.
    public var stftForwardSec: Double = 0

    /// Time spent building MLX graphs for all target models (CPU; lazy ops
    /// queued for the GPU but not yet materialized).
    public var modelGraphBuildSec: Double = 0

    /// Single GPU eval for all target models combined. With lazy launching the
    /// MLX scheduler interleaves kernel dispatches across stems, so this is the
    /// real GPU wall time (not the sum of per-target compute).
    public var modelEvalSec: Double = 0

    /// Per-target CPU unpack time (MLX → Swift nested arrays). GPU compute is
    /// in `modelEvalSec`; this is just the readback + reshape.
    public var modelSecByTarget: [SeparationTarget: Double] = [:]

    /// Wiener EM post-filter (zero if `wiener: false` or fewer than 2 targets).
    public var wienerSec: Double = 0

    /// Inverse STFT for all stems (sum across targets and channels).
    public var inverseStftSec: Double = 0

    /// End-to-end wall time for the `separate(...)` call.
    public var totalSec: Double = 0

    /// `totalSec / audioSeconds`. Lower is faster. <1.0 = faster than real-time.
    public var rtf: Double {
        guard audioSeconds > 0 else { return 0 }
        return totalSec / audioSeconds
    }

    /// Total model-related time: graph build + GPU eval + CPU unpack.
    public var modelTotalSec: Double {
        modelGraphBuildSec + modelEvalSec + modelSecByTarget.values.reduce(0, +)
    }

    public init() {}
}
