import Cmlx
import Foundation
import MLX

/// Metal GPU memory budget utilities.
public enum MetalBudget {

    /// Query real Metal headroom: recommended working set minus active allocations.
    /// Returns nil if Metal device info is unavailable.
    public static var availableBytes: Int? {
        let info = GPU.deviceInfo()
        let maxWorking = Int(info.maxRecommendedWorkingSetSize)
        guard maxWorking > 0 else { return nil }
        let active = Memory.activeMemory
        let overhead = 256 * 1024 * 1024  // 256 MB safety margin
        return max(0, maxWorking - active - overhead)
    }

    /// Total device memory in bytes.
    public static var totalMemory: Int {
        GPU.deviceInfo().memorySize
    }

    /// Maximum recommended working set size in bytes.
    public static var maxRecommendedWorkingSet: Int {
        Int(GPU.deviceInfo().maxRecommendedWorkingSetSize)
    }

    /// Currently active (non-cache) MLX memory in bytes.
    public static var activeMemory: Int {
        Memory.activeMemory
    }

    /// Fraction of the recommended working set wired by default.
    ///
    /// macOS keeps 90%: there, wiring genuinely prevents paging under
    /// memory pressure. iOS and the other embedded platforms pin nothing.
    /// ``maxRecommendedWorkingSetSize`` describes the *device's* GPU budget,
    /// but what terminates an app on those platforms is its own jetsam
    /// limit, which is far smaller and treats wired pages as
    /// non-reclaimable. Wiring 90% of the device working set there asks the
    /// OS to hold memory it may need back in order to keep the app alive.
    /// A caller that wants pinning anyway can still pass a fraction.
    #if os(macOS)
    public static let defaultPinFraction: Double = 0.9
    #else
    public static let defaultPinFraction: Double = 0.0
    #endif

    /// The wired limit implied by a working-set size and fraction, or `nil`
    /// when no limit should be set at all.
    ///
    /// Pure, so the policy is testable on any platform rather than only on
    /// the one running the tests.
    public static func wiredLimit(workingSet: Int, fraction: Double) -> Int? {
        guard fraction > 0, workingSet > 0 else { return nil }
        return Int(Double(workingSet) * fraction)
    }

    /// Pin GPU memory to prevent paging under pressure.
    /// Uses ``defaultPinFraction`` — 90% of the recommended working set on
    /// macOS, nothing on iOS. Only effective on macOS 15+ / iOS 18+.
    ///
    /// Returns the previous wired limit, or 0 when no limit was applied.
    @discardableResult
    public static func pinMemory(fraction: Double = defaultPinFraction) -> Int {
        guard let limit = wiredLimit(workingSet: maxRecommendedWorkingSet, fraction: fraction)
        else { return 0 }
        var previous: size_t = 0
        mlx_set_wired_limit(&previous, size_t(limit))
        return Int(previous)
    }

    /// Unpin GPU memory (set wired limit to 0).
    @discardableResult
    public static func unpinMemory() -> Int {
        var previous: size_t = 0
        mlx_set_wired_limit(&previous, 0)
        return Int(previous)
    }

    /// Check if a model of the given size (bytes) can fit in available GPU memory.
    public static func canFit(modelBytes: Int) -> Bool {
        guard let available = availableBytes else { return true }
        return modelBytes <= available
    }
}
