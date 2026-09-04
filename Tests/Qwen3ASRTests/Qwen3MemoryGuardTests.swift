import XCTest
import MLX
import MLXCommon
import Foundation
@testable import Qwen3ASR

/// Unit tests for the ``Qwen3ASRMemory`` helpers added by Bug 4b. These
/// pin the cache-limit math, the soft-warning threshold and the snapshot
/// formatter. They are pure — no model download, no GPU — so they run on
/// every CI shard.
final class Qwen3MemoryGuardTests: XCTestCase {

    // MARK: - Platform-aware cache ceiling (#479 hardening)

    /// On macOS only the 1.7B variant is capped — the case observed to swap.
    /// Leaving the 0.6B alone preserves existing desktop behaviour.
    func testDesktopCapsOnlyTheLargeVariant() {
        let sixteenGB = 16 * 1024 * 1024 * 1024

        XCTAssertNil(
            Qwen3ASRMemory.cacheLimitBytes(
                isLargeModel: false, isMemoryConstrainedPlatform: false,
                physicalMemoryBytes: sixteenGB),
            "the 0.6B model on a Mac should keep MLX's default cache limit")
        XCTAssertEqual(
            Qwen3ASRMemory.cacheLimitBytes(
                isLargeModel: true, isMemoryConstrainedPlatform: false,
                physicalMemoryBytes: sixteenGB),
            Qwen3ASRMemory.cacheLimitForLarge(physicalMemoryBytes: sixteenGB))
    }

    /// On a memory-constrained platform every variant is capped. MLX's
    /// default tracks the device's recommended working set, which has no
    /// relation to the app's jetsam budget, so the small model was running
    /// with no ceiling at all on iOS.
    func testConstrainedPlatformCapsEveryVariant() {
        let twelveGB = 12 * 1024 * 1024 * 1024
        let expected = Qwen3ASRMemory.cacheLimitForLarge(physicalMemoryBytes: twelveGB)

        for isLarge in [false, true] {
            XCTAssertEqual(
                Qwen3ASRMemory.cacheLimitBytes(
                    isLargeModel: isLarge, isMemoryConstrainedPlatform: true,
                    physicalMemoryBytes: twelveGB),
                expected,
                "isLargeModel=\(isLarge) should be capped on a constrained platform")
        }
    }

    // MARK: - Wired-memory pinning policy (#479 hardening)

    /// Pinning 90% of the *device* working set is a macOS policy. On iOS
    /// what kills the app is its own jetsam limit, and wired pages are not
    /// reclaimable, so the default there is to pin nothing.
    func testPinFractionZeroMeansNoWiredLimit() {
        XCTAssertNil(MetalBudget.wiredLimit(workingSet: 8 * 1024 * 1024 * 1024, fraction: 0),
                     "a zero fraction must set no wired limit at all")
        XCTAssertNil(MetalBudget.wiredLimit(workingSet: 0, fraction: 0.9),
                     "an unknown working set must set no wired limit")
        XCTAssertEqual(MetalBudget.wiredLimit(workingSet: 1000, fraction: 0.9), 900)
    }

    #if os(macOS)
    func testDesktopStillPinsNinetyPercent() {
        XCTAssertEqual(MetalBudget.defaultPinFraction, 0.9,
                       "macOS behaviour must be unchanged")
    }
    #else
    func testConstrainedPlatformPinsNothingByDefault() {
        XCTAssertEqual(MetalBudget.defaultPinFraction, 0.0)
    }
    #endif

    // MARK: - MLX errors are survivable (#479 hardening)

    /// mlx-swift's default handler calls `fatalError`, so any MLX failure
    /// took the host process with it. Transcription must degrade to a
    /// diagnostic string instead.
    func testMLXErrorsBecomeADiagnosticStringNotAProcessKill() {
        struct Boom: Error, LocalizedError {
            var errorDescription: String? { "simulated MLX failure" }
        }

        XCTAssertEqual(
            Qwen3ASRModel.catchingMLXErrors { "transcript" },
            "transcript",
            "the guard must be transparent when nothing fails")

        let result = Qwen3ASRModel.catchingMLXErrors { throw Boom() }
        XCTAssertTrue(result.hasPrefix("[Qwen3ASR MLX error:"),
                      "a failure must surface as a diagnostic, got: \(result)")
        XCTAssertTrue(result.contains("simulated MLX failure"),
                      "the diagnostic must carry the underlying reason")
    }

    // MARK: - cacheLimitForLarge

    func testCacheLimitForLarge_8GBMacReturnsQuarterRAM() {
        // 8 GB / 4 = 2 GB, which is below the 4 GB cap → quarter-RAM wins.
        let eightGB = 8 * 1024 * 1024 * 1024
        let twoGB = 2 * 1024 * 1024 * 1024
        XCTAssertEqual(
            Qwen3ASRMemory.cacheLimitForLarge(physicalMemoryBytes: eightGB),
            twoGB
        )
    }

    func testCacheLimitForLarge_16GBMacReturnsCap() {
        // 16 GB / 4 = 4 GB, exactly the cap. min picks 4 GB.
        let sixteenGB = 16 * 1024 * 1024 * 1024
        let fourGB = 4 * 1024 * 1024 * 1024
        XCTAssertEqual(
            Qwen3ASRMemory.cacheLimitForLarge(physicalMemoryBytes: sixteenGB),
            fourGB
        )
    }

    func testCacheLimitForLarge_24GBMacCapDominates() {
        // 24 GB / 4 = 6 GB; cap clamps to 4 GB.
        let twentyFourGB = 24 * 1024 * 1024 * 1024
        let fourGB = 4 * 1024 * 1024 * 1024
        XCTAssertEqual(
            Qwen3ASRMemory.cacheLimitForLarge(physicalMemoryBytes: twentyFourGB),
            fourGB
        )
    }

    func testCacheLimitForLarge_64GBMacCapDominates() {
        // 64 GB / 4 = 16 GB; cap clamps to 4 GB.
        let sixtyFourGB = 64 * 1024 * 1024 * 1024
        let fourGB = 4 * 1024 * 1024 * 1024
        XCTAssertEqual(
            Qwen3ASRMemory.cacheLimitForLarge(physicalMemoryBytes: sixtyFourGB),
            fourGB
        )
    }

    func testCacheLimitForLarge_ZeroReturnsZero() {
        // Edge case: 0 physical memory shouldn't underflow.
        XCTAssertEqual(
            Qwen3ASRMemory.cacheLimitForLarge(physicalMemoryBytes: 0),
            0
        )
    }

    func testCacheLimitForLarge_NegativeClampedToZero() {
        // The max(0, …) clamp must absorb pathological negatives.
        XCTAssertEqual(
            Qwen3ASRMemory.cacheLimitForLarge(physicalMemoryBytes: Int.min),
            0
        )
    }

    // MARK: - shouldWarnForLarge

    func testShouldWarnForLarge_8GBWarns() {
        let eightGB: UInt64 = 8 * 1024 * 1024 * 1024
        XCTAssertTrue(Qwen3ASRMemory.shouldWarnForLarge(physicalMemoryBytes: eightGB))
    }

    func testShouldWarnForLarge_16GBWarns() {
        let sixteenGB: UInt64 = 16 * 1024 * 1024 * 1024
        XCTAssertTrue(Qwen3ASRMemory.shouldWarnForLarge(physicalMemoryBytes: sixteenGB))
    }

    func testShouldWarnForLarge_JustBelow24GBWarns() {
        // 23.99 GB → still strictly less than threshold.
        let almost24GB = UInt64(23.99 * 1_073_741_824.0)
        XCTAssertTrue(Qwen3ASRMemory.shouldWarnForLarge(physicalMemoryBytes: almost24GB))
    }

    func testShouldWarnForLarge_Exactly24GBDoesNotWarn() {
        // Boundary is `<`, not `<=` → exactly 24 GB is safe.
        let twentyFourGB: UInt64 = 24 * 1024 * 1024 * 1024
        XCTAssertFalse(Qwen3ASRMemory.shouldWarnForLarge(physicalMemoryBytes: twentyFourGB))
    }

    func testShouldWarnForLarge_32GBDoesNotWarn() {
        let thirtyTwoGB: UInt64 = 32 * 1024 * 1024 * 1024
        XCTAssertFalse(Qwen3ASRMemory.shouldWarnForLarge(physicalMemoryBytes: thirtyTwoGB))
    }

    func testShouldWarnForLarge_64GBDoesNotWarn() {
        let sixtyFourGB: UInt64 = 64 * 1024 * 1024 * 1024
        XCTAssertFalse(Qwen3ASRMemory.shouldWarnForLarge(physicalMemoryBytes: sixtyFourGB))
    }

    // MARK: - Threshold constant

    func testThresholdConstantIs24GB() {
        // Pinning the doc'd constant so accidental edits surface here.
        XCTAssertEqual(Qwen3ASRMemory.largeModelRAMWarningThresholdGB, 24.0)
    }

    // MARK: - formatSnapshot

    func testFormatSnapshot_ContainsAllFieldsAndLabel() {
        // Synthetic snapshot: 100 MB active, 50 MB cache, 200 MB peak.
        // Uses the int-based overload so the test doesn't depend on
        // MLX.Memory.Snapshot's sealed initializer.
        let formatted = Qwen3ASRMemory.formatSnapshot(
            active: 100 * 1_048_576,
            cache: 50 * 1_048_576,
            peak: 200 * 1_048_576,
            label: "test-label")

        // Shape: label + each named field + MB suffix on each value.
        XCTAssertTrue(formatted.contains("test-label"),
                      "Expected label in output, got: \(formatted)")
        XCTAssertTrue(formatted.contains("active="),
                      "Expected active= in output, got: \(formatted)")
        XCTAssertTrue(formatted.contains("cache="),
                      "Expected cache= in output, got: \(formatted)")
        XCTAssertTrue(formatted.contains("peak="),
                      "Expected peak= in output, got: \(formatted)")
        XCTAssertTrue(formatted.contains("MB"),
                      "Expected MB suffix in output, got: \(formatted)")
    }

    func testFormatSnapshot_RendersExpectedMBValues() {
        // 100/50/200 MB inputs should appear verbatim in the rendered string.
        // Uses the int-based overload so the test doesn't depend on
        // MLX.Memory.Snapshot's sealed init.
        let formatted = Qwen3ASRMemory.formatSnapshot(
            active: 100 * 1_048_576,
            cache: 50 * 1_048_576,
            peak: 200 * 1_048_576,
            label: "pre-load")

        XCTAssertTrue(formatted.contains("active=100 MB"),
                      "Expected active=100 MB, got: \(formatted)")
        XCTAssertTrue(formatted.contains("cache=50 MB"),
                      "Expected cache=50 MB, got: \(formatted)")
        XCTAssertTrue(formatted.contains("peak=200 MB"),
                      "Expected peak=200 MB, got: \(formatted)")
        XCTAssertTrue(formatted.contains("pre-load"),
                      "Expected pre-load label, got: \(formatted)")
    }

    func testFormatSnapshot_ZeroSnapshotRendersZeroMB() {
        // Empty snapshot should not crash and should render 0 MB consistently.
        let formatted = Qwen3ASRMemory.formatSnapshot(
            active: 0, cache: 0, peak: 0, label: "empty")

        XCTAssertTrue(formatted.contains("active=0 MB"))
        XCTAssertTrue(formatted.contains("cache=0 MB"))
        XCTAssertTrue(formatted.contains("peak=0 MB"))
        XCTAssertTrue(formatted.contains("empty"))
    }

    // MARK: - cacheLimit save/restore (PersonaPlex regression coverage)

    /// Adversarial-review finding from the workflow: `MLX.Memory.cacheLimit`
    /// is process-global. The original fix added a per-instance
    /// `savedMLXCacheLimit` so `unload()` restores the prior cap and
    /// co-loaded models (PersonaPlex loads ASR + Mimi + LM in the same
    /// process) don't inherit our 4 GB ceiling. This test exercises the
    /// save/restore loop directly: simulate what `fromPretrained` does for
    /// `.large`, then call `unload()` and assert the cap is returned to
    /// the pre-load value. No model weights touched — pure plumbing test.
    func testUnload_RestoresPriorMLXCacheLimit() {
        let initialCap = 8 * 1024 * 1024 * 1024  // 8 GB starting state
        let appliedCap = 4 * 1024 * 1024 * 1024  // what .large would apply
        let priorCap = MLX.Memory.cacheLimit
        defer { MLX.Memory.cacheLimit = priorCap }  // never leak test state

        MLX.Memory.cacheLimit = initialCap

        let model = Qwen3ASRModel(
            audioConfig: ASRModelSize.large.audioConfig,
            textConfig: ASRModelSize.large.textConfig(bits: 8))

        // Simulate `fromPretrained`'s save-then-cap step (only the global
        // state and the instance var; we don't actually load weights).
        model.savedMLXCacheLimit = MLX.Memory.cacheLimit
        MLX.Memory.cacheLimit = appliedCap

        XCTAssertEqual(MLX.Memory.cacheLimit, appliedCap,
                       "precondition: cap should be applied at this point")
        XCTAssertEqual(model.savedMLXCacheLimit, initialCap,
                       "precondition: saved limit should match the prior value")

        model.unload()

        XCTAssertEqual(MLX.Memory.cacheLimit, initialCap,
                       "unload() must restore MLX.Memory.cacheLimit to the saved value — otherwise co-loaded models (PersonaPlex) inherit our 4 GB cap")
        XCTAssertNil(model.savedMLXCacheLimit,
                     "unload() must clear the saved value so a re-load doesn't restore a stale limit")
    }

    /// Small variant doesn't apply a cap (savedMLXCacheLimit stays nil), so
    /// unload() must NOT modify the global cache limit. Co-loaded models
    /// must see exactly what they configured themselves.
    func testUnload_DoesNotRestoreWhenNoSaveWasMade() {
        let observableCap = 6 * 1024 * 1024 * 1024  // some non-default value
        let priorCap = MLX.Memory.cacheLimit
        defer { MLX.Memory.cacheLimit = priorCap }

        MLX.Memory.cacheLimit = observableCap

        let model = Qwen3ASRModel(
            audioConfig: ASRModelSize.small.audioConfig,
            textConfig: ASRModelSize.small.textConfig(bits: 4))
        // Small variant: fromPretrained leaves savedMLXCacheLimit at nil.
        XCTAssertNil(model.savedMLXCacheLimit)

        model.unload()

        XCTAssertEqual(MLX.Memory.cacheLimit, observableCap,
                       "unload() on a non-large model must not touch MLX.Memory.cacheLimit")
    }
}
