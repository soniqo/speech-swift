import XCTest
import Foundation
@testable import Qwen3ASR

/// Unit tests for the ``Qwen3ASRMemory`` helpers added by Bug 4b. These
/// pin the cache-limit math, the soft-warning threshold and the snapshot
/// formatter. They are pure — no model download, no GPU — so they run on
/// every CI shard.
final class Qwen3MemoryGuardTests: XCTestCase {

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

    // MARK: - cacheLimitForSmall

    func testCacheLimitForSmall_8GBMacReturnsEighthRAM() {
        // 8 GB / 8 = 1 GB, exactly the cap. min picks 1 GB.
        let eightGB = 8 * 1024 * 1024 * 1024
        let oneGB = 1 * 1024 * 1024 * 1024
        XCTAssertEqual(
            Qwen3ASRMemory.cacheLimitForSmall(physicalMemoryBytes: eightGB),
            oneGB
        )
    }

    func testCacheLimitForSmall_4GBMacReturnsEighthRAM() {
        // 4 GB / 8 = 512 MB, which is below the 1 GB cap → eighth-RAM wins.
        let fourGB = 4 * 1024 * 1024 * 1024
        let fiveTwelveMB = 512 * 1024 * 1024
        XCTAssertEqual(
            Qwen3ASRMemory.cacheLimitForSmall(physicalMemoryBytes: fourGB),
            fiveTwelveMB
        )
    }

    func testCacheLimitForSmall_16GBMacCapDominates() {
        // 16 GB / 8 = 2 GB; cap clamps to 1 GB.
        let sixteenGB = 16 * 1024 * 1024 * 1024
        let oneGB = 1 * 1024 * 1024 * 1024
        XCTAssertEqual(
            Qwen3ASRMemory.cacheLimitForSmall(physicalMemoryBytes: sixteenGB),
            oneGB
        )
    }

    func testCacheLimitForSmall_64GBMacCapDominates() {
        // 64 GB / 8 = 8 GB; cap clamps to 1 GB.
        let sixtyFourGB = 64 * 1024 * 1024 * 1024
        let oneGB = 1 * 1024 * 1024 * 1024
        XCTAssertEqual(
            Qwen3ASRMemory.cacheLimitForSmall(physicalMemoryBytes: sixtyFourGB),
            oneGB
        )
    }

    func testCacheLimitForSmall_ZeroReturnsZero() {
        // Edge case: 0 physical memory shouldn't underflow.
        XCTAssertEqual(
            Qwen3ASRMemory.cacheLimitForSmall(physicalMemoryBytes: 0),
            0
        )
    }

    func testCacheLimitForSmall_NegativeClampedToZero() {
        // The max(0, …) clamp must absorb pathological negatives.
        XCTAssertEqual(
            Qwen3ASRMemory.cacheLimitForSmall(physicalMemoryBytes: Int.min),
            0
        )
    }

    func testCacheLimitForSmall_StaysBelowCacheLimitForLarge() {
        // Sanity: the small-model cap must never exceed the large-model
        // cap at the same RAM size — the 0.6B decoder's working set is
        // smaller, so its ceiling should be tighter or equal, never looser.
        for physicalGB in [4, 8, 16, 24, 32, 64, 128] {
            let bytes = physicalGB * 1024 * 1024 * 1024
            XCTAssertLessThanOrEqual(
                Qwen3ASRMemory.cacheLimitForSmall(physicalMemoryBytes: bytes),
                Qwen3ASRMemory.cacheLimitForLarge(physicalMemoryBytes: bytes),
                "at \(physicalGB) GB RAM"
            )
        }
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

}
