import XCTest
import MLX
@testable import Qwen3ASR

/// Exercises the model's real unload/destruction wiring without downloads.
final class E2EQwen3ASRCacheLimitLifecycleTests: XCTestCase {
    func testUnloadPreservesRemainingModelAndRestoresCallerBudget() {
        let prior = Memory.cacheLimit
        defer { Memory.cacheLimit = prior }
        let initial = 8 * 1024 * 1024 * 1024
        Memory.cacheLimit = initial
        let large = Qwen3ASRModel()
        let small = Qwen3ASRModel()
        large.mlxCacheLimitLease = Qwen3ASRCacheLimitCoordinator.shared.acquire(
            ceiling: 4 * 1024 * 1024 * 1024)
        small.mlxCacheLimitLease = Qwen3ASRCacheLimitCoordinator.shared.acquire(
            ceiling: 1 * 1024 * 1024 * 1024)

        large.unload()
        XCTAssertEqual(Memory.cacheLimit, 1 * 1024 * 1024 * 1024)
        XCTAssertFalse(large.isLoaded)
        XCTAssertNil(large.mlxCacheLimitLease)
        small.unload()
        XCTAssertEqual(Memory.cacheLimit, initial)
        small.unload()
        XCTAssertEqual(Memory.cacheLimit, initial)
    }

    func testUnloadWithoutRegistrationLeavesCallerBudgetAlone() {
        let prior = Memory.cacheLimit
        defer { Memory.cacheLimit = prior }
        let initial = 6 * 1024 * 1024 * 1024
        Memory.cacheLimit = initial
        let model = Qwen3ASRModel()
        model.unload()
        XCTAssertEqual(Memory.cacheLimit, initial)
    }

    func testModelDestructionReleasesCacheCeiling() {
        let prior = Memory.cacheLimit
        defer { Memory.cacheLimit = prior }
        let initial = 8 * 1024 * 1024 * 1024
        Memory.cacheLimit = initial
        var model: Qwen3ASRModel? = Qwen3ASRModel()
        model?.mlxCacheLimitLease = Qwen3ASRCacheLimitCoordinator.shared.acquire(
            ceiling: 1 * 1024 * 1024 * 1024)
        XCTAssertEqual(Memory.cacheLimit, 1 * 1024 * 1024 * 1024)
        withExtendedLifetime(model) {}
        model = nil
        XCTAssertEqual(Memory.cacheLimit, initial)
    }
}
