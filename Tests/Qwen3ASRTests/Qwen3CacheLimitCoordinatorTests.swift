import Foundation
import XCTest
@testable import Qwen3ASR

/// Ownership and caller-budget regressions, with no Metal initialization.
final class Qwen3CacheLimitCoordinatorTests: XCTestCase {
    private final class Budget: @unchecked Sendable {
        var limit: Int
        init(_ limit: Int) { self.limit = limit }
    }

    private func makeCoordinator(_ budget: Budget) -> Qwen3ASRCacheLimitCoordinator {
        Qwen3ASRCacheLimitCoordinator(
            getLimit: { budget.limit }, setLimit: { budget.limit = $0 })
    }

    func testMixedCeilingsSurviveEitherUnloadOrder() {
        for smallFirst in [false, true] {
            for unloadSmallFirst in [false, true] {
                let budget = Budget(8)
                let coordinator = makeCoordinator(budget)
                let first = coordinator.acquire(ceiling: smallFirst ? 1 : 4)
                let second = coordinator.acquire(ceiling: smallFirst ? 4 : 1)
                XCTAssertEqual(budget.limit, 1)
                let small = smallFirst ? first : second
                let large = smallFirst ? second : first
                if unloadSmallFirst {
                    small?.release()
                    XCTAssertEqual(budget.limit, 4)
                    large?.release()
                } else {
                    large?.release()
                    XCTAssertEqual(budget.limit, 1)
                    small?.release()
                }
                XCTAssertEqual(budget.limit, 8)
            }
        }
    }

    func testEqualCeilingsRemainUntilLastOwnerInEitherOrder() {
        for releaseFirst in [false, true] {
            let budget = Budget(8)
            let coordinator = makeCoordinator(budget)
            let first = coordinator.acquire(ceiling: 1)
            let second = coordinator.acquire(ceiling: 1)
            (releaseFirst ? first : second)?.release()
            XCTAssertEqual(budget.limit, 1)
            (releaseFirst ? second : first)?.release()
            XCTAssertEqual(budget.limit, 8)
        }
    }

    func testPreexistingCallerLimitIsNeverRaised() {
        for callerLimit in [0, 1, 2] {
            let budget = Budget(callerLimit)
            let coordinator = makeCoordinator(budget)
            let lease = coordinator.acquire(ceiling: 4)
            XCTAssertEqual(budget.limit, callerLimit)
            lease?.release()
            XCTAssertEqual(budget.limit, callerLimit)
        }
    }

    func testTighterCallerLimitDuringUseSurvivesUnload() {
        let budget = Budget(8)
        let coordinator = makeCoordinator(budget)
        let large = coordinator.acquire(ceiling: 4)
        let small = coordinator.acquire(ceiling: 1)
        budget.limit = 0
        small?.release()
        XCTAssertEqual(budget.limit, 0)
        large?.release()
        XCTAssertEqual(budget.limit, 0)
    }

    func testCallerBudgetChangedDuringUseStillHonorsRemainingModels() {
        let budget = Budget(8)
        let coordinator = makeCoordinator(budget)
        let large = coordinator.acquire(ceiling: 4)
        let small = coordinator.acquire(ceiling: 1)
        budget.limit = 2
        large?.release()
        XCTAssertEqual(budget.limit, 1)
        small?.release()
        XCTAssertEqual(budget.limit, 2)
    }

    func testAcquiringAfterExternalChangeKeepsTheNewCallerBudget() {
        let budget = Budget(8)
        let coordinator = makeCoordinator(budget)
        let first = coordinator.acquire(ceiling: 4)
        budget.limit = 2
        let second = coordinator.acquire(ceiling: 1)
        second?.release()
        XCTAssertEqual(budget.limit, 2)
        first?.release()
        XCTAssertEqual(budget.limit, 2)
    }

    func testReleaseIsIdempotent() {
        let budget = Budget(8)
        let coordinator = makeCoordinator(budget)
        let lease = coordinator.acquire(ceiling: 1)
        lease?.release()
        budget.limit = 6
        lease?.release()
        XCTAssertEqual(budget.limit, 6)
    }

    func testLeaseDestructionReleasesItsCeiling() {
        let budget = Budget(8)
        let coordinator = makeCoordinator(budget)
        var lease = coordinator.acquire(ceiling: 1)
        XCTAssertEqual(budget.limit, 1)
        withExtendedLifetime(lease) {}
        lease = nil
        XCTAssertEqual(budget.limit, 8)
    }

    func testLaterLoadCapturesFreshCallerBudget() {
        let budget = Budget(8)
        let coordinator = makeCoordinator(budget)
        coordinator.acquire(ceiling: 1)?.release()
        budget.limit = 6
        let lease = coordinator.acquire(ceiling: 4)
        XCTAssertEqual(budget.limit, 4)
        lease?.release()
        XCTAssertEqual(budget.limit, 6)
    }

    func testNonPositiveCeilingDoesNotChangeCallerBudget() {
        let budget = Budget(8)
        let coordinator = makeCoordinator(budget)
        XCTAssertNil(coordinator.acquire(ceiling: 0))
        XCTAssertNil(coordinator.acquire(ceiling: -1))
        XCTAssertEqual(budget.limit, 8)
    }

    func testConcurrentRegistrationAndReleasePreserveTheBudget() {
        final class Owners: @unchecked Sendable {
            let lock = NSLock()
            var leases: [Qwen3ASRCacheLimitCoordinator.Lease] = []
            func append(_ lease: Qwen3ASRCacheLimitCoordinator.Lease) {
                lock.lock()
                defer { lock.unlock() }
                leases.append(lease)
            }
        }
        let budget = Budget(8)
        let coordinator = makeCoordinator(budget)
        let owners = Owners()
        DispatchQueue.concurrentPerform(iterations: 32) { index in
            if let lease = coordinator.acquire(ceiling: index.isMultiple(of: 2) ? 1 : 4) {
                owners.append(lease)
            }
        }
        XCTAssertEqual(budget.limit, 1)
        let leases = owners.leases
        DispatchQueue.concurrentPerform(iterations: leases.count) { index in
            leases[index].release()
        }
        XCTAssertEqual(budget.limit, 8)
    }
}
