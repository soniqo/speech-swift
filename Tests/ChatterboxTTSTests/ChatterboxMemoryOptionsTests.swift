import MLX
import XCTest

@testable import ChatterboxTTS

final class ChatterboxMemoryOptionsTests: XCTestCase {
    func testBalancedDefaultsBoundCacheAndClearStages() {
        let options = ChatterboxMemoryOptions.balanced

        XCTAssertEqual(options.cacheLimitBytes, 512 * 1024 * 1024)
        XCTAssertTrue(options.clearCacheBetweenStages)
        XCTAssertTrue(options.clearCacheOnCompletion)
    }

    func testUnrestrictedPreservesMLXCacheBehavior() {
        let options = ChatterboxMemoryOptions.unrestricted

        XCTAssertNil(options.cacheLimitBytes)
        XCTAssertFalse(options.clearCacheBetweenStages)
        XCTAssertFalse(options.clearCacheOnCompletion)
    }

    func testWithOptionsRestoresPriorMLXCacheLimit() {
        let priorCap = Memory.cacheLimit
        defer { Memory.cacheLimit = priorCap }

        let initialCap = 3 * 1024 * 1024 * 1024
        let boundedCap = 512 * 1024 * 1024
        Memory.cacheLimit = initialCap

        ChatterboxMemory.withOptions(.balanced) {
            XCTAssertEqual(Memory.cacheLimit, boundedCap)
        }

        XCTAssertEqual(Memory.cacheLimit, initialCap)
    }

    func testWithOptionsDoesNotRaiseStricterCallerCacheLimit() {
        let priorCap = Memory.cacheLimit
        defer { Memory.cacheLimit = priorCap }

        let stricterCap = 256 * 1024 * 1024
        Memory.cacheLimit = stricterCap

        ChatterboxMemory.withOptions(.balanced) {
            XCTAssertEqual(Memory.cacheLimit, stricterCap)
        }

        XCTAssertEqual(Memory.cacheLimit, stricterCap)
    }
}
