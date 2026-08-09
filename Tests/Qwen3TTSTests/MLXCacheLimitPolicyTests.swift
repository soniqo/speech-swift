import MLX
import XCTest

private final class CacheLimitProbeCase: E2ETestCase {
    override class var mlxCacheLimitBytes: Int { 512 << 20 }
}

final class MLXCacheLimitPolicyTests: XCTestCase {
    func testAppliesBoundAndRestoresPreviousLimit() {
        exercisePolicy(
            initialLimit: 1 << 30,
            expectedAppliedLimit: 512 << 20)
    }

    func testPreservesStricterExistingLimit() {
        exercisePolicy(
            initialLimit: 256 << 20,
            expectedAppliedLimit: 256 << 20)
    }

    private func exercisePolicy(
        initialLimit: Int,
        expectedAppliedLimit: Int,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let processLimit = MLX.Memory.cacheLimit
        defer {
            MLX.Memory.cacheLimit = processLimit
            MLX.Memory.clearCache()
        }

        MLX.Memory.cacheLimit = initialLimit
        CacheLimitProbeCase.setUp()

        XCTAssertEqual(
            MLX.Memory.cacheLimit,
            expectedAppliedLimit,
            file: file,
            line: line)

        CacheLimitProbeCase.tearDown()

        XCTAssertEqual(
            MLX.Memory.cacheLimit,
            initialLimit,
            file: file,
            line: line)
    }
}
