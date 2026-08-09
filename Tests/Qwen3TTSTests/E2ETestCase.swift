import MLX
import XCTest

/// Bounds reusable MLX buffers while a Qwen3-TTS E2E suite runs, then restores
/// the process-wide cache policy for subsequent suites.
class E2ETestCase: XCTestCase {

    /// Maximum cache allowance. An existing stricter limit is preserved.
    class var mlxCacheLimitBytes: Int { 512 << 20 }

    private static var previousMLXCacheLimit: Int?

    override class func setUp() {
        super.setUp()
        let previousLimit = MLX.Memory.cacheLimit
        previousMLXCacheLimit = previousLimit
        MLX.Memory.cacheLimit = min(previousLimit, max(0, mlxCacheLimitBytes))
        MLX.Memory.clearCache()
    }

    override func tearDown() {
        MLX.Memory.clearCache()
        super.tearDown()
    }

    override class func tearDown() {
        MLX.Memory.clearCache()
        if let previousLimit = previousMLXCacheLimit {
            MLX.Memory.cacheLimit = previousLimit
            previousMLXCacheLimit = nil
        }
        super.tearDown()
    }
}
