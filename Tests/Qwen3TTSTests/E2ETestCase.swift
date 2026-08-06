import MLX
import XCTest

/// Base class for E2E suites, which bounds MLX's buffer pool.
///
/// MLX does not return freed buffers to the system; it pools them for reuse,
/// and the pool's default ceiling is a fraction of physical RAM (~0.95x). In a
/// long-lived xctest process that ceiling is never reached in a useful sense —
/// the pool simply grows for the length of the run and the machine runs out of
/// memory first.
///
/// Measured on a 16 GB machine before this change, `E2ETTSTests` alone, one
/// shared model, phys_footprint after each test:
///
///     testBaseModelNoDefaultInstruct   4029 MB   (mlx cache 2187 MB)
///     testEnglishLatency               7094 MB   (mlx cache 5298 MB)
///     testEnglishRoundTrip             8916 MB   (mlx cache 6359 MB)
///     testEnglishSynthesis            10317 MB   (mlx cache 7763 MB)
///
/// The `active` figure — the models themselves — stayed flat at 1.6-2.3 GB
/// throughout. All of the growth was pooled buffers. The full nine-test suite
/// could not finish: it passed 9 GB five tests in and was still climbing.
///
/// With the cap below, the same nine tests peak at 3089 MB and complete in 32 s,
/// the pool pinned at its limit and the footprint returning after every test.
///
/// This does not replace `scripts/test_e2e_isolated.sh`. That bounds memory
/// across suites; this bounds it within one. Both are needed — the per-suite
/// bound is only as good as the largest suite, and the largest suite was itself
/// larger than the machine.
class E2ETestCase: XCTestCase {

    /// Enough for a decode step's working set on the largest model here, and
    /// ~3% of a 16 GB machine. MLX's own guidance is that small caches usually
    /// perform as well as large ones; the risk of going lower is extra trips to
    /// the Metal allocator, not correctness.
    ///
    /// Override in a subclass that genuinely needs more.
    class var mlxCacheLimitBytes: Int { 512 << 20 }

    override class func setUp() {
        super.setUp()
        MLX.Memory.cacheLimit = mlxCacheLimitBytes
    }

    /// Hand the pool back between tests. The cap alone bounds the high-water
    /// mark; this keeps the resident figure near what the suite is actually
    /// using, which matters when several suites run back to back.
    override func tearDown() {
        super.tearDown()
        MLX.Memory.clearCache()
    }
}
