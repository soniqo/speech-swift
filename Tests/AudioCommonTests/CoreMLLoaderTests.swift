import XCTest
@testable import AudioCommon

/// Smoke tests for ``CoreMLLoader`` — only the logic that doesn't
/// require an actual compiled ``.mlmodelc`` on disk. The full load
/// path is exercised transitively whenever a module that uses the
/// loader runs E2E tests.
final class CoreMLLoaderTests: XCTestCase {

    func testResetWarningStateIsCallable() {
        // Simple smoke: the public reset hook runs without crashing and
        // is idempotent across repeated calls.
        CoreMLLoader.resetWarningState()
        CoreMLLoader.resetWarningState()
    }

    func testLoadRejectsMissingFile() {
        // Passing a nonexistent URL must throw — the loader delegates to
        // ``MLModel(contentsOf:)`` which will fail first. Important that
        // we don't accidentally swallow the error in the instrumented
        // wrapper.
        let url = URL(fileURLWithPath: "/tmp/nonexistent-\(UUID().uuidString).mlmodelc")
        XCTAssertThrowsError(try CoreMLLoader.load(url: url, computeUnits: .cpuOnly, name: "probe"))
    }
}
