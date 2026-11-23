import XCTest
@testable import AudioCLILib

final class InklingCommandTests: XCTestCase {
    func testParsesLocalTextAndImageRequests() throws {
        let command = try InklingCommand.parse(["Describe this", "--model-dir", "/tmp/inkling",
            "--image", "photo.png", "--max-tokens", "32", "--reasoning-effort", "0.2"])
        XCTAssertEqual(command.modelDir, "/tmp/inkling")
        XCTAssertEqual(command.image, "photo.png")
        XCTAssertEqual(command.maxTokens, 32)
        XCTAssertEqual(command.reasoningEffort, 0.2)
        XCTAssertEqual(command.backend, .mlx)
    }
    func testParsesCoreMLBackendAndComputeUnits() throws {
        let command = try InklingCommand.parse(["Hello", "--model-dir", "/tmp/coreml-inkling",
            "--backend", "coreml", "--coreml-compute-units", "gpu"])
        XCTAssertEqual(command.backend, .coreml)
        XCTAssertEqual(command.coremlComputeUnits, .gpu)
        XCTAssertThrowsError(try InklingCommand.parse(["Hello", "--model-dir", "/tmp/inkling", "--backend", "unknown"]))
        XCTAssertThrowsError(try InklingCommand.parse(["Hello", "--model-dir", "/tmp/inkling", "--coreml-compute-units", "unknown"]))
    }
    func testRejectsMissingModelAndInvalidBudgets() {
        XCTAssertThrowsError(try InklingCommand.parse(["Hello"]))
        for arguments in [["--max-tokens", "0"], ["--temperature", "-1"],
                          ["--reasoning-effort", "1"], ["--context-limit", "128"]] {
            XCTAssertThrowsError(try InklingCommand.parse(["Hello", "--model-dir", "/tmp/inkling"] + arguments))
        }
    }
}
