import ArgumentParser
import XCTest
@testable import AudioCLILib

final class ClefCommandTests: XCTestCase {
    func testDefaultsAndLocalOptions() throws {
        let command = try XCTUnwrap(try AudioCLI.parseAsRoot([
            "clef", "decide", "request.json", "--model-dir", "/models/clef", "--offline"
        ]) as? ClefDecideCommand)
        XCTAssertEqual(command.request, "request.json")
        XCTAssertEqual(command.modelDir, "/models/clef")
        XCTAssertTrue(command.offline)
        XCTAssertEqual(command.maxTokens, 4096)
    }

    func testRequiresRequestAndValidTokenLimit() {
        XCTAssertThrowsError(try AudioCLI.parseAsRoot(["clef", "decide"]))
        for value in ["0", "16385"] {
            XCTAssertThrowsError(try AudioCLI.parseAsRoot([
                "clef", "decide", "request.json", "--max-tokens", value]))
        }
    }
}
