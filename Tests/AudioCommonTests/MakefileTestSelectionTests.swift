import Foundation
import XCTest

final class MakefileTestSelectionTests: XCTestCase {
    func testCuratedTestTargetExplicitlyExcludesE2E() throws {
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let makefile = try String(
            contentsOf: packageRoot.appendingPathComponent("Makefile"),
            encoding: .utf8)

        guard let recipe = makefile
            .split(separator: "\n")
            .first(where: { $0.contains("swift test") && $0.contains("--filter") })
        else {
            return XCTFail("Makefile test target has no filtered swift test recipe")
        }

        XCTAssertTrue(
            recipe.contains("--skip E2E"),
            "The curated make test target must never select model-backed E2E suites")
        XCTAssertTrue(
            recipe.contains("PersonaPlexTests[.]PersonaPlexTests/"),
            "PersonaPlex unit selection must identify the class, not match its E2E class by suffix")
    }
}
