#if os(macOS)
import XCTest
@testable import PersonaPlexDemo

final class PersonaPlexDemoCachePolicyTests: XCTestCase {
    func testCacheCompleteRequiresRequiredFiles() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }

        try touch("model.safetensors", in: dir)
        try touch("vocab.json", in: dir)

        XCTAssertFalse(
            PersonaPlexDemoCachePolicy.cacheComplete(
                in: dir,
                requiredFiles: ["vocab.json", "merges.txt"],
                requiresWeights: true))

        try touch("merges.txt", in: dir)

        XCTAssertTrue(
            PersonaPlexDemoCachePolicy.cacheComplete(
                in: dir,
                requiredFiles: ["vocab.json", "merges.txt"],
                requiresWeights: true))
    }

    func testCacheCompleteRequiresWeightWhenRequested() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }

        try touch("config.json", in: dir)

        XCTAssertFalse(
            PersonaPlexDemoCachePolicy.cacheComplete(
                in: dir,
                requiredFiles: ["config.json"],
                requiresWeights: true))

        try touch("weights.safetensors", in: dir)

        XCTAssertTrue(
            PersonaPlexDemoCachePolicy.cacheComplete(
                in: dir,
                requiredFiles: ["config.json"],
                requiresWeights: true))
    }

    func testCacheCompleteRequiresDirectoryEntries() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }

        try touch("temporal.safetensors", in: dir)
        try touch("config.json", in: dir)
        let voices = dir.appendingPathComponent("voices", isDirectory: true)
        try FileManager.default.createDirectory(at: voices, withIntermediateDirectories: true)
        try touch("NATM0.safetensors", in: voices)

        let requirement = CacheDirectoryRequirement(
            relativePath: "voices",
            fileExtension: "safetensors",
            minimumCount: 2)

        XCTAssertFalse(
            PersonaPlexDemoCachePolicy.cacheComplete(
                in: dir,
                requiredFiles: ["temporal.safetensors", "config.json"],
                requiresWeights: false,
                directoryRequirements: [requirement]))

        try touch("NATF0.safetensors", in: voices)

        XCTAssertTrue(
            PersonaPlexDemoCachePolicy.cacheComplete(
                in: dir,
                requiredFiles: ["temporal.safetensors", "config.json"],
                requiresWeights: false,
                directoryRequirements: [requirement]))
    }

    private func makeTempDirectory() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("personaplex-demo-cache-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func touch(_ name: String, in directory: URL) throws {
        let url = directory.appendingPathComponent(name)
        try Data().write(to: url)
    }
}
#endif
