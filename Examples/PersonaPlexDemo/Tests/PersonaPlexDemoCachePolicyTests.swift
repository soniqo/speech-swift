#if os(macOS)
import XCTest
@testable import PersonaPlexDemo

final class PersonaPlexDemoCachePolicyTests: XCTestCase {
    func testCacheCompleteRequiresRequiredFiles() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }

        try write("model.safetensors", byteCount: 4, in: dir)
        try write("vocab.json", byteCount: 2, in: dir)

        XCTAssertFalse(
            PersonaPlexDemoCachePolicy.cacheComplete(
                in: dir,
                requiredFiles: [
                    CacheFileRequirement(relativePath: "vocab.json", byteCount: 2),
                    CacheFileRequirement(relativePath: "merges.txt", byteCount: 3),
                ],
                requiresWeights: true))

        try write("merges.txt", byteCount: 3, in: dir)

        XCTAssertTrue(
            PersonaPlexDemoCachePolicy.cacheComplete(
                in: dir,
                requiredFiles: [
                    CacheFileRequirement(relativePath: "vocab.json", byteCount: 2),
                    CacheFileRequirement(relativePath: "merges.txt", byteCount: 3),
                ],
                requiresWeights: true))
    }

    func testCacheCompleteRequiresWeightWhenRequested() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }

        try write("config.json", byteCount: 2, in: dir)

        XCTAssertFalse(
            PersonaPlexDemoCachePolicy.cacheComplete(
                in: dir,
                requiredFiles: [CacheFileRequirement(relativePath: "config.json", byteCount: 2)],
                requiresWeights: true))

        try write("weights.safetensors", byteCount: 5, in: dir)

        XCTAssertTrue(
            PersonaPlexDemoCachePolicy.cacheComplete(
                in: dir,
                requiredFiles: [CacheFileRequirement(relativePath: "config.json", byteCount: 2)],
                requiresWeights: true))
    }

    func testCacheCompleteRequiresExpectedByteCount() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }

        try write("config.json", byteCount: 2, in: dir)
        try write("model.safetensors", byteCount: 5, in: dir)

        XCTAssertFalse(
            PersonaPlexDemoCachePolicy.cacheComplete(
                in: dir,
                requiredFiles: [
                    CacheFileRequirement(relativePath: "config.json", byteCount: 3),
                    CacheFileRequirement(relativePath: "model.safetensors", byteCount: 5),
                ],
                requiresWeights: true))

        XCTAssertTrue(
            PersonaPlexDemoCachePolicy.cacheComplete(
                in: dir,
                requiredFiles: [
                    CacheFileRequirement(relativePath: "config.json", byteCount: 2),
                    CacheFileRequirement(relativePath: "model.safetensors", byteCount: 5),
                ],
                requiresWeights: true))
    }

    func testCacheCompleteRequiresNestedRequiredFiles() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }

        try write("temporal.safetensors", byteCount: 5, in: dir)
        try write("config.json", byteCount: 2, in: dir)
        let voices = dir.appendingPathComponent("voices", isDirectory: true)
        try FileManager.default.createDirectory(at: voices, withIntermediateDirectories: true)
        try write("NATF0.safetensors", byteCount: 7, in: voices)

        XCTAssertFalse(
            PersonaPlexDemoCachePolicy.cacheComplete(
                in: dir,
                requiredFiles: [
                    CacheFileRequirement(relativePath: "temporal.safetensors", byteCount: 5),
                    CacheFileRequirement(relativePath: "config.json", byteCount: 2),
                    CacheFileRequirement(relativePath: "voices/NATF0.safetensors", byteCount: 7),
                    CacheFileRequirement(relativePath: "voices/NATM0.safetensors", byteCount: 6),
                ],
                requiresWeights: false))

        try write("NATM0.safetensors", byteCount: 6, in: voices)

        XCTAssertTrue(
            PersonaPlexDemoCachePolicy.cacheComplete(
                in: dir,
                requiredFiles: [
                    CacheFileRequirement(relativePath: "temporal.safetensors", byteCount: 5),
                    CacheFileRequirement(relativePath: "config.json", byteCount: 2),
                    CacheFileRequirement(relativePath: "voices/NATF0.safetensors", byteCount: 7),
                    CacheFileRequirement(relativePath: "voices/NATM0.safetensors", byteCount: 6),
                ],
                requiresWeights: false))
    }

    private func makeTempDirectory() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("personaplex-demo-cache-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func write(_ name: String, byteCount: Int, in directory: URL) throws {
        let url = directory.appendingPathComponent(name)
        try Data(repeating: 0, count: byteCount).write(to: url)
    }
}
#endif
