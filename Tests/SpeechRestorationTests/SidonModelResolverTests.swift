import XCTest
@testable import SpeechRestoration

/// Unit tests for `SidonModelResolver` bundle resolution (no CoreML load): the
/// `.mlmodelc`-preferred / `.mlpackage`-fallback logic and the missing-artifact
/// error path, exercised against synthetic directory fixtures.
final class SidonModelResolverTests: XCTestCase {

    private var tmp: URL!

    override func setUpWithError() throws {
        tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("sidon-resolver-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let tmp { try? FileManager.default.removeItem(at: tmp) }
    }

    private func makeDir(_ name: String) throws -> URL {
        let url = tmp.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    func testPrefersCompiledWhenBothPresent() throws {
        let compiled = try makeDir(SidonConfig.predictorCompiledName)
        _ = try makeDir(SidonConfig.predictorPackageName)
        let resolved = try SidonModelResolver.resolve(
            directory: tmp,
            compiledName: SidonConfig.predictorCompiledName,
            packageName: SidonConfig.predictorPackageName)
        XCTAssertEqual(resolved.standardizedFileURL, compiled.standardizedFileURL,
            "resolver must prefer the precompiled .mlmodelc bundle")
    }

    func testReturnsCompiledWhenOnlyCompiledPresent() throws {
        let compiled = try makeDir(SidonConfig.vocoderCompiledName)
        let resolved = try SidonModelResolver.resolve(
            directory: tmp,
            compiledName: SidonConfig.vocoderCompiledName,
            packageName: SidonConfig.vocoderPackageName)
        XCTAssertEqual(resolved.lastPathComponent, SidonConfig.vocoderCompiledName)
        XCTAssertEqual(resolved.standardizedFileURL, compiled.standardizedFileURL)
    }

    func testThrowsWhenNeitherPresent() throws {
        XCTAssertThrowsError(
            try SidonModelResolver.resolve(
                directory: tmp,
                compiledName: SidonConfig.predictorCompiledName,
                packageName: SidonConfig.predictorPackageName)
        ) { error in
            guard case SidonModelError.modelNotFound = error else {
                return XCTFail("expected .modelNotFound, got \(error)")
            }
        }
    }
}
