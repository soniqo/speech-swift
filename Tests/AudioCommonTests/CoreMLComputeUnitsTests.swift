#if canImport(CoreML)
import XCTest
import CoreML
@testable import AudioCommon

final class CoreMLComputeUnitsResolverTests: XCTestCase {

    // The resolver reads a process env var, which we can't mutate per-test
    // reliably across platforms. Instead test the pure mapping by setting +
    // unsetting the var around each case via setenv/unsetenv.

    private func withEnv(_ value: String?, _ body: () -> Void) {
        let key = CoreMLComputeUnitsResolver.envKey
        let previous = ProcessInfo.processInfo.environment[key]
        if let value { setenv(key, value, 1) } else { unsetenv(key) }
        defer {
            if let previous { setenv(key, previous, 1) } else { unsetenv(key) }
        }
        body()
    }

    func testUnsetReturnsFallback() {
        withEnv(nil) {
            XCTAssertEqual(CoreMLComputeUnitsResolver.resolved(default: .cpuAndNeuralEngine), .cpuAndNeuralEngine)
            XCTAssertEqual(CoreMLComputeUnitsResolver.resolved(default: .all), .all)
        }
    }

    func testGpuOverride() {
        for v in ["cpuAndGPU", "gpu", "CPUANDGPU", " gpu "] {
            withEnv(v) {
                XCTAssertEqual(CoreMLComputeUnitsResolver.resolved(default: .cpuAndNeuralEngine), .cpuAndGPU,
                               "env '\(v)' should map to cpuAndGPU")
            }
        }
    }

    func testCpuAndAneAndAll() {
        withEnv("cpu") { XCTAssertEqual(CoreMLComputeUnitsResolver.resolved(default: .all), .cpuOnly) }
        withEnv("ane") { XCTAssertEqual(CoreMLComputeUnitsResolver.resolved(default: .cpuOnly), .cpuAndNeuralEngine) }
        withEnv("all") { XCTAssertEqual(CoreMLComputeUnitsResolver.resolved(default: .cpuOnly), .all) }
    }

    func testUnrecognizedFallsBack() {
        withEnv("banana") {
            XCTAssertEqual(CoreMLComputeUnitsResolver.resolved(default: .cpuAndNeuralEngine), .cpuAndNeuralEngine)
        }
    }

    // MARK: - parse / shortName (shared CLI + env vocabulary)

    func testParseAcceptsTheSharedVocabulary() {
        XCTAssertEqual(CoreMLComputeUnitsResolver.parse("ane"), .cpuAndNeuralEngine)
        XCTAssertEqual(CoreMLComputeUnitsResolver.parse("neuralEngine"), .cpuAndNeuralEngine)
        XCTAssertEqual(CoreMLComputeUnitsResolver.parse("cpuAndNeuralEngine"), .cpuAndNeuralEngine)
        XCTAssertEqual(CoreMLComputeUnitsResolver.parse("gpu"), .cpuAndGPU)
        XCTAssertEqual(CoreMLComputeUnitsResolver.parse("cpuAndGPU"), .cpuAndGPU)
        XCTAssertEqual(CoreMLComputeUnitsResolver.parse("cpu"), .cpuOnly)
        XCTAssertEqual(CoreMLComputeUnitsResolver.parse("cpuOnly"), .cpuOnly)
        XCTAssertEqual(CoreMLComputeUnitsResolver.parse("all"), .all)
        XCTAssertEqual(CoreMLComputeUnitsResolver.parse("  GPU\n"), .cpuAndGPU, "case/whitespace-insensitive")
    }

    func testParseRejectsUnknownAndEmpty() {
        XCTAssertNil(CoreMLComputeUnitsResolver.parse(""))
        XCTAssertNil(CoreMLComputeUnitsResolver.parse("   "))
        XCTAssertNil(CoreMLComputeUnitsResolver.parse("banana"))
        XCTAssertNil(CoreMLComputeUnitsResolver.parse("metal"))
    }

    func testShortNameRoundTripsThroughParse() {
        for units in [MLComputeUnits.cpuOnly, .cpuAndGPU, .cpuAndNeuralEngine, .all] {
            let name = CoreMLComputeUnitsResolver.shortName(units)
            XCTAssertEqual(CoreMLComputeUnitsResolver.parse(name), units, "shortName(\(name)) must parse back")
        }
        XCTAssertEqual(CoreMLComputeUnitsResolver.shortName(.cpuAndGPU), "gpu")
    }

    func testEmptyEnvFallsBack() {
        withEnv("") {
            XCTAssertEqual(CoreMLComputeUnitsResolver.resolved(default: .cpuAndGPU), .cpuAndGPU)
        }
    }

    func testCpuOnlyOverrideWinsOverGpuFallback() {
        withEnv("cpuOnly") {
            XCTAssertEqual(CoreMLComputeUnitsResolver.resolved(default: .cpuAndGPU), .cpuOnly)
        }
    }
}
#endif
