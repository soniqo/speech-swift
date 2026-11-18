import Foundation
import XCTest

final class IsolatedTestRunnerTests: XCTestCase {
    private var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func recordedCalls(emptyClass: String? = nil, expectedExitStatus: Int32 = 0) throws -> [[String]] {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("isolated-runner-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: root) }
        let scripts = root.appendingPathComponent("scripts")
        let bin = root.appendingPathComponent("bin")
        try fm.createDirectory(at: scripts, withIntermediateDirectories: true)
        try fm.createDirectory(at: bin, withIntermediateDirectories: true)
        try fm.copyItem(
            at: repositoryRoot.appendingPathComponent("scripts/test_e2e_isolated.sh"),
            to: scripts.appendingPathComponent("test_e2e_isolated.sh"))

        let swift = bin.appendingPathComponent("swift")
        let stub = #"""
        #!/bin/bash
        set -eu
        {
          printf 'CALL'
          printf '\t%s' "$@"
          printf '\n'
        } >> "$RUNNER_TRACE_FILE"
        if [[ "${1:-}" == test && "${2:-}" == list ]]; then
          printf 'Fixture.UnitTests/testOne\n'
          printf 'Fixture.E2EModelTests/testOne\n'
          printf 'Fixture.E2EModelTests/testTwo\n'
          printf 'Fixture.LegacyModelE2ETests/testOne\n'
          exit 0
        else
          if [[ -n "${RUNNER_EMPTY_CLASS:-}" ]]; then
            for ARG in "$@"; do
              if [[ "$ARG" == "$RUNNER_EMPTY_CLASS" ]]; then exit 0; fi
            done
          fi
          printf "Test Case '-[Fixture testOne]' passed (0.001 seconds).\n"
        fi
        """#
        try stub.write(to: swift, atomically: true, encoding: .utf8)
        try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: swift.path)
        let trace = root.appendingPathComponent("trace.tsv")
        var environment = ProcessInfo.processInfo.environment
        for key in ["E2E_SKIP_UNIT", "E2E_SKIP_FILE", "E2E_ONLY_FILTER"] {
            environment.removeValue(forKey: key)
        }
        environment["PATH"] = "\(bin.path):/usr/bin:/bin"
        environment["RUNNER_TRACE_FILE"] = trace.path
        environment["E2E_LOG_DIR"] = root.appendingPathComponent("logs").path
        environment["RUNNER_EMPTY_CLASS"] = emptyClass

        let output = Pipe()
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = ["scripts/test_e2e_isolated.sh"]
        process.currentDirectoryURL = root
        process.environment = environment
        process.standardOutput = output
        process.standardError = output
        try process.run()
        let messages = output.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, expectedExitStatus, String(decoding: messages, as: UTF8.self))
        return try String(contentsOf: trace, encoding: .utf8)
            .split(separator: "\n").map { $0.split(separator: "\t").dropFirst().map(String.init) }
    }

    func testUnitPhaseMatchesHostedCIExclusions() throws {
        let calls = try recordedCalls()
        let unit = try XCTUnwrap(calls.first { $0.contains("--skip") })
        let actual = Set(unit.indices.compactMap { index in
            unit[index] == "--skip" && index + 1 < unit.count ? unit[index + 1] : nil
        })
        let workflow = try String(
            contentsOf: repositoryRoot.appendingPathComponent(".github/workflows/tests.yml"),
            encoding: .utf8)
        let pattern = try NSRegularExpression(pattern: #"--skip\s+([A-Za-z][A-Za-z0-9]*)"#)
        let expected = Set(pattern.matches(in: workflow, range: NSRange(workflow.startIndex..., in: workflow))
            .compactMap { match -> String? in
                guard let range = Range(match.range(at: 1), in: workflow) else { return nil }
                return String(workflow[range])
            })
        XCTAssertFalse(expected.isEmpty)
        XCTAssertEqual(actual, expected)
        XCTAssertTrue(unit.contains("--no-parallel"))
        XCTAssertEqual(calls.filter { $0.contains("--skip") }.count, 1)
    }

    func testPrefixAndLegacySuffixClassesRunInSeparateProcesses() throws {
        let calls = try recordedCalls().filter { $0.contains("--filter") }
        XCTAssertEqual(calls.count, 2, "Methods in one class must share just one isolated invocation")
        let classes = Set(calls.compactMap { call -> String? in
            guard let index = call.firstIndex(of: "--filter"), index + 1 < call.count else { return nil }
            XCTAssertTrue(call.contains("--no-parallel"))
            return call[index + 1]
        })
        XCTAssertEqual(classes, ["Fixture.E2EModelTests", "Fixture.LegacyModelE2ETests"])
    }

    func testParakeetMelContractRunsInUnitPhaseWhileModelTestsStayExcluded() throws {
        let unit = try XCTUnwrap(recordedCalls().first { $0.contains("--skip") })
        let skipPatterns = unit.indices.compactMap { index in
            unit[index] == "--skip" && index + 1 < unit.count ? unit[index + 1] : nil
        }
        func isSkipped(_ test: String) -> Bool {
            skipPatterns.contains { test.range(of: $0, options: .regularExpression) != nil }
        }

        XCTAssertFalse(isSkipped(
            "ParakeetASRTests.MelContractTests/testQuietMelMatchesTheTrainingPreprocessor"),
            "The quiet-audio mel regression must run without model downloads")
        XCTAssertTrue(isSkipped(
            "ParakeetASRTests.E2EParakeetASRTests/testTranscription"),
            "Model-loading Parakeet tests must stay outside the unit phase")
    }

    func testSuccessfulProcessWithNoCasesIsNotReportedAsPassing() throws {
        let calls = try recordedCalls(emptyClass: "Fixture.E2EModelTests", expectedExitStatus: 1)
        XCTAssertEqual(calls.filter { $0.contains("--filter") }.count, 2)
    }
}
