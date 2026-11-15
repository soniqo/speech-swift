import Foundation
import XCTest

final class NightlyCrashReportTests: XCTestCase {
    func testSucceedsWithoutCrashReports() throws {
        let result = try runCollector()

        XCTAssertEqual(result.status, 0, result.output)
        XCTAssertTrue(result.output.contains("(no crash reports"), result.output)
    }

    func testSucceedsWithIPSReport() throws {
        let result = try runCollector { directory in
            try "fontd report fixture\n".write(
                to: directory.appendingPathComponent("fontd report.ips"),
                atomically: true, encoding: .utf8)
        }

        XCTAssertEqual(result.status, 0, result.output)
        XCTAssertTrue(result.output.contains("fontd report fixture"), result.output)
        XCTAssertFalse(result.output.contains("(no crash reports"), result.output)
    }

    func testSucceedsWithLegacyCrashReport() throws {
        let result = try runCollector { directory in
            try "legacy report fixture\n".write(
                to: directory.appendingPathComponent("legacy report.crash"),
                atomically: true, encoding: .utf8)
        }

        XCTAssertEqual(result.status, 0, result.output)
        XCTAssertTrue(result.output.contains("legacy report fixture"), result.output)
        XCTAssertFalse(result.output.contains("(no crash reports"), result.output)
    }

    func testPreservesReportReadFailures() throws {
        let result = try runCollector { directory in
            try FileManager.default.createDirectory(
                at: directory.appendingPathComponent("unreadable.ips", isDirectory: true),
                withIntermediateDirectories: true)
        }

        XCTAssertNotEqual(result.status, 0, result.output)
        XCTAssertTrue(result.output.contains("unreadable.ips"), result.output)
    }

    private func runCollector(
        prepare: (URL) throws -> Void = { _ in }
    ) throws -> (status: Int32, output: String) {
        let fileManager = FileManager.default
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let workflow = try String(
            contentsOf: packageRoot.appendingPathComponent(".github/workflows/nightly-e2e.yml"),
            encoding: .utf8)
        let stepStart = try XCTUnwrap(workflow.range(of: "      - name: Dump crash reports\n"))
        let remainder = workflow[stepStart.upperBound...]
        let stepEnd = try XCTUnwrap(remainder.range(of: "\n      - name:"))
        let step = remainder[..<stepEnd.lowerBound]
        let runStart = try XCTUnwrap(step.range(of: "        run: |\n"))
        let script = step[runStart.upperBound...]
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { String($0.dropFirst(10)) }
            .joined(separator: "\n")
            .replacingOccurrences(
                of: "~/Library/Logs/DiagnosticReports/",
                with: "\"$CRASH_REPORT_TEST_DIRECTORY\"/")
        let reportsDirectory = fileManager.temporaryDirectory
            .appendingPathComponent("nightly crash reports \(UUID().uuidString)", isDirectory: true)
        try fileManager.createDirectory(at: reportsDirectory, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: reportsDirectory) }
        try prepare(reportsDirectory)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = ["-e", "-c", script]
        var environment = ProcessInfo.processInfo.environment
        environment["CRASH_REPORT_TEST_DIRECTORY"] = reportsDirectory.path
        process.environment = environment
        let outputPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = outputPipe
        try process.run()
        let output = String(
            data: outputPipe.fileHandleForReading.readDataToEndOfFile(),
            encoding: .utf8) ?? ""
        process.waitUntilExit()
        return (process.terminationStatus, output)
    }
}
