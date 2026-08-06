import Foundation
import XCTest

final class DiagnosticsContractTests: XCTestCase {

    func testLibraryDiagnosticsUseCentralizedLoggingWithoutDirectStandardIO() throws {
        let sourceDirectory = packageRoot
            .appendingPathComponent("Sources", isDirectory: true)
            .appendingPathComponent("Qwen3TTS", isDirectory: true)
        let sourceFiles = try FileManager.default.contentsOfDirectory(
            at: sourceDirectory,
            includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "swift" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        XCTAssertFalse(sourceFiles.isEmpty)

        let forbiddenPatterns = [
            #"\bprint\s*\("#,
            #"\bdebugPrint\s*\("#,
            #"\bdump\s*\("#,
            #"FileHandle\s*\.\s*standard(?:Output|Error)"#,
            #"\bfputs\s*\("#,
            #"\bputs\s*\("#,
        ].map { try! NSRegularExpression(pattern: $0) }
        var violations: [String] = []
        var combinedSource = ""

        for file in sourceFiles {
            let source = try String(contentsOf: file, encoding: .utf8)
            combinedSource += source
            for (index, line) in source.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
                let text = String(line)
                guard !text.trimmingCharacters(in: .whitespaces).hasPrefix("//") else {
                    continue
                }
                let range = NSRange(text.startIndex..., in: text)
                if forbiddenPatterns.contains(where: {
                    $0.firstMatch(in: text, range: range) != nil
                }) {
                    violations.append("\(file.lastPathComponent):\(index + 1): \(text)")
                }
            }
        }

        XCTAssertTrue(
            violations.isEmpty,
            "Qwen3TTS library diagnostics must not write directly to stdout or stderr:\n"
                + violations.joined(separator: "\n"))
        XCTAssertTrue(combinedSource.contains("AudioLog.inference."))
        XCTAssertTrue(combinedSource.contains("AudioLog.modelLoading."))
    }

    func testDiagnosticPrivacySeparatesOperationalAndCallerControlledValues() throws {
        let modelSource = try source(named: "Qwen3TTS.swift")
        let iclSource = try source(named: "Qwen3TTS+ICL.swift")
        let weightSource = try source(named: "TTSWeightLoading.swift")

        XCTAssertTrue(
            weightSource.contains(#"\(message, privacy: .public)"#),
            "Package-owned weight diagnostics must not collapse to <private>")
        XCTAssertTrue(
            modelSource.contains(#"\(embedTime, privacy: .public)"#),
            "Package-owned timing values must remain visible")
        XCTAssertTrue(
            modelSource.contains(#"\(effectiveLanguage, privacy: .private)"#))
        XCTAssertTrue(
            modelSource.contains(#"\(speakerName, privacy: .private)"#))
        XCTAssertTrue(
            iclSource.contains(#"\(language, privacy: .private)"#))
        XCTAssertFalse(modelSource.contains(#"\(text, privacy: .public)"#))
        XCTAssertFalse(iclSource.contains(#"\(referenceText, privacy: .public)"#))
    }

    private func source(named name: String) throws -> String {
        let url = packageRoot
            .appendingPathComponent("Sources", isDirectory: true)
            .appendingPathComponent("Qwen3TTS", isDirectory: true)
            .appendingPathComponent(name)
        return try String(contentsOf: url, encoding: .utf8)
    }

    private var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }
}
