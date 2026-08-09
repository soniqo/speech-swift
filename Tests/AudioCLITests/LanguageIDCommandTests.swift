import ArgumentParser
import XCTest
@testable import AudioCLILib

final class LanguageIDCommandTests: XCTestCase {
    func testParsesDefaults() throws {
        let root = try AudioCLI.parseAsRoot(["language-id", "recording.wav"])
        let command = try XCTUnwrap(root as? LanguageIDCommand)
        XCTAssertEqual(command.audioFile, "recording.wav")
        XCTAssertEqual(command.engine, "mlx")
        XCTAssertEqual(command.top, 5)
        XCTAssertNil(command.model)
        XCTAssertFalse(command.json)
    }

    func testParsesCoreMLOptions() throws {
        let root = try AudioCLI.parseAsRoot([
            "language-id", "recording.wav",
            "--engine", "coreml",
            "--model", "org/custom-lid",
            "--top", "3",
            "--json",
        ])
        let command = try XCTUnwrap(root as? LanguageIDCommand)
        XCTAssertEqual(command.engine, "coreml")
        XCTAssertEqual(command.model, "org/custom-lid")
        XCTAssertEqual(command.top, 3)
        XCTAssertTrue(command.json)
    }

    func testRejectsUnknownEngine() {
        XCTAssertThrowsError(
            try AudioCLI.parseAsRoot([
                "language-id", "recording.wav", "--engine", "onnx",
            ])
        ) { error in
            XCTAssertEqual(AudioCLI.exitCode(for: error), .validationFailure)
        }
    }

    func testRejectsInvalidTopK() {
        XCTAssertThrowsError(
            try AudioCLI.parseAsRoot([
                "language-id", "recording.wav", "--top", "0",
            ])
        ) { error in
            XCTAssertEqual(AudioCLI.exitCode(for: error), .validationFailure)
        }
    }
}

final class E2ELanguageIDCommandTests: XCTestCase {
    func testPublishedMLXModelRunsThroughCLICommand() throws {
        guard let audioPath = ProcessInfo.processInfo.environment[
            "SPEECH_LANGUAGE_ID_TEST_AUDIO"
        ] else {
            throw XCTSkip("Set SPEECH_LANGUAGE_ID_TEST_AUDIO")
        }

        let root = try AudioCLI.parseAsRoot([
            "language-id", audioPath,
            "--model", "aufklarer/SpeechBrain-ECAPA-VoxLingua107-21M-MLX",
            "--top", "1",
            "--json",
        ])
        let command = try XCTUnwrap(root as? LanguageIDCommand)
        XCTAssertNoThrow(try command.run())
    }
}
