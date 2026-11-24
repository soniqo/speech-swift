import ArgumentParser
import XCTest
@testable import AudioCLILib

final class DenoiseCommandTests: XCTestCase {
    func testDeepFilterNetRemainsDefault() throws {
        let command = try XCTUnwrap(AudioCLI.parseAsRoot(["denoise", "noisy.wav"]) as? DenoiseCommand)
        XCTAssertEqual(command.engine, "coreml")
        XCTAssertNil(command.model)
    }

    func testInworldAcceptsExportDirectoryAndChunkOptions() throws {
        let command = try XCTUnwrap(AudioCLI.parseAsRoot([
            "denoise", "noisy.wav", "--engine", "inworld", "--model", "/tmp/export",
            "--chunk-seconds", "6", "--overlap-ms", "500", "-o", "voice.wav",
        ]) as? DenoiseCommand)
        XCTAssertEqual(command.engine, "inworld")
        XCTAssertEqual(command.model, "/tmp/export")
        XCTAssertEqual(command.chunkSeconds, 6)
        XCTAssertEqual(command.overlapMs, 500)
        XCTAssertEqual(command.output, "voice.wav")
    }

    func testInworldRequiresAnExportDirectory() {
        XCTAssertThrowsError(try AudioCLI.parseAsRoot(["denoise", "noisy.wav", "--engine", "inworld"])) { error in
            XCTAssertTrue(AudioCLI.message(for: error).contains("requires --model"))
        }
    }
}
