#if canImport(CoreML)
import CoreML
import Foundation
import XCTest

@testable import ChatterboxTTS

final class E2EChatterboxFlashCoreMLTests: XCTestCase {
    func testBundleLoadsAllGraphs() throws {
        let model = try ChatterboxFlashCoreMLModel(directory: try localBundleURL(), computeUnits: .cpuOnly)

        XCTAssertEqual(model.sampleRate, 24_000)
        XCTAssertEqual(model.t3.config.speechLen, 1024)
        XCTAssertEqual(model.t3.config.speechVocabSize, 8194)
        XCTAssertEqual(model.audio.config.melLen, 384)
    }

    func testExportedTokenizerMatchesReferenceIds() throws {
        let tokenizer = try ChatterboxFlashTokenizer(directory: try localBundleURL())

        XCTAssertEqual(
            tokenizer.encode("Core ML speech test."),
            [255, 279, 28, 46, 289, 288, 32, 124, 18, 71, 33, 218, 9, 0]
        )
    }

    func testT3SpeechTokenSmoke() throws {
        let model = try ChatterboxFlashCoreMLModel(directory: try localBundleURL(), computeUnits: .all)
        let conditioning = ChatterboxFlashT3Conditioning(
            speakerEmbedding: deterministicValues(count: 256, scale: 0.01),
            promptSpeechTokens: Array(repeating: Int32(0), count: model.t3.config.promptSpeechLen),
            emotionAdv: 0.5
        )
        let tokens = try model.generateSpeechTokens(
            text: "Core ML speech test.",
            conditioning: conditioning,
            options: ChatterboxFlashGenerationOptions(
                maxSpeechTokens: 16,
                numSteps: 2,
                temperature: 0,
                positionTemperature: 0,
                seed: 1234
            )
        )

        XCTAssertLessThanOrEqual(tokens.count, 16)
        XCTAssertTrue(tokens.allSatisfy { $0 >= 0 && $0 < model.t3.config.startSpeechToken })
    }

    func testAudioBackHalfSmoke() throws {
        let bundle = try localBundleURL()
        let audio = try ChatterboxFlashAudioGraphs(
            directory: bundle,
            config: ChatterboxFlashAudioConfig.fallback,
            computeUnits: .cpuOnly
        )

        let reference = ChatterboxFlashS3GenReference(
            embedding: deterministicValues(count: 192, scale: 0.01),
            promptToken: Array(repeating: Int32(1), count: 8),
            promptFeature: Array(repeating: 0, count: 16 * 80),
            promptFeatureFrames: 16
        )

        let waveform = try audio.synthesize(
            speechTokens: Array(repeating: 2, count: 8),
            reference: reference,
            seed: 1234
        )

        XCTAssertEqual(waveform.count, 16 * 480)
        XCTAssertTrue(waveform.allSatisfy { $0.isFinite })
        XCTAssertGreaterThan(waveform.map { abs($0) }.max() ?? 0, 0)
    }

    private func localBundleURL() throws -> URL {
        if let override = ProcessInfo.processInfo.environment["CHATTERBOX_FLASH_COREML_PATH"], !override.isEmpty {
            return URL(fileURLWithPath: override, isDirectory: true)
        }

        let fallback = URL(fileURLWithPath: "/tmp/chatterbox-flash-coreml-upload", isDirectory: true)
        guard FileManager.default.fileExists(atPath: fallback.path) else {
            throw XCTSkip("Set CHATTERBOX_FLASH_COREML_PATH to a Chatterbox Flash Core ML export bundle")
        }
        return fallback
    }

    private func deterministicValues(count: Int, scale: Float) -> [Float] {
        (0..<count).map { index in
            sin(Float(index) * 0.17) * scale
        }
    }
}
#endif
