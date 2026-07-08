import AudioCommon
@testable import IndexTTS2TTS
import XCTest

final class E2EIndexTTS2BundleTests: XCTestCase {
    func testExpandedBundleLoadsAndReportsCurrentSynthesisStatus() async throws {
        let model = try await loadModel()

        XCTAssertEqual(model.manifest.modelKey, IndexTTS2TTSModel.modelKey)
        XCTAssertEqual(model.manifest.displayName, "IndexTTS2")
        XCTAssertEqual(model.manifest.sampleRateHz, 24_000)
        XCTAssertEqual(model.manifest.auxiliaryModels.map(\.sourceRepo), [
            "facebook/w2v-bert-2.0",
            "amphion/MaskGCT",
            "funasr/campplus",
            "nvidia/bigvgan_v2_22khz_80band_256x",
        ])
        XCTAssertGreaterThan(model.memoryFootprint, 4_000_000_000)
        XCTAssertNotNil(model.tokenizer)

        do {
            _ = try await model.generate(
                text: "This is an IndexTTS2 native synthesis smoke test.",
                referenceAudio: URL(fileURLWithPath: "/tmp/index-tts2-reference.wav"),
                referenceText: "Reference voice sample.",
                language: "en")
            XCTFail("IndexTTS2 native synthesis should remain explicit until the Swift/MLX graph port lands")
        } catch let error as AudioModelError {
            guard case .inferenceFailed(let operation, let reason) = error else {
                return XCTFail("Unexpected AudioModelError: \(error)")
            }
            XCTAssertEqual(operation, "IndexTTS2 synthesis")
            XCTAssertTrue(reason.contains("native Swift inference"))
        }
    }

    private func loadModel() async throws -> IndexTTS2TTSModel {
        let env = ProcessInfo.processInfo.environment
        if let bundlePath = env["INDEXTTS2_E2E_BUNDLE"], !bundlePath.isEmpty {
            let bundle = URL(fileURLWithPath: bundlePath, isDirectory: true)
            guard FileManager.default.fileExists(atPath: bundle.path) else {
                throw XCTSkip("IndexTTS2 E2E bundle not found at \(bundle.path)")
            }
            return try await IndexTTS2TTSModel.fromBundle(bundle)
        }

        guard env["INDEXTTS2_E2E_DOWNLOAD"] == "1" else {
            throw XCTSkip("Set INDEXTTS2_E2E_BUNDLE or INDEXTTS2_E2E_DOWNLOAD=1 to run IndexTTS2 bundle E2E")
        }

        do {
            return try await IndexTTS2TTSModel.fromPretrained()
        } catch {
            throw XCTSkip("IndexTTS2 published bundle unavailable: \(error)")
        }
    }
}
