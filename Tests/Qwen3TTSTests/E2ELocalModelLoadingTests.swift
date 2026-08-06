import AudioCommon
import Foundation
import Qwen3TTS
import XCTest

/// Real-weight regression coverage for local bundle configuration resolution.
///
/// Set `QWEN3_TTS_LOCAL_MODEL_DIR` and `QWEN3_TTS_LOCAL_TOKENIZER_DIR` to override
/// the standard cache locations. The test skips instead of downloading because `fromLocal`
/// promises never to perform network or cache resolution.
final class E2ELocalModelLoadingTests: E2ETestCase {
    func testLocalLoaderUsesBundleMetadataAndSynthesizes() throws {
        let modelDirectory = try localDirectory(
            environmentKey: "QWEN3_TTS_LOCAL_MODEL_DIR",
            modelId: TTSModelVariant.base.rawValue)
        let tokenizerDirectory = try localDirectory(
            environmentKey: "QWEN3_TTS_LOCAL_TOKENIZER_DIR",
            modelId: "Qwen/Qwen3-TTS-Tokenizer-12Hz")

        guard HuggingFaceDownloader.weightsExist(in: modelDirectory),
              HuggingFaceDownloader.weightsExist(in: tokenizerDirectory)
        else {
            throw XCTSkip("Qwen3-TTS model and tokenizer bundles are not cached")
        }

        let model = try Qwen3TTSModel.fromLocal(
            modelDirectory: modelDirectory,
            tokenizerDirectory: tokenizerDirectory,
            configuration: .config(for: .large, bits: 0),
            wiredMemoryPolicy: .none)

        XCTAssertEqual(model.config.talker.hiddenSize, 1024)
        XCTAssertEqual(model.config.talker.bits, 8)
        XCTAssertEqual(model.config.codePredictor.embeddingDim, 1024)
        XCTAssertEqual(model.config.codePredictor.bits, 8)

        let samples = model.synthesize(
            text: "Hello from a local model bundle.",
            language: "english",
            sampling: SamplingConfig(
                temperature: 0,
                topK: 1,
                maxTokens: 80))
        XCTAssertFalse(samples.isEmpty)
        XCTAssertTrue(samples.allSatisfy(\.isFinite))
        XCTAssertGreaterThan(samples.map { abs($0) }.max() ?? 0, 0.001)
    }

    private func localDirectory(environmentKey: String, modelId: String) throws -> URL {
        if let path = ProcessInfo.processInfo.environment[environmentKey], !path.isEmpty {
            return URL(fileURLWithPath: path, isDirectory: true)
        }
        return try HuggingFaceDownloader.getCacheDirectory(for: modelId)
    }
}
