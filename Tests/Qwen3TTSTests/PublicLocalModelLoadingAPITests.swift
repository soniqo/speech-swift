import Foundation
import Qwen3TTS
import XCTest

final class PublicLocalModelLoadingAPITests: XCTestCase {

    func testPublicLocalAndPretrainedLoaderSignatures() {
        let localLoader: (URL, URL) throws -> Qwen3TTSModel = {
            modelDirectory, tokenizerDirectory in
            try Qwen3TTSModel.fromLocal(
                modelDirectory: modelDirectory,
                tokenizerDirectory: tokenizerDirectory)
        }
        let fallbackLocalLoader: (URL, URL, Qwen3TTSConfig) throws -> Qwen3TTSModel = {
            modelDirectory, tokenizerDirectory, configuration in
            try Qwen3TTSModel.fromLocal(
                modelDirectory: modelDirectory,
                tokenizerDirectory: tokenizerDirectory,
                configuration: configuration,
                wiredMemoryPolicy: .none)
        }
        let pretrainedLoader: (URL, URL) async throws -> Qwen3TTSModel = {
            modelDirectory, tokenizerDirectory in
            try await Qwen3TTSModel.fromPretrained(
                cacheDir: modelDirectory,
                tokenizerCacheDir: tokenizerDirectory)
        }
        let pretrainedEncoderLoader:
            (URL, URL) async throws -> (Qwen3TTSModel, SpeechTokenizerEncoder) = {
                modelDirectory, tokenizerDirectory in
                try await Qwen3TTSModel.fromPretrainedWithEncoder(
                    cacheDir: modelDirectory,
                    tokenizerCacheDir: tokenizerDirectory)
            }

        _ = localLoader
        _ = fallbackLocalLoader
        _ = pretrainedLoader
        _ = pretrainedEncoderLoader
    }
}
