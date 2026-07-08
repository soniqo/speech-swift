import AudioCommon
@testable import F5TTS
@testable import HiggsAudioTTS
@testable import IndexTTS2TTS
@testable import VoiceCloneTTSCommon
import XCTest

final class VoiceCloneCandidateTTSTests: XCTestCase {
    func testIndexTTS2BundleLoadsAndReportsMetadata() async throws {
        let dir = try makeBundle(
            modelKey: "indextts2",
            displayName: "IndexTTS2",
            parameterCount: "1.5B-class",
            sampleRate: 24_000,
            convertedFiles: ["gpt.safetensors", "s2mel.safetensors"])
        defer { try? FileManager.default.removeItem(at: dir) }

        let model = try await IndexTTS2TTSModel.fromBundle(dir)

        XCTAssertEqual(model.sampleRate, 24_000)
        XCTAssertEqual(model.manifest.displayName, "IndexTTS2")
        XCTAssertEqual(model.manifest.parameterCount, "1.5B-class")
        XCTAssertEqual(model.manifest.publishRepo, "aufklarer/IndexTTS2-MLX-fp16")
        XCTAssertEqual(model.memoryFootprint, 32)
        XCTAssertTrue(model.isLoaded)
    }

    func testIndexTTS2DocumentsAuxiliaryModelsNeededForNativePort() {
        let aux = IndexTTS2TTSModel.auxiliaryModels
        XCTAssertEqual(aux.map(\.repository), [
            "facebook/w2v-bert-2.0",
            "amphion/MaskGCT",
            "funasr/campplus",
            "nvidia/bigvgan_v2_22khz_80band_256x",
        ])
        XCTAssertTrue(aux[0].purpose.contains("semantic"))
        XCTAssertTrue(aux[3].files.contains("aux/bigvgan/bigvgan_generator.safetensors"))
    }

    func testManifestDecodesAuxiliaryModels() async throws {
        let dir = try makeBundle(
            modelKey: "indextts2",
            displayName: "IndexTTS2",
            parameterCount: "1.5B-class",
            sampleRate: 24_000,
            convertedFiles: ["gpt.safetensors"],
            copiedFiles: ["config.json"],
            auxiliaryModels: [[
                "key": "bigvgan",
                "display_name": "BigVGAN",
                "source_repo": "nvidia/bigvgan_v2_22khz_80band_256x",
                "source_revision": "aux-test",
                "purpose": "vocoder",
                "converted_files": ["aux/bigvgan/bigvgan_generator.safetensors"],
                "copied_files": ["aux/bigvgan/config.json"],
                "notes": ["unit test"],
            ]])
        defer { try? FileManager.default.removeItem(at: dir) }

        let model = try await IndexTTS2TTSModel.fromBundle(dir)

        XCTAssertEqual(model.manifest.auxiliaryModels.count, 1)
        XCTAssertEqual(model.manifest.auxiliaryModels[0].key, "bigvgan")
        XCTAssertEqual(model.manifest.auxiliaryModels[0].sourceRepo, "nvidia/bigvgan_v2_22khz_80band_256x")
        XCTAssertEqual(model.manifest.auxiliaryModels[0].convertedFiles, ["aux/bigvgan/bigvgan_generator.safetensors"])
    }

    func testBundleLoaderValidatesCopiedFiles() throws {
        let dir = try makeBundle(
            modelKey: "indextts2",
            displayName: "IndexTTS2",
            parameterCount: "1.5B-class",
            sampleRate: 24_000,
            convertedFiles: ["gpt.safetensors"],
            copiedFiles: ["config.json"],
            writeCopiedFiles: false)
        defer { try? FileManager.default.removeItem(at: dir) }

        XCTAssertThrowsError(
            try VoiceCloneBundleLoader.load(from: dir, expectedModelKey: "indextts2")
        ) { error in
            XCTAssertEqual(error as? VoiceCloneBundleError, .missingRequiredFile("config.json"))
        }
    }

    func testIndexTTS2TokenizerUsesSentencePieceScores() throws {
        let pieces = [
            SentencePieceModel.Piece(text: "<unk>", score: 0, type: SentencePieceModel.PieceType.unknown.rawValue),
            SentencePieceModel.Piece(text: "▁", score: -10, type: SentencePieceModel.PieceType.normal.rawValue),
            SentencePieceModel.Piece(text: "hello", score: -5, type: SentencePieceModel.PieceType.normal.rawValue),
            SentencePieceModel.Piece(text: "world", score: -5, type: SentencePieceModel.PieceType.normal.rawValue),
            SentencePieceModel.Piece(text: "▁hello", score: -1, type: SentencePieceModel.PieceType.normal.rawValue),
            SentencePieceModel.Piece(text: "▁world", score: -1, type: SentencePieceModel.PieceType.normal.rawValue),
        ]
        let tokenizer = try IndexTTS2Tokenizer(pieces: pieces)

        let ids = try tokenizer.encode("hello world")

        XCTAssertEqual(ids, [4, 5])
        XCTAssertEqual(tokenizer.decode(ids), "hello world")
    }

    func testHiggsBundleRejectsWrongManifestKey() async throws {
        let dir = try makeBundle(
            modelKey: "indextts2",
            displayName: "IndexTTS2",
            parameterCount: "1.5B-class",
            sampleRate: 24_000,
            convertedFiles: ["model.safetensors"])
        defer { try? FileManager.default.removeItem(at: dir) }

        do {
            _ = try await HiggsAudioTTSModel.fromBundle(dir)
            XCTFail("Expected wrong manifest key to fail")
        } catch let error as VoiceCloneBundleError {
            XCTAssertEqual(error, .unexpectedModelKey(expected: "higgs-audio-v3", actual: "indextts2"))
        }
    }

    func testF5GenerationFailsWithExplicitUnsupportedRuntime() async throws {
        let dir = try makeBundle(
            modelKey: "f5-tts-v1",
            displayName: "F5-TTS v1 Base",
            parameterCount: "335M-class",
            sampleRate: 24_000,
            convertedFiles: ["F5TTS_v1_Base/model_1250000.safetensors"])
        defer { try? FileManager.default.removeItem(at: dir) }

        let model = try await F5TTSModel.fromBundle(dir)

        do {
            _ = try await model.generate(text: "Hello", language: "en")
            XCTFail("Expected unsupported native inference error")
        } catch let error as AudioModelError {
            guard case .inferenceFailed(let operation, let reason) = error else {
                return XCTFail("Unexpected AudioModelError: \(error)")
            }
            XCTAssertEqual(operation, "F5-TTS v1 synthesis")
            XCTAssertTrue(reason.contains("native Swift inference"))
        }
    }

    func testUnloadClearsMemoryFootprint() async throws {
        let dir = try makeBundle(
            modelKey: "higgs-audio-v3",
            displayName: "Higgs Audio v3 TTS 4B",
            parameterCount: "4B",
            sampleRate: 24_000,
            convertedFiles: ["model.safetensors"])
        defer { try? FileManager.default.removeItem(at: dir) }

        let model = try await HiggsAudioTTSModel.fromBundle(dir)
        XCTAssertEqual(model.memoryFootprint, 16)

        model.unload()

        XCTAssertFalse(model.isLoaded)
        XCTAssertEqual(model.memoryFootprint, 0)
    }

    private func makeBundle(
        modelKey: String,
        displayName: String,
        parameterCount: String,
        sampleRate: Int,
        convertedFiles: [String],
        copiedFiles: [String] = [],
        auxiliaryModels: [[String: Any]] = [],
        writeCopiedFiles: Bool = true
    ) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        for relativePath in convertedFiles {
            let fileURL = dir.appendingPathComponent(relativePath)
            try FileManager.default.createDirectory(
                at: fileURL.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try Data(repeating: 0x7f, count: 16).write(to: fileURL)
        }

        if writeCopiedFiles {
            for relativePath in copiedFiles {
                let fileURL = dir.appendingPathComponent(relativePath)
                try FileManager.default.createDirectory(
                    at: fileURL.deletingLastPathComponent(),
                    withIntermediateDirectories: true)
                try Data("{}".utf8).write(to: fileURL)
            }
        }

        let manifest: [String: Any] = [
            "schema_version": 1,
            "artifact_type": "voice_cloning_tts_candidate",
            "model_key": modelKey,
            "display_name": displayName,
            "source_repo": "example/\(displayName)",
            "source_revision": "test",
            "publish_repo": "aufklarer/\(displayName)-MLX-fp16",
            "output_name": "\(displayName)-MLX-fp16",
            "license": "test",
            "license_posture": "test",
            "parameter_count": parameterCount,
            "sample_rate_hz": sampleRate,
            "languages": ["en"],
            "voice_conditioning": "reference audio",
            "streaming": "test",
            "format": "MLX fp16 safetensors",
            "runtime_status": "artifact-export; Swift native inference not implemented",
            "converted_files": convertedFiles,
            "copied_files": copiedFiles,
            "auxiliary_models": auxiliaryModels,
            "notes": ["unit test"],
            "files": (convertedFiles + copiedFiles).map { ["path": $0, "bytes": 16] },
        ]
        let data = try JSONSerialization.data(
            withJSONObject: manifest,
            options: [.prettyPrinted, .sortedKeys])
        try data.write(to: dir.appendingPathComponent("soniqo_manifest.json"))
        return dir
    }
}
