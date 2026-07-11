import AudioCommon
@testable import IndexTTS2TTS
import XCTest

final class IndexTTS2TTSTests: XCTestCase {
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

    func testIndexTTS2RuntimeConfigParsesUpstreamYamlShape() throws {
        let config = try IndexTTS2RuntimeConfig(
            document: IndexTTS2YAMLDocument(Self.indexTTS2ConfigYAML))

        XCTAssertEqual(config.dataset.sampleRate, 24_000)
        XCTAssertEqual(config.dataset.bpeModel, "bpe.model")
        XCTAssertEqual(config.gpt.modelDim, 1280)
        XCTAssertEqual(config.gpt.layers, 24)
        XCTAssertEqual(config.gpt.heads, 20)
        XCTAssertEqual(config.gpt.numberMelCodes, 8194)
        XCTAssertEqual(config.gpt.conditionType, "conformer_perceiver")
        XCTAssertEqual(config.gpt.conditionBlocks, 6)
        XCTAssertEqual(config.gpt.emotionConditionBlocks, 4)
        XCTAssertEqual(config.semanticCodec.codebookSize, 8192)
        XCTAssertEqual(config.semanticCodec.hiddenSize, 1024)
        XCTAssertEqual(config.s2Mel.sampleRate, 22_050)
        XCTAssertEqual(config.s2Mel.nMels, 80)
        XCTAssertEqual(config.s2Mel.depth, 13)
        XCTAssertEqual(config.emotionBucketCounts, [3, 17, 2, 8, 4, 5, 10, 24])
        XCTAssertEqual(config.qwenEmotionPath, "qwen0.6bemo4-merge/")
        XCTAssertEqual(config.outputSampleRate, 22_050)
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
            try IndexTTS2BundleLoader.load(from: dir, expectedModelKey: "indextts2")
        ) { error in
            XCTAssertEqual(error as? IndexTTS2BundleError, .missingRequiredFile("config.json"))
        }
    }

    func testIndexTTS2TokenizerUsesSentencePieceScores() throws {
        let pieces = [
            SentencePieceModel.Piece(text: "<unk>", score: 0, type: SentencePieceModel.PieceType.unknown.rawValue),
            SentencePieceModel.Piece(text: "▁", score: -10, type: SentencePieceModel.PieceType.normal.rawValue),
            SentencePieceModel.Piece(text: "HELLO", score: -5, type: SentencePieceModel.PieceType.normal.rawValue),
            SentencePieceModel.Piece(text: "WORLD", score: -5, type: SentencePieceModel.PieceType.normal.rawValue),
            SentencePieceModel.Piece(text: "▁HELLO", score: -1, type: SentencePieceModel.PieceType.normal.rawValue),
            SentencePieceModel.Piece(text: "▁WORLD", score: -1, type: SentencePieceModel.PieceType.normal.rawValue),
        ]
        let tokenizer = try IndexTTS2Tokenizer(pieces: pieces)

        let ids = try tokenizer.encode("Hello world")

        XCTAssertEqual(ids, [4, 5])
        XCTAssertEqual(tokenizer.decode(ids), "HELLO WORLD")
    }

    func testIndexTTS2EmotionPresetVectors() throws {
        let eager = try IndexTTS2EmotionControl(preset: .eager, weight: 0.5)

        XCTAssertEqual(IndexTTS2EmotionPreset(named: "excited"), .excited)
        XCTAssertEqual(IndexTTS2EmotionPreset(named: "fear"), .afraid)
        XCTAssertEqual(eager.vector, [0.65, 0, 0, 0, 0, 0, 0.15, 0])
        XCTAssertEqual(eager.scaledVectorSum, 0.4, accuracy: 0.0001)
    }

    func testIndexTTS2EmotionVectorValidation() {
        XCTAssertThrowsError(try IndexTTS2EmotionControl(vector: [0.5, 0.5, 0, 0, 0, 0, 0, 0])) { error in
            XCTAssertEqual(error as? IndexTTS2EmotionControlError, .invalidVectorSum(1.0))
        }
        XCTAssertThrowsError(try IndexTTS2EmotionControl(vector: [0.1], weight: 1.0)) { error in
            XCTAssertEqual(error as? IndexTTS2EmotionControlError, .invalidVectorCount(1))
        }
        XCTAssertThrowsError(try IndexTTS2EmotionControl(vector: [0, 0, 0, 0, 0, 0, 0, 0], weight: 1.2)) { error in
            XCTAssertEqual(error as? IndexTTS2EmotionControlError, .invalidWeight(1.2))
        }
    }

    func testIndexTTS2SynthesisOptionsValidateSpeakingRate() throws {
        let options = try IndexTTS2SynthesisOptions(
            speakingRate: 1.15,
            maxInternalPauseDuration: 0.18)
        XCTAssertEqual(options.speakingRate, 1.15, accuracy: 0.001)
        XCTAssertEqual(try XCTUnwrap(options.maxInternalPauseDuration), 0.18, accuracy: 0.001)
        XCTAssertThrowsError(try IndexTTS2SynthesisOptions(speakingRate: 2.0)) { error in
            guard case AudioModelError.invalidConfiguration(let model, let reason) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(model, "IndexTTS2")
            XCTAssertTrue(reason.contains("speakingRate"))
        }
        XCTAssertThrowsError(try IndexTTS2SynthesisOptions(maxInternalPauseDuration: 0.01)) { error in
            guard case AudioModelError.invalidConfiguration(let model, let reason) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(model, "IndexTTS2")
            XCTAssertTrue(reason.contains("maxInternalPauseDuration"))
        }
    }

    func testIndexTTS2PauseCompressorCapsInternalSilence() {
        let sampleRate = 1_000
        let voiced = Array(repeating: Float(0.2), count: 250)
        let silence = Array(repeating: Float(0), count: 500)
        let samples = voiced + silence + voiced

        let compressed = IndexTTS2PauseCompressor.compress(
            samples,
            sampleRate: sampleRate,
            maxPauseDuration: 0.12)

        XCTAssertLessThan(compressed.count, samples.count - 250)
        XCTAssertGreaterThan(compressed.count, voiced.count * 2 + 90)
    }

    func testIndexTTS2BundleRejectsWrongManifestKey() async throws {
        let dir = try makeBundle(
            modelKey: "other-tts",
            displayName: "Other TTS",
            parameterCount: "1.5B-class",
            sampleRate: 24_000,
            convertedFiles: ["model.safetensors"])
        defer { try? FileManager.default.removeItem(at: dir) }

        XCTAssertThrowsError(
            try IndexTTS2BundleLoader.load(from: dir, expectedModelKey: "indextts2")
        ) { error in
            XCTAssertEqual(error as? IndexTTS2BundleError, .unexpectedModelKey(
                expected: "indextts2",
                actual: "other-tts"))
        }
    }

    func testIndexTTS2ProtocolGenerateRequiresReferenceAudio() async throws {
        let dir = try makeBundle(
            modelKey: "indextts2",
            displayName: "IndexTTS2",
            parameterCount: "1.5B-class",
            sampleRate: 24_000,
            convertedFiles: ["gpt.safetensors", "s2mel.safetensors"])
        defer { try? FileManager.default.removeItem(at: dir) }

        let model = try await IndexTTS2TTSModel.fromBundle(dir)

        do {
            _ = try await model.generate(text: "Hello", language: "en")
            XCTFail("Expected reference-required error")
        } catch let error as AudioModelError {
            guard case .inferenceFailed(let operation, let reason) = error else {
                return XCTFail("Unexpected AudioModelError: \(error)")
            }
            XCTAssertEqual(operation, "IndexTTS2 synthesis")
            XCTAssertTrue(reason.contains("reference audio"))
        }
    }

    func testUnloadClearsMemoryFootprint() async throws {
        let dir = try makeBundle(
            modelKey: "indextts2",
            displayName: "IndexTTS2",
            parameterCount: "1.5B-class",
            sampleRate: 24_000,
            convertedFiles: ["gpt.safetensors"])
        defer { try? FileManager.default.removeItem(at: dir) }

        let model = try await IndexTTS2TTSModel.fromBundle(dir)
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

    private static let indexTTS2ConfigYAML = """
    dataset:
        bpe_model: bpe.model
        sample_rate: 24000
    gpt:
        model_dim: 1280
        max_mel_tokens: 1815
        max_text_tokens: 600
        heads: 20
        layers: 24
        number_text_tokens: 12000
        number_mel_codes: 8194
        start_mel_token: 8192
        stop_mel_token: 8193
        start_text_token: 0
        stop_text_token: 1
        condition_type: "conformer_perceiver"
        condition_module:
            num_blocks: 6
        emo_condition_module:
            num_blocks: 4
    semantic_codec:
        codebook_size: 8192
        hidden_size: 1024
        codebook_dim: 8
        vocos_dim: 384
        vocos_num_layers: 12
    s2mel:
        preprocess_params:
            sr: 22050
            spect_params:
                n_fft: 1024
                win_length: 1024
                hop_length: 256
                n_mels: 80
        DiT:
            hidden_dim: 512
            num_heads: 8
            depth: 13
            in_channels: 80
            content_dim: 512
    emo_num: [3, 17, 2, 8, 4, 5, 10, 24]
    qwen_emo_path: qwen0.6bemo4-merge/
    version: 2.0
    """
}
