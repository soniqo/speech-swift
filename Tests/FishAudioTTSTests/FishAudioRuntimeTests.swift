@testable import FishAudioTTS
import MLX
import XCTest

final class FishAudioRuntimeTests: XCTestCase {
    func testConfigParsesFishS2ProShape() throws {
        let dir = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }

        let configURL = dir.appendingPathComponent("config.json")
        let json = """
        {
          "model_type": "fish_qwen3_omni",
          "dtype": "float16",
          "eos_token_id": 151645,
          "pad_token_id": 151669,
          "audio_pad_token_id": 151677,
          "semantic_start_token_id": 151678,
          "semantic_end_token_id": 155773,
          "text_config": {
            "vocab_size": 155776,
            "n_layer": 36,
            "n_head": 32,
            "n_local_heads": 8,
            "head_dim": 128,
            "dim": 2560,
            "intermediate_size": 9728,
            "max_seq_len": 32768,
            "rope_base": 1000000,
            "norm_eps": 0.000001,
            "tie_word_embeddings": true,
            "attention_qk_norm": true,
            "attention_qkv_bias": false,
            "attention_o_bias": false
          },
          "audio_decoder_config": {
            "vocab_size": 4096,
            "num_codebooks": 10,
            "n_layer": 4,
            "n_head": 32,
            "n_local_heads": 8,
            "head_dim": 128,
            "dim": 2560,
            "text_dim": 2560,
            "intermediate_size": 9728,
            "max_seq_len": 11,
            "rope_base": 1000000,
            "norm_eps": 0.000001,
            "attention_qk_norm": false,
            "attention_qkv_bias": false,
            "attention_o_bias": false
          }
        }
        """
        try json.write(to: configURL, atomically: true, encoding: .utf8)

        let config = try FishAudioConfig.load(from: configURL)

        XCTAssertEqual(config.modelType, "fish_qwen3_omni")
        XCTAssertEqual(config.semanticTokenCount, 4_096)
        XCTAssertTrue(config.scaleCodebookEmbeddings)
        XCTAssertTrue(config.normFastLayerInput)
        XCTAssertEqual(config.text.hiddenSize, 2_560)
        XCTAssertEqual(config.text.numHiddenLayers, 36)
        XCTAssertEqual(config.text.numKeyValueHeads, 8)
        XCTAssertTrue(config.text.attentionQKNorm)
        XCTAssertEqual(config.audioDecoder.numCodebooks, 10)
        XCTAssertEqual(config.audioDecoder.numHiddenLayers, 4)
        XCTAssertFalse(config.audioDecoder.attentionQKNorm)
    }

    func testTokenizerLoadsFishSemanticAndSpecialTokens() throws {
        let dir = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeTokenizerJSON(vocab: completeFishVocab(), to: dir)

        let metadata = try FishAudioTokenizerMetadata.load(from: dir)

        XCTAssertEqual(metadata.semanticBeginId, 151_678)
        XCTAssertEqual(metadata.semanticEndId, 155_773)
        XCTAssertEqual(try metadata.semanticTokenId(for: 0), 151_678)
        XCTAssertEqual(try metadata.semanticTokenId(for: 4_095), 155_773)
        XCTAssertEqual(try metadata.tokenId(FishAudioToken.eos), 151_643)
        XCTAssertEqual(try metadata.tokenId(FishAudioToken.imStart), 151_644)
        XCTAssertEqual(try metadata.tokenId(FishAudioToken.audioPad), 151_677)
    }

    func testTokenizerRejectsIncompleteSemanticVocabulary() throws {
        var vocab = completeFishVocab()
        vocab.removeValue(forKey: FishAudioToken.semantic(12))

        XCTAssertThrowsError(try FishAudioTokenizerMetadata.parse(vocab: vocab)) { error in
            XCTAssertEqual(error as? FishAudioError, .missingToken(FishAudioToken.semantic(12)))
        }
    }

    func testPromptUsesFishChatTemplateAndSpeakerReferences() {
        let system = FishAudioPrompt.systemPrompt(referenceTexts: [
            "नमस्ते, यह संदर्भ है।",
            "<|speaker:9|>Already tagged.",
        ])
        let prompt = FishAudioPrompt.chatTemplate(
            system: system,
            user: "नमस्ते [excited]"
        )

        XCTAssertEqual(system, """
        convert the provided text to speech reference to the following:

        Text:
        <|speaker:0|>नमस्ते, यह संदर्भ है।
        <|speaker:9|>Already tagged.

        Speech:
        """)
        XCTAssertEqual(prompt, """
        <|im_start|>system
        convert the provided text to speech reference to the following:

        Text:
        <|speaker:0|>नमस्ते, यह संदर्भ है।
        <|speaker:9|>Already tagged.

        Speech:<|im_end|>
        <|im_start|>user
        नमस्ते [excited]<|im_end|>
        <|im_start|>assistant
        <|voice|>
        """)
    }

    func testReferenceTextTaggingMatchesFishInference() {
        XCTAssertEqual(
            FishAudioInputBuilder.taggedReferenceText([
                "पहला संदर्भ।",
                "<|speaker:7|>Already tagged.",
            ]),
            """
            <|speaker:0|>पहला संदर्भ।
            <|speaker:7|>Already tagged.
            """)
    }

    func testSamplerConstrainsAllowedTokens() {
        let logits: [Float] = [0, 100, 2, 3, 4, 5]
        let sampled = FishAudioSampler.sample(
            logits: logits,
            allowedTokenIds: [2, 4],
            config: .greedy)

        XCTAssertEqual(sampled, 4)
    }

    func testReferencePromptRejectsInvalidCodebookShape() throws {
        XCTAssertThrowsError(try FishAudioReferencePrompt(text: "bad", codes: [])) { error in
            XCTAssertEqual(
                error as? FishAudioError,
                .invalidCodebookShape("reference codes must not be empty"))
        }
        XCTAssertThrowsError(
            try FishAudioReferencePrompt(text: "bad", codes: [[1, 2], [1]])
        ) { error in
            XCTAssertEqual(
                error as? FishAudioError,
                .invalidCodebookShape("all codebooks must have the same frame count"))
        }
    }

    func testEmotionMarkerVocabularyMatchesExportManifest() {
        XCTAssertTrue(FishAudioEmotionMarker.supported.contains("[excited]"))
        XCTAssertTrue(FishAudioEmotionMarker.supported.contains("[angry]"))
        XCTAssertTrue(FishAudioEmotionMarker.supported.contains("[whisper]"))
        XCTAssertFalse(FishAudioEmotionMarker.supported.contains("<happy>"))
    }

    func testWeightRemapMatchesFishQwen3OmniExport() {
        let raw = [
            "text_model.model.embeddings.weight": 1,
            "text_model.model.layers.0.attention.wqkv.weight": 2,
            "audio_decoder.codebook_embeddings.weight": 3,
            "audio_decoder.embeddings.weight": 4,
            "audio_decoder.layers.0.attention.wqkv.weight": 5,
            "audio_decoder.output.weight": 6,
        ]

        let remapped = FishAudioWeightLoader.remapFishQwen3OmniKeys(raw)

        XCTAssertNotNil(remapped["embeddings.weight"])
        XCTAssertNotNil(remapped["layers.0.attention.wqkv.weight"])
        XCTAssertNotNil(remapped["codebook_embeddings.weight"])
        XCTAssertNotNil(remapped["fast_embeddings.weight"])
        XCTAssertNotNil(remapped["fast_layers.0.attention.wqkv.weight"])
        XCTAssertNotNil(remapped["fast_output.weight"])
        XCTAssertNil(remapped["audio_decoder.embeddings.weight"])
    }

    func testModelShardDiscoveryIgnoresCodecSafetensors() throws {
        let dir = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }

        for name in ["model-00002-of-00002.safetensors", "codec.safetensors", "model-00001-of-00002.safetensors"] {
            FileManager.default.createFile(
                atPath: dir.appendingPathComponent(name).path,
                contents: Data())
        }

        let files = try FishAudioWeightLoader.modelSafetensorFiles(in: dir)
            .map(\.lastPathComponent)

        XCTAssertEqual(files, [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ])
    }

    func testTinyDualARForwardShapes() throws {
        try XCTSkipIf(
            !hasCompiledMLXMetallib(),
            "Fish Audio MLX forward shape test requires scripts/build_mlx_metallib.sh debug")
        let config = tinyConfig()
        let model = FishAudioDualARModel(config: config)
        let tokens = MLXArray([
            Int32(1), Int32(20),
            Int32(0), Int32(1),
            Int32(0), Int32(2),
        ]).reshaped([1, config.audioDecoder.numCodebooks + 1, 2])

        let (slow, _) = model.forwardSlow(inputIds: tokens, state: model.initialSlowState())
        eval(slow.logits, slow.hiddenStates)

        XCTAssertEqual(slow.logits.shape, [1, 2, config.text.vocabSize])
        XCTAssertEqual(slow.hiddenStates.shape, [1, 2, config.text.hiddenSize])

        let lastHidden = slow.hiddenStates[0..., 1, 0...]
            .reshaped([1, 1, config.text.hiddenSize])
        let (fastLogits, _) = model.forwardFast(
            inputEmbeddings: lastHidden,
            state: model.initialFastState())
        eval(fastLogits)

        XCTAssertEqual(fastLogits.shape, [1, 1, config.audioDecoder.vocabSize])

        let input = try FishAudioModelInput(rows: [
            [1, 20],
            [0, 1],
            [0, 2],
        ])
        let generated = try model.generateCodebooks(
            from: input,
            sampling: FishAudioSamplingConfig(
                maxNewTokens: 2,
                temperature: 0,
                topK: 1,
                topP: 1,
                repetitionPenalty: 1))
        XCTAssertEqual(generated.codebookCount, config.audioDecoder.numCodebooks)
        XCTAssertLessThanOrEqual(generated.frameCount, 2)
        XCTAssertTrue(generated.codes.allSatisfy { $0.count == generated.frameCount })
    }
}

private func completeFishVocab() -> [String: Int] {
    var vocab = [
        FishAudioToken.eos: 151_643,
        FishAudioToken.pad: 151_669,
        FishAudioToken.imStart: 151_644,
        FishAudioToken.imEnd: 151_645,
        FishAudioToken.textModality: 151_672,
        FishAudioToken.voiceModality: 151_673,
        FishAudioToken.interleaveModality: 151_674,
        FishAudioToken.audioStart: 151_675,
        FishAudioToken.audioEnd: 151_676,
        FishAudioToken.audioPad: 151_677,
    ]
    for code in 0..<FishAudioDefaults.codebookSize {
        vocab[FishAudioToken.semantic(code)] = 151_678 + code
    }
    return vocab
}

private func writeTokenizerJSON(vocab: [String: Int], to directory: URL) throws {
    let addedTokens = vocab
        .sorted { $0.value < $1.value }
        .map { token, id -> [String: Any] in
            [
                "id": id,
                "content": token,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            ]
        }
    let root: [String: Any] = [
        "model": [
            "vocab": [
                "!": 0,
                "hello": 1,
            ]
        ],
        "added_tokens": addedTokens,
    ]
    let data = try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
    try data.write(to: directory.appendingPathComponent("tokenizer.json"))
}

private func temporaryDirectory() throws -> URL {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("fish-audio-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

private func hasCompiledMLXMetallib() -> Bool {
    let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let candidates = [
        cwd.appendingPathComponent(".build/debug/mlx.metallib"),
        cwd.appendingPathComponent(".build/arm64-apple-macosx/debug/mlx.metallib"),
    ]
    return candidates.contains { FileManager.default.fileExists(atPath: $0.path) }
}

private func tinyConfig() -> FishAudioConfig {
    let text = FishAudioTransformerConfig(
        vocabSize: 32,
        numHiddenLayers: 1,
        numAttentionHeads: 2,
        numKeyValueHeads: 1,
        headDim: 4,
        hiddenSize: 8,
        intermediateSize: 16,
        maxSequenceLength: 16,
        ropeTheta: 10_000,
        rmsNormEps: 1e-6,
        tieWordEmbeddings: true,
        attentionQKNorm: true,
        attentionQKVBias: false,
        attentionOutputBias: false
    )
    let decoder = FishAudioDecoderConfig(
        vocabSize: 4_096,
        numCodebooks: 2,
        numHiddenLayers: 1,
        numAttentionHeads: 2,
        numKeyValueHeads: 1,
        headDim: 4,
        hiddenSize: 8,
        textHiddenSize: 8,
        intermediateSize: 16,
        maxSequenceLength: 3,
        ropeTheta: 10_000,
        rmsNormEps: 1e-6,
        attentionQKNorm: false,
        attentionQKVBias: false,
        attentionOutputBias: false
    )
    return FishAudioConfig(
        modelType: "fish_qwen3_omni",
        dtype: "float16",
        eosTokenId: 2,
        padTokenId: 0,
        audioPadTokenId: 0,
        semanticStartTokenId: 20,
        semanticEndTokenId: 23,
        text: text,
        audioDecoder: decoder
    )
}
