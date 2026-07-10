@testable import HiggsTTS
import XCTest

final class HiggsTTSTests: XCTestCase {
    private let boc: Int32 = 1024
    private let eoc: Int32 = 1025

    func testDelayPatternMatchesReferenceLayout() throws {
        let codes: [[Int32]] = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ]
        let delayed = try HiggsTTSDelayPattern.apply(
            codes, codebooks: 4, bocId: boc, eocId: eoc)

        XCTAssertEqual(delayed, [
            [1, boc, boc, boc],
            [5, 2, boc, boc],
            [9, 6, 3, boc],
            [eoc, 10, 7, 4],
            [eoc, eoc, 11, 8],
            [eoc, eoc, eoc, 12],
        ])
    }

    func testDelayReverseRoundTrip() throws {
        var codes: [[Int32]] = []
        for t in 0..<25 {
            codes.append((0..<8).map { Int32((t * 8 + $0) % 1024) })
        }
        let delayed = try HiggsTTSDelayPattern.apply(
            codes, codebooks: 8, bocId: boc, eocId: eoc)
        XCTAssertEqual(delayed.count, 25 + 7)
        let restored = try HiggsTTSDelayPattern.reverse(delayed, codebooks: 8)
        XCTAssertEqual(restored, codes)
    }

    func testDelayRejectsInvalidShapes() {
        XCTAssertThrowsError(try HiggsTTSDelayPattern.apply(
            [], codebooks: 8, bocId: boc, eocId: eoc))
        XCTAssertThrowsError(try HiggsTTSDelayPattern.apply(
            [[1, 2]], codebooks: 8, bocId: boc, eocId: eoc))
        XCTAssertThrowsError(try HiggsTTSDelayPattern.reverse(
            [[Int32]](repeating: [Int32](repeating: 0, count: 8), count: 7),
            codebooks: 8))
    }

    func testSamplerRampForcesBOCOnUpperCodebooks() throws {
        var state = HiggsTTSSamplerState(codebooks: 4, bocId: boc, eocId: eoc)

        XCTAssertEqual(try state.advance([10, 20, 30, 40]), [10, boc, boc, boc])
        XCTAssertEqual(try state.advance([11, 21, 31, 41]), [11, 21, boc, boc])
        XCTAssertEqual(try state.advance([12, 22, 32, 42]), [12, 22, 32, boc])
        XCTAssertEqual(try state.advance([13, 23, 33, 43]), [13, 23, 33, 43])
        XCTAssertEqual(try state.advance([14, 24, 34, 44]), [14, 24, 34, 44])
        XCTAssertFalse(state.isDone)
    }

    func testSamplerEOCCountdownStopsAfterNMinusTwoSteps() throws {
        var state = HiggsTTSSamplerState(codebooks: 4, bocId: boc, eocId: eoc)
        for _ in 0..<4 {
            _ = try state.advance([1, 2, 3, 4])
        }

        _ = try state.advance([eoc, 5, 6, 7])
        XCTAssertFalse(state.isDone)
        _ = try state.advance([eoc, eoc, 8, 9])
        XCTAssertFalse(state.isDone)
        _ = try state.advance([eoc, eoc, eoc, 10])
        XCTAssertTrue(state.isDone)
    }

    func testSamplerTwoCodebooksStopsImmediatelyOnEOC() throws {
        var state = HiggsTTSSamplerState(codebooks: 2, bocId: boc, eocId: eoc)
        for _ in 0..<2 {
            _ = try state.advance([1, 2])
        }
        _ = try state.advance([eoc, 3])
        XCTAssertTrue(state.isDone)
    }

    func testSamplerRejectsWrongWidth() {
        var state = HiggsTTSSamplerState(codebooks: 8, bocId: boc, eocId: eoc)
        XCTAssertThrowsError(try state.advance([1, 2, 3]))
    }

    func testPromptBuilderMatchesReferenceLayout() throws {
        let specials = HiggsTTSSpecialTokens(
            tts: 200, refAudio: 201, refText: 202, text: 203, audio: 204)
        let builder = HiggsTTSPromptBuilder(specials: specials) { text in
            text.utf8.map { Int32($0) }
        }
        let delayed: [[Int32]] = [[1, boc], [2, 3], [eoc, 4]]

        let prompt = builder.build(
            text: "Hi",
            references: [HiggsTTSReference(delayedCodes: delayed, text: "ok")])

        let ph = HiggsTTSPrompt.audioPlaceholderId
        XCTAssertEqual(prompt.tokenIds, [
            200,                    // <|tts|>
            202, 111, 107,          // <|ref_text|> "ok"
            201, ph, ph, ph,        // <|ref_audio|> + 3 code rows
            203, 72, 105,           // <|text|> "Hi"
            204,                    // <|audio|>
        ])
        XCTAssertEqual(prompt.audioSegments.count, 1)
        XCTAssertEqual(prompt.audioSegments[0].start, 5)
        XCTAssertEqual(prompt.audioSegments[0].delayedCodes, delayed)
    }

    func testPromptBuilderWithoutReferenceOrRefTextToken() {
        let specials = HiggsTTSSpecialTokens(
            tts: 200, refAudio: 201, refText: nil, text: 203, audio: 204)
        let builder = HiggsTTSPromptBuilder(specials: specials) { _ in [9] }

        let plain = builder.build(text: "x")
        XCTAssertEqual(plain.tokenIds, [200, 203, 9, 204])
        XCTAssertTrue(plain.audioSegments.isEmpty)

        // Without a <|ref_text|> id the transcript is omitted, codes are kept.
        let cloned = builder.build(
            text: "x",
            references: [HiggsTTSReference(delayedCodes: [[1, 2]], text: "ignored")])
        XCTAssertEqual(cloned.tokenIds, [
            200, 201, HiggsTTSPrompt.audioPlaceholderId, 203, 9, 204,
        ])
        XCTAssertEqual(cloned.audioSegments[0].start, 2)
    }

    func testWeightMapRoutesUpstreamKeys() {
        func route(_ key: String) -> (HiggsTTSWeightComponent, String)? {
            HiggsTTSWeightMap.remap(key).map { ($0.component, $0.key) }
        }

        XCTAssertEqual(
            route("body.layers.0.self_attn.q_proj.weight")?.1,
            "layers.0.self_attn.q_proj.weight")
        XCTAssertEqual(route("body.layers.35.mlp.gate_proj.weight")?.0, .backbone)
        XCTAssertEqual(route("body.norm.weight")?.1, "norm.weight")
        XCTAssertEqual(
            route("tied.embedding.text_embedding.weight")?.1,
            "embed_tokens.weight")
        XCTAssertEqual(
            route("tied.embedding.modality_embeddings.0.embedding.weight")?.1,
            "weight")
        XCTAssertEqual(
            route("tied.embedding.modality_embeddings.0.embedding.weight")?.0,
            .fusedEmbedding)
        XCTAssertEqual(
            route("tied.embedding.modality_embeddings.0.model.acoustic_decoder.conv1.weight")?.0,
            .codec)
        XCTAssertNil(HiggsTTSWeightMap.remap("tied.head.text_head.weight"))
        XCTAssertNil(HiggsTTSWeightMap.remap("tied.head.modality_heads.0.weight"))
        XCTAssertNil(HiggsTTSWeightMap.remap("unrelated.weight"))
    }

    func testConfigParsesUpstreamShape() throws {
        let json = """
        {
          "model_type": "higgs_multimodal_qwen3",
          "audio_token_id": -100,
          "audio_encoder_config": {
            "num_codebooks": 8,
            "vocab_size": 1026,
            "use_delay_pattern": true
          },
          "text_config": {
            "hidden_size": 2560,
            "num_hidden_layers": 36,
            "intermediate_size": 9728,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "vocab_size": 151936,
            "tie_word_embeddings": true
          }
        }
        """
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("higgs-config-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("config.json")
        try json.write(to: url, atomically: true, encoding: .utf8)

        let config = try HiggsTTSConfig.load(from: url)
        XCTAssertEqual(config.audioNumCodebooks, 8)
        XCTAssertEqual(config.audioCodebookSize, 1026)
        XCTAssertEqual(config.audioBOCTokenId, 1024)
        XCTAssertEqual(config.audioEOCTokenId, 1025)
        XCTAssertEqual(config.sampleRate, 24_000)
        XCTAssertTrue(config.useDelayPattern)
        XCTAssertEqual(config.textConfig.numHiddenLayers, 36)
        XCTAssertEqual(config.textConfig.hiddenSize, 2560)
        XCTAssertEqual(config.textConfig.numKeyValueHeads, 8)
        XCTAssertTrue(config.textConfig.tieWordEmbeddings)
    }

    func testConfigRejectsUnknownModelType() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("higgs-config-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("config.json")
        try #"{"model_type": "not-higgs"}"#.write(to: url, atomically: true, encoding: .utf8)

        XCTAssertThrowsError(try HiggsTTSConfig.load(from: url)) { error in
            guard case HiggsTTSError.unexpectedModelType = error else {
                return XCTFail("expected unexpectedModelType, got \(error)")
            }
        }
    }
}
