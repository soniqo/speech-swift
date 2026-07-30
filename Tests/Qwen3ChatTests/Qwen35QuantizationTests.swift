import XCTest
import MLX
import MLXNN
import MLXCommon
@testable import Qwen3Chat

/// Regression tests for INT4-only loading.
///
/// `Qwen35MLXModel` used to build every quantized layer with literal
/// `groupSize: 64, bits: 4` and never read the checkpoint's declared quantization,
/// so an INT8 or INT5 export was interpreted as INT4: same tensor names, same row
/// count, a packed row of the wrong width, and no error — mlx-swift's
/// `update(parameters:)` verifies nothing.
final class Qwen35QuantizationConfigTests: XCTestCase {

    /// A minimal Qwen3.5 config with the quantization statement under test spliced in.
    private func decodeConfig(quantizationFields: String) throws -> Qwen3ChatConfig {
        let json = """
        {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "intermediate_size": 3584,
            "vocab_size": 248320,
            "max_seq_len": 2048,
            "rope_theta": 10000000,
            "rms_norm_eps": 1e-6,
            "eos_token_id": 248046,
            "pad_token_id": 248044,
            "model_type": "qwen3_5_text"\(quantizationFields.isEmpty ? "" : ",")
            \(quantizationFields)
        }
        """
        return try JSONDecoder().decode(Qwen3ChatConfig.self, from: Data(json.utf8))
    }

    func testExplicitNumericFieldsAreAuthoritative() throws {
        let config = try decodeConfig(quantizationFields: """
            "quantization": "int4",
            "quantization_bits": 8,
            "quantization_group_size": 32
            """)

        // The numeric fields win over a stale label — the label is only a display string.
        XCTAssertEqual(config.quantBits, 8)
        XCTAssertEqual(config.quantGroupSize, 32)
    }

    func testFallsBackToQuantizationLabel() throws {
        let config = try decodeConfig(quantizationFields: #""quantization": "int8""#)

        XCTAssertEqual(config.quantBits, 8)
        // The label carries no group size, so MLX's default stands.
        XCTAssertEqual(config.quantGroupSize, 64)
    }

    func testReadsNestedMLXCommunityQuantizationObject() throws {
        let config = try decodeConfig(
            quantizationFields: #""quantization": {"group_size": 32, "bits": 5}"#)

        XCTAssertEqual(config.quantBits, 5)
        XCTAssertEqual(config.quantGroupSize, 32)
        XCTAssertEqual(config.quantization, "int5")
    }

    func testCheckpointPredatingBothFieldsLoadsAsInt4() throws {
        let config = try decodeConfig(quantizationFields: "")

        XCTAssertEqual(config.quantBits, 4)
        XCTAssertEqual(config.quantGroupSize, 64)
        XCTAssertEqual(config.quantization, "int4")
    }

    func testLabelThatNamesNoUsableWidthFallsBackToInt4() throws {
        // "bf16" parses to 16, which MLX cannot pack — ignore it rather than trust it.
        let config = try decodeConfig(quantizationFields: #""quantization": "bf16""#)

        XCTAssertEqual(config.quantBits, 4)
    }

    func testExistingInt4CheckpointIsUnchanged() throws {
        let config = try decodeConfig(quantizationFields: #""quantization": "int4""#)

        XCTAssertEqual(config.quantBits, 4)
        XCTAssertEqual(config.quantGroupSize, 64)
        XCTAssertEqual(Qwen3ChatConfig.qwen35_08B.quantBits, 4)
        XCTAssertEqual(Qwen3ChatConfig.qwen35_08B.quantGroupSize, 64)
    }

    func testExplicitFieldsSurviveEncodeDecode() throws {
        let original = try decodeConfig(quantizationFields: """
            "quantization": "int8",
            "quantization_bits": 8,
            "quantization_group_size": 32
            """)
        let decoded = try JSONDecoder().decode(
            Qwen3ChatConfig.self, from: JSONEncoder().encode(original))

        XCTAssertEqual(decoded.quantBits, 8)
        XCTAssertEqual(decoded.quantGroupSize, 32)
    }
}

/// Which variant `Qwen35MLXChat.fromPretrained` downloads when the caller names none.
///
/// The default is not cosmetic: `fromPretrained` builds the download path from
/// `Quantization.rawValue`, so this value decides which subdirectory of the model repo is
/// fetched. INT4 was retired for measured quality loss (+19.90% wikitext perplexity, 78.3%
/// top-1 agreement, 18-of-24 strict JSON against the bf16 reference) and the repo's `int4/`
/// directory is being removed, so a default that drifted back to `.int4` would not merely load
/// a worse model — it would 404.
final class Qwen35MLXQuantizationDefaultTests: XCTestCase {

    func testDefaultQuantizationIsInt5() {
        XCTAssertEqual(Qwen35MLXChat.defaultQuantization, .int5)
    }

    /// The raw value is a path segment in the model repo, so the spelling is load-bearing.
    func testDefaultVariantNamesThePublishedSubdirectory() {
        XCTAssertEqual(Qwen35MLXChat.defaultQuantization.rawValue, "int5")
    }

    /// INT4 stays reachable for a local checkpoint exported before INT5 existed; the point of
    /// the change is that nothing defaults to it, not that it became unloadable.
    func testInt4RemainsSelectable() {
        XCTAssertEqual(Qwen35MLXChat.Quantization.int4.rawValue, "int4")
        XCTAssertEqual(Qwen35MLXChat.Quantization(rawValue: "int4"), .int4)
    }

    func testEveryVariantRoundTripsThroughItsRawValue() {
        for variant: Qwen35MLXChat.Quantization in [.int4, .int5, .int8] {
            XCTAssertEqual(Qwen35MLXChat.Quantization(rawValue: variant.rawValue), variant)
        }
    }

    /// The download variant and the width the loader builds its layers at have to agree, or
    /// the checkpoint fetched from `int5/` fails the load-time shape check. `int5/config.json`
    /// carries the label, and `ChatQuantization` has to read 5 bits out of it.
    func testDefaultVariantLabelResolvesToItsBitWidth() {
        XCTAssertEqual(
            ChatQuantization.bits(fromLabel: Qwen35MLXChat.defaultQuantization.rawValue), 5)
    }
}

/// The layers a config actually builds, and the load-time check that refuses a
/// checkpoint packed at some other width.
final class Qwen35QuantizedLayerTests: XCTestCase {

    /// A small hybrid config (one DeltaNet layer, one GatedAttention layer) so the test
    /// exercises the real module tree without allocating an 0.8B model.
    private func tinyConfig(bits: Int, groupSize: Int) -> Qwen3ChatConfig {
        Qwen3ChatConfig(
            hiddenSize: 128,
            numHiddenLayers: 2,
            numAttentionHeads: 2,
            numKeyValueHeads: 1,
            headDim: 64,
            intermediateSize: 256,
            vocabSize: 512,
            maxSeqLen: 128,
            ropeTheta: 10_000_000,
            rmsNormEps: 1e-6,
            eosTokenId: 2,
            padTokenId: 0,
            quantization: "int\(bits)",
            quantizationBits: bits,
            quantizationGroupSize: groupSize,
            modelType: .qwen35,
            layerTypes: ["linear_attention", "full_attention"],
            fullAttentionInterval: 2,
            linearNumKeyHeads: 2,
            linearKeyHeadDim: 32,
            linearNumValueHeads: 2,
            linearValueHeadDim: 32,
            linearConvKernelDim: 4,
            partialRotaryFactor: 0.25,
            tieWordEmbeddings: true)
    }

    func testLayersAreBuiltAtTheDeclaredBitWidth() {
        let model = Qwen35MLXModel(config: tinyConfig(bits: 8, groupSize: 32))

        XCTAssertEqual(model.embedTokens.bits, 8)
        XCTAssertEqual(model.embedTokens.groupSize, 32)
        // Packed embedding row: dimensions * bits / 32 = 128 * 8 / 32 = 32 uint32 words.
        XCTAssertEqual(model.embedTokens.weight.shape, [512, 32])

        let deltaNet = model.layers[0].deltaNet!
        XCTAssertEqual(deltaNet.inProjQKV.bits, 8)
        XCTAssertEqual(deltaNet.outProj.groupSize, 32)

        let attn = model.layers[1].gatedAttn!
        XCTAssertEqual(attn.qProj.bits, 8)
        XCTAssertEqual(attn.oProj.bits, 8)

        XCTAssertEqual(model.layers[0].mlp.gateProj.bits, 8)
        XCTAssertEqual(model.layers[0].mlp.downProj.groupSize, 32)
    }

    func testInt4ConfigStillBuildsInt4Layers() {
        let model = Qwen35MLXModel(config: tinyConfig(bits: 4, groupSize: 64))

        XCTAssertEqual(model.embedTokens.bits, 4)
        XCTAssertEqual(model.embedTokens.groupSize, 64)
        XCTAssertEqual(model.embedTokens.weight.shape, [512, 16])
        XCTAssertEqual(model.layers[1].gatedAttn!.qProj.bits, 4)
    }

    // MARK: - Load-time verification

    /// Tensors as a checkpoint quantized at `bits`/`groupSize` would store them.
    private func packedWeights(
        prefix: String, rows: Int, columns: Int, bits: Int, groupSize: Int
    ) -> [String: MLXArray] {
        [
            "\(prefix).weight": MLXArray.zeros([rows, columns * bits / 32], dtype: .uint32),
            "\(prefix).scales": MLXArray.zeros([rows, columns / groupSize], dtype: .bfloat16),
            "\(prefix).biases": MLXArray.zeros([rows, columns / groupSize], dtype: .bfloat16),
        ]
    }

    func testMatchingTensorsLoad() throws {
        let linear = QuantizedLinear(128, 256, bias: false, groupSize: 64, bits: 8)
        let weights = packedWeights(
            prefix: "mlp.gate_proj", rows: 256, columns: 128, bits: 8, groupSize: 64)

        try CommonWeightLoader.applyCheckedQuantizedLinearWeights(
            to: linear, prefix: "mlp.gate_proj", from: weights)

        XCTAssertEqual(linear.weight.shape, [256, 32])
    }

    func testInt4TensorsAreRejectedByAnInt8Layer() {
        let linear = QuantizedLinear(128, 256, bias: false, groupSize: 64, bits: 8)
        let weights = packedWeights(
            prefix: "mlp.gate_proj", rows: 256, columns: 128, bits: 4, groupSize: 64)

        XCTAssertThrowsError(
            try CommonWeightLoader.applyCheckedQuantizedLinearWeights(
                to: linear, prefix: "mlp.gate_proj", from: weights)
        ) { error in
            guard case QuantizedWeightMismatch.bits(_, let declared, let implied) = error else {
                return XCTFail("expected a bit-width mismatch, got \(error)")
            }
            XCTAssertEqual(declared, 8)
            XCTAssertEqual(implied, 4)
        }
    }

    /// The shape this bug actually produced: an INT8 checkpoint read by INT4 layers.
    func testInt8TensorsAreRejectedByAnInt4Layer() {
        let linear = QuantizedLinear(128, 256, bias: false, groupSize: 64, bits: 4)
        let weights = packedWeights(
            prefix: "self_attn.q_proj", rows: 256, columns: 128, bits: 8, groupSize: 64)

        XCTAssertThrowsError(
            try CommonWeightLoader.applyCheckedQuantizedLinearWeights(
                to: linear, prefix: "self_attn.q_proj", from: weights)
        ) { error in
            guard case QuantizedWeightMismatch.bits(_, let declared, let implied) = error else {
                return XCTFail("expected a bit-width mismatch, got \(error)")
            }
            XCTAssertEqual(declared, 4)
            XCTAssertEqual(implied, 8)
        }
        // The layer keeps its own shape rather than silently taking the wrong one.
        XCTAssertEqual(linear.weight.shape, [256, 16])
    }

    func testWrongGroupSizeIsRejected() {
        let linear = QuantizedLinear(128, 256, bias: false, groupSize: 64, bits: 4)
        let weights = packedWeights(
            prefix: "mlp.up_proj", rows: 256, columns: 128, bits: 4, groupSize: 32)

        XCTAssertThrowsError(
            try CommonWeightLoader.applyCheckedQuantizedLinearWeights(
                to: linear, prefix: "mlp.up_proj", from: weights)
        ) { error in
            guard case QuantizedWeightMismatch.groupSize(_, let declared, let implied) = error
            else {
                return XCTFail("expected a group-size mismatch, got \(error)")
            }
            XCTAssertEqual(declared, 64)
            XCTAssertEqual(implied, 32)
        }
    }

    func testPackedWeightWithoutScalesIsRejected() {
        let linear = QuantizedLinear(128, 256, bias: false, groupSize: 64, bits: 4)
        var weights = packedWeights(
            prefix: "mlp.down_proj", rows: 256, columns: 128, bits: 4, groupSize: 64)
        weights["mlp.down_proj.scales"] = nil

        XCTAssertThrowsError(
            try CommonWeightLoader.applyCheckedQuantizedLinearWeights(
                to: linear, prefix: "mlp.down_proj", from: weights)
        ) { error in
            guard case QuantizedWeightMismatch.missingScales = error else {
                return XCTFail("expected a missing-scales error, got \(error)")
            }
        }
    }

    func testEmbeddingBitWidthIsChecked() {
        let embedding = PreQuantizedEmbedding(
            embeddingCount: 512, dimensions: 128, groupSize: 64, bits: 4)
        let weights = packedWeights(
            prefix: "embed_tokens", rows: 512, columns: 128, bits: 8, groupSize: 64)

        XCTAssertThrowsError(
            try CommonWeightLoader.applyCheckedQuantizedEmbeddingWeights(
                to: embedding, prefix: "embed_tokens", from: weights)
        ) { error in
            guard case QuantizedWeightMismatch.bits(_, let declared, let implied) = error else {
                return XCTFail("expected a bit-width mismatch, got \(error)")
            }
            XCTAssertEqual(declared, 4)
            XCTAssertEqual(implied, 8)
        }
    }
}
