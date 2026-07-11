import Foundation
import MLXCommon
import MLX
import MLXNN
import AudioCommon

/// Weight loading for CosyVoice3 TTS components (LLM, Flow/DiT, HiFi-GAN).
///
/// Loads from three separate safetensors files produced by the conversion script:
/// - `llm.safetensors`: Qwen2.5-0.5B based speech token generator
/// - `flow.safetensors`: Conditional flow matching with DiT decoder
/// - `hifigan.safetensors`: Neural source filter vocoder
///
/// All weight-bearing matmul modules in the LLM and DiT are declared as
/// `Linear` and swapped to `QuantizedLinear` per-path at load time when the
/// safetensors carries matching `.scales`. This lets one runtime serve
/// quantised bundles (8-bit, 8-bit-full) and the unquantised bf16 bundle from
/// the same module hierarchy.
public enum CosyVoiceWeightLoader {

    // MARK: - LLM

    /// Load LLM weights from llm.safetensors into the CosyVoiceLLM module.
    ///
    /// Expected keys (after conversion script remapping):
    /// - text_embedding.weight
    /// - speech_embedding.weight
    /// - layers.{i}.self_attn.q_proj.weight (+ optional .scales/.biases for quant)
    /// - layers.{i}.self_attn.q_proj.bias (Qwen2.5 has biased q/k/v projections)
    /// - layers.{i}.self_attn.k_proj/v_proj/o_proj (same pattern; o_proj has no bias)
    /// - layers.{i}.input_layernorm.weight
    /// - layers.{i}.post_attention_layernorm.weight
    /// - layers.{i}.mlp.gate_proj/up_proj/down_proj (no Linear bias)
    /// - norm.weight
    /// - speech_head.weight (+ optional .scales/.biases for quant)
    public static func loadLLM(_ llm: CosyVoiceLLM, from url: URL) throws {
        let weights = try CommonWeightLoader.loadSafetensors(url: url)

        // ─── Phase 1: per-projection dispatch ──────────────────────────────────
        //
        // Every projection that *might* be quantised gets one of two treatments:
        //
        //   1. If the safetensors has `<prefix>.scales`, the projection is
        //      packed: build a fully-initialised `QuantizedLinear` from the
        //      pre-loaded tensors and stage it for `update(modules:)` so the
        //      existing `Linear` is swapped in place.
        //
        //   2. Otherwise (bf16 bundle), leave the `Linear` in place and apply
        //      `.weight` + optional `.bias` via the common helper.
        //
        // Doing the swap through `update(modules:)` (rather than MLX's
        // `quantize(model:filter:)`) keeps us in control of dtype: the
        // QuantizedLinear is constructed with the loaded fp16 scales/biases
        // already, so the matmul site cannot land an fp16 weight where it
        // expects a packed uint32 — the bug that the 8-bit refactor flushed out.
        let perLayerProj: [(String, String, Bool)] = [
            ("self_attn.q_proj", "selfAttn.qProj", true),
            ("self_attn.k_proj", "selfAttn.kProj", true),
            ("self_attn.v_proj", "selfAttn.vProj", true),
            ("self_attn.o_proj", "selfAttn.oProj", false),
            ("mlp.gate_proj",    "mlp.gateProj",    false),
            ("mlp.up_proj",      "mlp.upProj",      false),
            ("mlp.down_proj",    "mlp.downProj",    false),
        ]

        var qReplacements: [String: Module] = [:]

        func handleProjection(
            stPrefix: String,
            modPath: String,
            owner: Linear,
            hasLinearBias: Bool,
            llmConfig: CosyVoiceLLMConfig
        ) throws {
            if let scales = weights["\(stPrefix).scales"],
               let w = weights["\(stPrefix).weight"] {
                let bits = llmConfig.bits
                let groupSize = llmConfig.groupSize
                try validate8BitQuantizationLayout(
                    weight: w, scales: scales, bits: bits, groupSize: groupSize)
                let quantBiases = weights["\(stPrefix).biases"]
                let linearBias  = hasLinearBias ? weights["\(stPrefix).bias"] : nil
                qReplacements[modPath] = QuantizedLinear(
                    weight: w, bias: linearBias,
                    scales: scales, biases: quantBiases,
                    groupSize: groupSize, bits: bits)
            } else {
                CommonWeightLoader.applyLinearWeights(to: owner, prefix: stPrefix, from: weights)
            }
        }

        for (i, block) in llm.layers.enumerated() {
            for (sfxSnake, sfxCamel, hasLinearBias) in perLayerProj {
                let stPrefix = "layers.\(i).\(sfxSnake)"
                let modPath  = "layers.\(i).\(sfxCamel)"
                let owner: Linear
                switch sfxSnake {
                case "self_attn.q_proj": owner = block.selfAttn.qProj
                case "self_attn.k_proj": owner = block.selfAttn.kProj
                case "self_attn.v_proj": owner = block.selfAttn.vProj
                case "self_attn.o_proj": owner = block.selfAttn.oProj
                case "mlp.gate_proj":    owner = block.mlp.gateProj
                case "mlp.up_proj":      owner = block.mlp.upProj
                case "mlp.down_proj":    owner = block.mlp.downProj
                default: continue
                }
                try handleProjection(
                    stPrefix: stPrefix,
                    modPath: modPath,
                    owner: owner,
                    hasLinearBias: hasLinearBias,
                    llmConfig: llm.config)
            }
        }

        // Speech head sits at the LLM's top level; no Linear bias upstream.
        try handleProjection(
            stPrefix: "speech_head",
            modPath: "speechHead",
            owner: llm.speechHead,
            hasLinearBias: false,
            llmConfig: llm.config)

        if !qReplacements.isEmpty {
            let nested = NestedDictionary<String, Module>.unflattened(qReplacements)
            llm.update(modules: nested)
        }

        // ─── Phase 2: non-quantized parameters (embeddings, layer norms) ───────
        CommonWeightLoader.applyEmbeddingWeights(
            to: llm.textEmbedding, prefix: "text_embedding", from: weights)
        CommonWeightLoader.applyEmbeddingWeights(
            to: llm.speechEmbedding, prefix: "speech_embedding", from: weights)
        for (i, layer) in llm.layers.enumerated() {
            CommonWeightLoader.applyRMSNormWeights(
                to: layer.inputLayerNorm,
                prefix: "layers.\(i).input_layernorm", from: weights)
            CommonWeightLoader.applyRMSNormWeights(
                to: layer.postAttentionLayerNorm,
                prefix: "layers.\(i).post_attention_layernorm", from: weights)
        }
        CommonWeightLoader.applyRMSNormWeights(
            to: llm.norm, prefix: "norm", from: weights)
    }

    // MARK: - Flow (DiT Decoder)

    /// Load flow weights from flow.safetensors into the CosyVoiceFlowModel.
    public static func loadFlow(_ flow: CosyVoiceFlowModel, from url: URL) throws {
        let weights = try CommonWeightLoader.loadSafetensors(url: url)

        CommonWeightLoader.applyEmbeddingWeights(
            to: flow.inputEmbedding, prefix: "input_embedding", from: weights)

        CommonWeightLoader.applyLinearWeights(
            to: flow.spkEmbedAffineLayer, prefix: "spk_embed_affine_layer", from: weights)

        CommonWeightLoader.applyConv1dWeights(
            to: flow.preLookaheadLayer.conv1.conv,
            prefix: "pre_lookahead_layer.conv1", from: weights, transpose: false)
        CommonWeightLoader.applyConv1dWeights(
            to: flow.preLookaheadLayer.conv2.conv,
            prefix: "pre_lookahead_layer.conv2", from: weights, transpose: false)

        try loadDiT(flow.decoder.decoder, prefix: "decoder", from: weights, ditConfig: flow.decoder.decoder.config)
    }

    /// Load DiT weights. Performs the same per-projection quantised/plain
    /// dispatch as the LLM loader, building one `update(modules:)` swap for
    /// every projection that ships with `.scales`.
    static func loadDiT(
        _ dit: DiT,
        prefix: String,
        from weights: [String: MLXArray],
        ditConfig: CosyVoiceDiTConfig
    ) throws {
        var qReplacements: [String: Module] = [:]

        func handleProjection(
            stPrefix: String,
            modPath: String,
            owner: Linear,
            hasLinearBias: Bool = true
        ) throws {
            if let scales = weights["\(stPrefix).scales"],
               let w = weights["\(stPrefix).weight"] {
                let bits = ditConfig.bits
                let groupSize = ditConfig.groupSize
                try validate8BitQuantizationLayout(
                    weight: w, scales: scales, bits: bits, groupSize: groupSize)
                let quantBiases = weights["\(stPrefix).biases"]
                let linearBias  = hasLinearBias ? weights["\(stPrefix).bias"] : nil
                qReplacements[modPath] = QuantizedLinear(
                    weight: w, bias: linearBias,
                    scales: scales, biases: quantBiases,
                    groupSize: groupSize, bits: bits)
            } else {
                CommonWeightLoader.applyLinearWeights(to: owner, prefix: stPrefix, from: weights)
            }
        }

        // `modPath` here is the module-tree path used by `Module.update
        // (modules:)`. DiT declares its sub-modules with explicit `@ModuleInfo
        // (key: "time_embed")` etc. — those keys are what the children
        // dictionary uses, so the paths are snake_case at every level whose
        // owner module declared an explicit key. Sub-fields without an
        // explicit key fall back to the Swift property name.

        try handleProjection(
            stPrefix: "\(prefix).time_embed.time_mlp.0",
            modPath: "time_embed.linear1",
            owner: dit.timeEmbed.linear1)
        try handleProjection(
            stPrefix: "\(prefix).time_embed.time_mlp.2",
            modPath: "time_embed.linear2",
            owner: dit.timeEmbed.linear2)

        try handleProjection(
            stPrefix: "\(prefix).input_embed.proj",
            modPath: "input_embed.proj",
            owner: dit.inputEmbed.proj)

        // Conv position embedding stays Conv1d (no quantisation, kernel=31).
        CommonWeightLoader.applyConv1dWeights(
            to: dit.inputEmbed.convPosEmbed.conv1,
            prefix: "\(prefix).input_embed.conv_pos_embed.conv1.0",
            from: weights, transpose: false)
        CommonWeightLoader.applyConv1dWeights(
            to: dit.inputEmbed.convPosEmbed.conv2,
            prefix: "\(prefix).input_embed.conv_pos_embed.conv2.0",
            from: weights, transpose: false)

        for (i, block) in dit.transformerBlocks.enumerated() {
            let blockPrefix = "\(prefix).transformer_blocks.\(i)"
            let modBase = "transformer_blocks.\(i)"

            try handleProjection(
                stPrefix: "\(blockPrefix).attn_norm.linear",
                modPath: "\(modBase).attn_norm.linear",
                owner: block.attnNorm.linear)

            try handleProjection(
                stPrefix: "\(blockPrefix).attn.to_q",
                modPath: "\(modBase).attn.to_q",
                owner: block.attn.toQ)
            try handleProjection(
                stPrefix: "\(blockPrefix).attn.to_k",
                modPath: "\(modBase).attn.to_k",
                owner: block.attn.toK)
            try handleProjection(
                stPrefix: "\(blockPrefix).attn.to_v",
                modPath: "\(modBase).attn.to_v",
                owner: block.attn.toV)
            try handleProjection(
                stPrefix: "\(blockPrefix).attn.to_out.0",
                modPath: "\(modBase).attn.to_out",
                owner: block.attn.toOut)

            // Feedforward (GELU MLP). Python keys: ff.ff.0.0 / ff.ff.2.
            try handleProjection(
                stPrefix: "\(blockPrefix).ff.ff.0.0",
                modPath: "\(modBase).ff.linear1",
                owner: block.ff.linear1)
            try handleProjection(
                stPrefix: "\(blockPrefix).ff.ff.2",
                modPath: "\(modBase).ff.linear2",
                owner: block.ff.linear2)
        }

        // Final adaptive norm projection.
        try handleProjection(
            stPrefix: "\(prefix).norm_out.linear",
            modPath: "norm_out.linear",
            owner: dit.normOut.linear)

        if !qReplacements.isEmpty {
            let nested = NestedDictionary<String, Module>.unflattened(qReplacements)
            dit.update(modules: nested)
        }

        // Output projection: model dim → 80 mel dims. Never quantised
        // (out_features=80 isn't divisible by group_size=64).
        CommonWeightLoader.applyLinearWeights(
            to: dit.projOut, prefix: "\(prefix).proj_out", from: weights)
    }

    // MARK: - HiFi-GAN

    public static func loadHiFiGAN(_ hifigan: HiFiGANGenerator, from url: URL) throws {
        let weights = try CommonWeightLoader.loadSafetensors(url: url)

        CommonWeightLoader.applyConv1dWeights(
            to: hifigan.convPre.conv, prefix: "conv_pre", from: weights, transpose: false)

        for (i, up) in hifigan.ups.enumerated() {
            CommonWeightLoader.applyConv1dWeights(
                to: up.conv.conv, prefix: "ups.\(i)", from: weights, transpose: false)
        }

        for (i, down) in hifigan.sourceDowns.enumerated() {
            if let downSample = down as? CausalConv1dDownSample {
                CommonWeightLoader.applyConv1dWeights(
                    to: downSample.conv, prefix: "source_downs.\(i)", from: weights, transpose: false)
            } else if let causalConv = down as? CausalDilatedConv1d {
                CommonWeightLoader.applyConv1dWeights(
                    to: causalConv.conv, prefix: "source_downs.\(i)", from: weights, transpose: false)
            }
        }

        var flatIdx = 0
        for stage in hifigan.resblocks {
            for resblock in stage {
                loadResBlock(resblock, prefix: "resblocks.\(flatIdx)", from: weights)
                flatIdx += 1
            }
        }

        for (i, resblock) in hifigan.sourceResblocks.enumerated() {
            loadResBlock(resblock, prefix: "source_resblocks.\(i)", from: weights)
        }

        CommonWeightLoader.applyConv1dWeights(
            to: hifigan.convPost.conv, prefix: "conv_post", from: weights, transpose: false)

        CommonWeightLoader.applyLinearWeights(
            to: hifigan.source.linearMerge, prefix: "m_source.l_linear", from: weights)

        for (i, conv) in hifigan.f0Predictor.condnet.enumerated() {
            CommonWeightLoader.applyConv1dWeights(
                to: conv.conv, prefix: "f0_predictor.condnet.\(i * 2)", from: weights, transpose: false)
        }
        CommonWeightLoader.applyLinearWeights(
            to: hifigan.f0Predictor.classifier, prefix: "f0_predictor.classifier", from: weights)
    }

    static func loadResBlock(_ resblock: ResBlock, prefix: String, from weights: [String: MLXArray]) {
        for (j, conv) in resblock.convs1.enumerated() {
            CommonWeightLoader.applyConv1dWeights(
                to: conv.conv, prefix: "\(prefix).convs1.\(j)", from: weights, transpose: false)
        }
        for (j, conv) in resblock.convs2.enumerated() {
            CommonWeightLoader.applyConv1dWeights(
                to: conv.conv, prefix: "\(prefix).convs2.\(j)", from: weights, transpose: false)
        }

        for (j, act) in resblock.activations1.enumerated() {
            loadSnakeActivation(act, prefix: "\(prefix).activations1.\(j)", from: weights)
        }
        for (j, act) in resblock.activations2.enumerated() {
            loadSnakeActivation(act, prefix: "\(prefix).activations2.\(j)", from: weights)
        }
    }

    static func loadSnakeActivation(
        _ act: SnakeActivation, prefix: String, from weights: [String: MLXArray]
    ) {
        if let alpha = weights["\(prefix).alpha"] {
            act.update(parameters: ModuleParameters(values: ["alpha": .value(alpha)]))
        }
    }

    // MARK: - Speech Tokenizer (zero-shot voice cloning)

    public static func loadSpeechTokenizer(
        _ tokenizer: SpeechTokenizerModel, from url: URL
    ) throws {
        let weights = try CommonWeightLoader.loadSafetensors(url: url)

        CommonWeightLoader.applyConv1dWeights(
            to: tokenizer.encoder.conv1, prefix: "encoder.conv1", from: weights, transpose: false)
        CommonWeightLoader.applyConv1dWeights(
            to: tokenizer.encoder.conv2, prefix: "encoder.conv2", from: weights, transpose: false)

        for (i, block) in tokenizer.encoder.blocks.enumerated() {
            let p = "encoder.blocks.\(i)"

            CommonWeightLoader.applyLinearWeights(
                to: block.attn.query, prefix: "\(p).attn.query", from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: block.attn.key, prefix: "\(p).attn.key", from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: block.attn.value, prefix: "\(p).attn.value", from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: block.attn.out, prefix: "\(p).attn.out", from: weights)

            CommonWeightLoader.applyConv1dWeights(
                to: block.attn.fsmnBlock, prefix: "\(p).attn.fsmn_block",
                from: weights, transpose: false)

            CommonWeightLoader.applyLayerNormWeights(
                to: block.attnLN, prefix: "\(p).attn_ln", from: weights)
            CommonWeightLoader.applyLayerNormWeights(
                to: block.mlpLN, prefix: "\(p).mlp_ln", from: weights)

            CommonWeightLoader.applyLinearWeights(
                to: block.mlpFc1, prefix: "\(p).mlp.0", from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: block.mlpFc2, prefix: "\(p).mlp.2", from: weights)
        }

        CommonWeightLoader.applyLinearWeights(
            to: tokenizer.quantizer.codebook.projectDown,
            prefix: "quantizer._codebook.project_down", from: weights)
    }

    // MARK: - Quantization inference helpers

    /// Validate MLX's packed 8-bit QuantizedLinear layout. bf16 bundles do not
    /// enter this path because they omit `.scales` entirely.
    static func validate8BitQuantizationLayout(
        weight: MLXArray,
        scales: MLXArray,
        bits: Int,
        groupSize: Int
    ) throws {
        guard bits == 8 else {
            throw CosyVoiceTTSError.modelLoadFailed(
                "CosyVoice packed quantized weights must be 8-bit; 16-bit/bf16 bundles must not include .scales.")
        }
        guard weight.ndim == 2, scales.ndim == 2 else {
            throw CosyVoiceTTSError.modelLoadFailed(
                "CosyVoice quantized weights must be rank-2 MLX QuantizedLinear tensors.")
        }
        let packedCols = weight.dim(1)
        let numGroups = scales.dim(1)
        guard numGroups > 0, packedCols > 0 else {
            throw CosyVoiceTTSError.modelLoadFailed(
                "CosyVoice quantized weights have an invalid packed shape.")
        }
        let elementsPerWord = 32 / bits
        guard groupSize % elementsPerWord == 0 else {
            throw CosyVoiceTTSError.modelLoadFailed(
                "CosyVoice quantized group size \(groupSize) is not compatible with 8-bit packing.")
        }
        let expectedPackedColsPerGroup = groupSize / elementsPerWord
        guard packedCols == numGroups * expectedPackedColsPerGroup else {
            throw CosyVoiceTTSError.modelLoadFailed(
                "CosyVoice quantized weights are not 8-bit packed; re-export this bundle as bf16 or 8-bit.")
        }
    }
}
