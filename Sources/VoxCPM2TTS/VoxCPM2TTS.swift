import AudioCommon
import Foundation
import Hub
import MLX
import MLXCommon
import MLXFast
import MLXNN
import Tokenizers

public final class ScalarQuantizationLayer: Module {
    @ModuleInfo(key: "in_proj") public var in_proj: Linear
    @ModuleInfo(key: "out_proj") public var out_proj: Linear
    public let scale: Int

    public init(inDim: Int, outDim: Int, latentDim: Int = 64, scale: Int = 9) {
        self.scale = scale
        self._in_proj = ModuleInfo(wrappedValue: zeroLinear(inDim, latentDim), key: "in_proj")
        self._out_proj = ModuleInfo(wrappedValue: zeroLinear(latentDim, outDim), key: "out_proj")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = in_proj(x)
        let quantized = round(tanh(h) * MLXArray(Float(scale))) / MLXArray(Float(scale))
        return out_proj(quantized)
    }
}

public final class VoxCPM2TTSModel: Module {
    public let args: ModelArgs
    public let outputSampleRate: Int
    private static let officialTokenizerModelId = "openbmb/VoxCPM2"
    private static let tokenizerOverlayClassName = "LlamaTokenizer"
    private static let officialTokenizerFiles = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenization_voxcpm2.py",
        "special_tokens_map.json",
        "generation_config.json"
    ]

    @ModuleInfo public var base_lm: MiniCPMModel
    @ModuleInfo public var residual_lm: MiniCPMModel
    @ModuleInfo public var feat_encoder: VoxCPMLocEnc
    @ModuleInfo public var feat_decoder: UnifiedCFM
    @ModuleInfo public var fsq_layer: ScalarQuantizationLayer
    @ModuleInfo public var enc_to_lm_proj: Linear
    @ModuleInfo public var lm_to_dit_proj: Linear
    @ModuleInfo public var res_to_dit_proj: Linear
    @ModuleInfo public var fusion_concat_proj: Linear
    @ModuleInfo public var stop_proj: Linear
    @ModuleInfo public var stop_head: Linear
    @ModuleInfo public var audio_vae: AudioVAE

    private var tokenizer: Tokenizer?
    private var voxCPM2TokenizerSplitMap: [Int: [Int]] = [:]
    private var _isLoaded: Bool = true
    private static var debugVerboseEnabled: Bool {
        ProcessInfo.processInfo.environment["VOXCPM2_DEBUG_VERBOSE"] == "1"
    }

    public init(args: ModelArgs) {
        self.args = args
        self.outputSampleRate = args.audioVAEConfig.outSampleRate
        let debugInitEnabled = ProcessInfo.processInfo.environment["VOXCPM2_DEBUG_INIT"] == "1"
        @inline(__always)
        func logInit(_ message: String) {
            guard debugInitEnabled else { return }
            let line = "  VoxCPM2 init: \(message)\n"
            if let data = line.data(using: .utf8) {
                FileHandle.standardOutput.write(data)
            }
        }

        let lmConfig = args.lmConfig
        logInit("base_lm")
        self._base_lm = ModuleInfo(wrappedValue: MiniCPMModel(lmConfig))

        var residualConfig = lmConfig
        residualConfig.numHiddenLayers = args.residualLMNumLayers
        residualConfig.vocabSize = 0
        residualConfig.noRope = args.residualLMNoRope
        logInit("residual_lm")
        self._residual_lm = ModuleInfo(wrappedValue: MiniCPMModel(residualConfig))

        var encoderConfig = lmConfig
        encoderConfig.hiddenSize = args.encoderConfig.hiddenDim
        encoderConfig.intermediateSize = args.encoderConfig.ffnDim
        encoderConfig.numAttentionHeads = args.encoderConfig.numHeads
        encoderConfig.numHiddenLayers = args.encoderConfig.numLayers
        encoderConfig.kvChannels = args.encoderConfig.kvChannels
        encoderConfig.vocabSize = 0
        logInit("feat_encoder")
        self._feat_encoder = ModuleInfo(wrappedValue: VoxCPMLocEnc(config: encoderConfig, inputDim: args.featDim))

        var ditConfig = lmConfig
        ditConfig.hiddenSize = args.ditConfig.hiddenDim
        ditConfig.intermediateSize = args.ditConfig.ffnDim
        ditConfig.numAttentionHeads = args.ditConfig.numHeads
        ditConfig.numHiddenLayers = args.ditConfig.numLayers
        ditConfig.kvChannels = args.ditConfig.kvChannels
        ditConfig.vocabSize = 0
        logInit("feat_decoder")
        let estimator = VoxCPMLocDiTV2(config: ditConfig, inChannels: args.featDim)
        self._feat_decoder = ModuleInfo(wrappedValue: UnifiedCFM(
            inChannels: args.featDim,
            cfmParams: args.ditConfig.cfmConfig,
            estimator: estimator,
            meanMode: args.ditConfig.ditMeanMode
        ))

        logInit("fsq_layer")
        self._fsq_layer = ModuleInfo(wrappedValue: ScalarQuantizationLayer(
            inDim: args.lmConfig.hiddenSize,
            outDim: args.lmConfig.hiddenSize,
            latentDim: args.scalarQuantizationLatentDim,
            scale: args.scalarQuantizationScale
        ))

        logInit("projection heads")
        self._enc_to_lm_proj = ModuleInfo(wrappedValue: zeroLinear(args.encoderConfig.hiddenDim, args.lmConfig.hiddenSize))
        self._lm_to_dit_proj = ModuleInfo(wrappedValue: zeroLinear(args.lmConfig.hiddenSize, args.ditConfig.hiddenDim))
        self._res_to_dit_proj = ModuleInfo(wrappedValue: zeroLinear(args.lmConfig.hiddenSize, args.ditConfig.hiddenDim))
        self._fusion_concat_proj = ModuleInfo(wrappedValue: zeroLinear(args.lmConfig.hiddenSize * 2, args.lmConfig.hiddenSize))
        self._stop_proj = ModuleInfo(wrappedValue: zeroLinear(args.lmConfig.hiddenSize, args.lmConfig.hiddenSize))
        self._stop_head = ModuleInfo(wrappedValue: zeroLinear(args.lmConfig.hiddenSize, 2, bias: false))
        logInit("audio_vae")
        self._audio_vae = ModuleInfo(wrappedValue: AudioVAE(args.audioVAEConfig))

        super.init()
    }

    private static var shouldPromoteRuntimeParametersToFloat32: Bool {
        GPU.deviceInfo().architecture.lowercased().contains("apple")
    }

    private func promoteRuntimeParametersToFloat32() {
        _ = apply(filter: { module, _, _ in
            !(module is Quantized)
        }) { array in
            array.asType(.float32)
        }
    }

    // MARK: - Loading

    public static func fromPretrained(
        modelId: String = "aufklarer/VoxCPM2-MLX-bf16",
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> VoxCPM2TTSModel {
        let modelCacheDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        let tokenizerCacheDir = try HuggingFaceDownloader.getCacheDirectory(
            for: Self.officialTokenizerModelId,
            cacheDirName: "qwen3-speech-voxcpm2-tokenizer"
        )
        let tokenizerNeedsRefresh = try tokenizerSnapshotNeedsRefresh(in: tokenizerCacheDir)
        if !HuggingFaceDownloader.weightsExist(in: modelCacheDir)
            || !FileManager.default.fileExists(atPath: modelCacheDir.appendingPathComponent("config.json").path)
        {
            let refreshMessage = "Downloading \(modelId)..."
            progressHandler?(0.0, refreshMessage)
            try await HuggingFaceDownloader.downloadWeights(
                modelId: modelId,
                to: modelCacheDir,
                additionalFiles: [
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "tokenization_voxcpm2.py",
                    "special_tokens_map.json",
                    "generation_config.json"
                ],
                offlineMode: offlineMode
                ) { fraction in
                progressHandler?(fraction * 0.8, "Downloading model...")
            }
        }

        if tokenizerNeedsRefresh {
            progressHandler?(0.0, "Refreshing VoxCPM2 tokenizer snapshot...")
            do {
                try await HuggingFaceDownloader.downloadFiles(
                    modelId: Self.officialTokenizerModelId,
                    to: tokenizerCacheDir,
                    files: Self.officialTokenizerFiles,
                    offlineMode: offlineMode
                ) { fraction in
                    progressHandler?(fraction * 0.8, "Refreshing tokenizer snapshot...")
                }
            } catch {
                // A refresh failure is only survivable when a usable snapshot
                // is already on disk. Without one, swallowing here just defers
                // the failure to loadTokenizer, which throws a confusing
                // tokenizer error instead of the actionable DownloadError.
                let fm = FileManager.default
                let hasUsableSnapshot = fm.fileExists(
                    atPath: tokenizerCacheDir.appendingPathComponent("tokenizer.json").path)
                    && fm.fileExists(
                        atPath: tokenizerCacheDir.appendingPathComponent("tokenizer_config.json").path)
                guard hasUsableSnapshot else { throw error }
                let warning = "VoxCPM2 tokenizer refresh failed, continuing with cached snapshot: \(error)"
                logToStderr("[VoxCPM2] \(warning)")
            }
        }

        try Self.applyTokenizerConfigOverlay(in: tokenizerCacheDir)

        progressHandler?(0.82, "Loading config...")
        let args = try ModelArgs.load(from: modelCacheDir)
        let model = VoxCPM2TTSModel(args: args)

        progressHandler?(0.88, "Loading tokenizer...")
        let tokenizer = try await Self.loadTokenizer(from: tokenizerCacheDir)
        model.setTokenizer(tokenizer)

        progressHandler?(0.92, "Loading weights...")
        try model.loadWithDiagnostics("loadWeights(from:)") {
            try model.loadWeights(from: modelCacheDir)
        }
        if ProcessInfo.processInfo.environment["VOXCPM2_DEBUG_WLOAD"] == "1" {
            let q = model.base_lm.layers[0].selfAttn.qProj.weight
            let mAbs = q.abs().mean().item(Float.self)
            logToStderr("[VoxCPM2 WLOAD pre-promote] base_lm L0 q_proj mean_abs=\(mAbs) dtype=\(q.dtype)")
        }
        // Upstream promotes low-precision VoxCPM2 runtime dtypes to float32 on
        // Apple Silicon because bf16/fp16 can glitch the diffusion loop.
        if Self.shouldPromoteRuntimeParametersToFloat32 {
            model.promoteRuntimeParametersToFloat32()
        } else {
            model.audio_vae.castParametersToFloat32()
        }
        if ProcessInfo.processInfo.environment["VOXCPM2_DEBUG_WLOAD"] == "1" {
            let q = model.base_lm.layers[0].selfAttn.qProj.weight
            let mAbs = q.abs().mean().item(Float.self)
            logToStderr("[VoxCPM2 WLOAD post-promote] base_lm L0 q_proj mean_abs=\(mAbs) dtype=\(q.dtype)")
        }

        progressHandler?(0.98, "Evaluating parameters...")
        // NOTE: MLX can trip over some parameter views during eager evaluation
        // even when the model loads cleanly. We defer evaluation until first use.

        progressHandler?(1.0, "Ready")
        return model
    }

    private func swapQuantizedLinears(from weights: [String: MLXArray]) throws {
        let quantizedPaths = Set(weights.keys.flatMap { key -> [String] in
            guard key.hasSuffix(".scales") else { return [] }
            let base = String(key.dropLast(".scales".count))
            return Self.quantizedPathVariants(for: base)
        })
        guard !quantizedPaths.isEmpty else { return }

        let bits: Int
        if let cfg = args.quantization, cfg.isQuantized {
            bits = cfg.bits
        } else {
            bits = inferQuantizationBits(from: weights) ?? 4
        }
        let groupSize = args.quantization?.groupSize ?? 64
        guard bits == 4 || bits == 8 else {
            throw NSError(domain: "VoxCPM2TTSModel", code: 6, userInfo: [
                NSLocalizedDescriptionKey: "Unsupported quantization bits: \(bits) (expected 4 or 8)"
            ])
        }

        // Walk every leaf module; convert each Linear whose path appears in the
        // safetensors as a quantized weight. `quantize(model:filter:)` from MLXNN
        // does the Linear → QuantizedLinear swap in place via update(modules:).
        // The placeholder quantized parameters are overwritten by the subsequent
        // update(parameters:) calls in loadWeights().
        quantize(
            model: self,
            groupSize: groupSize,
            bits: bits,
            filter: { path, _ in quantizedPaths.contains(path) }
        )
    }

    private static func quantizedPathVariants(for path: String) -> [String] {
        let components = path.split(separator: ".").map(String.init)
        guard let last = components.last else { return [path] }

        var variants = [path]
        let camelLast = camelCaseKey(last)
        if camelLast != last {
            let camelPath = (components.dropLast() + [camelLast]).joined(separator: ".")
            variants.append(camelPath)
        }
        return variants
    }

    private static func camelCaseKey(_ key: String) -> String {
        let parts = key.split(separator: "_").map(String.init)
        guard parts.count > 1 else { return key }

        let head = parts[0]
        let tail = parts.dropFirst().map { component in
            guard let first = component.first else { return component }
            return first.uppercased() + component.dropFirst()
        }
        return ([head] + tail).joined()
    }

    private func inferQuantizationBits(from weights: [String: MLXArray]) -> Int? {
        for key in weights.keys where key.hasSuffix(".scales") {
            let prefix = String(key.dropLast(".scales".count))
            guard let weight = weights["\(prefix).weight"],
                  let scales = weights["\(prefix).scales"],
                  weight.ndim == 2, scales.ndim == 2 else { continue }
            let packedCols = weight.dim(1)
            let numGroups = scales.dim(1)
            guard numGroups > 0 else { continue }
            let ratio = packedCols / numGroups
            if ratio == 8 { return 4 }
            if ratio == 16 { return 8 }
        }
        return nil
    }

    private static func loadTokenizer(from directory: URL) async throws -> Tokenizer {
        do {
            let hubApi = HubApi()
            let tokenizerConfigURL = directory.appendingPathComponent("tokenizer_config.json")
            let tokenizerDataURL = directory.appendingPathComponent("tokenizer.json")
            guard FileManager.default.fileExists(atPath: tokenizerConfigURL.path),
                  FileManager.default.fileExists(atPath: tokenizerDataURL.path)
            else {
                return try await AutoTokenizer.from(modelFolder: directory, strict: false)
            }

            var tokenizerConfig = try hubApi.configuration(fileURL: tokenizerConfigURL)
            let tokenizerData = try hubApi.configuration(fileURL: tokenizerDataURL)

            if shouldUseVoxCPM2TokenizerCompatibilityPath(tokenizerConfig) {
                tokenizerConfig = compatibleTokenizerConfig(for: tokenizerConfig)
            }

            return try AutoTokenizer.from(
                tokenizerConfig: tokenizerConfig,
                tokenizerData: tokenizerData,
                strict: false
            )
        } catch {
            if ProcessInfo.processInfo.environment["VOXCPM2_DEBUG_VERBOSE"] == "1" {
                logToStderr("[VoxCPM2] tokenizer compatibility load failed, falling back to default loader: \(error)")
            }
            return try await AutoTokenizer.from(modelFolder: directory, strict: false)
        }
    }

    /// Diagnostics must NEVER go to stdout: consumers like the Speech Studio
    /// sidecar use stdout as an NDJSON protocol channel, and a stray line
    /// there fails the in-flight request and desyncs every one after it.
    private static func logToStderr(_ message: String) {
        if let data = (message + "\n").data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
    }

    static func compatibleTokenizerConfig(for tokenizerConfig: Config) -> Config {
        guard shouldUseVoxCPM2TokenizerCompatibilityPath(tokenizerConfig) else {
            return tokenizerConfig
        }

        var dictionary = tokenizerConfig.dictionary(or: [:])
        dictionary[Config.Key("tokenizer_class")] = "LlamaTokenizer"
        return Config(dictionary)
    }

    static func shouldUseVoxCPM2TokenizerCompatibilityPath(_ tokenizerConfig: Config) -> Bool {
        let dictionary = tokenizerConfig.dictionary(or: [:])

        if dictionary[Config.Key("tokenizer_class")]?.string() == "VoxCPM2Tokenizer" {
            return true
        }

        guard let autoMap = dictionary[Config.Key("auto_map")]?.dictionary(),
              let tokenizerEntry = autoMap[Config.Key("AutoTokenizer")]?.array(),
              let first = tokenizerEntry.first?.string()
        else {
            return false
        }

        return first.contains("tokenization_voxcpm2.VoxCPM2Tokenizer")
    }

    private func loadWeights(from directory: URL) throws {
        let allWeights = try CommonWeightLoader.loadAllSafetensors(from: directory)
        let sanitized = audio_vae.sanitize(allWeights)
        try swapQuantizedLinears(from: sanitized)
        try loadWeights(into: base_lm, prefix: "base_lm", from: sanitized)
        try loadWeights(into: residual_lm, prefix: "residual_lm", from: sanitized)
        if let specialToken = sanitized["feat_encoder.special_token"] {
            try loadWithDiagnostics("feat_encoder.special_token") {
                feat_encoder.loadSpecialToken(specialToken)
            }
        }
        try loadWithDiagnostics("feat_encoder.in_proj") {
            try loadLinearWeights(to: feat_encoder.inProj, prefix: "feat_encoder.in_proj", from: sanitized)
        }
        try loadWeights(into: feat_encoder.encoder, prefix: "feat_encoder.encoder", from: sanitized)
        try loadWithDiagnostics("feat_decoder") {
            try loadLinearWeights(
                to: feat_decoder.estimator.inProj,
                prefix: "feat_decoder.estimator.in_proj",
                from: sanitized
            )
            try loadLinearWeights(
                to: feat_decoder.estimator.condProj,
                prefix: "feat_decoder.estimator.cond_proj",
                from: sanitized
            )
            try loadLinearWeights(
                to: feat_decoder.estimator.outProj,
                prefix: "feat_decoder.estimator.out_proj",
                from: sanitized
            )
            try loadWeights(
                into: feat_decoder.estimator.timeMlp,
                prefix: "feat_decoder.estimator.time_mlp",
                from: sanitized
            )
            try loadWeights(
                into: feat_decoder.estimator.deltaTimeMlp,
                prefix: "feat_decoder.estimator.delta_time_mlp",
                from: sanitized
            )
            // Use the generic recursive loader (same path as base_lm). The
            // explicit loadMiniCPMModel below only loaded .weight per Linear
            // and silently skipped .scales / .biases — which left every
            // quantized Linear in the DiT decoder at random init in the
            // int8 / int4 bundles, producing near-no-op layers.
            try loadWeights(
                into: feat_decoder.estimator.decoder,
                prefix: "feat_decoder.estimator.decoder",
                from: sanitized
            )
        }
        try loadWithDiagnostics("fsq_layer") {
            try loadWeights(into: fsq_layer, prefix: "fsq_layer", from: sanitized)
        }
        try loadWithDiagnostics("enc_to_lm_proj") {
            try loadLinearWeights(to: enc_to_lm_proj, prefix: "enc_to_lm_proj", from: sanitized)
        }
        try loadWithDiagnostics("lm_to_dit_proj") {
            try loadLinearWeights(to: lm_to_dit_proj, prefix: "lm_to_dit_proj", from: sanitized)
        }
        try loadWithDiagnostics("res_to_dit_proj") {
            try loadLinearWeights(to: res_to_dit_proj, prefix: "res_to_dit_proj", from: sanitized)
        }
        try loadWithDiagnostics("fusion_concat_proj") {
            try loadLinearWeights(to: fusion_concat_proj, prefix: "fusion_concat_proj", from: sanitized)
        }
        try loadWithDiagnostics("stop_proj") {
            try loadLinearWeights(to: stop_proj, prefix: "stop_proj", from: sanitized)
        }
        try loadWithDiagnostics("stop_head") {
            try loadLinearWeights(to: stop_head, prefix: "stop_head", from: sanitized)
        }
        try loadWithDiagnostics("audio_vae.encoder") {
            try loadWeights(into: audio_vae.encoder, prefix: "audio_vae.encoder", from: sanitized)
        }
        try loadWithDiagnostics("audio_vae.encoder.conv_in") {
            if let weight = sanitized["audio_vae.encoder.conv_in.weight"] {
                try ensureShape(weight, matches: audio_vae.encoder.conv_in.weight.shape, label: "audio_vae.encoder.conv_in.weight")
                audio_vae.encoder.conv_in.weight = weight
            }
            if let bias = sanitized["audio_vae.encoder.conv_in.bias"] {
                if let currentBias = audio_vae.encoder.conv_in.bias {
                    try ensureShape(bias, matches: currentBias.shape, label: "audio_vae.encoder.conv_in.bias")
                }
                audio_vae.encoder.conv_in.bias = bias
            }
        }
        try loadWithDiagnostics("audio_vae.decoder.conv_in") {
            try loadDecoderConvStack(audio_vae.decoder.conv_in, prefix: "audio_vae.decoder.conv_in", from: sanitized)
        }
        for (index, block) in audio_vae.decoder.blocks.layers.enumerated() {
            try loadWithDiagnostics("audio_vae.decoder.blocks.layers.\(index)") {
                try loadDecoderBlock(block, prefix: "audio_vae.decoder.blocks.layers.\(index)", from: sanitized)
            }
        }
        try loadWithDiagnostics("audio_vae.decoder.sr_cond_layers") {
            try loadSampleRateConditionLayerStack(
                audio_vae.decoder.srCondLayers,
                prefix: "audio_vae.decoder.sr_cond_layers",
                from: sanitized
            )
        }
        if let snakeOut = sanitized["audio_vae.decoder.snake_out.alpha"] {
            try loadWithDiagnostics("audio_vae.decoder.snake_out.alpha") {
                try ensureShape(
                    snakeOut,
                    matches: audio_vae.decoder.snake_out.alpha.shape,
                    label: "audio_vae.decoder.snake_out.alpha"
                )
                audio_vae.decoder.snake_out.loadAlpha(snakeOut)
            }
        }
        if let convOutWeight = sanitized["audio_vae.decoder.conv_out.weight"] {
            try loadWithDiagnostics("audio_vae.decoder.conv_out.weight") {
                try ensureShape(
                    convOutWeight,
                    matches: audio_vae.decoder.conv_out.weight.shape,
                    label: "audio_vae.decoder.conv_out.weight"
                )
                audio_vae.decoder.conv_out.weight = convOutWeight
            }
        }
        if let convOutBias = sanitized["audio_vae.decoder.conv_out.bias"] {
            try loadWithDiagnostics("audio_vae.decoder.conv_out.bias") {
                if let currentBias = audio_vae.decoder.conv_out.bias {
                    try ensureShape(
                        convOutBias,
                        matches: currentBias.shape,
                        label: "audio_vae.decoder.conv_out.bias"
                    )
                }
                audio_vae.decoder.conv_out.bias = convOutBias
            }
        }
        if ProcessInfo.processInfo.environment["VOXCPM2_DEBUG_STATS"] == "1" {
            let baseQ = base_lm.layers[0].selfAttn.qProj.weight.asArray(Float.self)
            let baseGate = base_lm.layers[0].mlp.gateProj.weight.asArray(Float.self)
            let residualQ = residual_lm.layers[0].selfAttn.qProj.weight.asArray(Float.self)
            let encoderQ = feat_encoder.encoder.layers[0].selfAttn.qProj.weight.asArray(Float.self)
            let encoderGate = feat_encoder.encoder.layers[0].mlp.gateProj.weight.asArray(Float.self)
            let featIn = feat_encoder.inProj.weight.asArray(Float.self)
            let featDecoderIn = feat_decoder.estimator.inProj.weight.asArray(Float.self)
            let featDecoderCond = feat_decoder.estimator.condProj.weight.asArray(Float.self)
            let featDecoderOut = feat_decoder.estimator.outProj.weight.asArray(Float.self)
            if let data = "  VoxCPM2 loaded weights: base_q[min=\(baseQ.min() ?? 0), max=\(baseQ.max() ?? 0)], base_gate[min=\(baseGate.min() ?? 0), max=\(baseGate.max() ?? 0)], residual_q[min=\(residualQ.min() ?? 0), max=\(residualQ.max() ?? 0)], encoder_q[min=\(encoderQ.min() ?? 0), max=\(encoderQ.max() ?? 0)], encoder_gate[min=\(encoderGate.min() ?? 0), max=\(encoderGate.max() ?? 0)], feat_in[min=\(featIn.min() ?? 0), max=\(featIn.max() ?? 0)]\n".data(using: .utf8) {
                FileHandle.standardOutput.write(data)
            }
            if let data = "  VoxCPM2 loaded feat_decoder weights: in_proj[min=\(featDecoderIn.min() ?? 0), max=\(featDecoderIn.max() ?? 0)], cond_proj[min=\(featDecoderCond.min() ?? 0), max=\(featDecoderCond.max() ?? 0)], out_proj[min=\(featDecoderOut.min() ?? 0), max=\(featDecoderOut.max() ?? 0)]\n".data(using: .utf8) {
                FileHandle.standardOutput.write(data)
            }
        }
    }

    private func loadMiniCPMModel(
        _ model: MiniCPMModel,
        prefix: String,
        from weights: [String: MLXArray]
    ) throws {
        if let embedTokens = weights["\(prefix).embed_tokens.weight"] {
            try loadWithDiagnostics("\(prefix).embed_tokens") {
                model.update(
                    modules: ModuleChildren.unflattened([
                        ("embed_tokens", Embedding(weight: embedTokens))
                    ])
                )
            }
        }

        if let rope = model.rope {
            try loadMiniCPMRope(rope, prefix: "\(prefix).rope", from: weights)
        }

        if let normWeight = weights["\(prefix).norm.weight"] {
            try loadWithDiagnostics("\(prefix).norm") {
                try ensureShape(normWeight, matches: model.norm.weight.shape, label: "\(prefix).norm.weight")
                try model.norm.update(parameters: ModuleParameters.unflattened(["weight": normWeight]), verify: .shapeMismatch)
            }
        }

        for (index, layer) in model.layers.enumerated() {
            try loadMiniCPMDecoderLayer(layer, prefix: "\(prefix).layers.\(index)", from: weights)
        }
    }

    private func loadMiniCPMRope(
        _ rope: MiniCPMLongRoPE,
        prefix: String,
        from weights: [String: MLXArray]
    ) throws {
        if let invFreq = weights["\(prefix).inv_freq"] {
            try loadWithDiagnostics("\(prefix).inv_freq") {
                try ensureShape(invFreq, matches: rope.invFreq.shape, label: "\(prefix).inv_freq")
                try rope.update(parameters: ModuleParameters.unflattened(["inv_freq": invFreq]), verify: .shapeMismatch)
            }
        }
        if let shortFactor = weights["\(prefix).short_factor"] {
            try loadWithDiagnostics("\(prefix).short_factor") {
                try ensureShape(shortFactor, matches: rope.shortFactor.shape, label: "\(prefix).short_factor")
                try rope.update(parameters: ModuleParameters.unflattened(["short_factor": shortFactor]), verify: .shapeMismatch)
            }
        }
        if let longFactor = weights["\(prefix).long_factor"] {
            try loadWithDiagnostics("\(prefix).long_factor") {
                try ensureShape(longFactor, matches: rope.longFactor.shape, label: "\(prefix).long_factor")
                try rope.update(parameters: ModuleParameters.unflattened(["long_factor": longFactor]), verify: .shapeMismatch)
            }
        }
    }

    private func loadMiniCPMDecoderLayer(
        _ layer: MiniCPMDecoderLayer,
        prefix: String,
        from weights: [String: MLXArray]
    ) throws {
        if let inputLayerNorm = weights["\(prefix).input_layernorm.weight"] {
            try loadWithDiagnostics("\(prefix).input_layernorm") {
                try ensureShape(inputLayerNorm, matches: layer.inputLayerNorm.weight.shape, label: "\(prefix).input_layernorm.weight")
                try layer.inputLayerNorm.update(parameters: ModuleParameters.unflattened(["weight": inputLayerNorm]), verify: .shapeMismatch)
            }
        }

        if let qWeight = weights["\(prefix).self_attn.q_proj.weight"] {
            try loadWithDiagnostics("\(prefix).self_attn.q_proj") {
                try ensureShape(qWeight, matches: layer.selfAttn.qProj.weight.shape, label: "\(prefix).self_attn.q_proj.weight")
                try layer.selfAttn.qProj.update(parameters: ModuleParameters.unflattened(["weight": qWeight]), verify: .shapeMismatch)
            }
        }
        if let kWeight = weights["\(prefix).self_attn.k_proj.weight"] {
            try loadWithDiagnostics("\(prefix).self_attn.k_proj") {
                try ensureShape(kWeight, matches: layer.selfAttn.kProj.weight.shape, label: "\(prefix).self_attn.k_proj.weight")
                try layer.selfAttn.kProj.update(parameters: ModuleParameters.unflattened(["weight": kWeight]), verify: .shapeMismatch)
            }
        }
        if let vWeight = weights["\(prefix).self_attn.v_proj.weight"] {
            try loadWithDiagnostics("\(prefix).self_attn.v_proj") {
                try ensureShape(vWeight, matches: layer.selfAttn.vProj.weight.shape, label: "\(prefix).self_attn.v_proj.weight")
                try layer.selfAttn.vProj.update(parameters: ModuleParameters.unflattened(["weight": vWeight]), verify: .shapeMismatch)
            }
        }
        if let oWeight = weights["\(prefix).self_attn.o_proj.weight"] {
            try loadWithDiagnostics("\(prefix).self_attn.o_proj") {
                try ensureShape(oWeight, matches: layer.selfAttn.oProj.weight.shape, label: "\(prefix).self_attn.o_proj.weight")
                try layer.selfAttn.oProj.update(parameters: ModuleParameters.unflattened(["weight": oWeight]), verify: .shapeMismatch)
            }
        }
        if let gateWeight = weights["\(prefix).mlp.gate_proj.weight"] {
            try loadWithDiagnostics("\(prefix).mlp.gate_proj") {
                try ensureShape(gateWeight, matches: layer.mlp.gateProj.weight.shape, label: "\(prefix).mlp.gate_proj.weight")
                try layer.mlp.gateProj.update(parameters: ModuleParameters.unflattened(["weight": gateWeight]), verify: .shapeMismatch)
            }
        }
        if let upWeight = weights["\(prefix).mlp.up_proj.weight"] {
            try loadWithDiagnostics("\(prefix).mlp.up_proj") {
                try ensureShape(upWeight, matches: layer.mlp.upProj.weight.shape, label: "\(prefix).mlp.up_proj.weight")
                try layer.mlp.upProj.update(parameters: ModuleParameters.unflattened(["weight": upWeight]), verify: .shapeMismatch)
            }
        }
        if let downWeight = weights["\(prefix).mlp.down_proj.weight"] {
            try loadWithDiagnostics("\(prefix).mlp.down_proj") {
                try ensureShape(downWeight, matches: layer.mlp.downProj.weight.shape, label: "\(prefix).mlp.down_proj.weight")
                try layer.mlp.downProj.update(parameters: ModuleParameters.unflattened(["weight": downWeight]), verify: .shapeMismatch)
            }
        }

        if let postAttentionLayerNorm = weights["\(prefix).post_attention_layernorm.weight"] {
            try loadWithDiagnostics("\(prefix).post_attention_layernorm") {
                try ensureShape(postAttentionLayerNorm, matches: layer.postAttentionLayerNorm.weight.shape, label: "\(prefix).post_attention_layernorm.weight")
                try layer.postAttentionLayerNorm.update(parameters: ModuleParameters.unflattened(["weight": postAttentionLayerNorm]), verify: .shapeMismatch)
            }
        }
    }

    @discardableResult
    private func loadWithDiagnostics<T>(_ label: String, _ body: () throws -> T) throws -> T {
        do {
            return try withErrorHandler({ message in
                if let data = "[VoxCPM2] MLX error in \(label): \(message)\n".data(using: .utf8) {
                    FileHandle.standardOutput.write(data)
                }
            }) {
                try withError { error in
                    let value = try body()
                    try error.check()
                    return value
                }
            }
        } catch {
            if let data = "[VoxCPM2] Swift error in \(label): \(error)\n".data(using: .utf8) {
                FileHandle.standardOutput.write(data)
            }
            throw error
        }
    }

    private func loadLinearWeights(
        to linear: Linear,
        prefix: String,
        from weights: [String: MLXArray]
    ) throws {
        var params: [String: NestedItem<String, MLXArray>] = [:]
        if let weight = weights["\(prefix).weight"] {
            params["weight"] = .value(weight)
        }
        if let bias = weights["\(prefix).bias"] {
            params["bias"] = .value(bias)
        }
        // QuantizedLinear also has `scales` and `biases` parameters. Without
        // loading these the dequantised matmul gets random values from
        // `quantize()`'s placeholder init and the layer becomes a near-no-op.
        if let scales = weights["\(prefix).scales"] {
            params["scales"] = .value(scales)
        }
        if let qBiases = weights["\(prefix).biases"] {
            params["biases"] = .value(qBiases)
        }
        guard !params.isEmpty else { return }
        try linear.update(parameters: ModuleParameters(values: params), verify: .shapeMismatch)
    }

    private func loadWeights<T: Module>(
        into module: T,
        prefix: String,
        from weights: [String: MLXArray],
        stripPrefix: String? = nil
    ) throws {
        let prefixWithDot = prefix + "."
        let stripPrefix = stripPrefix ?? prefixWithDot
        let filtered = weights.reduce(into: [String: MLXArray]()) { result, entry in
            if entry.key == prefix {
                result[""] = entry.value
            } else if entry.key.hasPrefix(prefixWithDot) {
                result[String(entry.key.dropFirst(stripPrefix.count))] = entry.value
            }
        }

        guard !filtered.isEmpty else {
            return
        }

        if let snake = module as? Snake1d {
            if let alpha = filtered["alpha"] ?? filtered[""] {
                try loadWithDiagnostics(prefix) {
                    if Self.debugVerboseEnabled {
                        if let data = "  snake current alpha shape: \(snake.alpha.shape)\n".data(using: .utf8) {
                            FileHandle.standardOutput.write(data)
                        }
                        if let data = "  snake loaded alpha shape: \(alpha.shape)\n".data(using: .utf8) {
                            FileHandle.standardOutput.write(data)
                        }
                    }
                    snake.loadAlpha(alpha)
                }
                return
            }
        }

        if Self.debugVerboseEnabled, let data = "Loading \(prefix)...\n".data(using: .utf8) {
            FileHandle.standardOutput.write(data)
        }
        let params = ModuleParameters.unflattened(filtered)
        _ = try loadWithDiagnostics(prefix) {
            try module.update(parameters: params, verify: .shapeMismatch)
        }
    }

    private func loadDecoderConvStack(
        _ stack: ConvStack1d,
        prefix: String,
        from weights: [String: MLXArray]
    ) throws {
        for (index, layer) in stack.layers.enumerated() {
            let layerPrefix = "\(prefix).layers.\(index)"
            if let weight = weights["\(layerPrefix).weight"] {
                try ensureShape(weight, matches: layer.weight.shape, label: "\(layerPrefix).weight")
                layer.weight = weight
            }
            if let bias = weights["\(layerPrefix).bias"] {
                if let currentBias = layer.bias {
                    try ensureShape(bias, matches: currentBias.shape, label: "\(layerPrefix).bias")
                }
                layer.bias = bias
            }
        }
    }

    private func loadResidualUnit(
        _ unit: CausalResidualUnit,
        prefix: String,
        from weights: [String: MLXArray]
    ) throws {
        if let alpha = weights["\(prefix).snake1.alpha"] {
            try ensureShape(alpha, matches: unit.snake1.alpha.shape, label: "\(prefix).snake1.alpha")
            unit.snake1.loadAlpha(alpha)
        }
        if let weight = weights["\(prefix).conv1.weight"] {
            try ensureShape(weight, matches: unit.conv1.weight.shape, label: "\(prefix).conv1.weight")
            unit.conv1.weight = weight
        }
        if let bias = weights["\(prefix).conv1.bias"] {
            if let currentBias = unit.conv1.bias {
                try ensureShape(bias, matches: currentBias.shape, label: "\(prefix).conv1.bias")
            }
            unit.conv1.bias = bias
        }
        if let alpha = weights["\(prefix).snake2.alpha"] {
            try ensureShape(alpha, matches: unit.snake2.alpha.shape, label: "\(prefix).snake2.alpha")
            unit.snake2.loadAlpha(alpha)
        }
        if let weight = weights["\(prefix).conv2.weight"] {
            try ensureShape(weight, matches: unit.conv2.weight.shape, label: "\(prefix).conv2.weight")
            unit.conv2.weight = weight
        }
        if let bias = weights["\(prefix).conv2.bias"] {
            if let currentBias = unit.conv2.bias {
                try ensureShape(bias, matches: currentBias.shape, label: "\(prefix).conv2.bias")
            }
            unit.conv2.bias = bias
        }
    }

    private func loadDecoderBlock(
        _ block: CausalDecoderBlock,
        prefix: String,
        from weights: [String: MLXArray]
    ) throws {
        if let alpha = weights["\(prefix).snake.alpha"] {
            try ensureShape(alpha, matches: block.snake.alpha.shape, label: "\(prefix).snake.alpha")
            block.snake.loadAlpha(alpha)
        }
        if let weight = weights["\(prefix).conv_t.weight"] {
            try ensureShape(weight, matches: block.conv_t.weight.shape, label: "\(prefix).conv_t.weight")
            block.conv_t.weight = weight
        }
        if let bias = weights["\(prefix).conv_t.bias"] {
            if let currentBias = block.conv_t.bias {
                try ensureShape(bias, matches: currentBias.shape, label: "\(prefix).conv_t.bias")
            }
            block.conv_t.bias = bias
        }
        try loadResidualUnit(block.res1, prefix: "\(prefix).res1", from: weights)
        try loadResidualUnit(block.res2, prefix: "\(prefix).res2", from: weights)
        try loadResidualUnit(block.res3, prefix: "\(prefix).res3", from: weights)
    }

    private func loadSampleRateConditionLayerStack(
        _ stack: SampleRateConditionLayerStack,
        prefix: String,
        from weights: [String: MLXArray]
    ) throws {
        for (index, layer) in stack.layers.enumerated() {
            let layerPrefix = "\(prefix).\(index)"
            if let scaleWeight = weights["\(layerPrefix).scale_embed.weight"] {
                try loadWithDiagnostics("\(layerPrefix).scale_embed") {
                    try ensureTableEmbeddingShape(
                        scaleWeight,
                        matches: layer.scale_embed.weight.shape,
                        label: "\(layerPrefix).scale_embed.weight"
                    )
                    layer.loadScaleWeight(scaleWeight)
                }
            }
            if let biasWeight = weights["\(layerPrefix).bias_embed.weight"] {
                try loadWithDiagnostics("\(layerPrefix).bias_embed") {
                    try ensureTableEmbeddingShape(
                        biasWeight,
                        matches: layer.bias_embed.weight.shape,
                        label: "\(layerPrefix).bias_embed.weight"
                    )
                    layer.loadBiasWeight(biasWeight)
                }
            }
        }
    }

    private func ensureTableEmbeddingShape(
        _ array: MLXArray,
        matches expectedShape: [Int],
        label: String
    ) throws {
        if array.shape == expectedShape {
            return
        }
        guard expectedShape.count == 3, array.shape.count == 2 else {
            throw NSError(
                domain: "VoxCPM2TTSModel",
                code: 6,
                userInfo: [
                    NSLocalizedDescriptionKey: "\(label) shape mismatch: expected \(expectedShape), got \(array.shape)"
                ]
            )
        }
        guard array.shape[0] == expectedShape[0],
              array.shape[1] == expectedShape[1] * expectedShape[2]
        else {
            throw NSError(
                domain: "VoxCPM2TTSModel",
                code: 6,
                userInfo: [
                    NSLocalizedDescriptionKey: "\(label) shape mismatch: expected \(expectedShape), got \(array.shape)"
                ]
            )
        }
    }

    private func ensureShape(
        _ array: MLXArray,
        matches expectedShape: [Int],
        label: String
    ) throws {
        guard array.shape == expectedShape else {
            throw NSError(
                domain: "VoxCPM2TTSModel",
                code: 5,
                userInfo: [
                    NSLocalizedDescriptionKey: "\(label) shape mismatch: expected \(expectedShape), got \(array.shape)"
                ]
            )
        }
    }

    public func setTokenizer(_ tokenizer: Tokenizer) {
        self.tokenizer = tokenizer
        self.voxCPM2TokenizerSplitMap = Self.buildVoxCPM2TokenizerSplitMap(
            tokenizer: tokenizer,
            vocabSize: args.lmConfig.vocabSize
        )
    }

    // MARK: - Shape Helpers

    func tokenize(_ text: String) throws -> [Int32] {
        guard let tokenizer else {
            throw NSError(domain: "VoxCPM2TTSModel", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Tokenizer not loaded"
            ])
        }
        let tokens = tokenizer.tokenize(text: text)
        var ids: [Int] = []
        ids.reserveCapacity(tokens.count)
        for token in tokens {
            if let expansion = Self.expandVoxCPM2TokenizerToken(token, tokenizer: tokenizer) {
                ids.append(contentsOf: expansion)
            } else if let id = tokenizer.convertTokenToId(token) {
                ids.append(id)
            }
        }
        return ids.map(Int32.init)
    }

    static func tokenizerSnapshotNeedsRefresh(in directory: URL) throws -> Bool {
        let fm = FileManager.default
        let tokenizerConfigURL = directory.appendingPathComponent("tokenizer_config.json")
        let tokenizerDataURL = directory.appendingPathComponent("tokenizer.json")

        guard fm.fileExists(atPath: tokenizerConfigURL.path) else { return true }
        guard fm.fileExists(atPath: tokenizerDataURL.path) else { return true }

        let data = try Data(contentsOf: tokenizerConfigURL)
        guard
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return true
        }

        guard let tokenizerClass = json["tokenizer_class"] as? String else {
            return true
        }

        let normalizedClass = tokenizerClass.replacingOccurrences(of: "Fast", with: "")
        if normalizedClass == Self.tokenizerOverlayClassName || normalizedClass == "VoxCPM2Tokenizer" {
            return false
        }

        return true
    }

    static func patchedTokenizerConfigOverlay(from data: Data) throws -> Data {
        guard var json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return data
        }

        let currentClass = json["tokenizer_class"] as? String
        let needsOverlay = currentClass != Self.tokenizerOverlayClassName || json["auto_map"] != nil
        guard needsOverlay else { return data }

        json["tokenizer_class"] = Self.tokenizerOverlayClassName
        json.removeValue(forKey: "auto_map")
        return try JSONSerialization.data(withJSONObject: json, options: [.sortedKeys])
    }

    static func applyTokenizerConfigOverlay(in directory: URL) throws {
        let tokenizerConfigURL = directory.appendingPathComponent("tokenizer_config.json")
        guard FileManager.default.fileExists(atPath: tokenizerConfigURL.path) else { return }

        let originalData = try Data(contentsOf: tokenizerConfigURL)
        let patchedData = try patchedTokenizerConfigOverlay(from: originalData)
        guard patchedData != originalData else { return }

        try patchedData.write(to: tokenizerConfigURL, options: .atomic)
    }

    static func buildVoxCPM2TokenizerSplitMap(
        tokenizer: Tokenizer,
        vocabSize: Int
    ) -> [Int: [Int]] {
        guard vocabSize > 0 else { return [:] }

        var splitMap: [Int: [Int]] = [:]
        splitMap.reserveCapacity(max(vocabSize / 256, 1))

        for id in 0..<vocabSize {
            guard let token = tokenizer.convertIdToToken(id) else { continue }
            guard let expansion = expandVoxCPM2TokenizerToken(token, tokenizer: tokenizer) else { continue }
            splitMap[id] = expansion
        }

        return splitMap
    }

    static func expandVoxCPM2TokenizerIds(
        _ ids: [Int],
        using splitMap: [Int: [Int]]
    ) -> [Int] {
        guard !splitMap.isEmpty else { return ids }

        var expanded: [Int] = []
        expanded.reserveCapacity(ids.count)
        for id in ids {
            if let expansion = splitMap[id] {
                expanded.append(contentsOf: expansion)
            } else {
                expanded.append(id)
            }
        }
        return expanded
    }

    static func expandVoxCPM2TokenizerToken(
        _ token: String,
        tokenizer: Tokenizer
    ) -> [Int]? {
        let clean = token.replacingOccurrences(of: "▁", with: "")
        guard clean.count >= 2 else { return nil }
        guard clean.unicodeScalars.allSatisfy({ Self.isVoxCPM2CJKScalar($0) }) else { return nil }

        let charIds = clean.compactMap { tokenizer.convertTokenToId(String($0)) }
        guard charIds.count == clean.count else { return nil }
        return charIds
    }

    static func isVoxCPM2CJKScalar(_ scalar: UnicodeScalar) -> Bool {
        switch scalar.value {
        case 0x4E00...0x9FFF,
             0x3400...0x4DBF,
             0xF900...0xFAFF,
             0x20000...0x2A6DF:
            return true
        default:
            return false
        }
    }

    private func encodeAudio(
        _ audio: [Float],
        sampleRate: Int,
        paddingMode: String = "right"
    ) throws -> MLXArray {
        var mono = audio
        if sampleRate != audio_vae.sampleRate {
            mono = AudioFileLoader.resample(audio, from: sampleRate, to: audio_vae.sampleRate)
        }
        guard !mono.isEmpty else {
            throw NSError(domain: "VoxCPM2TTSModel", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Reference/prompt audio is empty"
            ])
        }

        let patchLen = args.patchSize * audio_vae.chunkSize
        let remainder = mono.count % patchLen
        if remainder != 0 {
            let pad = patchLen - remainder
            if paddingMode == "left" {
                mono = [Float](repeating: 0, count: pad) + mono
            } else {
                mono += [Float](repeating: 0, count: pad)
            }
        }

        let input = MLXArray(mono).reshaped([1, mono.count, 1])
        let feat = audio_vae.encode(input, sampleRate: audio_vae.sampleRate).squeezed(axis: 0)
        let audioLength = feat.dim(0) / args.patchSize
        let trimmed = feat[
            0..<(audioLength * args.patchSize),
            0...
        ]
        return trimmed.reshaped([audioLength, args.patchSize, audio_vae.latentDim])
    }

    private func makeTimeSpan(_ timesteps: Int) -> [Float] {
        return makeUnifiedCFMTimeSpan(
            timesteps: timesteps,
            scheduler: args.ditConfig.cfmConfig.tScheduler,
            sigmaMin: args.ditConfig.cfmConfig.sigmaMin
        )
    }

    // MARK: - Generation

    public func generate(text: String, language: String? = nil) async throws -> [Float] {
        try await generateVoxCPM2(
            text: text,
            language: language,
            maxTokens: 2000,
            minTokens: 2,
            refText: nil,
            refAudio: nil,
            promptText: nil,
            promptAudio: nil,
            inferenceTimesteps: 10,
            cfgValue: 2.0,
            streamingPrefixLen: 4,
            warmupPatches: 0,
            instruct: nil
        )
    }

    public func generateVoxCPM2(
        text: String,
        language: String? = nil,
        maxTokens: Int = 2000,
        minTokens: Int = 2,
        refText: String? = nil,
        refAudio: [Float]? = nil,
        promptText: String? = nil,
        promptAudio: [Float]? = nil,
        inferenceTimesteps: Int = 10,
        cfgValue: Float = 2.0,
        streamingPrefixLen: Int = 4,
        warmupPatches: Int = 0,
        instruct: String? = nil
    ) async throws -> [Float] {
        guard tokenizer != nil else {
            throw NSError(domain: "VoxCPM2TTSModel", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "Tokenizer not loaded"
            ])
        }
        _ = language
        let debugVerboseEnabled = ProcessInfo.processInfo.environment["VOXCPM2_DEBUG_VERBOSE"] == "1"
        @inline(__always)
        func logGen(_ message: String) {
            guard debugVerboseEnabled else { return }
            if let data = "[VoxCPM2] \(message)\n".data(using: .utf8) {
                FileHandle.standardOutput.write(data)
            }
        }

        var workingText = text
        var effectiveWarmup = warmupPatches
        if let instruct, !instruct.isEmpty {
            workingText = "(\(instruct))\(workingText)"
            effectiveWarmup = min(effectiveWarmup, 1)
        }

        let scaleEmb = args.lmConfig.useMup ? Float(args.lmConfig.scaleEmb) : 1.0
        let latentDim = audio_vae.latentDim
        let hasRef = refAudio != nil
        let hasPrompt = promptAudio != nil && promptText != nil

        let textIds: [Int32]
        var textToken: MLXArray
        var audioFeat: MLXArray
        var textMask: MLXArray
        var audioMask: MLXArray

        if hasRef && hasPrompt {
            let combinedText = (promptText ?? "") + workingText
            textIds = try tokenize(combinedText)
            let textLength = textIds.count + 1
            textToken = MLXArray(textIds + [Int32(101)]).reshaped([1, textLength])

            let refFeat = try encodeAudio(
                refAudio ?? [],
                sampleRate: audio_vae.sampleRate,
                paddingMode: "right"
            )
            let promptFeat = try encodeAudio(
                promptAudio ?? [],
                sampleRate: audio_vae.sampleRate,
                paddingMode: "left"
            )
            let promptLen = promptFeat.dim(0)

            let refTokens = MLXArray(
                [Int32(103)]
                + Array(repeating: Int32(0), count: refFeat.dim(0))
                + [Int32(104)]
            )
            let refFeats = concatenated(
                [
                    MLXArray.zeros([1, args.patchSize, latentDim]),
                    refFeat,
                    MLXArray.zeros([1, args.patchSize, latentDim])
                ],
                axis: 0
            )
            let refTMask = MLXArray(
                [Float(1.0)]
                + Array(repeating: Float(0.0), count: refFeat.dim(0))
                + [Float(1.0)]
            )
            let refAMask = MLXArray(
                [Float(0.0)]
                + Array(repeating: Float(1.0), count: refFeat.dim(0))
                + [Float(0.0)]
            )

            let textPadFeat = MLXArray.zeros([textLength, args.patchSize, latentDim])
            let promptPadToken = MLXArray.zeros([promptLen], dtype: .int32)

            let fullText = concatenated([refTokens, textToken.squeezed(axis: 0), promptPadToken], axis: 0)
            let fullAudio = concatenated([refFeats, textPadFeat, promptFeat], axis: 0)
            textMask = concatenated(
                [
                    refTMask,
                    MLXArray.ones([textLength], dtype: .float32),
                    MLXArray.zeros([promptLen], dtype: .float32)
                ],
                axis: 0
            )
            audioMask = concatenated(
                [
                    refAMask,
                    MLXArray.zeros([textLength], dtype: .float32),
                    MLXArray.ones([promptLen], dtype: .float32)
                ],
                axis: 0
            )

            textToken = fullText.reshaped([1, fullText.dim(0)])
            audioFeat = fullAudio.reshaped([1, fullAudio.dim(0), args.patchSize, latentDim])
        } else if hasRef {
            textIds = try tokenize(workingText)
            let textLength = textIds.count + 1
            textToken = MLXArray(textIds + [Int32(101)]).reshaped([1, textLength])

            let refFeat = try encodeAudio(
                refAudio ?? [],
                sampleRate: audio_vae.sampleRate,
                paddingMode: "right"
            )
            let refTokens = MLXArray(
                [Int32(103)]
                + Array(repeating: Int32(0), count: refFeat.dim(0))
                + [Int32(104)]
            )
            let refFeats = concatenated(
                [
                    MLXArray.zeros([1, args.patchSize, latentDim]),
                    refFeat,
                    MLXArray.zeros([1, args.patchSize, latentDim])
                ],
                axis: 0
            )
            let refTMask = MLXArray(
                [Float(1.0)]
                + Array(repeating: Float(0.0), count: refFeat.dim(0))
                + [Float(1.0)]
            )
            let refAMask = MLXArray(
                [Float(0.0)]
                + Array(repeating: Float(1.0), count: refFeat.dim(0))
                + [Float(0.0)]
            )

            let textPadFeat = MLXArray.zeros([textLength, args.patchSize, latentDim])
            let fullText = concatenated([refTokens, textToken.squeezed(axis: 0)], axis: 0)
            let fullAudio = concatenated([refFeats, textPadFeat], axis: 0)
            textMask = concatenated([refTMask, MLXArray.ones([textLength], dtype: .float32)], axis: 0)
            audioMask = concatenated([refAMask, MLXArray.zeros([textLength], dtype: .float32)], axis: 0)

            textToken = fullText.reshaped([1, fullText.dim(0)])
            audioFeat = fullAudio.reshaped([1, fullAudio.dim(0), args.patchSize, latentDim])
        } else if hasPrompt {
            let combinedText = (promptText ?? "") + workingText
            textIds = try tokenize(combinedText)
            let textLength = textIds.count + 1
            textToken = MLXArray(textIds + [Int32(101)]).reshaped([1, textLength])

            let promptFeat = try encodeAudio(
                promptAudio ?? [],
                sampleRate: audio_vae.sampleRate,
                paddingMode: "left"
            )
            let promptLen = promptFeat.dim(0)

            let textPadFeat = MLXArray.zeros([textLength, args.patchSize, latentDim])
            let promptPadToken = MLXArray.zeros([promptLen], dtype: .int32)

            let fullText = concatenated([textToken.squeezed(axis: 0), promptPadToken], axis: 0)
            let fullAudio = concatenated([textPadFeat, promptFeat], axis: 0)
            textMask = concatenated(
                [
                    MLXArray.ones([textLength], dtype: .float32),
                    MLXArray.zeros([promptLen], dtype: .float32)
                ],
                axis: 0
            )
            audioMask = concatenated(
                [
                    MLXArray.zeros([textLength], dtype: .float32),
                    MLXArray.ones([promptLen], dtype: .float32)
                ],
                axis: 0
            )

            textToken = fullText.reshaped([1, fullText.dim(0)])
            audioFeat = fullAudio.reshaped([1, fullAudio.dim(0), args.patchSize, latentDim])
        } else {
            textIds = try tokenize(workingText)
            let textLength = textIds.count + 1
            textToken = MLXArray(textIds + [Int32(101)]).reshaped([1, textLength])
            audioFeat = MLXArray.zeros([1, textLength, args.patchSize, latentDim])
            textMask = MLXArray.ones([1, textLength], dtype: .float32)
            audioMask = MLXArray.zeros([1, textLength], dtype: .float32)
        }

        let textTokenB = textToken
        let audioFeatB = audioFeat
        let textMaskB = textMask.shape.count == 1 ? textMask.reshaped([1, textMask.dim(0)]) : textMask
        let audioMaskB = audioMask.shape.count == 1 ? audioMask.reshaped([1, audioMask.dim(0)]) : audioMask
        let textMask3 = textMaskB.expandedDimensions(axis: 2)
        let audioMask3 = audioMaskB.expandedDimensions(axis: 2)

        logGen("building embeddings")
        let featEmbed = enc_to_lm_proj(feat_encoder(audioFeatB))
        let textEmbed = base_lm.embedTokens!(textTokenB) * MLXArray(scaleEmb)
        let combinedEmbed = textMask3 * textEmbed + audioMask3 * featEmbed

        let lastFeatIndex = audioFeatB.dim(1) - 1
        var prefixFeatCond = audioFeatB[
            0...,
            lastFeatIndex...(lastFeatIndex),
            0...,
            0...
        ].squeezed(axis: 1)

        let (encOutputs, initialLmCache) = base_lm(inputsEmbeds: combinedEmbed)
        var lmCache = initialLmCache
        let encOutputsFSQ = fsq_layer(encOutputs)
        let maskedEnc = encOutputsFSQ * audioMask3 + encOutputs * textMask3
        var lmHidden = maskedEnc[
            0...,
            (maskedEnc.dim(1) - 1)...(maskedEnc.dim(1) - 1),
            0...
        ].squeezed(axis: 1)

        let residualInput = fusion_concat_proj(
            concatenated([maskedEnc, audioMask3 * featEmbed], axis: -1)
        )
        let (resOutputs, initialResCache) = residual_lm(inputsEmbeds: residualInput)
        var resCache = initialResCache
        var residualHidden = resOutputs[
            0...,
            (resOutputs.dim(1) - 1)...(resOutputs.dim(1) - 1),
            0...
        ].squeezed(axis: 1)

        let hasContinuation = hasPrompt
        var predFeatSeq: [MLXArray] = []
        if hasContinuation {
            let audioIndices = (0..<audioMaskB.dim(1)).filter { idx in
                audioMaskB[0, idx].item(Float.self) > 0.5
            }
            let contextLen = min(streamingPrefixLen - 1, audioIndices.count)
            for idx in audioIndices.suffix(contextLen) {
                let slice = audioFeatB[
                    0...,
                    idx..<(idx + 1),
                    0...,
                    0...
                ]
                predFeatSeq.append(slice)
            }
        }

        let warmupCount = hasContinuation ? 0 : effectiveWarmup
        for step in 0..<(maxTokens + warmupCount) {
            if step == 0 {
                logGen("sampling first audio patch")
            }
            let ditMu = concatenated([
                lm_to_dit_proj(lmHidden),
                res_to_dit_proj(residualHidden)
            ], axis: -1)
            let condIn = prefixFeatCond.transposed(0, 2, 1)

            var predFeat = feat_decoder.sample(
                mu: ditMu,
                nTimesteps: inferenceTimesteps,
                patchSize: args.patchSize,
                cond: condIn,
                cfgValue: cfgValue
            )

            predFeat = predFeat.transposed(0, 2, 1)

            if step >= warmupCount {
                predFeatSeq.append(predFeat.expandedDimensions(axis: 1))
            }

            let currEmbed = enc_to_lm_proj(
                feat_encoder(predFeat.expandedDimensions(axis: 1))
            )

            let stopLogits = stop_head(silu(stop_proj(lmHidden)))
            let stopFlag = argMax(stopLogits, axis: -1).squeezed().item(Int32.self)
            let realSteps = step - warmupCount
            if realSteps > minTokens && stopFlag == 1 {
                break
            }

            let (newLmOut, nextLmCache) = base_lm(
                inputsEmbeds: currEmbed,
                cache: lmCache
            )
            lmCache = nextLmCache
            lmHidden = fsq_layer(newLmOut[
                0...,
                (newLmOut.dim(1) - 1)...(newLmOut.dim(1) - 1),
                0...
            ].squeezed(axis: 1))

            let currResidualInput = fusion_concat_proj(
                concatenated([lmHidden.expandedDimensions(axis: 1), currEmbed], axis: -1)
            )
            let (newResOut, nextResCache) = residual_lm(
                inputsEmbeds: currResidualInput,
                cache: resCache
            )
            resCache = nextResCache
            residualHidden = newResOut[
                0...,
                (newResOut.dim(1) - 1)...(newResOut.dim(1) - 1),
                0...
            ].squeezed(axis: 1)

            prefixFeatCond = predFeat
        }

        guard !predFeatSeq.isEmpty else {
            throw NSError(domain: "VoxCPM2TTSModel", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "No audio patches were generated"
            ])
        }

        logGen("decoding audio")
        var allFeat = concatenated(predFeatSeq, axis: 1)
        allFeat = allFeat.reshaped([allFeat.dim(0), -1, args.featDim])

        var audio = audio_vae.decode(allFeat.asType(.float32))
        audio = audio.flattened()

        if hasContinuation {
            let decodePatchLen = args.patchSize * audio_vae.decodeChunkSize
            let trimAudioSamples = decodePatchLen * (streamingPrefixLen - 1)
            if trimAudioSamples < audio.count {
                audio = audio[trimAudioSamples...]
            }
        }

        eval(audio)
        return audio.asArray(Float.self)
    }
}

extension VoxCPM2TTSModel: SpeechGenerationModel {
    public var sampleRate: Int { outputSampleRate }
}

extension VoxCPM2TTSModel: ModelMemoryManageable {
    public var isLoaded: Bool { _isLoaded }

    public func unload() {
        guard _isLoaded else { return }
        base_lm.clearParameters()
        residual_lm.clearParameters()
        feat_encoder.clearParameters()
        feat_decoder.clearParameters()
        fsq_layer.clearParameters()
        enc_to_lm_proj.clearParameters()
        lm_to_dit_proj.clearParameters()
        res_to_dit_proj.clearParameters()
        fusion_concat_proj.clearParameters()
        stop_proj.clearParameters()
        stop_head.clearParameters()
        audio_vae.clearParameters()
        _isLoaded = false
    }

    public var memoryFootprint: Int {
        guard _isLoaded else { return 0 }
        return base_lm.parameterMemoryBytes()
            + residual_lm.parameterMemoryBytes()
            + feat_encoder.parameterMemoryBytes()
            + feat_decoder.parameterMemoryBytes()
            + fsq_layer.parameterMemoryBytes()
            + enc_to_lm_proj.parameterMemoryBytes()
            + lm_to_dit_proj.parameterMemoryBytes()
            + res_to_dit_proj.parameterMemoryBytes()
            + fusion_concat_proj.parameterMemoryBytes()
            + stop_proj.parameterMemoryBytes()
            + stop_head.parameterMemoryBytes()
            + audio_vae.parameterMemoryBytes()
    }
}
