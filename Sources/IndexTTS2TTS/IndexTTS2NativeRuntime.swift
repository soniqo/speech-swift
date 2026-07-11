import Foundation
import ChatterboxTTS
import MLX
import MLXCommon

public struct IndexTTS2RuntimeInventory: Equatable, Sendable {
    public let outputSampleRate: Int
    public let gptTensorCount: Int
    public let s2MelRuntimeTensorCount: Int
    public let s2MelIgnoredTensorCount: Int
    public let w2vBertTensorCount: Int
    public let semanticCodecTensorCount: Int
    public let campPlusTensorCount: Int
    public let bigVGANTensorCount: Int
    public let wavStatTensorCount: Int
    public let speakerMatrixShape: [Int]
    public let emotionMatrixShape: [Int]
    public let qwenEmotionTensorCount: Int

    public var totalRuntimeTensorCount: Int {
        gptTensorCount +
            s2MelRuntimeTensorCount +
            w2vBertTensorCount +
            semanticCodecTensorCount +
            campPlusTensorCount +
            bigVGANTensorCount +
            wavStatTensorCount +
            2 +
            qwenEmotionTensorCount
    }
}

public enum IndexTTS2RuntimeError: Error, LocalizedError, Equatable {
    case missingWeightFile(String)
    case missingTensor(component: String, key: String)
    case invalidTensorShape(component: String, key: String, expected: [Int], actual: [Int])
    case emptyRuntimeComponent(String)

    public var errorDescription: String? {
        switch self {
        case .missingWeightFile(let path):
            return "IndexTTS2 runtime weight file is missing: \(path)"
        case .missingTensor(let component, let key):
            return "IndexTTS2 \(component) weights are missing tensor: \(key)"
        case .invalidTensorShape(let component, let key, let expected, let actual):
            return "IndexTTS2 \(component) tensor \(key) has shape \(actual), expected \(expected)"
        case .emptyRuntimeComponent(let component):
            return "IndexTTS2 runtime component has no tensors after filtering: \(component)"
        }
    }
}

final class IndexTTS2NativeRuntime {
    let config: IndexTTS2RuntimeConfig
    let gpt: [String: MLXArray]
    let semanticGPT: IndexTTS2SemanticGPT
    let s2Mel: [String: MLXArray]
    let lengthRegulator: IndexTTS2S2MelLengthRegulator
    let s2MelFlow: IndexTTS2S2MelFlow
    let w2vBert: [String: MLXArray]
    let wav2Vec2BertModel: IndexTTS2Wav2Vec2Bert
    let semanticCodec: [String: MLXArray]
    let semanticEncoder: IndexTTS2SemanticEncoder
    let semanticQuantizer: IndexTTS2SemanticQuantizer
    let campPlus: [String: MLXArray]
    let campPlusEncoder: CAMPPlus
    let bigVGAN: [String: MLXArray]
    let vocoder: IndexTTS2BigVGAN
    let wavStats: [String: MLXArray]
    let speakerMatrix: MLXArray
    let emotionMatrix: MLXArray
    let qwenEmotion: [String: MLXArray]
    let ignoredS2MelTensorCount: Int

    var inventory: IndexTTS2RuntimeInventory {
        IndexTTS2RuntimeInventory(
            outputSampleRate: config.outputSampleRate,
            gptTensorCount: gpt.count,
            s2MelRuntimeTensorCount: s2Mel.count,
            s2MelIgnoredTensorCount: ignoredS2MelTensorCount,
            w2vBertTensorCount: w2vBert.count,
            semanticCodecTensorCount: semanticCodec.count,
            campPlusTensorCount: campPlus.count,
            bigVGANTensorCount: bigVGAN.count,
            wavStatTensorCount: wavStats.count,
            speakerMatrixShape: speakerMatrix.shape,
            emotionMatrixShape: emotionMatrix.shape,
            qwenEmotionTensorCount: qwenEmotion.count)
    }

    private init(
        config: IndexTTS2RuntimeConfig,
        gpt: [String: MLXArray],
        semanticGPT: IndexTTS2SemanticGPT,
        s2Mel: [String: MLXArray],
        lengthRegulator: IndexTTS2S2MelLengthRegulator,
        s2MelFlow: IndexTTS2S2MelFlow,
        ignoredS2MelTensorCount: Int,
        w2vBert: [String: MLXArray],
        wav2Vec2BertModel: IndexTTS2Wav2Vec2Bert,
        semanticCodec: [String: MLXArray],
        semanticEncoder: IndexTTS2SemanticEncoder,
        semanticQuantizer: IndexTTS2SemanticQuantizer,
        campPlus: [String: MLXArray],
        campPlusEncoder: CAMPPlus,
        bigVGAN: [String: MLXArray],
        vocoder: IndexTTS2BigVGAN,
        wavStats: [String: MLXArray],
        speakerMatrix: MLXArray,
        emotionMatrix: MLXArray,
        qwenEmotion: [String: MLXArray]
    ) {
        self.config = config
        self.gpt = gpt
        self.semanticGPT = semanticGPT
        self.s2Mel = s2Mel
        self.lengthRegulator = lengthRegulator
        self.s2MelFlow = s2MelFlow
        self.ignoredS2MelTensorCount = ignoredS2MelTensorCount
        self.w2vBert = w2vBert
        self.wav2Vec2BertModel = wav2Vec2BertModel
        self.semanticCodec = semanticCodec
        self.semanticEncoder = semanticEncoder
        self.semanticQuantizer = semanticQuantizer
        self.campPlus = campPlus
        self.campPlusEncoder = campPlusEncoder
        self.bigVGAN = bigVGAN
        self.vocoder = vocoder
        self.wavStats = wavStats
        self.speakerMatrix = speakerMatrix
        self.emotionMatrix = emotionMatrix
        self.qwenEmotion = qwenEmotion
    }

    static func load(
        from directory: URL,
        config: IndexTTS2RuntimeConfig,
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws -> IndexTTS2NativeRuntime {
        progressHandler?(0.02, "Loading IndexTTS2 GPT weights")
        let gpt = try loadSafetensors("gpt.safetensors", from: directory)
        try require(gpt, component: "GPT", key: "mel_embedding.weight", shape: [config.gpt.numberMelCodes, config.gpt.modelDim])
        try require(gpt, component: "GPT", key: "text_embedding.weight", shape: [config.gpt.numberTextTokens + 1, config.gpt.modelDim])
        try require(gpt, component: "GPT", key: "gpt.h.0.attn.c_attn.weight", shape: [config.gpt.modelDim, config.gpt.modelDim * 3])
        try require(gpt, component: "GPT", key: "conditioning_encoder.embed.conv.0.weight", shape: [512, 1, 3, 3])
        try require(gpt, component: "GPT", key: "perceiver_encoder.latents", shape: [32, config.gpt.modelDim])
        try require(gpt, component: "GPT", key: "emo_perceiver_encoder.latents", shape: [1, 1024])
        try require(gpt, component: "GPT", key: "speed_emb.weight", shape: [2, config.gpt.modelDim])
        let semanticGPT = IndexTTS2SemanticGPT(weights: gpt, config: config)

        progressHandler?(0.16, "Loading IndexTTS2 S2Mel weights")
        let rawS2Mel = try loadSafetensors("s2mel.safetensors", from: directory)
        let s2Mel = stripRuntimePrefix(rawS2Mel, prefix: "net.")
        guard !s2Mel.isEmpty else {
            throw IndexTTS2RuntimeError.emptyRuntimeComponent("S2Mel")
        }
        try require(s2Mel, component: "S2Mel", key: "gpt_layer.0.weight", shape: [256, config.gpt.modelDim])
        try require(s2Mel, component: "S2Mel", key: "length_regulator.content_in_proj.weight", shape: [512, config.semanticCodec.hiddenSize])
        try require(s2Mel, component: "S2Mel", key: "cfm.estimator.transformer.layers.0.attention.wqkv.weight", shape: [config.s2Mel.hiddenDim * 3, config.s2Mel.hiddenDim])
        let lengthRegulator = try IndexTTS2S2MelLengthRegulator.load(from: s2Mel)
        let s2MelFlow = IndexTTS2S2MelFlow(weights: s2Mel, config: config)

        progressHandler?(0.30, "Loading IndexTTS2 reference encoders")
        let w2vBert = try loadSafetensors("aux/w2v-bert-2.0/model.safetensors", from: directory)
        try require(w2vBert, component: "w2v-BERT", key: "feature_projection.projection.weight", shape: [config.semanticCodec.hiddenSize, 160])
        try require(w2vBert, component: "w2v-BERT", key: "encoder.layers.0.self_attn.linear_q.weight", shape: [config.semanticCodec.hiddenSize, config.semanticCodec.hiddenSize])
        let wav2Vec2BertModel = try IndexTTS2Wav2Vec2Bert.load(from: w2vBert)

        let semanticCodec = try loadSafetensors("aux/maskgct/semantic_codec/model.safetensors", from: directory)
        try require(semanticCodec, component: "semantic codec", key: "quantizer.quantizers.0.codebook.weight", shape: [config.semanticCodec.codebookSize, config.semanticCodec.codebookDim])
        try require(semanticCodec, component: "semantic codec", key: "encoder.0.embed.weight", shape: [config.semanticCodec.vocosDim, config.semanticCodec.hiddenSize, 7])
        try require(semanticCodec, component: "semantic codec", key: "encoder.1.weight", shape: [config.semanticCodec.hiddenSize, config.semanticCodec.vocosDim])
        try require(semanticCodec, component: "semantic codec", key: "quantizer.quantizers.0.out_project.weight_v", shape: [config.semanticCodec.hiddenSize, config.semanticCodec.codebookDim, 1])
        let semanticEncoder = try IndexTTS2SemanticEncoder.load(from: semanticCodec)
        let semanticQuantizer = try IndexTTS2SemanticQuantizer.load(from: semanticCodec)

        progressHandler?(0.48, "Loading IndexTTS2 style and vocoder weights")
        let campPlus = try loadSafetensors("aux/campplus/campplus_cn_common.safetensors", from: directory)
        try require(campPlus, component: "CAMPPlus", key: "head.conv1.weight", shape: [32, 1, 3, 3])
        try require(campPlus, component: "CAMPPlus", key: "xvector.tdnn.linear.weight", shape: [128, 320, 5])
        let campPlusEncoder = try IndexTTS2CAMPPlusAdapter.load(from: campPlus)

        let bigVGAN = try loadSafetensors("aux/bigvgan/bigvgan_generator.safetensors", from: directory)
        try require(bigVGAN, component: "BigVGAN", key: "generator.conv_pre.weight_v", shape: [1536, config.s2Mel.nMels, 7])
        try require(bigVGAN, component: "BigVGAN", key: "generator.conv_post.weight_v", shape: [1, 24, 7])
        let vocoder = IndexTTS2BigVGAN()
        try vocoder.loadWeights(bigVGAN)

        progressHandler?(0.70, "Loading IndexTTS2 conditioning statistics")
        let wavStats = try loadSafetensors("wav2vec2bert_stats.safetensors", from: directory)
        try require(wavStats, component: "wav2vec2bert stats", key: "mean", shape: [config.semanticCodec.hiddenSize])
        try require(wavStats, component: "wav2vec2bert stats", key: "var", shape: [config.semanticCodec.hiddenSize])

        let speakerMatrixWeights = try loadSafetensors("feat1.safetensors", from: directory)
        let emotionMatrixWeights = try loadSafetensors("feat2.safetensors", from: directory)
        guard let speakerMatrix = speakerMatrixWeights["tensor"] else {
            throw IndexTTS2RuntimeError.missingTensor(component: "speaker matrix", key: "tensor")
        }
        guard let emotionMatrix = emotionMatrixWeights["tensor"] else {
            throw IndexTTS2RuntimeError.missingTensor(component: "emotion matrix", key: "tensor")
        }
        try requireShape(speakerMatrix, component: "speaker matrix", key: "tensor", expected: [73, 192])
        try requireShape(emotionMatrix, component: "emotion matrix", key: "tensor", expected: [73, config.gpt.modelDim])

        progressHandler?(0.86, "Loading IndexTTS2 emotion text model weights")
        let qwenEmotion = try loadSafetensors("qwen0.6bemo4-merge/model.safetensors", from: directory)
        try require(qwenEmotion, component: "Qwen emotion", key: "model.embed_tokens.weight", shape: [151_936, 1024])

        progressHandler?(1.0, "IndexTTS2 runtime weights ready")
        return IndexTTS2NativeRuntime(
            config: config,
            gpt: gpt,
            semanticGPT: semanticGPT,
            s2Mel: s2Mel,
            lengthRegulator: lengthRegulator,
            s2MelFlow: s2MelFlow,
            ignoredS2MelTensorCount: rawS2Mel.count - s2Mel.count,
            w2vBert: w2vBert,
            wav2Vec2BertModel: wav2Vec2BertModel,
            semanticCodec: semanticCodec,
            semanticEncoder: semanticEncoder,
            semanticQuantizer: semanticQuantizer,
            campPlus: campPlus,
            campPlusEncoder: campPlusEncoder,
            bigVGAN: bigVGAN,
            vocoder: vocoder,
            wavStats: wavStats,
            speakerMatrix: speakerMatrix,
            emotionMatrix: emotionMatrix,
            qwenEmotion: qwenEmotion)
    }

    private static func loadSafetensors(_ relativePath: String, from directory: URL) throws -> [String: MLXArray] {
        let url = directory.appendingPathComponent(relativePath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw IndexTTS2RuntimeError.missingWeightFile(relativePath)
        }
        return try CommonWeightLoader.loadSafetensors(url: url)
    }

    private static func stripRuntimePrefix(
        _ weights: [String: MLXArray],
        prefix: String
    ) -> [String: MLXArray] {
        var filtered: [String: MLXArray] = [:]
        filtered.reserveCapacity(weights.count)
        for (key, value) in weights where key.hasPrefix(prefix) {
            filtered[String(key.dropFirst(prefix.count))] = value
        }
        return filtered
    }

    private static func require(
        _ weights: [String: MLXArray],
        component: String,
        key: String,
        shape expected: [Int]
    ) throws {
        guard let value = weights[key] else {
            throw IndexTTS2RuntimeError.missingTensor(component: component, key: key)
        }
        try requireShape(value, component: component, key: key, expected: expected)
    }

    private static func requireShape(
        _ value: MLXArray,
        component: String,
        key: String,
        expected: [Int]
    ) throws {
        let actual = value.shape
        guard actual == expected else {
            throw IndexTTS2RuntimeError.invalidTensorShape(
                component: component,
                key: key,
                expected: expected,
                actual: actual)
        }
    }
}
