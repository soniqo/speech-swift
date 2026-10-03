import Foundation
import MLX
import MLXCommon
import Qwen3Chat
import AudioCommon
import Tokenizers
import Hub

/// Local, text-only Clef-flash decisions. Use each instance serially.
public final class Clef {
    public static let defaultModelID = "aufklarer/Clef-flash-9B-MLX-4bit"
    private let backbone: Qwen35MLXModel
    private let lexical: PreQuantizedEmbedding
    private let head: ClefHead
    private let tokenizer: Tokenizer

    private init(backbone: Qwen35MLXModel, lexical: PreQuantizedEmbedding,
                 head: ClefHead, tokenizer: Tokenizer) {
        self.backbone = backbone; self.lexical = lexical; self.head = head; self.tokenizer = tokenizer
    }

    public static func fromPretrained(cacheDir: URL? = nil, offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil) async throws -> Clef {
        let directory = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: defaultModelID)
        let files = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors",
                     "joint_head.safetensors", "joint_head_config.json", "LICENSE"]
        try await HuggingFaceDownloader.downloadFiles(modelId: defaultModelID, to: directory,
            files: files, offlineMode: offlineMode,
            progressHandler: { progressHandler?($0 * 0.8, "Downloading Clef-flash") })
        return try load(from: directory, progressHandler: progressHandler)
    }

    /// Load a text-only affine 4-bit MLX Clef-flash export, including the joint head.
    public static func load(from directory: URL,
        progressHandler: ((Double, String) -> Void)? = nil) throws -> Clef {
        let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("config.json")))
        guard let root = raw as? [String: Any], var text = root["text_config"] as? [String: Any],
              root["model_type"] as? String == "qwen3_5",
              let quant = root["quantization"] as? [String: Any],
              quant["bits"] as? Int == 4, quant["group_size"] as? Int == 64,
              quant["mode"] as? String == "affine",
              text["hidden_size"] as? Int == 4096, text["num_hidden_layers"] as? Int == 32,
              text["linear_num_key_heads"] as? Int == 16, text["linear_num_value_heads"] as? Int == 32,
              text["linear_key_head_dim"] as? Int == 128, text["linear_value_head_dim"] as? Int == 128,
              text["tie_word_embeddings"] as? Bool == false else {
            throw ClefError.invalidModel("Expected the text-only affine 4-bit Clef-flash 9B export.")
        }
        let rope = text["rope_parameters"] as? [String: Any]
        text["rope_theta"] = rope?["rope_theta"] ?? 10000000
        text["max_seq_len"] = 16384
        text["pad_token_id"] = 0
        text["quantization"] = quant
        let config = try JSONDecoder().decode(Qwen3ChatConfig.self, from: JSONSerialization.data(withJSONObject: text))
        guard config.layerTypes?.count == config.numHiddenLayers,
              config.layerTypes?.allSatisfy({ ["linear_attention", "full_attention"].contains($0) }) == true else {
            throw ClefError.invalidModel("Invalid backbone layer types.")
        }
        let hc = try JSONDecoder().decode(ClefHeadConfig.self, from: Data(contentsOf: directory.appendingPathComponent("joint_head_config.json")))
        guard hc.hidden_size == config.hiddenSize else { throw ClefError.invalidModel("Joint head does not match backbone.") }
        let head = try ClefHead(config: hc, weights: MLX.loadArrays(url: directory.appendingPathComponent("joint_head.safetensors")))
        let weights = try MLX.loadArrays(url: directory.appendingPathComponent("model.safetensors"))
        try validateBackbone(weights, config: config)
        let backbone = Qwen35MLXModel(config: config, optimizedDeltaNet: true)
        try Qwen35WeightLoader.loadWeights(into: backbone, weights: weights, progressHandler: progressHandler)
        let output = PreQuantizedEmbedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize, groupSize: 64, bits: 4)
        let prefix = weights["language_model.lm_head.weight"] != nil ? "language_model.lm_head" : "lm_head"
        try CommonWeightLoader.applyCheckedQuantizedEmbeddingWeights(to: output, prefix: prefix, from: weights)
        guard var tc = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("tokenizer_config.json"))) as? [NSString: Any],
              let td = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("tokenizer.json"))) as? [NSString: Any] else {
            throw ClefError.invalidModel("Invalid tokenizer files.")
        }
        // Use the dependency's generic BPE implementation with the checkpoint's
        // pre-tokenizer, vocabulary, merges, and added tokens unchanged.
        tc["tokenizer_class"] = "Qwen2Tokenizer"
        let tokenizer = try AutoTokenizer.from(tokenizerConfig: Config(tc), tokenizerData: Config(td))
        eval(output)
        return Clef(backbone: backbone, lexical: output, head: head, tokenizer: tokenizer)
    }

    private static func validateBackbone(_ weights: [String: MLXArray], config: Qwen3ChatConfig) throws {
        let prefix = weights["language_model.model.norm.weight"] != nil ? "language_model.model." : "model."
        var required: [String: [Int]] = ["norm.weight": [config.hiddenSize]]
        for i in 0..<config.numHiddenLayers {
            let p = "layers.\(i)."
            required[p + "input_layernorm.weight"] = [config.hiddenSize]
            required[p + "post_attention_layernorm.weight"] = [config.hiddenSize]
            if config.layerTypes![i] == "linear_attention" {
                required[p + "linear_attn.conv1d.weight"] = [8192, 4, 1]
                required[p + "linear_attn.dt_bias"] = [32]
                required[p + "linear_attn.A_log"] = [32]
                required[p + "linear_attn.norm.weight"] = [128]
            } else {
                required[p + "self_attn.q_norm.weight"] = [config.headDim]
                required[p + "self_attn.k_norm.weight"] = [config.headDim]
            }
        }
        for (key, shape) in required {
            guard weights[prefix + key]?.shape == shape else {
                throw ClefError.invalidModel("Missing or incorrectly shaped backbone tensor: \(prefix + key)")
            }
        }
    }

    func encode(state: String, questions: [ClefQuestion], maxTokens: Int) throws -> ClefEncoding {
        try ClefEncoding.encode(state: state, questions: questions, maxTokens: maxTokens,
            tokenize: { self.tokenizer.encode(text: $0, addSpecialTokens: false) })
    }

    /// Score all fields together. Oversized input is rejected, never truncated.
    /// `state` is plain text (or a caller-rendered JSON string). No media is accepted.
    public func decide(state: String, questions: [ClefQuestion], maxTokens: Int = 4096) throws -> ClefResult {
        guard (1...16384).contains(maxTokens) else { throw ClefError.invalidRequest("maxTokens must be in 1...16384.") }
        let encoding = try encode(state: state, questions: questions, maxTokens: maxTokens)
        guard encoding.tokens.allSatisfy({ $0 >= 0 && $0 < backbone.config.vocabSize }) else {
            throw ClefError.invalidModel("Tokenizer produced an out-of-vocabulary ID.")
        }
        var inferenceState = Qwen35MLXModel.InferenceState.initial(config: backbone.config)
        var chunks: [MLXArray] = []
        // Bound the recurrent graph size while retaining every token for span pooling.
        for start in stride(from: 0, to: encoding.tokens.count, by: 512) {
            let ids = MLXArray(encoding.tokens[start..<min(start + 512, encoding.tokens.count)].map(Int32.init)).expandedDimensions(axis: 0)
            let (hidden, next) = backbone.forwardHidden(inputIds: ids, state: inferenceState)
            eval(hidden)
            inferenceState = next
            chunks.append(hidden[0])
        }
        let logits = head.logits(hidden: concatenated(chunks), encoding: encoding,
            lexical: { self.lexical(MLXArray($0.map(Int32.init))) })
        let decisions = try zip(questions, logits).map { question, logits -> ClefDecision in
            let probabilities = softmax(logits).asArray(Float.self)
            guard probabilities.allSatisfy({ $0.isFinite }) else { throw ClefError.invalidModel("Nonfinite decision probabilities.") }
            let options = question.options
            let best = probabilities.indices.max(by: { probabilities[$0] < probabilities[$1] })!
            let score: Float?
            if case .score = question.kind {
                score = probabilities.enumerated().reduce(0) { $0 + Float($1.offset) * $1.element }
            } else { score = nil }
            let noul: Float? = question.typeID == 0 ? probabilities[0] : nil
            return ClefDecision(id: question.id, probabilities: Dictionary(uniqueKeysWithValues: zip(options.map(\.0), probabilities)),
                selectedOption: options[best].0, confidence: probabilities[best], score: score, noul: noul)
        }
        return ClefResult(decisions: decisions, inputTokens: encoding.tokens.count)
    }
}
