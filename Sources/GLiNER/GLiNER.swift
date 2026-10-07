import Foundation
import MLX
import Tokenizers
import Hub
import AudioCommon

/// Local schema-conditioned classification and entity spans. One instance must
/// be used serially. Model scores are not guarantees of semantic correctness.
public final class GLiNER {
    let network: GLiNERNetwork
    let tokenizer: Tokenizer
    let maxWidth: Int
    let addedTokenIDs: [String:Int]
    private var tokenCache = [String: [Int]]()

    /// Published MLX conversions of fastino/GLiNER2.5-Decide at revision 7ee5da4c.
    public enum Variant: String, CaseIterable, Sendable {
        case fp32, fp16, int8
        public var modelID: String {
            switch self {
            case .fp32: return "aufklarer/GLiNER2.5-Decide-340M-MLX"
            case .fp16: return "aufklarer/GLiNER2.5-Decide-340M-MLX-fp16"
            case .int8: return "aufklarer/GLiNER2.5-Decide-340M-MLX-8bit"
            }
        }
        var files: [String] {
            let base = ["config.json", "encoder_config/config.json", "tokenizer.json", "tokenizer_config.json",
                        "special_tokens_map.json", "export.json", "weights.safetensors"]
            return self == .int8 ? base + ["quantization.json"] : base
        }
    }

    /// Download (or reuse from cache) a published variant and load it.
    /// `modelID` overrides the repository for a variant's file layout.
    public static func fromPretrained(
        variant: Variant = .int8,
        modelID: String? = nil,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        evaluateLayers: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> GLiNER {
        let id = modelID ?? variant.modelID
        let directory = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: id)
        progressHandler?(0, "Downloading GLiNER...")
        try await HuggingFaceDownloader.downloadFiles(
            modelId: id, to: directory, files: variant.files, offlineMode: offlineMode,
            progressHandler: { progressHandler?($0 * 0.9, "Downloading model files...") })
        // The downloader skips requested files a repository does not have, so
        // an incomplete or wrong repository must be caught here, not at load.
        if let missing = variant.files.first(where: {
            !FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path)
        }) {
            throw GLiNERError.missingWeight("\(id) does not provide \(missing); expected a \(variant.rawValue) GLiNER bundle.")
        }
        progressHandler?(0.9, "Loading GLiNER...")
        let model = try await load(from: directory, evaluateLayers: evaluateLayers)
        progressHandler?(1, "Ready")
        return model
    }
    public static func load(from directory: URL, evaluateLayers: Bool = false) async throws -> GLiNER {
        let config = try GLiNERConfig.load(from: directory.appendingPathComponent("encoder_config/config.json"))
        let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("config.json"))) as? [String: Any]
        guard raw?["architecture"] as? String == "span", raw?["counting_layer"] as? String == "count_lstm",
              let width = raw?["max_width"] as? Int, width > 0, width <= 32,
              (raw?["token_pooling"] as? String ?? "first") == "first",
              ((raw?["span_head"] as? [String: Any])?["span_mode"] as? String ?? "markerV0") == "markerV0" else {
            throw GLiNERError.invalidConfiguration("Only span/count_lstm checkpoints are supported.")
        }
        guard var tokenizerConfig = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("tokenizer_config.json"))) as? [NSString: Any],
              let tokenizerData = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("tokenizer.json"))) as? [NSString: Any],
              let tokenizerModel = tokenizerData["model"] as? [String: Any], tokenizerModel["type"] as? String == "Unigram" else {
            throw GLiNERError.invalidConfiguration("Expected a Unigram tokenizer export.")
        }
        // The dependency dispatches by tokenizer class. Select its generic
        // Unigram implementation while preserving the original tokenizer data.
        tokenizerConfig["tokenizer_class"] = "XLMRobertaTokenizer"
        let tokenizer = try AutoTokenizer.from(tokenizerConfig: Config(tokenizerConfig),tokenizerData: Config(tokenizerData))
        let added = (tokenizerData["added_tokens"] as? [[String:Any]] ?? []).compactMap { entry -> (String,Int)? in
            guard let token = entry["content"] as? String, let id = entry["id"] as? Int else { return nil }
            return (token,id)
        }
        var addedIDs = [String:Int]()
        for (token,id) in added {
            guard id >= 0, id < config.vocabSize, addedIDs[token] == nil else { throw GLiNERError.invalidConfiguration("Invalid added-token mapping.") }
            addedIDs[token] = id
        }
        for marker in ["[P]","[L]","[E]","[SEP_TEXT]","[DESCRIPTION]"] {
            guard addedIDs[marker] != nil else { throw GLiNERError.invalidConfiguration("Missing schema marker: \(marker)") }
        }
        let quantizationURL = directory.appendingPathComponent("quantization.json")
        let quantization = FileManager.default.fileExists(atPath:quantizationURL.path)
            ? try GLiNERQuantization.load(from:quantizationURL) : nil
        let weights = try MLX.loadArrays(url: directory.appendingPathComponent("weights.safetensors"))
        let net = try GLiNERNetwork(config: config,weights: weights,evaluateLayers:evaluateLayers,quantization:quantization)
        eval(Array(weights.values))
        return GLiNER(network: net,tokenizer: tokenizer,maxWidth: width,addedTokenIDs:addedIDs)
    }
    init(network: GLiNERNetwork, tokenizer: Tokenizer, maxWidth: Int, addedTokenIDs: [String:Int]) {
        self.network = network; self.tokenizer = tokenizer; self.maxWidth = maxWidth; self.addedTokenIDs = addedTokenIDs
    }
    static func validateLabels(_ labels: [String]) throws {
        guard !labels.isEmpty, labels.count <= 255, Set(labels).count == labels.count,
              labels.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) else {
            throw GLiNERError.invalidSchema("Supply 1–255 distinct, nonempty labels.")
        }
    }
    struct Word { let text: String; let range: NSRange }
    private static let wordPattern = try! NSRegularExpression(
        pattern: #"(?:https?://[^\s]+|www\.[^\s]+)|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|@[a-z0-9_]+|\w+(?:[-_]\w+)*|\S"#,
        options: [.caseInsensitive])
    static func words(_ text: String) -> [Word] {
        let source = text as NSString
        return wordPattern.matches(in: text,range: NSRange(location: 0,length: source.length)).map {
            Word(text: source.substring(with: $0.range).lowercased(),range: $0.range)
        }
    }
    private func tokens(_ value: String) throws -> [Int] {
        if let cached = tokenCache[value] { return cached }
        // The generic Unigram model only maps its base vocabulary. Preserve
        // the added GLiNER marker IDs instead of converting them to unknown.
        let ids = try tokenizer.tokenize(text: value).map { token in
            guard let id = addedTokenIDs[token] ?? tokenizer.convertTokenToId(token) else { throw GLiNERError.invalidSchema("Unmapped token") }
            return id
        }
        guard !ids.isEmpty else { throw GLiNERError.invalidSchema("Input fragment produced no tokens.") }
        if tokenCache.count >= 4096 { tokenCache.removeAll(keepingCapacity: true) }
        tokenCache[value] = ids; return ids
    }
    func prepare(text: String, task: String, labels: [String], descriptions: [String:String], marker: String) throws -> (ids: [Int], parent: Int, labels: [Int], words: [Int], ranges: [Word]) {
        try Self.validateLabels(labels)
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw GLiNERError.invalidSchema("Text must not be empty.") }
        var prompt = task
        for label in labels { if let description = descriptions[label] { prompt += " [DESCRIPTION] \(label): \(description)" } }
        var ids = try tokens("(")
        let parent = ids.count; ids += try tokens("[P]"); ids += try tokens(prompt); ids += try tokens("(")
        var positions = [Int]()
        for label in labels { positions.append(ids.count); ids += try tokens(marker); ids += try tokens(label) }
        ids += try tokens(")"); ids += try tokens(")"); ids += try tokens("[SEP_TEXT]")
        let words = Self.words(text); var wordPositions = [Int]()
        for word in words { wordPositions.append(ids.count); ids += try tokens(word.text) }
        guard ids.count <= 512 else { throw GLiNERError.inputTooLong(ids.count) }
        return (ids,parent,positions,wordPositions,words)
    }
    /// Single-label classification. Returns all alternatives in supplied order.
    public func classify(_ text: String, task: String = "action", labels: [String], descriptions: [String:String] = [:]) throws -> [GLiNERChoice] {
        let input = try prepare(text: text,task: task,labels: labels,descriptions: descriptions,marker: "[L]")
        let encoded = network.encode(input.ids)
        let scores = network.classify(encoded[MLXArray(input.labels.map(Int32.init))]).asArray(Float.self)
        return zip(labels,scores).map { GLiNERChoice(label: $0.0,probability: $0.1) }
    }
    /// Entity mentions, not normalized dates or resolved tool arguments.
    /// Greedy overlap removal is performed independently for each entity label.
    public func extractEntities(_ text: String, labels: [String], descriptions: [String:String] = [:], threshold: Float = 0.5) throws -> [String:[GLiNERSpan]] {
        guard threshold.isFinite, threshold >= 0, threshold <= 1 else { throw GLiNERError.invalidSchema("Threshold must be between zero and one.") }
        let input = try prepare(text: text,task: "entities",labels: labels,descriptions: descriptions,marker: "[E]")
        let encoded = network.encode(input.ids)
        var result = Dictionary(uniqueKeysWithValues: labels.map { ($0,[GLiNERSpan]()) })
        guard let scores = network.spans(words: encoded[MLXArray(input.words.map(Int32.init))],parent: encoded[input.parent].expandedDimensions(axis: 0),fields: encoded[MLXArray(input.labels.map(Int32.init))],maxWidth: maxWidth) else { return result }
        let values = scores.asArray(Float.self), n = input.words.count, source = text as NSString
        for (j,label) in labels.enumerated() {
            var candidates = [GLiNERSpan]()
            for i in 0..<n { for w in 0..<min(maxWidth,n-i) {
                let score = values[(j*n+i)*maxWidth+w]
                if score >= threshold {
                    let start = input.ranges[i].range.location, end = NSMaxRange(input.ranges[i+w].range)
                    candidates.append(GLiNERSpan(text: source.substring(with: NSRange(location: start,length: end-start)),score: score,start: start,end: end))
                }
            }}
            candidates.sort { $0.score == $1.score ? $0.start < $1.start : $0.score > $1.score }
            var selected = [GLiNERSpan]()
            for span in candidates where !selected.contains(where: { span.start < $0.end && span.end > $0.start }) { selected.append(span) }
            result[label] = selected
        }
        return result
    }
}
