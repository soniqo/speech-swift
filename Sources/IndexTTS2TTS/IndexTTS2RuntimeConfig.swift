import Foundation

public struct IndexTTS2DatasetConfig: Equatable, Sendable {
    public let sampleRate: Int
    public let bpeModel: String
}

public struct IndexTTS2GPTConfig: Equatable, Sendable {
    public let modelDim: Int
    public let maxMelTokens: Int
    public let maxTextTokens: Int
    public let heads: Int
    public let layers: Int
    public let numberTextTokens: Int
    public let numberMelCodes: Int
    public let startMelToken: Int
    public let stopMelToken: Int
    public let startTextToken: Int
    public let stopTextToken: Int
    public let conditionType: String
    public let conditionBlocks: Int
    public let emotionConditionBlocks: Int
}

public struct IndexTTS2SemanticCodecConfig: Equatable, Sendable {
    public let codebookSize: Int
    public let hiddenSize: Int
    public let codebookDim: Int
    public let vocosDim: Int
    public let vocosNumLayers: Int
}

public struct IndexTTS2S2MelConfig: Equatable, Sendable {
    public let sampleRate: Int
    public let nFFT: Int
    public let winLength: Int
    public let hopLength: Int
    public let nMels: Int
    public let hiddenDim: Int
    public let numHeads: Int
    public let depth: Int
    public let inChannels: Int
    public let contentDim: Int
}

public struct IndexTTS2RuntimeConfig: Equatable, Sendable {
    public let dataset: IndexTTS2DatasetConfig
    public let gpt: IndexTTS2GPTConfig
    public let semanticCodec: IndexTTS2SemanticCodecConfig
    public let s2Mel: IndexTTS2S2MelConfig
    public let emotionBucketCounts: [Int]
    public let qwenEmotionPath: String
    public let version: String

    public var outputSampleRate: Int { s2Mel.sampleRate }

    public init(
        dataset: IndexTTS2DatasetConfig,
        gpt: IndexTTS2GPTConfig,
        semanticCodec: IndexTTS2SemanticCodecConfig,
        s2Mel: IndexTTS2S2MelConfig,
        emotionBucketCounts: [Int],
        qwenEmotionPath: String,
        version: String
    ) {
        self.dataset = dataset
        self.gpt = gpt
        self.semanticCodec = semanticCodec
        self.s2Mel = s2Mel
        self.emotionBucketCounts = emotionBucketCounts
        self.qwenEmotionPath = qwenEmotionPath
        self.version = version
    }

    public static let fallback = IndexTTS2RuntimeConfig(
        dataset: IndexTTS2DatasetConfig(sampleRate: 24_000, bpeModel: "bpe.model"),
        gpt: IndexTTS2GPTConfig(
            modelDim: 1280,
            maxMelTokens: 1815,
            maxTextTokens: 600,
            heads: 20,
            layers: 24,
            numberTextTokens: 12_000,
            numberMelCodes: 8194,
            startMelToken: 8192,
            stopMelToken: 8193,
            startTextToken: 0,
            stopTextToken: 1,
            conditionType: "conformer_perceiver",
            conditionBlocks: 6,
            emotionConditionBlocks: 4),
        semanticCodec: IndexTTS2SemanticCodecConfig(
            codebookSize: 8192,
            hiddenSize: 1024,
            codebookDim: 8,
            vocosDim: 384,
            vocosNumLayers: 12),
        s2Mel: IndexTTS2S2MelConfig(
            sampleRate: 22_050,
            nFFT: 1024,
            winLength: 1024,
            hopLength: 256,
            nMels: 80,
            hiddenDim: 512,
            numHeads: 8,
            depth: 13,
            inChannels: 80,
            contentDim: 512),
        emotionBucketCounts: [3, 17, 2, 8, 4, 5, 10, 24],
        qwenEmotionPath: "qwen0.6bemo4-merge/",
        version: "2.0")

    public static func load(from directory: URL) throws -> IndexTTS2RuntimeConfig {
        let configURL = directory.appendingPathComponent("config.yaml")
        let text = try String(contentsOf: configURL, encoding: .utf8)
        return try IndexTTS2RuntimeConfig(document: IndexTTS2YAMLDocument(text))
    }

    init(document doc: IndexTTS2YAMLDocument) throws {
        dataset = IndexTTS2DatasetConfig(
            sampleRate: try doc.requireInt("dataset.sample_rate"),
            bpeModel: try doc.requireString("dataset.bpe_model"))

        gpt = IndexTTS2GPTConfig(
            modelDim: try doc.requireInt("gpt.model_dim"),
            maxMelTokens: try doc.requireInt("gpt.max_mel_tokens"),
            maxTextTokens: try doc.requireInt("gpt.max_text_tokens"),
            heads: try doc.requireInt("gpt.heads"),
            layers: try doc.requireInt("gpt.layers"),
            numberTextTokens: try doc.requireInt("gpt.number_text_tokens"),
            numberMelCodes: try doc.requireInt("gpt.number_mel_codes"),
            startMelToken: try doc.requireInt("gpt.start_mel_token"),
            stopMelToken: try doc.requireInt("gpt.stop_mel_token"),
            startTextToken: try doc.requireInt("gpt.start_text_token"),
            stopTextToken: try doc.requireInt("gpt.stop_text_token"),
            conditionType: try doc.requireString("gpt.condition_type"),
            conditionBlocks: try doc.requireInt("gpt.condition_module.num_blocks"),
            emotionConditionBlocks: try doc.requireInt("gpt.emo_condition_module.num_blocks"))

        semanticCodec = IndexTTS2SemanticCodecConfig(
            codebookSize: try doc.requireInt("semantic_codec.codebook_size"),
            hiddenSize: try doc.requireInt("semantic_codec.hidden_size"),
            codebookDim: try doc.requireInt("semantic_codec.codebook_dim"),
            vocosDim: try doc.requireInt("semantic_codec.vocos_dim"),
            vocosNumLayers: try doc.requireInt("semantic_codec.vocos_num_layers"))

        s2Mel = IndexTTS2S2MelConfig(
            sampleRate: try doc.requireInt("s2mel.preprocess_params.sr"),
            nFFT: try doc.requireInt("s2mel.preprocess_params.spect_params.n_fft"),
            winLength: try doc.requireInt("s2mel.preprocess_params.spect_params.win_length"),
            hopLength: try doc.requireInt("s2mel.preprocess_params.spect_params.hop_length"),
            nMels: try doc.requireInt("s2mel.preprocess_params.spect_params.n_mels"),
            hiddenDim: try doc.requireInt("s2mel.DiT.hidden_dim"),
            numHeads: try doc.requireInt("s2mel.DiT.num_heads"),
            depth: try doc.requireInt("s2mel.DiT.depth"),
            inChannels: try doc.requireInt("s2mel.DiT.in_channels"),
            contentDim: try doc.requireInt("s2mel.DiT.content_dim"))

        emotionBucketCounts = try doc.requireIntArray("emo_num")
        qwenEmotionPath = try doc.requireString("qwen_emo_path")
        version = try doc.requireString("version")
    }
}

enum IndexTTS2ConfigError: Error, LocalizedError, Equatable {
    case malformedLine(String)
    case missingValue(String)
    case invalidInt(path: String, value: String)
    case invalidIntArray(path: String, value: String)

    var errorDescription: String? {
        switch self {
        case .malformedLine(let line):
            return "IndexTTS2 config contains an unsupported YAML line: \(line)"
        case .missingValue(let path):
            return "IndexTTS2 config is missing required value: \(path)"
        case .invalidInt(let path, let value):
            return "IndexTTS2 config value \(path) is not an integer: \(value)"
        case .invalidIntArray(let path, let value):
            return "IndexTTS2 config value \(path) is not an integer array: \(value)"
        }
    }
}

struct IndexTTS2YAMLDocument {
    private let scalars: [String: String]

    init(_ text: String) throws {
        var values: [String: String] = [:]
        var stack: [(indent: Int, key: String)] = []

        for rawLine in text.split(separator: "\n", omittingEmptySubsequences: false) {
            let withoutComment = rawLine.split(separator: "#", maxSplits: 1, omittingEmptySubsequences: false).first ?? ""
            guard !withoutComment.trimmingCharacters(in: .whitespaces).isEmpty else { continue }

            let line = String(withoutComment)
            let indent = line.prefix { $0 == " " }.count
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard let colon = trimmed.firstIndex(of: ":") else {
                throw IndexTTS2ConfigError.malformedLine(trimmed)
            }

            while let last = stack.last, last.indent >= indent {
                stack.removeLast()
            }

            let key = String(trimmed[..<colon]).trimmingCharacters(in: .whitespaces)
            let rawValue = String(trimmed[trimmed.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
            guard !key.isEmpty else {
                throw IndexTTS2ConfigError.malformedLine(trimmed)
            }

            if rawValue.isEmpty {
                stack.append((indent, key))
            } else {
                let path = (stack.map(\.key) + [key]).joined(separator: ".")
                values[path] = rawValue
            }
        }

        self.scalars = values
    }

    func requireString(_ path: String) throws -> String {
        guard let raw = scalars[path] else {
            throw IndexTTS2ConfigError.missingValue(path)
        }
        return unquoted(raw)
    }

    func requireInt(_ path: String) throws -> Int {
        let raw = try requireString(path)
        guard let value = Int(raw) else {
            throw IndexTTS2ConfigError.invalidInt(path: path, value: raw)
        }
        return value
    }

    func requireIntArray(_ path: String) throws -> [Int] {
        let raw = try requireString(path)
        guard raw.hasPrefix("["), raw.hasSuffix("]") else {
            throw IndexTTS2ConfigError.invalidIntArray(path: path, value: raw)
        }
        let inner = raw.dropFirst().dropLast()
        if inner.trimmingCharacters(in: .whitespaces).isEmpty { return [] }
        return try inner.split(separator: ",").map { part in
            let item = part.trimmingCharacters(in: .whitespaces)
            guard let value = Int(item) else {
                throw IndexTTS2ConfigError.invalidIntArray(path: path, value: raw)
            }
            return value
        }
    }

    private func unquoted(_ raw: String) -> String {
        let value = raw.trimmingCharacters(in: .whitespaces)
        if value.count >= 2,
           (value.first == "\"" && value.last == "\"") ||
           (value.first == "'" && value.last == "'")
        {
            return String(value.dropFirst().dropLast()).trimmingCharacters(in: .whitespaces)
        }
        return value
    }
}
