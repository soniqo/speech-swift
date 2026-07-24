import AudioCommon
import Foundation

/// Published Core ML variants supported by the native MOSS runtime.
public enum MossModelVariant: String, CaseIterable, Sendable {
    case int8
    case fp16

    public var modelId: String {
        switch self {
        case .int8:
            return "aufklarer/MOSS-Transcribe-Diarize-0.9B-CoreML-INT8"
        case .fp16:
            return "aufklarer/MOSS-Transcribe-Diarize-0.9B-CoreML-FP16"
        }
    }
}

/// Quantized decoder variants for the long-context MLX runtime.
///
/// Both bundles keep the Whisper encoder and VQ adaptor in FP16. Only the
/// Qwen3 decoder weights differ, so quality comparisons isolate decoder
/// quantization.
public enum MossMLXVariant: String, CaseIterable, Sendable {
    case int5
    case int8

    public var modelId: String {
        switch self {
        case .int5:
            return "aufklarer/MOSS-Transcribe-Diarize-0.9B-MLX-5bit"
        case .int8:
            return "aufklarer/MOSS-Transcribe-Diarize-0.9B-MLX-INT8"
        }
    }

    public var quantizationBits: Int {
        switch self {
        case .int5: return 5
        case .int8: return 8
        }
    }
}

/// Precision used for the dynamically growing MLX key/value cache.
///
/// Decoder weight precision and cache precision are intentionally separate.
/// Use `.float16` when comparing INT5 and INT8 model quality. Quantized cache
/// can affect quality while reducing long-context memory.
public enum MossMLXKVCachePrecision: String, CaseIterable, Sendable {
    case float16 = "fp16"
    case int8

    public var bitsPerElement: Int {
        switch self {
        case .float16: return 16
        case .int8: return 8
        }
    }
}

/// Greedy decoding and memory controls for the MOSS MLX runtime.
public struct MossMLXDecodingOptions: Sendable, Equatable {
    public var maxTokens: Int
    public var encoderBatchSize: Int
    public var prefillChunkSize: Int
    public var kvCachePrecision: MossMLXKVCachePrecision

    public init(
        maxTokens: Int = 5_120,
        encoderBatchSize: Int = 4,
        prefillChunkSize: Int = 512,
        kvCachePrecision: MossMLXKVCachePrecision = .float16
    ) {
        self.maxTokens = maxTokens
        self.encoderBatchSize = encoderBatchSize
        self.prefillChunkSize = prefillChunkSize
        self.kvCachePrecision = kvCachePrecision
    }
}

/// Analytical MOSS key/value-cache sizing.
public enum MossMLXCacheMemory {
    /// Estimated bytes occupied by all decoder-layer K/V tensors.
    ///
    /// Affine quantized estimates include FP16 scale and bias tensors for
    /// each group. Runtime allocator padding is not included.
    public static func estimatedBytes(
        tokenCount: Int,
        precision: MossMLXKVCachePrecision,
        layers: Int = 28,
        keyValueHeads: Int = 8,
        headDimension: Int = 128,
        groupSize: Int = 64
    ) -> Int {
        guard
            tokenCount > 0,
            layers > 0,
            keyValueHeads > 0,
            headDimension > 0
        else {
            return 0
        }
        let elementCount =
            tokenCount * layers * 2 * keyValueHeads * headDimension
        switch precision {
        case .float16:
            return elementCount * MemoryLayout<Float16>.stride
        case .int8:
            let packedBits = elementCount * precision.bitsPerElement
            let packedBytes = (packedBits + 7) / 8
            let groups = (elementCount + max(groupSize, 1) - 1)
                / max(groupSize, 1)
            let affineMetadataBytes =
                groups * 2 * MemoryLayout<Float16>.stride
            return packedBytes + affineMetadataBytes
        }
    }
}

/// One timestamped, speaker-attributed MOSS transcript segment.
public struct MossTranscriptSegment: Sendable, Equatable {
    public let startTime: Double
    public let endTime: Double
    public let speaker: String
    public let text: String

    public init(
        startTime: Double,
        endTime: Double,
        speaker: String,
        text: String
    ) {
        self.startTime = startTime
        self.endTime = endTime
        self.speaker = speaker
        self.text = text
    }

    public var duration: Double { endTime - startTime }
}

/// Why greedy generation stopped.
public enum MossGenerationStopReason: String, Sendable, Equatable {
    case endOfSequence
    case maximumTokens
    case contextLimit
}

/// Per-call timings reported by the native runtime.
public struct MossTranscriptionMetrics: Sendable, Equatable {
    public let preprocessingSeconds: Double
    public let audioEncoderSeconds: Double
    public let decoderPrefillSeconds: Double
    public let tokenDecodeSeconds: Double
    public let totalSeconds: Double
    public let audioDurationSeconds: Double
    public let promptTokens: Int
    public let generatedTokens: Int
    public let stopReason: MossGenerationStopReason

    public init(
        preprocessingSeconds: Double,
        audioEncoderSeconds: Double,
        decoderPrefillSeconds: Double,
        tokenDecodeSeconds: Double,
        totalSeconds: Double,
        audioDurationSeconds: Double,
        promptTokens: Int,
        generatedTokens: Int,
        stopReason: MossGenerationStopReason
    ) {
        self.preprocessingSeconds = preprocessingSeconds
        self.audioEncoderSeconds = audioEncoderSeconds
        self.decoderPrefillSeconds = decoderPrefillSeconds
        self.tokenDecodeSeconds = tokenDecodeSeconds
        self.totalSeconds = totalSeconds
        self.audioDurationSeconds = audioDurationSeconds
        self.promptTokens = promptTokens
        self.generatedTokens = generatedTokens
        self.stopReason = stopReason
    }

    /// Real-time factor. Lower is faster.
    public var realTimeFactor: Double {
        totalSeconds / max(audioDurationSeconds, 1e-9)
    }

    /// Audio seconds processed per wall-clock second. Higher is faster.
    public var realtimeThroughput: Double {
        totalSeconds > 0 ? audioDurationSeconds / totalSeconds : 0
    }

    /// Autoregressive decoder throughput, excluding prompt prefill.
    public var decodedTokensPerSecond: Double {
        tokenDecodeSeconds > 0 ? Double(generatedTokens) / tokenDecodeSeconds : 0
    }
}

/// Structured output from MOSS transcription and diarization.
public struct MossTranscription: Sendable, Equatable {
    /// Decoded upstream wire format, for example
    /// `[0.42][S01] Hello.[1.10]`.
    public let rawText: String
    /// Plain transcript formed by joining the parsed segment text.
    ///
    /// If the model emits malformed structured output, this contains the raw
    /// decoded text rather than silently discarding it.
    public let text: String
    public let segments: [MossTranscriptSegment]
    public let metrics: MossTranscriptionMetrics

    public init(
        rawText: String,
        text: String,
        segments: [MossTranscriptSegment],
        metrics: MossTranscriptionMetrics
    ) {
        self.rawText = rawText
        self.text = text
        self.segments = segments
        self.metrics = metrics
    }
}

public extension MossTranscription {
    /// Convert model speaker labels into contiguous integer identities for
    /// diarization scoring and shared `AudioCommon` consumers.
    ///
    /// Speaker names are assigned in first-appearance order because MOSS
    /// labels are anonymous within each recording.
    func diarizedSegments(
        audioDuration: Double? = nil
    ) -> [DiarizedSegment] {
        var speakerIDs: [String: Int] = [:]
        var nextSpeakerID = 0
        return segments.compactMap { segment in
            let start = max(0, segment.startTime)
            let end = min(
                max(start, segment.endTime),
                audioDuration ?? segment.endTime
            )
            guard end > start else { return nil }
            let speakerID: Int
            if let existing = speakerIDs[segment.speaker] {
                speakerID = existing
            } else {
                speakerID = nextSpeakerID
                speakerIDs[segment.speaker] = speakerID
                nextSpeakerID += 1
            }
            return DiarizedSegment(
                startTime: Float(start),
                endTime: Float(end),
                speakerId: speakerID
            )
        }
    }
}

public enum MossTranscribeError: Error, LocalizedError {
    case invalidAudio(String)
    case missingModelFile(String)
    case invalidConfiguration(String)
    case missingTokenizerToken(String)
    case promptTooLong(actual: Int, maximum: Int)
    case audioEmbeddingMismatch(placeholders: Int, embeddings: Int)
    case missingModelOutput(String)
    case unsupportedArrayType(String)
    case inferenceFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidAudio(let reason):
            return "Invalid MOSS audio: \(reason)"
        case .missingModelFile(let path):
            return "MOSS model file is missing: \(path)"
        case .invalidConfiguration(let reason):
            return "Invalid MOSS model configuration: \(reason)"
        case .missingTokenizerToken(let token):
            return "MOSS tokenizer is missing required token \(token)"
        case .promptTooLong(let actual, let maximum):
            return "MOSS prompt has \(actual) tokens, exceeding the model context limit of \(maximum)"
        case .audioEmbeddingMismatch(let placeholders, let embeddings):
            return "MOSS audio placeholder/embedding mismatch: \(placeholders) placeholders, \(embeddings) embeddings"
        case .missingModelOutput(let name):
            return "MOSS Core ML model did not return \(name)"
        case .unsupportedArrayType(let type):
            return "MOSS Core ML model returned unsupported array type \(type)"
        case .inferenceFailed(let reason):
            return "MOSS inference failed: \(reason)"
        }
    }
}

/// Parser for the upstream `[start][Sxx]text[end]` output format.
public enum MossTranscriptParser {
    private static let expression = try! NSRegularExpression(
        pattern: #"\[(\d+(?:\.\d+)?)\]\[(S\d+)\]([\s\S]*?)\[(\d+(?:\.\d+)?)\]"#
    )

    public static func parse(_ rawText: String) -> [MossTranscriptSegment] {
        let fullRange = NSRange(rawText.startIndex..<rawText.endIndex, in: rawText)
        return expression.matches(in: rawText, range: fullRange).compactMap { match in
            guard
                let startRange = Range(match.range(at: 1), in: rawText),
                let speakerRange = Range(match.range(at: 2), in: rawText),
                let textRange = Range(match.range(at: 3), in: rawText),
                let endRange = Range(match.range(at: 4), in: rawText),
                let start = Double(rawText[startRange]),
                let end = Double(rawText[endRange]),
                end >= start
            else {
                return nil
            }

            let content = rawText[textRange].trimmingCharacters(in: .whitespacesAndNewlines)
            guard !content.isEmpty else { return nil }
            return MossTranscriptSegment(
                startTime: start,
                endTime: end,
                speaker: String(rawText[speakerRange]),
                text: content
            )
        }
    }

    public static func plainText(from rawText: String) -> (
        text: String,
        segments: [MossTranscriptSegment]
    ) {
        let trimmed = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
        let segments = parse(trimmed)
        guard !segments.isEmpty else { return (trimmed, []) }
        return (segments.map(\.text).joined(separator: " "), segments)
    }
}
