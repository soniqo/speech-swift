import CoreML
import Foundation
import AudioCommon

/// Errors specific to the Canary runtime.
public enum CanaryError: Error, LocalizedError {
    case emptyAudio
    case missingPrompt(language: String)
    case unexpectedOutput(String)

    public var errorDescription: String? {
        switch self {
        case .emptyAudio:
            return "Audio buffer was empty or shorter than one STFT frame"
        case .missingPrompt(let language):
            return "Bundle has no prompt token for language '\(language)'"
        case .unexpectedOutput(let name):
            return "Model produced no output named '\(name)'"
        }
    }
}

/// NVIDIA Canary 180M Flash — Core ML automatic speech recognition.
///
/// FastConformer encoder with an autoregressive Transformer decoder, in three
/// compiled models: the encoder runs once over the utterance, a prefill pass
/// consumes the prompt with an empty cache, then a step model emits one token
/// at a time against the growing cache.
///
/// Offline per utterance — the encoder needs the whole segment before the first
/// token, so there is no streaming mode. English, German, Spanish and French,
/// with translation between them from the same graphs.
///
/// - Warning: This class is not thread-safe. Create separate instances for
///   concurrent use.
public class CanaryASRModel {
    /// Model configuration, read from the bundle.
    public let config: CanaryConfig

    /// Default HuggingFace model id.
    public static let defaultModelId = "aufklarer/Canary-180M-Flash-CoreML"

    let encoder: MLModel
    let prefill: MLModel
    let step: MLModel
    let vocabulary: CanaryVocabulary
    private let preprocessor: MelPreprocessor

    /// Source language for the decode prompt.
    public var language: String
    /// Target language. Differing from [language] requests translation.
    public var targetLanguage: String

    /// Mel frames the encoder accepts in one pass.
    var encoderWindow: Int { config.coreml?.encoderMelFrames ?? 3000 }

    /// Seconds of audio the encoder window covers.
    public var maxUtteranceSeconds: Double {
        Double(encoderWindow * config.hopLength) / Double(config.sampleRate)
    }

    init(
        config: CanaryConfig,
        encoder: MLModel,
        prefill: MLModel,
        step: MLModel,
        vocabulary: CanaryVocabulary,
        language: String = "en"
    ) {
        self.config = config
        self.encoder = encoder
        self.prefill = prefill
        self.step = step
        self.vocabulary = vocabulary
        self.language = language
        self.targetLanguage = language
        self.preprocessor = MelPreprocessor(config: config)
    }

    /// Transcribe one utterance.
    ///
    /// Audio longer than the encoder window is truncated rather than chunked:
    /// an attention-encoder-decoder has no cache to carry across windows, and
    /// stitching independent decodes produces duplicated and dropped words at
    /// the seams. Segment with VAD before calling.
    public func transcribeAudio(
        _ audio: [Float],
        sampleRate: Int = 16000,
        language: String? = nil,
        maxTokens: Int = 256
    ) throws -> TranscriptionResult {
        let source = language ?? self.language
        guard let prompt = config.prompt(source: source, target: targetLanguage) else {
            throw CanaryError.missingPrompt(language: source)
        }

        let samples = sampleRate == config.sampleRate
            ? audio
            : CanaryASRModel.resample(audio, from: sampleRate, to: config.sampleRate)
        guard !samples.isEmpty else { return TranscriptionResult(text: "") }

        let (mel, frames) = try preprocessor.extract(samples, window: encoderWindow)

        let encoded = try encoder.prediction(from: try MLDictionaryFeatureProvider(dictionary: [
            "audio_signal": MLFeatureValue(multiArray: mel),
            "length": MLFeatureValue(multiArray: try MLMultiArray.scalarInt32(frames)),
        ]))
        guard let embeddings = encoded.featureValue(for: "encoder_embeddings")?.multiArrayValue,
              let mask = encoded.featureValue(for: "encoder_mask")?.multiArrayValue else {
            throw CanaryError.unexpectedOutput("encoder_embeddings")
        }

        // Prefill consumes the whole prompt against an empty cache.
        let promptArray = try MLMultiArray(shape: [1, NSNumber(value: prompt.count)],
                                           dataType: .int32)
        for (i, id) in prompt.enumerated() { promptArray[i] = NSNumber(value: Int32(id)) }

        var output = try prefill.prediction(from: try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: promptArray),
            "encoder_embeddings": MLFeatureValue(multiArray: embeddings),
            "encoder_mask": MLFeatureValue(multiArray: mask),
        ]))

        var tokens: [Int] = []
        var scoreSum: Double = 0
        let endOfText = config.endOfTextId

        for _ in 0..<maxTokens {
            guard let logits = output.featureValue(for: "logits")?.multiArrayValue,
                  let cache = output.featureValue(for: "decoder_hidden_states")?.multiArrayValue
            else { throw CanaryError.unexpectedOutput("logits") }

            let (best, score) = CanaryASRModel.argmax(logits)
            scoreSum += Double(score)
            if best == endOfText { break }
            tokens.append(best)

            let next = try MLMultiArray(shape: [1, 1], dataType: .int32)
            next[0] = NSNumber(value: Int32(best))
            // The cache length is the decoder's position offset. It is an input
            // rather than something the graph reads off the cache tensor: the
            // tracer folds a shape read into a constant, which would freeze the
            // positional encoding and leave every step after the first wrong —
            // while still producing fluent text.
            let cacheLength = cache.shape[2].intValue

            output = try step.prediction(from: try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(multiArray: next),
                "encoder_embeddings": MLFeatureValue(multiArray: embeddings),
                "encoder_mask": MLFeatureValue(multiArray: mask),
                "decoder_mems": MLFeatureValue(multiArray: cache),
                "start_pos": MLFeatureValue(multiArray: try MLMultiArray.scalarInt32(cacheLength)),
            ]))
        }

        // The head applies log_softmax, so the mean greedy score is a mean log
        // probability and exp() puts it back on 0…1.
        let confidence: Float = tokens.isEmpty ? 0
            : (config.logitsAreLogProbs
                ? Float(exp(scoreSum / Double(tokens.count)))
                : Float(1 / (1 + exp(-scoreSum / Double(tokens.count) * 0.1))))

        return TranscriptionResult(
            text: vocabulary.decode(tokens),
            language: source,
            confidence: min(max(confidence, 0), 1)
        )
    }

    private static func argmax(_ logits: MLMultiArray) -> (index: Int, score: Float) {
        let count = logits.count
        // The graph emits one position, so the last `vocab` values are the
        // distribution to sample even if a future export returns more.
        let vocab = logits.shape.last?.intValue ?? count
        let offset = count - vocab

        // Read at the array's own precision. These models are float16, and
        // reading that buffer as Float32 walks past its end — it segfaults
        // rather than returning something merely wrong.
        func scan<T: BinaryFloatingPoint>(_ pointer: UnsafeMutablePointer<T>) -> (Int, Float) {
            var best = 0
            var bestScore = pointer[offset]
            for i in 1..<vocab where pointer[offset + i] > bestScore {
                bestScore = pointer[offset + i]
                best = i
            }
            return (best, Float(bestScore))
        }

        switch logits.dataType {
        case .float16:
            return scan(logits.dataPointer.assumingMemoryBound(to: Float16.self))
        case .double:
            return scan(logits.dataPointer.assumingMemoryBound(to: Float64.self))
        default:
            return scan(logits.dataPointer.assumingMemoryBound(to: Float32.self))
        }
    }

    /// Linear resample. Callers on the voice pipeline already deliver 16 kHz;
    /// this exists so a one-off buffer at another rate transcribes rather than
    /// returning nothing.
    private static func resample(_ audio: [Float], from: Int, to: Int) -> [Float] {
        guard from > 0, to > 0, from != to, !audio.isEmpty else { return audio }
        let outputCount = Int((Int64(audio.count) * Int64(to)) / Int64(from))
        guard outputCount > 0 else { return [] }
        return (0..<outputCount).map { i in
            let position = Double(i) * Double(from) / Double(to)
            let low = min(Int(position), audio.count - 1)
            let high = min(low + 1, audio.count - 1)
            let fraction = Float(position - Double(low))
            return audio[low] * (1 - fraction) + audio[high] * fraction
        }
    }
}

extension MLMultiArray {
    /// One-element Int32 array, which is how the graphs take `length` and
    /// `start_pos`.
    static func scalarInt32(_ value: Int) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1], dataType: .int32)
        array[0] = NSNumber(value: Int32(value))
        return array
    }
}
