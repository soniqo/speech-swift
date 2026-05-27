#if canImport(CoreML)
import CoreML
import Foundation
import MLX
import AudioCommon

/// CoreML text decoder for Qwen3-ASR with MLState KV cache.
///
/// Runs the full text decoder on Neural Engine via CoreML instead of GPU via MLX.
/// Uses fixed-size KV cache with attention masking. Requires macOS 15+ / iOS 18+
/// for MLState support.
///
/// The decoder is split into three CoreML models so the 28-layer Qwen3
/// transformer fits within the ANE compiler's per-model graph-size
/// budget (the full 28-layer monolithic decoder fails with
/// ``ANECCompile() FAILED``):
///   - **embedding**:     Token ID → embedding vector lookup
///   - **decoder_part1**: Embedding → hidden state, layers 0..(split-1),
///                        keeps half the KV cache states in its MLState
///   - **decoder_part2**: Hidden state → logits, layers split..N-1 plus
///                        the final RMSNorm + tied LM head, keeps the
///                        remaining KV cache states in its MLState
///
/// Each part has its own ``MLState``. ``decoderStep`` runs the two parts
/// sequentially and returns the final logits.
public class CoreMLTextDecoder {
    private let embeddingModel: MLModel
    private let decoderPart1Model: MLModel
    private let decoderPart2Model: MLModel
    private let maxSeqLength: Int
    private let vocabSize: Int
    private let hiddenSize: Int

    /// One MLState per decoder part, each holds that part's KV caches.
    private var part1State: MLState
    private var part2State: MLState

    /// Current position in the KV cache (incremented per step).
    private var currentPosition: Int = 0

    public static let defaultModelId = "aufklarer/Qwen3-ASR-CoreML"

    public init(
        embeddingModel: MLModel,
        decoderPart1Model: MLModel,
        decoderPart2Model: MLModel,
        maxSeqLength: Int = 1024,
        vocabSize: Int = 151936,
        hiddenSize: Int = 1024
    ) {
        self.embeddingModel = embeddingModel
        self.decoderPart1Model = decoderPart1Model
        self.decoderPart2Model = decoderPart2Model
        self.maxSeqLength = maxSeqLength
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.part1State = decoderPart1Model.makeState()
        self.part2State = decoderPart2Model.makeState()
    }

    /// Load decoder models from a directory containing
    /// ``embedding.mlmodelc``, ``decoder_part1.mlmodelc`` and ``decoder_part2.mlmodelc``.
    public static func load(
        from directory: URL,
        computeUnits: MLComputeUnits = .all
    ) throws -> CoreMLTextDecoder {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits

        var maxSeq = 1024
        var vocabSize = 151936
        var hiddenSize = 1024
        let configPath = directory.appendingPathComponent("config.json")
        if let data = try? Data(contentsOf: configPath),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            maxSeq = json["max_seq_length"] as? Int ?? 1024
            vocabSize = json["vocab_size"] as? Int ?? 151936
            hiddenSize = json["hidden_size"] as? Int ?? 1024
        }

        let embURL = findModel(named: "embedding", in: directory)
        let p1URL = findModel(named: "decoder_part1", in: directory)
        let p2URL = findModel(named: "decoder_part2", in: directory)

        guard let embURL else {
            throw AudioModelError.modelLoadFailed(
                modelId: "embedding",
                reason: "CoreML embedding not found in \(directory.path)")
        }
        guard let p1URL else {
            throw AudioModelError.modelLoadFailed(
                modelId: "decoder_part1",
                reason: "CoreML decoder_part1 not found in \(directory.path)")
        }
        guard let p2URL else {
            throw AudioModelError.modelLoadFailed(
                modelId: "decoder_part2",
                reason: "CoreML decoder_part2 not found in \(directory.path)")
        }

        let embModel = try MLModel(contentsOf: embURL, configuration: config)
        let p1Model = try MLModel(contentsOf: p1URL, configuration: config)
        let p2Model = try MLModel(contentsOf: p2URL, configuration: config)

        return CoreMLTextDecoder(
            embeddingModel: embModel,
            decoderPart1Model: p1Model,
            decoderPart2Model: p2Model,
            maxSeqLength: maxSeq,
            vocabSize: vocabSize,
            hiddenSize: hiddenSize
        )
    }

    /// Load from HuggingFace.
    public static func fromPretrained(
        modelId: String = defaultModelId,
        computeUnits: MLComputeUnits = .all,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> CoreMLTextDecoder {
        let cacheDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)

        progressHandler?(0.0, "Downloading CoreML decoder...")
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: cacheDir,
            additionalFiles: [
                "embedding.mlmodelc/**",
                "decoder_part1.mlmodelc/**",
                "decoder_part2.mlmodelc/**",
                "config.json",
            ],
            offlineMode: offlineMode
        ) { fraction in
            progressHandler?(fraction * 0.8, "Downloading CoreML decoder...")
        }

        progressHandler?(0.9, "Loading CoreML decoder...")
        let decoder = try load(from: cacheDir, computeUnits: computeUnits)
        progressHandler?(1.0, "Ready")
        return decoder
    }

    /// Warm up all three models with dummy inputs.
    public func warmUp() throws {
        // Embedding
        let dummyToken = try MLMultiArray(shape: [1, 1], dataType: .int32)
        dummyToken[0] = 0
        _ = try embeddingModel.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "token_id": MLFeatureValue(multiArray: dummyToken),
        ]))

        // Decoder parts — use throwaway MLStates so we don't pollute the live caches.
        let dummyEmbed = try MLMultiArray(shape: [1, 1, hiddenSize as NSNumber], dataType: .float32)
        let dummyPos = try MLMultiArray(shape: [1], dataType: .int32)
        dummyPos[0] = 0
        let dummyMask = try MLMultiArray(shape: [1, 1, 1, maxSeqLength as NSNumber], dataType: .float32)
        let mptr = dummyMask.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<maxSeqLength { mptr[i] = -1e4 }
        mptr[0] = 0

        let warmP1 = decoderPart1Model.makeState()
        let warmP2 = decoderPart2Model.makeState()

        let dummyInputs = try MLDictionaryFeatureProvider(dictionary: [
            "input_embeds": MLFeatureValue(multiArray: dummyEmbed),
            "position": MLFeatureValue(multiArray: dummyPos),
            "attention_mask": MLFeatureValue(multiArray: dummyMask),
        ])
        let p1Out = try decoderPart1Model.prediction(from: dummyInputs, using: warmP1)
        guard let hidden = p1Out.featureValue(for: "hidden_state")?.multiArrayValue else {
            throw AudioModelError.inferenceFailed(
                operation: "CoreML decoder part1 warmup",
                reason: "Missing hidden_state output")
        }
        let p2Inputs = try MLDictionaryFeatureProvider(dictionary: [
            "input_embeds": MLFeatureValue(multiArray: hidden),
            "position": MLFeatureValue(multiArray: dummyPos),
            "attention_mask": MLFeatureValue(multiArray: dummyMask),
        ])
        _ = try decoderPart2Model.prediction(from: p2Inputs, using: warmP2)
    }

    /// Reset the KV caches in both parts for a new transcription.
    public func resetCache() {
        currentPosition = 0
        part1State = decoderPart1Model.makeState()
        part2State = decoderPart2Model.makeState()
    }

    // MARK: - Token Operations

    /// Look up embedding for a token ID.
    public func embed(tokenId: Int32) throws -> MLMultiArray {
        let tokenArray = try MLMultiArray(shape: [1, 1], dataType: .int32)
        tokenArray[0] = NSNumber(value: tokenId)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "token_id": MLFeatureValue(multiArray: tokenArray),
        ])
        let output = try embeddingModel.prediction(from: input)

        guard let embedding = output.featureValue(for: "embedding")?.multiArrayValue else {
            throw AudioModelError.inferenceFailed(
                operation: "CoreML embedding", reason: "Missing embedding output")
        }
        return embedding
    }

    /// Run one decoder step.
    ///
    /// Chains the two CoreML parts: ``embedding → part1 → part2 → logits``.
    /// Each part updates its own KV cache via its own ``MLState``. The
    /// hidden state handed from part1 to part2 has shape ``[1, 1, hidden]``.
    public func decoderStep(embedding: MLMultiArray) throws -> MLMultiArray {
        guard currentPosition < maxSeqLength else {
            throw AudioModelError.inferenceFailed(
                operation: "CoreML decoder",
                reason: "Sequence length \(currentPosition) exceeds max \(maxSeqLength)")
        }

        let mask = try MLMultiArray(shape: [1, 1, 1, maxSeqLength as NSNumber], dataType: .float32)
        let maskPtr = mask.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0...currentPosition {
            maskPtr[i] = 0
        }
        for i in (currentPosition + 1)..<maxSeqLength {
            maskPtr[i] = -1e4
        }

        let position = try MLMultiArray(shape: [1], dataType: .int32)
        position[0] = NSNumber(value: Int32(currentPosition))

        let part1Input = try MLDictionaryFeatureProvider(dictionary: [
            "input_embeds": MLFeatureValue(multiArray: embedding),
            "position": MLFeatureValue(multiArray: position),
            "attention_mask": MLFeatureValue(multiArray: mask),
        ])
        let part1Out = try decoderPart1Model.prediction(from: part1Input, using: part1State)
        guard let hidden = part1Out.featureValue(for: "hidden_state")?.multiArrayValue else {
            throw AudioModelError.inferenceFailed(
                operation: "CoreML decoder part1",
                reason: "Missing hidden_state output")
        }

        let part2Input = try MLDictionaryFeatureProvider(dictionary: [
            "input_embeds": MLFeatureValue(multiArray: hidden),
            "position": MLFeatureValue(multiArray: position),
            "attention_mask": MLFeatureValue(multiArray: mask),
        ])
        let part2Out = try decoderPart2Model.prediction(from: part2Input, using: part2State)

        currentPosition += 1

        guard let logits = part2Out.featureValue(for: "logits")?.multiArrayValue else {
            throw AudioModelError.inferenceFailed(
                operation: "CoreML decoder part2",
                reason: "Missing logits output")
        }
        return logits
    }

    /// Get argmax token ID from logits.
    ///
    /// Stride-aware: walks `vocabSize` (the logical last-dim length) using
    /// `strides.last` as the step, so this is correct for CoreML outputs
    /// where the runtime decides to return a strided buffer (e.g. ANE
    /// padding). NaN-safe — NaN values are skipped, so a single bad logit
    /// can't poison the argmax (the previous flat `ptr[i]` loop combined
    /// with `maxVal = -Float.infinity` would silently keep `maxIdx = 0`
    /// if NaN propagated through the comparison).
    public func argmax(logits: MLMultiArray) -> Int32 {
        let vocab = logits.shape.last?.intValue ?? logits.count
        let lastStride = logits.strides.last?.intValue ?? 1
        var maxVal: Float = -Float.infinity
        var maxIdx: Int32 = 0
        var nanCount: Int = 0

        switch logits.dataType {
        case .float16:
            let ptr = logits.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<vocab {
                let val = Float(ptr[i * lastStride])
                if val.isNaN { nanCount += 1; continue }
                if val > maxVal {
                    maxVal = val
                    maxIdx = Int32(i)
                }
            }
        case .float32:
            let ptr = logits.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<vocab {
                let val = ptr[i * lastStride]
                if val.isNaN { nanCount += 1; continue }
                if val > maxVal {
                    maxVal = val
                    maxIdx = Int32(i)
                }
            }
        default:
            let ptr = logits.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<vocab {
                let val = ptr[i * lastStride]
                if val.isNaN { nanCount += 1; continue }
                if val > maxVal {
                    maxVal = val
                    maxIdx = Int32(i)
                }
            }
        }

        return maxIdx
    }

    // MARK: - Audio Embedding Injection

    /// Convert MLXArray audio embeddings to MLMultiArray for decoder input.
    ///
    /// Audio embeddings from the CoreML encoder are [1, T, 1024].
    /// Each position is fed to the decoder one at a time during prefill.
    public func audioEmbeddingToMultiArray(_ embedding: MLXArray, at index: Int) throws -> MLMultiArray {
        // embedding: [1, T, hidden_size] — extract [1, 1, hidden_size] at index
        let hidden = embedding.dim(2)
        let result = try MLMultiArray(shape: [1, 1, hidden as NSNumber], dataType: .float32)
        let ptr = result.dataPointer.assumingMemoryBound(to: Float.self)

        // Extract the slice at index
        let slice = embedding[0..., index..<(index + 1), 0...]
        let data: [Float] = slice.asArray(Float.self)
        for i in 0..<hidden {
            ptr[i] = data[i]
        }

        return result
    }

    /// Extract audio embedding at index from MLMultiArray (no MLX dependency).
    ///
    /// This is the MLX-free equivalent of `audioEmbeddingToMultiArray(_:at:)`.
    /// Used by `transcribeWithoutMLX()` to avoid any Metal GPU evaluation,
    /// making it safe for iOS background execution.
    ///
    /// - Parameters:
    ///   - embeddings: Audio embeddings as MLMultiArray with shape `[1, T, hidden_size]`
    ///   - index: Time-step index to extract (0..<T)
    /// - Returns: MLMultiArray with shape `[1, 1, hidden_size]` for a single time step
    public func audioEmbeddingFromMultiArray(_ embeddings: MLMultiArray, at index: Int) throws -> MLMultiArray {
        let hidden = embeddings.shape[2].intValue
        let result = try MLMultiArray(shape: [1, 1, hidden as NSNumber], dataType: .float32)
        let srcPtr = embeddings.dataPointer.assumingMemoryBound(to: Float.self)
        let dstPtr = result.dataPointer.assumingMemoryBound(to: Float.self)
        let offset = index * hidden
        for i in 0..<hidden {
            dstPtr[i] = srcPtr[offset + i]
        }
        return result
    }

    // MARK: - Helpers

    private static func findModel(named name: String, in directory: URL) -> URL? {
        // Only pre-compiled ``.mlmodelc`` is supported — on-device
        // ``MLModel.compileModel`` drifts per runtime, and the published
        // ``aufklarer/Qwen3-ASR-CoreML`` repo ships compiled bundles only.
        let compiled = directory.appendingPathComponent("\(name).mlmodelc", isDirectory: true)
        if FileManager.default.fileExists(atPath: compiled.path) {
            return compiled
        }
        return nil
    }
}
#endif
