#if canImport(CoreML)
import CoreML
import Foundation
import MLX
import AudioCommon

/// CoreML text decoder for Qwen3-ASR with MLState KV cache.
///
/// Runs the full text decoder on Neural Engine via CoreML instead of GPU via MLX.
/// Requires macOS 15+ / iOS 18+ for MLState support.
///
/// Architecture (three CoreML models for ANE compile-budget reasons):
///   - **embedding**:     Token ID → embedding vector lookup
///   - **decoder_part1**: Layers 0..(split-1), embedding-in → hidden-out
///   - **decoder_part2**: Layers split..N-1 + norm + lm_head, hidden-in → logits
///
/// Each part keeps its own ``MLState`` pool of KV caches. Hidden state
/// flows part1 → part2.
///
/// **Batched dispatch.** Both decoder parts are converted with a fixed
/// ``T`` (per ``config.json: enumerated_t`` — typically 128). One ANE
/// dispatch processes T tokens at once: prefill chunks a contiguous run
/// of new positions through a single call; single-token generation
/// reserves indices ``[0, T-2]`` for *scratch* positions (the last T-1
/// slots of the KV cache, addresses ``maxSeqLength - (T-1)..maxSeqLength``)
/// whose writes are discarded by attention masking. This collapses ~250
/// single-token audio-prefill dispatches into ~2 batched calls.
///
/// EnumeratedShapes(T) is **not** ANE-compatible at this layer count,
/// so the model is fixed-T and we wrap step calls with scratch padding
/// rather than running a smaller variant.
public class CoreMLTextDecoder {
    private let embeddingModel: MLModel
    private let decoderPart1Model: MLModel
    private let decoderPart2Model: MLModel
    private let maxSeqLength: Int
    private let vocabSize: Int
    private let hiddenSize: Int

    /// Fixed batch size used by both decoder parts. Loaded from
    /// ``config.json: enumerated_t``. Single-token decode pads to this.
    private let batchSize: Int

    /// One MLState per decoder part, each holds that part's KV caches.
    private var part1State: MLState
    private var part2State: MLState

    /// Current real position in the KV cache (incremented per real token).
    private var currentPosition: Int = 0

    /// First slot index reserved for scratch writes by partial / step calls.
    /// Cache writes at these positions are garbage; attention masks them.
    /// Range: ``[scratchStart, maxSeqLength)``, length ``batchSize - 1``.
    private var scratchStart: Int { maxSeqLength - (batchSize - 1) }

    public static let defaultModelId = "aufklarer/Qwen3-ASR-CoreML"

    public init(
        embeddingModel: MLModel,
        decoderPart1Model: MLModel,
        decoderPart2Model: MLModel,
        maxSeqLength: Int = 1024,
        vocabSize: Int = 151936,
        hiddenSize: Int = 1024,
        batchSize: Int = 128
    ) {
        self.embeddingModel = embeddingModel
        self.decoderPart1Model = decoderPart1Model
        self.decoderPart2Model = decoderPart2Model
        self.maxSeqLength = maxSeqLength
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.batchSize = batchSize
        self.part1State = decoderPart1Model.makeState()
        self.part2State = decoderPart2Model.makeState()
    }

    /// Load decoder models from a directory containing
    /// ``embedding.mlmodelc``, ``decoder_part1.mlmodelc`` and ``decoder_part2.mlmodelc``.
    public static func load(
        from directory: URL,
        computeUnits: MLComputeUnits = CoreMLComputeUnitsResolver.resolved(default: .cpuAndNeuralEngine)
    ) throws -> CoreMLTextDecoder {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits

        var maxSeq = 1024
        var vocabSize = 151936
        var hiddenSize = 1024
        var batchSize = 128
        let configPath = directory.appendingPathComponent("config.json")
        if let data = try? Data(contentsOf: configPath),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            maxSeq = json["max_seq_length"] as? Int ?? 1024
            vocabSize = json["vocab_size"] as? Int ?? 151936
            hiddenSize = json["hidden_size"] as? Int ?? 1024
            if let ts = json["enumerated_t"] as? [Int], let t = ts.first {
                batchSize = t
            }
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
            hiddenSize: hiddenSize,
            batchSize: batchSize
        )
    }

    /// Load from HuggingFace.
    public static func fromPretrained(
        modelId: String = defaultModelId,
        computeUnits: MLComputeUnits = CoreMLComputeUnitsResolver.resolved(default: .cpuAndNeuralEngine),
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

    /// Warm up the three models so the first real call doesn't pay
    /// the ANE compile / load latency. Uses throwaway MLStates so the
    /// live KV cache stays untouched.
    public func warmUp() throws {
        let dummyToken = try MLMultiArray(shape: [1, 1], dataType: .int32)
        dummyToken[0] = 0
        _ = try embeddingModel.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "token_id": MLFeatureValue(multiArray: dummyToken),
        ]))

        let warmP1 = decoderPart1Model.makeState()
        let warmP2 = decoderPart2Model.makeState()
        let dummyEmbeds = try MLMultiArray(shape: [1, batchSize as NSNumber, hiddenSize as NSNumber],
                                            dataType: .float32)
        let dummyPositions = try MLMultiArray(shape: [batchSize as NSNumber], dataType: .int32)
        for i in 0..<batchSize { dummyPositions[i] = NSNumber(value: Int32(i)) }
        let dummyMask = try MLMultiArray(shape: [1, 1, batchSize as NSNumber, maxSeqLength as NSNumber],
                                          dataType: .float32)
        let mptr = dummyMask.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<(batchSize * maxSeqLength) { mptr[i] = -1e4 }

        let warmInputs = try MLDictionaryFeatureProvider(dictionary: [
            "input_embeds": MLFeatureValue(multiArray: dummyEmbeds),
            "positions": MLFeatureValue(multiArray: dummyPositions),
            "attention_mask": MLFeatureValue(multiArray: dummyMask),
        ])
        let p1Out = try decoderPart1Model.prediction(from: warmInputs, using: warmP1)
        guard let hidden = p1Out.featureValue(for: "hidden_state")?.multiArrayValue else {
            throw AudioModelError.inferenceFailed(
                operation: "CoreML decoder part1 warmup",
                reason: "Missing hidden_state output")
        }
        let p2Inputs = try MLDictionaryFeatureProvider(dictionary: [
            "input_embeds": MLFeatureValue(multiArray: hidden),
            "positions": MLFeatureValue(multiArray: dummyPositions),
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

    /// Look up the embedding vector for a single token id.
    /// Returns shape ``[1, 1, hidden_size]``.
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

    /// Run one decoder step on a single embedding.
    ///
    /// Packs the one real token into the last input slot and fills the
    /// remaining ``batchSize - 1`` slots with scratch positions whose
    /// outputs and cache writes are discarded by masking. Single-token
    /// decode pays the same ANE dispatch cost as a full chunked prefill.
    public func decoderStep(embedding: MLMultiArray) throws -> MLMultiArray {
        let bufs = try writeChunk(realEmbeddingsSource: { (slot, dstPtr) in
            Self.copyRow(from: embedding, sourceRow: 0, hidden: self.hiddenSize,
                         to: dstPtr, destSlot: slot)
        }, realCount: 1)
        return try runParts(embeds: bufs.embeds, positions: bufs.positions, mask: bufs.mask)
    }

    /// Copy one ``hidden``-length row from an MLMultiArray (any float
    /// dtype, any strides) into a contiguous Float32 destination row.
    /// The decoder input is declared Float32, but CoreML model *outputs*
    /// (embedding lookup, part1 hidden) may come back as Float16 with
    /// padded strides on ANE; a raw ``assumingMemoryBound(to: Float)``
    /// copy would then read garbage.
    private static func copyRow(from src: MLMultiArray, sourceRow: Int, hidden: Int,
                                to dst: UnsafeMutablePointer<Float>, destSlot: Int) {
        let rowStride = src.strides.count >= 2 ? src.strides[src.strides.count - 2].intValue : hidden
        let lastStride = src.strides.last?.intValue ?? 1
        let base = sourceRow * rowStride
        let dstBase = destSlot * hidden
        switch src.dataType {
        case .float16:
            let p = src.dataPointer.assumingMemoryBound(to: Float16.self)
            for j in 0..<hidden { dst[dstBase + j] = Float(p[base + j * lastStride]) }
        case .float32:
            let p = src.dataPointer.assumingMemoryBound(to: Float.self)
            for j in 0..<hidden { dst[dstBase + j] = p[base + j * lastStride] }
        default:
            let p = src.dataPointer.assumingMemoryBound(to: Float.self)
            for j in 0..<hidden { dst[dstBase + j] = p[base + j * lastStride] }
        }
    }

    /// Run a chunked multi-token prefill, source from MLMultiArray rows.
    ///
    /// ``embeddings`` is shape ``[1, N, hidden]`` containing the real
    /// next-N tokens to commit to positions ``[currentPosition,
    /// currentPosition + N)``. ``N`` must be ``<= batchSize``. The
    /// returned MLMultiArray is the logits for the last real position.
    @discardableResult
    public func decoderPrefill(embeddings: MLMultiArray, realCount n: Int) throws -> MLMultiArray {
        precondition(n > 0 && n <= batchSize,
                     "realCount \(n) must be in 1...\(batchSize)")
        let bufs = try writeChunk(realEmbeddingsSource: { (slot, dstPtr) in
            let srcPtr = embeddings.dataPointer.assumingMemoryBound(to: Float.self)
            for t in 0..<n {
                let srcOff = t * self.hiddenSize
                let dstOff = (slot + t) * self.hiddenSize
                for j in 0..<self.hiddenSize {
                    dstPtr[dstOff + j] = srcPtr[srcOff + j]
                }
            }
        }, realCount: n)
        return try runParts(embeds: bufs.embeds, positions: bufs.positions, mask: bufs.mask)
    }

    /// Embed a contiguous run of token ids and prefill them in batched
    /// chunks of ``batchSize``. Used for the chat-template prefix/suffix
    /// runs, replacing per-token ``embed`` + ``decoderStep`` loops with
    /// one ANE dispatch per chunk. Returns the logits for the last token.
    @discardableResult
    public func decoderPrefillTokens(_ tokenIds: [Int32]) throws -> MLMultiArray {
        precondition(!tokenIds.isEmpty, "decoderPrefillTokens requires at least one token")
        var lastLogits: MLMultiArray!
        var consumed = 0
        while consumed < tokenIds.count {
            let n = min(batchSize, tokenIds.count - consumed)
            // Pack n token embeddings into a [1, n, hidden] Float32 buffer.
            let packed = try MLMultiArray(shape: [1, n as NSNumber, hiddenSize as NSNumber],
                                           dataType: .float32)
            let pptr = packed.dataPointer.assumingMemoryBound(to: Float.self)
            for k in 0..<n {
                let emb = try embed(tokenId: tokenIds[consumed + k])
                Self.copyRow(from: emb, sourceRow: 0, hidden: hiddenSize,
                             to: pptr, destSlot: k)
            }
            lastLogits = try decoderPrefill(embeddings: packed, realCount: n)
            consumed += n
        }
        return lastLogits
    }

    /// Run a chunked prefill where embeddings come from a Float buffer
    /// (typical: bulk-extracted MLX audio embeddings).
    @discardableResult
    public func decoderPrefill(flatEmbeddings: [Float], offset: Int, realCount n: Int) throws -> MLMultiArray {
        precondition(n > 0 && n <= batchSize,
                     "realCount \(n) must be in 1...\(batchSize)")
        let bufs = try writeChunk(realEmbeddingsSource: { (slot, dstPtr) in
            flatEmbeddings.withUnsafeBufferPointer { buf in
                let src = buf.baseAddress!
                for t in 0..<n {
                    let srcOff = (offset + t) * self.hiddenSize
                    let dstOff = (slot + t) * self.hiddenSize
                    for j in 0..<self.hiddenSize {
                        dstPtr[dstOff + j] = src[srcOff + j]
                    }
                }
            }
        }, realCount: n)
        return try runParts(embeds: bufs.embeds, positions: bufs.positions, mask: bufs.mask)
    }

    // MARK: - Internal: chunked dispatch primitives

    /// Set up the input buffers (positions, mask, embeds) for a chunk of
    /// ``realCount`` real tokens placed in the LAST ``realCount`` input
    /// slots, with the remaining slots filled by scratch positions. The
    /// ``realEmbeddingsSource`` callback is given the slot where real
    /// data should land and a pointer into the embeds buffer.
    ///
    /// Allocates FRESH MLMultiArrays each call. Reusing buffers across
    /// calls was tried and produces wrong outputs — CoreML appears to
    /// hold references into the input buffer past prediction return,
    /// so mutating it before the next call corrupts the stored KV state.
    /// Per-call allocation costs ~0.5 ms (mask is 64 KB) which is well
    /// below the dispatch budget.
    private func writeChunk(
        realEmbeddingsSource: (Int, UnsafeMutablePointer<Float>) -> Void,
        realCount n: Int
    ) throws -> (embeds: MLMultiArray, positions: MLMultiArray, mask: MLMultiArray) {
        precondition(currentPosition + n <= scratchStart,
                     "Cache overflow: would write real position \(currentPosition + n - 1) into scratch range starting at \(scratchStart)")

        let T = batchSize
        let firstRealSlot = T - n

        let embeds = try MLMultiArray(shape: [1, T as NSNumber, hiddenSize as NSNumber],
                                       dataType: .float32)
        let positions = try MLMultiArray(shape: [T as NSNumber], dataType: .int32)
        let mask = try MLMultiArray(shape: [1, 1, T as NSNumber, maxSeqLength as NSNumber],
                                     dataType: .float32)

        // Positions: scratch slots fill 0..firstRealSlot-1, real fills firstRealSlot..T-1
        for i in 0..<firstRealSlot {
            positions[i] = NSNumber(value: Int32(scratchStart + i))
        }
        for i in 0..<n {
            positions[firstRealSlot + i] = NSNumber(value: Int32(currentPosition + i))
        }

        // Mask: shape [1, 1, T, MAX_SEQ]
        let mptr = mask.dataPointer.assumingMemoryBound(to: Float.self)
        let rowSize = maxSeqLength
        let scratchStartLocal = scratchStart
        for t in 0..<T {
            let rowBase = t * rowSize
            if t < firstRealSlot {
                for j in 0..<rowSize {
                    mptr[rowBase + j] = -1e4
                }
            } else {
                let realIdxInChunk = t - firstRealSlot
                let absPosition = currentPosition + realIdxInChunk
                for j in 0..<rowSize {
                    if j <= absPosition && j < scratchStartLocal {
                        mptr[rowBase + j] = 0
                    } else {
                        mptr[rowBase + j] = -1e4
                    }
                }
            }
        }

        // Embeddings: zero the scratch rows + populate the real rows.
        let eptr = embeds.dataPointer.assumingMemoryBound(to: Float.self)
        for s in 0..<firstRealSlot {
            for j in 0..<hiddenSize { eptr[s * hiddenSize + j] = 0 }
        }
        realEmbeddingsSource(firstRealSlot, eptr)

        currentPosition += n
        return (embeds, positions, mask)
    }

    /// Dispatch part1 then part2 over the given input buffers. Returns
    /// the part2 logits MLMultiArray (shape ``[1, 1, vocab]`` — the last
    /// real position's next-token distribution).
    private func runParts(embeds: MLMultiArray, positions: MLMultiArray, mask: MLMultiArray) throws -> MLMultiArray {
        let p1Input = try MLDictionaryFeatureProvider(dictionary: [
            "input_embeds": MLFeatureValue(multiArray: embeds),
            "positions": MLFeatureValue(multiArray: positions),
            "attention_mask": MLFeatureValue(multiArray: mask),
        ])
        let p1Out = try decoderPart1Model.prediction(from: p1Input, using: part1State)
        guard let hidden = p1Out.featureValue(for: "hidden_state")?.multiArrayValue else {
            throw AudioModelError.inferenceFailed(
                operation: "CoreML decoder part1",
                reason: "Missing hidden_state output")
        }

        let p2Input = try MLDictionaryFeatureProvider(dictionary: [
            "input_embeds": MLFeatureValue(multiArray: hidden),
            "positions": MLFeatureValue(multiArray: positions),
            "attention_mask": MLFeatureValue(multiArray: mask),
        ])
        let p2Out = try decoderPart2Model.prediction(from: p2Input, using: part2State)
        guard let logits = p2Out.featureValue(for: "logits")?.multiArrayValue else {
            throw AudioModelError.inferenceFailed(
                operation: "CoreML decoder part2",
                reason: "Missing logits output")
        }
        return logits
    }

    /// Expose the fixed batch size so callers can chunk audio prefill.
    public var prefillBatchSize: Int { batchSize }

    /// Get argmax token ID, optionally excluding a single token.
    ///
    /// When ``skipToken`` is non-nil, the slot at that index is ignored —
    /// used by the ASR generation loop to suppress premature ``<|im_end|>``
    /// (see `CoreMLASRModel.transcribe`). When it's nil this is the plain
    /// argmax over the full vocab.
    public func argmax(logits: MLMultiArray, skipping skipToken: Int32? = nil) -> Int32 {
        return argmaxImpl(logits: logits, skipIdx: skipToken.map { Int($0) })
    }

    /// Read a single logit by token index. Stride-aware (matches `argmax`).
    public func logit(_ logits: MLMultiArray, at index: Int32) -> Float {
        let lastStride = logits.strides.last?.intValue ?? 1
        let i = Int(index) * lastStride
        switch logits.dataType {
        case .float16:
            let ptr = logits.dataPointer.assumingMemoryBound(to: Float16.self)
            return Float(ptr[i])
        case .float32:
            let ptr = logits.dataPointer.assumingMemoryBound(to: Float.self)
            return ptr[i]
        default:
            let ptr = logits.dataPointer.assumingMemoryBound(to: Float.self)
            return ptr[i]
        }
    }

    /// Get argmax token ID from logits.
    ///
    /// Stride-aware: walks ``vocabSize`` (the logical last-dim length)
    /// using ``strides.last`` as the step, correct for CoreML outputs
    /// that may be strided (e.g. ANE padding). NaN-safe — NaN values
    /// are skipped, so one bad logit can't poison the argmax (the
    /// previous flat ``ptr[i]`` loop with ``maxVal = -Float.infinity``
    /// would silently keep ``maxIdx = 0`` since IEEE-754 ``NaN > x``
    /// is always false).
    private func argmaxImpl(logits: MLMultiArray, skipIdx: Int?) -> Int32 {
        let vocab = logits.shape.last?.intValue ?? logits.count
        let lastStride = logits.strides.last?.intValue ?? 1
        var maxVal: Float = -Float.infinity
        var maxIdx: Int32 = 0
        var nanCount: Int = 0

        switch logits.dataType {
        case .float16:
            let ptr = logits.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<vocab where i != skipIdx {
                let val = Float(ptr[i * lastStride])
                if val.isNaN { nanCount += 1; continue }
                if val > maxVal {
                    maxVal = val
                    maxIdx = Int32(i)
                }
            }
        case .float32:
            let ptr = logits.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<vocab where i != skipIdx {
                let val = ptr[i * lastStride]
                if val.isNaN { nanCount += 1; continue }
                if val > maxVal {
                    maxVal = val
                    maxIdx = Int32(i)
                }
            }
        default:
            let ptr = logits.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<vocab where i != skipIdx {
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
    public func audioEmbeddingToMultiArray(_ embedding: MLXArray, at index: Int) throws -> MLMultiArray {
        let hidden = embedding.dim(2)
        let result = try MLMultiArray(shape: [1, 1, hidden as NSNumber], dataType: .float32)
        let ptr = result.dataPointer.assumingMemoryBound(to: Float.self)
        let slice = embedding[0..., index..<(index + 1), 0...]
        let data: [Float] = slice.asArray(Float.self)
        for i in 0..<hidden {
            ptr[i] = data[i]
        }
        return result
    }

    /// Extract audio embedding at index from MLMultiArray (no MLX dependency).
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
        let compiled = directory.appendingPathComponent("\(name).mlmodelc", isDirectory: true)
        if FileManager.default.fileExists(atPath: compiled.path) {
            return compiled
        }
        return nil
    }
}
#endif
