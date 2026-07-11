#if canImport(CoreML)
import Accelerate
import CoreML
import Foundation
import AudioCommon

/// Forced aligner running entirely on CoreML (Neural Engine + GPU).
///
/// Pipeline (mirrors the MLX ``Qwen3ForcedAligner.align`` path but with no
/// MLX dependency at runtime):
///
///   audio → mel (Accelerate) → audio_encoder.mlpackage → audio embeddings
///   text  → tokenizer → <ts> word <ts> slots
///   tokens → embedding.mlpackage → token embeddings
///   embeddings (token slots + audio slot) → text_decoder.mlpackage (single pass)
///   logits at <ts> positions → argmax → LIS correction → ``[AlignedWord]``
///
/// The text decoder is non-autoregressive (one forward pass per align call)
/// and has no KV cache — input is the full assembled sequence and the
/// causal mask is provided as a model input so we can use a finite ``-1e4``
/// fill value (the upstream fp16 graph cannot survive ``-inf`` softmax).
///
/// Two variants are supported:
///   ``coremlFP16`` — full precision, ~1.0 GB on disk, highest fidelity.
///   ``coremlInt8`` — 8-bit kmeans-palettized, ~570 MB on disk.
public final class CoreMLForcedAligner {
    public let audioEncoder: CoreMLForcedAlignerEncoder
    public let embedding: CoreMLForcedAlignerEmbedding
    public let textDecoder: CoreMLForcedAlignerDecoder
    public let featureExtractor: WhisperFeatureExtractor
    public var tokenizer: Qwen3Tokenizer?

    private let classifyNum: Int
    private let segmentTime: Float

    public init(
        audioEncoder: CoreMLForcedAlignerEncoder,
        embedding: CoreMLForcedAlignerEmbedding,
        textDecoder: CoreMLForcedAlignerDecoder,
        classifyNum: Int = 5000,
        segmentTime: Float = 0.08
    ) {
        self.audioEncoder = audioEncoder
        self.embedding = embedding
        self.textDecoder = textDecoder
        self.featureExtractor = WhisperFeatureExtractor()
        self.classifyNum = classifyNum
        self.segmentTime = segmentTime
    }

    /// Load all three component bundles + tokenizer from a local directory.
    /// Useful for testing a freshly produced converter output before uploading
    /// to HuggingFace.
    public static func fromDirectory(
        _ directory: URL,
        encoderComputeUnits: MLComputeUnits = CoreMLComputeUnitsResolver.resolved(default: .cpuAndNeuralEngine),
        decoderComputeUnits: MLComputeUnits = CoreMLComputeUnitsResolver.resolved(default: .all)
    ) throws -> CoreMLForcedAligner {
        let enc = try CoreMLForcedAlignerEncoder.load(
            from: directory, computeUnits: encoderComputeUnits)
        let emb = try CoreMLForcedAlignerEmbedding.load(from: directory)
        let dec = try CoreMLForcedAlignerDecoder.load(
            from: directory, computeUnits: decoderComputeUnits)

        var classifyNum = 5000
        var segmentTime: Float = 0.08
        let cfgURL = directory.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: cfgURL.path),
           let data = try? Data(contentsOf: cfgURL),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let n = json["classify_num"] as? Int { classifyNum = n }
            if let t = json["timestamp_segment_time"] as? Double { segmentTime = Float(t) }
        }

        let aligner = CoreMLForcedAligner(
            audioEncoder: enc, embedding: emb, textDecoder: dec,
            classifyNum: classifyNum, segmentTime: segmentTime)

        let vocabURL = directory.appendingPathComponent("vocab.json")
        if FileManager.default.fileExists(atPath: vocabURL.path) {
            let tokenizer = Qwen3Tokenizer()
            try tokenizer.load(from: vocabURL)
            aligner.tokenizer = tokenizer
        }
        return aligner
    }

    /// Download all three component bundles + tokenizer files from one HuggingFace repo.
    public static func fromPretrained(
        modelId: String = "aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-FP16",
        encoderComputeUnits: MLComputeUnits = CoreMLComputeUnitsResolver.resolved(default: .cpuAndNeuralEngine),
        decoderComputeUnits: MLComputeUnits = CoreMLComputeUnitsResolver.resolved(default: .all),
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> CoreMLForcedAligner {
        let dir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)

        progressHandler?(0.0, "Downloading CoreML aligner...")
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: dir,
            additionalFiles: [
                "audio_encoder.mlpackage/**",
                "text_decoder.mlpackage/**",
                "audio_encoder.mlmodelc/**",
                "text_decoder.mlmodelc/**",
                "embed_tokens.fp16.bin",
                "config.json",
                "vocab.json",
                "merges.txt",
                "tokenizer_config.json",
            ],
            offlineMode: offlineMode
        ) { fraction in
            progressHandler?(fraction * 0.8, "Downloading CoreML aligner...")
        }

        progressHandler?(0.8, "Loading CoreML models...")
        let enc = try CoreMLForcedAlignerEncoder.load(from: dir, computeUnits: encoderComputeUnits)
        let emb = try CoreMLForcedAlignerEmbedding.load(from: dir)
        let dec = try CoreMLForcedAlignerDecoder.load(from: dir, computeUnits: decoderComputeUnits)

        // Read config.json for classify_num + segment time if available.
        var classifyNum = 5000
        var segmentTime: Float = 0.08
        let cfgURL = dir.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: cfgURL.path),
           let data = try? Data(contentsOf: cfgURL),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let n = json["classify_num"] as? Int { classifyNum = n }
            if let t = json["timestamp_segment_time"] as? Double { segmentTime = Float(t) }
        }

        let aligner = CoreMLForcedAligner(
            audioEncoder: enc,
            embedding: emb,
            textDecoder: dec,
            classifyNum: classifyNum,
            segmentTime: segmentTime
        )

        let vocabURL = dir.appendingPathComponent("vocab.json")
        if FileManager.default.fileExists(atPath: vocabURL.path) {
            let tokenizer = Qwen3Tokenizer()
            try tokenizer.load(from: vocabURL)
            aligner.tokenizer = tokenizer
        }

        progressHandler?(1.0, "Ready")
        return aligner
    }

    public func warmUp() throws {
        try audioEncoder.warmUp()
        try embedding.warmUp()
        try textDecoder.warmUp()
    }

    /// Align text to audio in one CoreML forward pass.
    public func align(
        audio: [Float],
        text: String,
        sampleRate: Int = 16000,
        language: String = "English"
    ) throws -> [AlignedWord] {
        guard let tokenizer = tokenizer else {
            throw AudioModelError.inferenceFailed(
                operation: "CoreML forced aligner", reason: "Tokenizer not loaded")
        }

        let profile = ProcessInfo.processInfo.environment["COREML_ALIGN_PROFILE"] == "1"
        let t0 = profile ? CFAbsoluteTimeGetCurrent() : 0

        // 1. Mel features (pure CPU via Accelerate) → audio encoder.
        let mel = featureExtractor.processRaw(audio, sampleRate: sampleRate)
        let t1 = profile ? CFAbsoluteTimeGetCurrent() : 0
        let encoded = try audioEncoder.encode(
            melData: mel.data, melBins: mel.melBins, timeFrames: mel.timeFrames)
        let t2 = profile ? CFAbsoluteTimeGetCurrent() : 0
        let numAudioTokens = encoded.outputLength
        let audioEmbeds = encoded.embeddings  // MLMultiArray [1, paddedAudioTokens, hidden]

        // 2. Insert <timestamp> slots between words.
        let slotted = TextPreprocessor.prepareForAlignment(
            text: text, tokenizer: tokenizer, language: language)
        guard !slotted.words.isEmpty else { return [] }

        // 3. Build the full prompt token sequence.
        let prefix = buildPrefixTokens()
        let suffix = buildSuffixTokens()
        // Layout: [prefix] + [audioPad × numAudioTokens] + [suffix] + [slotted]
        var inputIds: [Int] = prefix
        let audioStart = inputIds.count
        for _ in 0..<numAudioTokens {
            inputIds.append(Qwen3ASRTokens.audioTokenId)
        }
        inputIds.append(contentsOf: suffix)
        let slottedStart = inputIds.count
        inputIds.append(contentsOf: slotted.tokenIds)

        let seqLen = inputIds.count
        let fixedT = textDecoder.fixedT
        guard seqLen <= fixedT else {
            throw AudioModelError.inferenceFailed(
                operation: "CoreML forced aligner",
                reason: "Prompt length \(seqLen) exceeds the exported fixed T=\(fixedT). "
                + "Trim text or chunk the audio (the 30 s audio encoder takes care of "
                + "audio-side chunking, but a single align() pass still has to fit T tokens).")
        }

        // 4. Embed the token sequence (with audioPad placeholders) via the embedding bundle.
        let hiddenSize = audioEncoder.hiddenSize
        let tokenEmbeds = try embedding.embed(
            tokenIds: inputIds, fixedT: fixedT, hiddenSize: hiddenSize)
        let t3 = profile ? CFAbsoluteTimeGetCurrent() : 0

        // 5. Splice the audio embeddings into the audio_pad slot.
        try Self.spliceAudioEmbeddings(
            into: tokenEmbeds,
            audioEmbeds: audioEmbeds,
            tokenStart: audioStart,
            audioTokenCount: numAudioTokens,
            hiddenSize: hiddenSize
        )
        let t4 = profile ? CFAbsoluteTimeGetCurrent() : 0

        // 6. One forward pass through text_decoder + classify head. The
        // causal mask is baked into the exported graph as a constant.
        let logits = try textDecoder.run(inputsEmbeds: tokenEmbeds)
        let t5 = profile ? CFAbsoluteTimeGetCurrent() : 0

        // 8. Extract argmax at <timestamp> positions.
        let absolutePositions = slotted.timestampPositions.map { $0 + slottedStart }
        let rawIndices = Self.argmaxAtPositions(
            logits: logits, positions: absolutePositions, classifyNum: classifyNum)
        let t6 = profile ? CFAbsoluteTimeGetCurrent() : 0

        // 9. LIS correction + pair into [AlignedWord].
        let corrected = TimestampCorrection.enforceMonotonicity(rawIndices)
        let t7 = profile ? CFAbsoluteTimeGetCurrent() : 0

        if profile {
            let ms = { (a: CFAbsoluteTime, b: CFAbsoluteTime) in (b - a) * 1000 }
            print(String(format: "[COREML-ALIGN-PROFILE] mel=%.1fms encoder=%.1fms embedding=%.1fms splice=%.1fms decoder=%.1fms argmax=%.1fms lis=%.1fms total=%.1fms (audio=%.1fs, %d audioTokens, T=%d)",
                ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4), ms(t4, t5),
                ms(t5, t6), ms(t6, t7), ms(t0, t7),
                Float(audio.count) / Float(sampleRate),
                numAudioTokens, fixedT))
        }

        if ProcessInfo.processInfo.environment["ALIGN_DEBUG"] == "1" {
            print("[coreml-align-debug] seqLen=\(seqLen) fixedT=\(fixedT) "
                + "numAudioTokens=\(numAudioTokens) audioStart=\(audioStart) "
                + "slottedStart=\(slottedStart) "
                + "timestamps=\(absolutePositions.count) words=\(slotted.words.count)")
            print("[coreml-align-debug] raw:       \(rawIndices)")
            print("[coreml-align-debug] corrected: \(corrected)")
        }

        var aligned: [AlignedWord] = []
        aligned.reserveCapacity(slotted.words.count)
        for (i, word) in slotted.words.enumerated() {
            let s = i * 2, e = i * 2 + 1
            guard e < corrected.count else { break }
            let st = Float(corrected[s]) * segmentTime
            let en = Float(corrected[e]) * segmentTime
            aligned.append(AlignedWord(text: word, startTime: st, endTime: max(en, st)))
        }
        return aligned
    }

    // MARK: - Token sequence construction (matches MLX path)

    private func buildPrefixTokens() -> [Int] {
        let imStart = Qwen3ASRTokens.imStartTokenId
        let imEnd = Qwen3ASRTokens.imEndTokenId
        let audioStart = Qwen3ASRTokens.audioStartTokenId
        let newline = 198, systemId = 8948, userId = 872
        return [imStart, systemId, newline, imEnd, newline,
                imStart, userId, newline, audioStart]
    }

    private func buildSuffixTokens() -> [Int] {
        let imStart = Qwen3ASRTokens.imStartTokenId
        let imEnd = Qwen3ASRTokens.imEndTokenId
        let audioEnd = Qwen3ASRTokens.audioEndTokenId
        let newline = 198, assistantId = 77091
        return [audioEnd, imEnd, newline, imStart, assistantId, newline]
    }

    // MARK: - MLMultiArray helpers

    /// Overwrite ``tokenEmbeds[0, tokenStart..tokenStart+audioTokenCount, :]`` with
    /// the corresponding slice of ``audioEmbeds[0, 0..audioTokenCount, :]``.
    static func spliceAudioEmbeddings(
        into tokenEmbeds: MLMultiArray,
        audioEmbeds: MLMultiArray,
        tokenStart: Int,
        audioTokenCount: Int,
        hiddenSize: Int
    ) throws {
        guard audioTokenCount > 0 else { return }

        let tShape = tokenEmbeds.shape.map { $0.intValue }
        let aShape = audioEmbeds.shape.map { $0.intValue }
        guard tShape.count == 3, aShape.count == 3,
              tShape[2] == hiddenSize, aShape[2] == hiddenSize
        else {
            throw AudioModelError.inferenceFailed(
                operation: "CoreML forced aligner",
                reason: "Embedding shape mismatch: token=\(tShape) audio=\(aShape) hidden=\(hiddenSize)")
        }
        guard tokenStart + audioTokenCount <= tShape[1],
              audioTokenCount <= aShape[1]
        else {
            throw AudioModelError.inferenceFailed(
                operation: "CoreML forced aligner",
                reason: "Audio splice out of bounds: tokenStart=\(tokenStart) "
                + "audioTokenCount=\(audioTokenCount) tokenT=\(tShape[1]) audioT=\(aShape[1])")
        }

        // Use the per-axis strides from each array — CoreML pads inner dims
        // for alignment so a tight ``t * hiddenSize`` offset would walk into
        // the wrong row. Both arrays declare contiguous strides for axis 2
        // (= 1) but the per-position stride often differs from ``hiddenSize``.
        let dstStrides = tokenEmbeds.strides.map { $0.intValue }
        let srcStrides = audioEmbeds.strides.map { $0.intValue }
        let dstPosStride = dstStrides[1]
        let srcPosStride = srcStrides[1]
        let dstPtr = tokenEmbeds.dataPointer.assumingMemoryBound(to: Float.self)

        switch audioEmbeds.dataType {
        case .float32:
            // Per-row memcpy. ``dstPosStride`` may be > ``hiddenSize`` because
            // the destination MLMultiArray rounds the inner dim up for
            // alignment; that padding belongs to the dst row and shouldn't be
            // touched.
            let srcPtr = audioEmbeds.dataPointer.assumingMemoryBound(to: Float.self)
            let bytesPerRow = hiddenSize * MemoryLayout<Float>.stride
            for t in 0..<audioTokenCount {
                let dst = dstPtr.advanced(by: (tokenStart + t) * dstPosStride)
                let src = srcPtr.advanced(by: t * srcPosStride)
                memcpy(dst, src, bytesPerRow)
            }
        case .float16:
            // vImage's vImageConvert_Planar16FtoPlanarF unpacks fp16→fp32 on
            // CPU SIMD in one shot. Doing it row by row keeps the call within
            // the destination's row padding without spilling into the next row.
            let srcPtr = audioEmbeds.dataPointer.assumingMemoryBound(to: Float16.self)
            for t in 0..<audioTokenCount {
                let dstRow = dstPtr.advanced(by: (tokenStart + t) * dstPosStride)
                let srcRow = srcPtr.advanced(by: t * srcPosStride)
                var srcBuf = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: srcRow),
                    height: 1,
                    width: vImagePixelCount(hiddenSize),
                    rowBytes: hiddenSize * MemoryLayout<Float16>.stride)
                var dstBuf = vImage_Buffer(
                    data: dstRow,
                    height: 1,
                    width: vImagePixelCount(hiddenSize),
                    rowBytes: hiddenSize * MemoryLayout<Float>.stride)
                _ = vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)
            }
        default:
            throw AudioModelError.inferenceFailed(
                operation: "CoreML forced aligner",
                reason: "Unsupported audio embeddings dtype \(audioEmbeds.dataType.rawValue)")
        }
    }

    static func argmaxAtPositions(
        logits: MLMultiArray, positions: [Int], classifyNum: Int
    ) -> [Int] {
        let shape = logits.shape.map { $0.intValue }
        // Expected [1, T, classifyNum]. Honour the reported strides — CoreML
        // rounds the inner class dimension up for alignment (5000 → 5024
        // floats per row in our case), so a tight pos * classifyNum offset
        // walks off-by-stride and reads garbage.
        guard shape.count == 3 else { return [] }
        let strides = logits.strides.map { $0.intValue }
        let posStride = strides[1]
        let classStride = strides[2]
        var out: [Int] = []
        out.reserveCapacity(positions.count)
        switch logits.dataType {
        case .float32:
            let p = logits.dataPointer.assumingMemoryBound(to: Float.self)
            for pos in positions {
                let base = pos * posStride
                var best = -Float.greatestFiniteMagnitude
                var bestIdx = 0
                for c in 0..<classifyNum {
                    let v = p[base + c * classStride]
                    if v > best { best = v; bestIdx = c }
                }
                out.append(bestIdx)
            }
        case .float16:
            let p = logits.dataPointer.assumingMemoryBound(to: Float16.self)
            for pos in positions {
                let base = pos * posStride
                var best = -Float.greatestFiniteMagnitude
                var bestIdx = 0
                for c in 0..<classifyNum {
                    let v = Float(p[base + c * classStride])
                    if v > best { best = v; bestIdx = c }
                }
                out.append(bestIdx)
            }
        default:
            return Array(repeating: 0, count: positions.count)
        }
        return out
    }
}

// MARK: - Audio encoder bundle

public final class CoreMLForcedAlignerEncoder {
    public static let paddedMelLength = 3000
    public static let paddedAudioTokens = 390

    public let hiddenSize: Int
    private let model: MLModel

    public init(model: MLModel, hiddenSize: Int = 1024) {
        self.model = model
        self.hiddenSize = hiddenSize
    }

    public struct EncodedAudio {
        public let embeddings: MLMultiArray
        public let outputLength: Int
    }

    public static func load(
        from directory: URL,
        computeUnits: MLComputeUnits = CoreMLComputeUnitsResolver.resolved(default: .cpuAndNeuralEngine)
    ) throws -> CoreMLForcedAlignerEncoder {
        let url = try Self.resolveURL(in: directory, name: "audio_encoder")
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        let model = try MLModel(contentsOf: url, configuration: cfg)
        return CoreMLForcedAlignerEncoder(model: model)
    }

    public func warmUp() throws {
        _ = try encode(
            melData: [Float](repeating: 0, count: 128 * Self.paddedMelLength),
            melBins: 128, timeFrames: 100)
    }

    public func encode(melData: [Float], melBins: Int, timeFrames: Int) throws -> EncodedAudio {
        let padded = Self.paddedMelLength
        guard timeFrames <= padded else {
            throw AudioModelError.inferenceFailed(
                operation: "CoreML forced aligner encoder",
                reason: "Audio too long: \(timeFrames) mel frames exceeds fixed \(padded). "
                + "Segment with VAD or process in 30 s windows.")
        }

        let mel = try MLMultiArray(
            shape: [1, melBins as NSNumber, padded as NSNumber], dataType: .float32)
        let mp = mel.dataPointer.assumingMemoryBound(to: Float.self)
        for bin in 0..<melBins {
            let src = bin * timeFrames
            let dst = bin * padded
            for t in 0..<timeFrames { mp[dst + t] = melData[src + t] }
            for t in timeFrames..<padded { mp[dst + t] = 0 }
        }
        let len = try MLMultiArray(shape: [1], dataType: .int32)
        len[0] = NSNumber(value: Int32(timeFrames))

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: mel),
            "mel_length": MLFeatureValue(multiArray: len),
        ])
        let output = try model.prediction(from: input)

        guard let emb = output.featureValue(for: "audio_embeddings")?.multiArrayValue,
              let lout = output.featureValue(for: "output_length")?.multiArrayValue
        else {
            throw AudioModelError.inferenceFailed(
                operation: "CoreML forced aligner encoder",
                reason: "Missing audio_embeddings or output_length output")
        }
        let outLen = max(0, Int(lout[0].int32Value))
        return EncodedAudio(embeddings: emb, outputLength: outLen)
    }

    static func resolveURL(in directory: URL, name: String) throws -> URL {
        let compiled = directory.appendingPathComponent("\(name).mlmodelc", isDirectory: true)
        let pkg = directory.appendingPathComponent("\(name).mlpackage", isDirectory: true)
        if FileManager.default.fileExists(atPath: compiled.path) { return compiled }
        if FileManager.default.fileExists(atPath: pkg.path) { return pkg }
        throw AudioModelError.modelLoadFailed(
            modelId: name,
            reason: "Neither \(compiled.path) nor \(pkg.path) exists")
    }
}

// MARK: - Embedding table (raw fp16 gather)

/// Token embedding lookup that bypasses CoreML entirely.
///
/// The converter writes the embed_tokens weight as a raw little-endian fp16
/// blob of shape ``[vocab_size, hidden_size]`` (≈ 304 MB for the
/// 152 064 × 1024 Qwen3 vocabulary). At ``align()`` time we mmap the file
/// and copy + dequantize the requested rows into the output buffer with
/// vImage. A single CoreML embedding mlpackage was previously costing
/// ~70 ms per align call to do the same lookup — pure overhead per
/// profile measurement.
public final class CoreMLForcedAlignerEmbedding {
    /// Filename produced by the converter alongside the .mlmodelc bundles.
    public static let binFilename = "embed_tokens.fp16.bin"

    private let fileURL: URL
    /// Memory-mapped fp16 table, shape ``[vocabSize, hiddenSize]`` row-major.
    private let table: Data
    public let vocabSize: Int
    public let hiddenSize: Int

    public init(fileURL: URL, vocabSize: Int, hiddenSize: Int) throws {
        self.fileURL = fileURL
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        // Memory-map the binary so multiple aligners can share pages and we
        // don't double the RSS by reading the file into RAM.
        self.table = try Data(contentsOf: fileURL, options: [.mappedIfSafe])
        let expected = vocabSize * hiddenSize * MemoryLayout<Float16>.stride
        guard table.count == expected else {
            throw AudioModelError.modelLoadFailed(
                modelId: fileURL.lastPathComponent,
                reason: "Embedding binary is \(table.count) bytes, expected "
                + "\(expected) (\(vocabSize) × \(hiddenSize) × 2)")
        }
    }

    public static func load(
        from directory: URL,
        vocabSize: Int = 152064,
        hiddenSize: Int = 1024
    ) throws -> CoreMLForcedAlignerEmbedding {
        let binURL = directory.appendingPathComponent(binFilename)
        guard FileManager.default.fileExists(atPath: binURL.path) else {
            throw AudioModelError.modelLoadFailed(
                modelId: binFilename,
                reason: "Missing \(binFilename) in \(directory.path). The CoreML "
                + "aligner runtime no longer reads embedding.mlpackage — re-export "
                + "with the latest convert_coreml.py.")
        }
        return try CoreMLForcedAlignerEmbedding(
            fileURL: binURL, vocabSize: vocabSize, hiddenSize: hiddenSize)
    }

    public func warmUp() throws {
        _ = try embed(tokenIds: [0, 1, 2], fixedT: 16, hiddenSize: hiddenSize)
    }

    /// Gather rows for ``tokenIds`` into a fresh float32 ``[1, fixedT, hiddenSize]``
    /// MLMultiArray. Trailing slots beyond ``tokenIds.count`` are filled with
    /// the embedding of token 0 (matches what the old mlpackage did when fed
    /// a zero-padded id sequence).
    public func embed(tokenIds: [Int], fixedT: Int, hiddenSize: Int) throws -> MLMultiArray {
        guard hiddenSize == self.hiddenSize else {
            throw AudioModelError.inferenceFailed(
                operation: "CoreML forced aligner embedding",
                reason: "Hidden size mismatch: requested \(hiddenSize), "
                + "table has \(self.hiddenSize)")
        }
        guard tokenIds.count <= fixedT else {
            throw AudioModelError.inferenceFailed(
                operation: "CoreML forced aligner embedding",
                reason: "Token count \(tokenIds.count) exceeds fixed T \(fixedT)")
        }

        let fp32 = try MLMultiArray(
            shape: [1, fixedT as NSNumber, hiddenSize as NSNumber], dataType: .float32)
        let dstStrides = fp32.strides.map { $0.intValue }
        let dstPosStride = dstStrides[1]
        let dst = fp32.dataPointer.assumingMemoryBound(to: Float.self)

        let rowFp16Bytes = hiddenSize * MemoryLayout<Float16>.stride
        let rowFp32Bytes = hiddenSize * MemoryLayout<Float>.stride

        table.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            guard let basePtr = raw.baseAddress?.assumingMemoryBound(to: Float16.self) else {
                return
            }
            for t in 0..<fixedT {
                let id = t < tokenIds.count ? tokenIds[t] : 0
                // Guard against bad token ids in case the tokenizer ever
                // emits something past the embed_tokens range.
                let safeId = (id >= 0 && id < self.vocabSize) ? id : 0
                let srcRow = basePtr.advanced(by: safeId * hiddenSize)
                let dstRow = dst.advanced(by: t * dstPosStride)
                var srcBuf = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: srcRow),
                    height: 1, width: vImagePixelCount(hiddenSize),
                    rowBytes: rowFp16Bytes)
                var dstBuf = vImage_Buffer(
                    data: dstRow, height: 1, width: vImagePixelCount(hiddenSize),
                    rowBytes: rowFp32Bytes)
                _ = vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)
            }
        }
        return fp32
    }
}

// MARK: - Text decoder + classify head bundle

public final class CoreMLForcedAlignerDecoder {
    private let model: MLModel
    public let fixedT: Int

    public init(model: MLModel, fixedT: Int = 768) {
        self.model = model
        self.fixedT = fixedT
    }

    public static func load(
        from directory: URL,
        computeUnits: MLComputeUnits = CoreMLComputeUnitsResolver.resolved(default: .all)
    ) throws -> CoreMLForcedAlignerDecoder {
        let url = try CoreMLForcedAlignerEncoder.resolveURL(in: directory, name: "text_decoder")
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        let model = try MLModel(contentsOf: url, configuration: cfg)
        // Read fixedT from the model's declared input shape so a runtime
        // override (T=512 / T=1024) doesn't need a separate code path.
        var fixedT = 768
        if let desc = model.modelDescription.inputDescriptionsByName["inputs_embeds"],
           let constraint = desc.multiArrayConstraint {
            let shape = constraint.shape.map { $0.intValue }
            if shape.count == 3 && shape[1] > 0 { fixedT = shape[1] }
        }
        return CoreMLForcedAlignerDecoder(model: model, fixedT: fixedT)
    }

    public func warmUp() throws {
        let hidden = 1024
        let dummy = try MLMultiArray(
            shape: [1, fixedT as NSNumber, hidden as NSNumber], dataType: .float32)
        _ = try run(inputsEmbeds: dummy)
    }

    public func run(inputsEmbeds: MLMultiArray) throws -> MLMultiArray {
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "inputs_embeds": MLFeatureValue(multiArray: inputsEmbeds),
        ])
        let output = try model.prediction(from: input)
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw AudioModelError.inferenceFailed(
                operation: "CoreML forced aligner decoder",
                reason: "Missing logits output")
        }
        return logits
    }
}

#endif
