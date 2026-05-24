import CoreML
import Foundation
import AudioCommon
import MagpieTTS

/// CoreML port of Magpie-TTS Multilingual 357M, backed by the
/// `aufklarer/Magpie-TTS-Multilingual-357M-CoreML-8bit` bundle.
///
/// Pipeline per synthesis call:
///   1. Per-language tokenizer (reused from MagpieTTS) → shared 2360-token vocab IDs.
///   2. `text_encoder.mlmodelc` runs once over the prompt.
///   3. `decoder_prefill.mlmodelc` selects the baked speaker by index,
///      seeds the 12-layer self-attention KV cache (110 + 1 BOS = 111
///      positions), and emits the precomputed cross-attention K/V.
///   4. AR loop: ``MagpieCoreMLLocalTransformer`` (pure-Swift Accelerate)
///      samples the 8 codebook tokens per frame from `decoder_step`'s
///      `h_last`. Audio embeddings are averaged in Swift to build the
///      next `audio_emb` input. **No MLX dispatch per frame** — MLX is
///      only used once at init to snapshot the LT weights + audio
///      embedding tables out of the MLX MagpieTTS module.
///   5. `nanocodec_decoder.mlmodelc` decodes the (T, 8) code matrix in
///      64-frame windows to 22.05 kHz mono PCM.
///
/// Streaming is **not** supported because the codec is traced at a fixed
/// 64-frame window. ``synthesize(...)`` returns the complete waveform.
public final class MagpieTTSCoreML {

    public let textEncoder: MagpieCoreMLTextEncoder
    public let decoder: MagpieCoreMLDecoder
    /// Stateful KV cache variant of decoder_step. Cache lives as
    /// CoreML state (fp16, ANE-resident) instead of being passed as
    /// inputs+outputs each step — saves ~85 MB of per-step IO transfer.
    /// Default path; set ``useStatefulDecoder = false`` to bisect
    /// against the cache-as-IO model.
    public let statefulDecoder: MagpieCoreMLStatefulDecoderStep
    /// 64-frame batch codec, used by ``synthesize(...)``. One call per
    /// ~3 s of audio; lowest dispatch overhead per second of audio.
    public let nanoCodec: MagpieCoreMLNanoCodec
    /// 8-frame streaming codec, used by ``synthesizeStream(...)``. Higher
    /// per-frame overhead but ~370 ms first-packet latency.
    public let nanoCodecStreaming: MagpieCoreMLNanoCodec

    /// Toggle between the two `decoder_step` variants. Reads
    /// `MAGPIE_COREML_STATEFUL=0` to opt OUT (bisecting / A/B). Default
    /// uses the stateful path.
    public var useStatefulDecoder: Bool =
        ProcessInfo.processInfo.environment["MAGPIE_COREML_STATEFUL"] != "0"

    /// Pure-Swift LocalTransformer + sampler. Weights were extracted from
    /// the MLX MagpieTTS module once at ``fromPretrained`` time.
    public let localTransformer: MagpieCoreMLLocalTransformer
    /// 8 × `[2024 * 768]` Float32 — averaged per frame for the next
    /// `audio_emb` input. Extracted from the MLX bundle alongside LT.
    public let audioEmbeddings: [[Float]]

    private let tokenizerDir: URL
    private var tokenizers: [MagpieCoreMLLanguage: MagpieTokenizer] = [:]
    private let tokenizerLock = NSLock()

    public static let sampleRate = MagpieCoreMLConstants.sampleRate

    public init(textEncoder: MagpieCoreMLTextEncoder,
                decoder: MagpieCoreMLDecoder,
                statefulDecoder: MagpieCoreMLStatefulDecoderStep,
                nanoCodec: MagpieCoreMLNanoCodec,
                nanoCodecStreaming: MagpieCoreMLNanoCodec,
                localTransformer: MagpieCoreMLLocalTransformer,
                audioEmbeddings: [[Float]],
                tokenizerDir: URL) {
        self.textEncoder = textEncoder
        self.decoder = decoder
        self.statefulDecoder = statefulDecoder
        self.nanoCodec = nanoCodec
        self.nanoCodecStreaming = nanoCodecStreaming
        self.localTransformer = localTransformer
        self.audioEmbeddings = audioEmbeddings
        self.tokenizerDir = tokenizerDir
    }

    // MARK: - Loading

    public static func fromPretrained(
        progressHandler: ((Double) -> Void)? = nil
    ) async throws -> MagpieTTSCoreML {
        let debug = ProcessInfo.processInfo.environment["MAGPIE_COREML_PROFILE"] == "1"
        func tick(_ label: String, _ t0: CFAbsoluteTime) {
            if debug {
                let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                let padded = label.padding(toLength: 24, withPad: " ", startingAt: 0)
                print("[LOAD] \(padded) \(String(format: "%6.0f", ms)) ms")
            }
        }

        let tDownload = CFAbsoluteTimeGetCurrent()
        let paths = try await MagpieCoreMLDownloader.ensureDownloaded(
            progressHandler: progressHandler)
        tick("ensureDownloaded", tDownload)

        let tTE = CFAbsoluteTimeGetCurrent()
        let te = try MagpieCoreMLTextEncoder(url: paths.textEncoderCompiled)
        tick("load text_encoder", tTE)

        let tDec = CFAbsoluteTimeGetCurrent()
        let dec = try MagpieCoreMLDecoder(
            prefillURL: paths.decoderPrefillCompiled,
            stepURL: paths.decoderStepCompiled)
        tick("load decoder pre+step", tDec)

        let tStateful = CFAbsoluteTimeGetCurrent()
        let statefulDec = try MagpieCoreMLStatefulDecoderStep(
            url: paths.decoderStepStatefulCompiled)
        tick("load decoder stateful", tStateful)

        let tCodec = CFAbsoluteTimeGetCurrent()
        let codec = try MagpieCoreMLNanoCodec(
            url: paths.nanocodecCompiled,
            windowFrames: MagpieCoreMLConstants.nanocodecBatchFrames)
        tick("load nanocodec", tCodec)

        let tStreamCodec = CFAbsoluteTimeGetCurrent()
        let codecStream = try MagpieCoreMLNanoCodec(
            url: paths.nanocodecStreamingCompiled,
            windowFrames: MagpieCoreMLConstants.nanocodecStreamFrames)
        tick("load nanocodec stream", tStreamCodec)

        let cacheURL = paths.bundleRoot.appendingPathComponent("extracted_weights.bin")
        let ltWeights: MagpieCoreMLLocalTransformerWeights
        let audioEmbeds: [[Float]]
        let tCache = CFAbsoluteTimeGetCurrent()
        if let cached = MagpieCoreMLWeightCache.load(from: cacheURL) {
            ltWeights = cached.lt
            audioEmbeds = cached.audioEmbeds
            tick("load weight cache", tCache)
        } else {
            tick("cache MISS", tCache)
            let tMLX = CFAbsoluteTimeGetCurrent()
            let mlxModel = try await MagpieTTS.fromPretrained(variant: .int4)
            tick("MLX MagpieTTS load", tMLX)
            let tExtract = CFAbsoluteTimeGetCurrent()
            (ltWeights, audioEmbeds) = try MagpieCoreMLWeightExtractor.extract(from: mlxModel)
            tick("extract weights", tExtract)
            let tSave = CFAbsoluteTimeGetCurrent()
            try? MagpieCoreMLWeightCache.save(to: cacheURL,
                                               lt: ltWeights,
                                               audioEmbeds: audioEmbeds)
            tick("save weight cache", tSave)
        }
        let lt = MagpieCoreMLLocalTransformer(weights: ltWeights)
        // Pre-warm is OFF by default. It pays the ANE JIT/binding cost at
        // load time (~3.8 s) instead of during the first decoder_step
        // call (~2 s spike on the first frame). For a CLI doing one
        // synthesis per process the cost stays the same — slightly worse,
        // even — so we skip it. Long-lived apps that synthesize
        // repeatedly should call `model.decoder.prewarm()` explicitly
        // after `fromPretrained`.
        return MagpieTTSCoreML(
            textEncoder: te, decoder: dec, statefulDecoder: statefulDec,
            nanoCodec: codec, nanoCodecStreaming: codecStream,
            localTransformer: lt,
            audioEmbeddings: audioEmbeds,
            tokenizerDir: paths.mlxTokenizerDir)
    }

    // MARK: - Tokenizer access

    public func tokenizer(for language: MagpieCoreMLLanguage) throws -> MagpieTokenizer {
        tokenizerLock.lock()
        defer { tokenizerLock.unlock() }
        if let cached = tokenizers[language] { return cached }
        let url = tokenizerDir.appendingPathComponent("\(language.rawValue).json")
        let tok = try MagpieTokenizer.load(from: url, language: language.mlx)
        tokenizers[language] = tok
        return tok
    }

    // MARK: - Synthesis

    public func synthesize(text: String,
                            speaker: MagpieCoreMLSpeaker = .sofia,
                            language: MagpieCoreMLLanguage = .english,
                            prephonemized: Bool = false,
                            params: MagpieCoreMLParams = MagpieCoreMLParams()) throws -> [Float] {
        let tok = try tokenizer(for: language)
        let ids = tok.tokenize(text, prephonemized: prephonemized)
        if ids.isEmpty {
            throw MagpieCoreMLError.inferenceFailed(
                stage: "tokenize", underlying: "empty token sequence")
        }
        let encOut = try textEncoder.encode(tokens: ids.map(Int32.init))
        let prefill = try decoder.prefill(
            speakerIndex: speaker.rawValue,
            encoderOutput: encOut.encoderOutput,
            encoderMask: encOut.encoderMask)

        var rng: any RandomNumberGenerator =
            params.seed.map { SeededRng(seed: $0) as any RandomNumberGenerator }
                ?? SystemRandomNumberGenerator()

        if useStatefulDecoder {
            try statefulDecoder.resetState(
                prefillSaK: prefill.saK, prefillSaV: prefill.saV)
        }

        var saK = prefill.saK
        var saV = prefill.saV
        let xaK = prefill.xaK
        let xaV = prefill.xaV
        var position = prefill.initialPosition
        var audioEmb = averageBosEmbedding()
        var frames: [[Int32]] = []
        let stepCap = min(params.maxSteps, MagpieCoreMLConstants.maxARSteps)

        let debug = ProcessInfo.processInfo.environment["MAGPIE_COREML_PROFILE"] == "1"
        var totalStep: Double = 0
        var totalSample: Double = 0
        var totalEmbed: Double = 0

        for step in 0..<stepCap {
            let t0 = debug ? CFAbsoluteTimeGetCurrent() : 0
            let hLastArr: MLMultiArray
            if useStatefulDecoder {
                let (_, h) = try statefulDecoder.step(
                    audioEmbedding: audioEmb, position: position,
                    encoderOutput: encOut.encoderOutput,
                    encoderMask: encOut.encoderMask,
                    xaK: xaK, xaV: xaV)
                hLastArr = h
            } else {
                let stepOut = try decoder.step(
                    audioEmbedding: audioEmb, position: position,
                    encoderOutput: encOut.encoderOutput,
                    encoderMask: encOut.encoderMask,
                    saK: saK, saV: saV, xaK: xaK, xaV: xaV)
                saK = stepOut.saK
                saV = stepOut.saV
                hLastArr = stepOut.hLast
            }
            if debug { totalStep += CFAbsoluteTimeGetCurrent() - t0 }
            position += 1
            let t1 = debug ? CFAbsoluteTimeGetCurrent() : 0
            let hLastFlat = MagpieCoreMLBridge.toFloat32(hLastArr)
            let codes = sampleFromLocalTransformer(
                hLastFlat: hLastFlat,
                forbidEos: step < params.minFrames,
                temperature: params.temperature, topK: params.topK,
                rng: &rng)
            if debug { totalSample += CFAbsoluteTimeGetCurrent() - t1 }
            if step >= params.minFrames, codes.contains(MagpieCoreMLConstants.audioEosId) {
                break
            }
            frames.append(codes)
            let t2 = debug ? CFAbsoluteTimeGetCurrent() : 0
            audioEmb = averageEmbedding(codes: codes)
            if debug { totalEmbed += CFAbsoluteTimeGetCurrent() - t2 }
        }
        let tCodec0 = debug ? CFAbsoluteTimeGetCurrent() : 0
        let audio = try nanoCodec.decode(codes: frames)
        if debug {
            let codecTime = CFAbsoluteTimeGetCurrent() - tCodec0
            let n = max(frames.count, 1)
            print("[MAGPIE-COREML-PROFILE] \(frames.count) frames  (decoder=\(useStatefulDecoder ? "stateful" : "cache-IO"))")
            print(String(format: "  decoder_step: %6.0f ms total  %5.2f ms/frame", totalStep * 1000, totalStep * 1000 / Double(n)))
            print(String(format: "  LT+sample:    %6.0f ms total  %5.2f ms/frame", totalSample * 1000, totalSample * 1000 / Double(n)))
            print(String(format: "  audio_emb:    %6.0f ms total  %5.2f ms/frame", totalEmbed * 1000, totalEmbed * 1000 / Double(n)))
            print(String(format: "  nanocodec:    %6.0f ms (one call)", codecTime * 1000))
            if ProcessInfo.processInfo.environment["MAGPIE_COREML_PROFILE_STEP"] == "1" {
                print(String(format: "  └─ step IO setup:  %6.0f ms  (%5.2f ms/frame)",
                              MagpieCoreMLDecoder.setupAccum * 1000,
                              MagpieCoreMLDecoder.setupAccum * 1000 / Double(n)))
                print(String(format: "  └─ step predict:   %6.0f ms  (%5.2f ms/frame)",
                              MagpieCoreMLDecoder.predAccum * 1000,
                              MagpieCoreMLDecoder.predAccum * 1000 / Double(n)))
                print(String(format: "  └─ step extract:   %6.0f ms  (%5.2f ms/frame)",
                              MagpieCoreMLDecoder.extractAccum * 1000,
                              MagpieCoreMLDecoder.extractAccum * 1000 / Double(n)))
            }
        }
        return audio
    }

    // MARK: - Pre-warm (optional, recommended for streaming apps)

    /// Pre-warm all CoreML models on the ANE so the first real synthesis
    /// call doesn't pay the JIT/compile cost. Adds ~4–5 s to load time
    /// but makes subsequent first-packet latency on streaming drop from
    /// several seconds (cold compile) to ~100–200 ms (warm).
    ///
    /// Worth calling for:
    /// - Streaming apps that synthesize repeatedly in one process
    /// - Voice agents that need low first-packet latency on every turn
    ///
    /// **Not** worth calling for:
    /// - One-shot CLI invocations (the pre-warm cost is paid every time;
    ///   total wall time is the same or slightly worse)
    public func prewarm() {
        decoder.prewarm()
        nanoCodec.prewarm()
        nanoCodecStreaming.prewarm()
    }

    // MARK: - Streaming synthesis

    /// Streaming variant of ``synthesize``. Emits an
    /// ``AudioChunk`` every `framesPerChunk` codec frames (default 8 —
    /// the streaming codec's window size). First-packet latency is
    /// roughly `framesPerChunk × decoder_step_ms + one codec call`,
    /// which on M4 Pro is ~150–250 ms after model load.
    public func synthesizeStream(
        text: String,
        speaker: MagpieCoreMLSpeaker = .sofia,
        language: MagpieCoreMLLanguage = .english,
        prephonemized: Bool = false,
        params: MagpieCoreMLParams = MagpieCoreMLParams(),
        framesPerChunk: Int = MagpieCoreMLConstants.nanocodecStreamFrames
    ) -> AsyncThrowingStream<AudioChunk, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { [weak self] in
                guard let self else {
                    continuation.finish()
                    return
                }
                do {
                    try self.runStreaming(
                        text: text, speaker: speaker, language: language,
                        prephonemized: prephonemized, params: params,
                        framesPerChunk: framesPerChunk,
                        continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    private func runStreaming(
        text: String,
        speaker: MagpieCoreMLSpeaker,
        language: MagpieCoreMLLanguage,
        prephonemized: Bool,
        params: MagpieCoreMLParams,
        framesPerChunk: Int,
        continuation: AsyncThrowingStream<AudioChunk, Error>.Continuation
    ) throws {
        let chunkSize = min(framesPerChunk, nanoCodecStreaming.windowFrames)
        precondition(chunkSize > 0, "framesPerChunk must be > 0")
        let tok = try tokenizer(for: language)
        let ids = tok.tokenize(text, prephonemized: prephonemized)
        if ids.isEmpty { return }
        let encOut = try textEncoder.encode(tokens: ids.map(Int32.init))
        let prefill = try decoder.prefill(
            speakerIndex: speaker.rawValue,
            encoderOutput: encOut.encoderOutput,
            encoderMask: encOut.encoderMask)

        var rng: any RandomNumberGenerator =
            params.seed.map { SeededRng(seed: $0) as any RandomNumberGenerator }
                ?? SystemRandomNumberGenerator()

        var saK = prefill.saK
        var saV = prefill.saV
        let xaK = prefill.xaK
        let xaV = prefill.xaV
        var position = prefill.initialPosition
        var audioEmb = averageBosEmbedding()
        let stepCap = min(params.maxSteps, MagpieCoreMLConstants.maxARSteps)

        let t0 = CFAbsoluteTimeGetCurrent()
        var pendingFrames: [[Int32]] = []
        pendingFrames.reserveCapacity(chunkSize)
        var totalFrames = 0
        var stoppedOnEos = false

        for step in 0..<stepCap {
            let stepOut = try decoder.step(
                audioEmbedding: audioEmb, position: position,
                encoderOutput: encOut.encoderOutput,
                encoderMask: encOut.encoderMask,
                saK: saK, saV: saV, xaK: xaK, xaV: xaV)
            saK = stepOut.saK
            saV = stepOut.saV
            position += 1
            let hLastFlat = MagpieCoreMLBridge.toFloat32(stepOut.hLast)
            let codes = sampleFromLocalTransformer(
                hLastFlat: hLastFlat,
                forbidEos: step < params.minFrames,
                temperature: params.temperature, topK: params.topK,
                rng: &rng)
            if step >= params.minFrames, codes.contains(MagpieCoreMLConstants.audioEosId) {
                stoppedOnEos = true
                break
            }
            pendingFrames.append(codes)
            totalFrames += 1
            audioEmb = averageEmbedding(codes: codes)

            if pendingFrames.count >= chunkSize {
                try emitChunk(&pendingFrames,
                              continuation: continuation,
                              t0: t0,
                              totalFrames: totalFrames,
                              isFinal: false)
            }
        }
        // Final partial chunk if any.
        if !pendingFrames.isEmpty {
            try emitChunk(&pendingFrames,
                          continuation: continuation,
                          t0: t0,
                          totalFrames: totalFrames,
                          isFinal: true)
        } else if stoppedOnEos || totalFrames > 0 {
            continuation.yield(AudioChunk(
                samples: [], sampleRate: Self.sampleRate,
                frameIndex: totalFrames, isFinal: true,
                elapsedTime: CFAbsoluteTimeGetCurrent() - t0))
        }
    }

    private func emitChunk(
        _ pending: inout [[Int32]],
        continuation: AsyncThrowingStream<AudioChunk, Error>.Continuation,
        t0: CFAbsoluteTime,
        totalFrames: Int,
        isFinal: Bool
    ) throws {
        let pcm = try nanoCodecStreaming.decodeWindow(frames: pending[...])
        let validSamples = pending.count * MagpieCoreMLConstants.samplesPerFrame
        let chunkSamples = Array(pcm.prefix(validSamples))
        pending.removeAll(keepingCapacity: true)
        continuation.yield(AudioChunk(
            samples: chunkSamples, sampleRate: Self.sampleRate,
            frameIndex: totalFrames, isFinal: isFinal,
            elapsedTime: CFAbsoluteTimeGetCurrent() - t0))
    }

    // MARK: - LT sampling + audio_emb averaging

    /// AR-loop bootstrap frame: average of the 8 codebooks' BOS rows.
    private func averageBosEmbedding() -> [Float] {
        let codes = [Int32](repeating: MagpieCoreMLConstants.audioBosId,
                            count: MagpieCoreMLConstants.numCodebooks)
        return averageEmbedding(codes: codes)
    }

    /// Mirror of NeMo's `Decoder.embed_audio_frame`: per-codebook embedding
    /// lookup, sum across codebooks, divide by K.
    private func averageEmbedding(codes: [Int32]) -> [Float] {
        let D = MagpieCoreMLConstants.dModel
        let K = MagpieCoreMLConstants.numCodebooks
        var out = [Float](repeating: 0, count: D)
        for k in 0..<K {
            let row = Int(codes[k]) * D
            let table = audioEmbeddings[k]
            // Manual unroll over the row — 768 muls + 768 adds. Profiling
            // showed vDSP_vadd's setup cost dominates at this size.
            for d in 0..<D { out[d] += table[row + d] }
        }
        let inv = 1.0 / Float(K)
        for d in 0..<D { out[d] *= inv }
        return out
    }

    /// Drive the Accelerate LocalTransformer for one frame: 8 codebooks
    /// sampled sequentially with top-k + Gumbel-max.
    private func sampleFromLocalTransformer(
        hLastFlat: [Float],
        forbidEos: Bool,
        temperature: Float,
        topK: Int,
        rng: inout any RandomNumberGenerator
    ) -> [Int32] {
        let K = MagpieCoreMLConstants.numCodebooks
        let D = MagpieCoreMLConstants.dModel
        let LD = MagpieCoreMLConstants.localTransformerDim
        precondition(hLastFlat.count == D)

        // Initial LT input = inProj(h_last).
        var seq = localTransformer.projectInput(hidden: hLastFlat)
        var length = 1

        var codes = [Int32](repeating: 0, count: K)
        let forbidden = forbiddenTokens(eosMasked: forbidEos)

        for cb in 0..<K {
            let out = localTransformer.forward(sequence: seq, length: length)
            let lastOff = (length - 1) * LD
            let lastHidden = Array(out[lastOff..<(lastOff + LD)])
            var logits = localTransformer.codebookLogits(
                lastHidden: lastHidden, codebook: cb)
            for tok in forbidden where Int(tok) < logits.count {
                logits[Int(tok)] = -.infinity
            }
            let sampled = sampleTopK(
                logits: logits, temperature: temperature, topK: topK, rng: &rng)
            codes[cb] = Int32(sampled)

            if cb < K - 1 {
                // Embed the sampled token → project → append to LT input
                // sequence for the next codebook step.
                let row = Int(sampled) * D
                let table = audioEmbeddings[cb]
                let hiddenSlice = Array(table[row..<(row + D)])
                let nextInput = localTransformer.projectInput(hidden: hiddenSlice)
                seq.append(contentsOf: nextInput)
                length += 1
            }
        }
        return codes
    }

    private func forbiddenTokens(eosMasked: Bool) -> [Int32] {
        if eosMasked {
            return [MagpieCoreMLConstants.audioEosId]
                + MagpieCoreMLConstants.forbiddenAudioIds
        } else {
            return MagpieCoreMLConstants.forbiddenAudioIds
        }
    }

    private func sampleTopK(
        logits raw: [Float], temperature: Float, topK: Int,
        rng: inout any RandomNumberGenerator
    ) -> Int {
        if temperature <= 1e-3 {
            return raw.indices.max(by: { raw[$0] < raw[$1] }) ?? 0
        }
        var logits = raw
        if topK > 0 && topK < logits.count {
            let threshold = topKThreshold(values: logits, k: topK)
            for i in 0..<logits.count where logits[i] < threshold {
                logits[i] = -.infinity
            }
        }
        let t = max(temperature, 1e-8)
        var bestVal: Float = -.infinity
        var bestIdx = 0
        for i in 0..<logits.count {
            if !logits[i].isFinite { continue }
            let u = Float(Double.random(in: Double(Float.ulpOfOne)..<1, using: &rng))
            let g = -log(-log(u))
            let s = logits[i] / t + g
            if s > bestVal { bestVal = s; bestIdx = i }
        }
        return bestIdx
    }

    private func topKThreshold(values: [Float], k: Int) -> Float {
        var heap = [Float](repeating: 0, count: k)
        for i in 0..<k {
            heap[i] = values[i]
            var j = i
            while j > 0 {
                let parent = (j - 1) >> 1
                if heap[j] < heap[parent] {
                    heap.swapAt(j, parent)
                    j = parent
                } else { break }
            }
        }
        for i in k..<values.count {
            let v = values[i]
            if v <= heap[0] { continue }
            heap[0] = v
            var j = 0
            while true {
                let left = 2 * j + 1
                let right = left + 1
                var smallest = j
                if left < k && heap[left] < heap[smallest] { smallest = left }
                if right < k && heap[right] < heap[smallest] { smallest = right }
                if smallest == j { break }
                heap.swapAt(j, smallest)
                j = smallest
            }
        }
        return heap[0]
    }
}

/// Splitmix64 — small, well-distributed, reproducible. Used when callers
/// pass `--seed N`.
private struct SeededRng: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { self.state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
