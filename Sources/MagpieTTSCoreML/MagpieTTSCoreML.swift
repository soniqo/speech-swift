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
    public let nanoCodec: MagpieCoreMLNanoCodec

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
                nanoCodec: MagpieCoreMLNanoCodec,
                localTransformer: MagpieCoreMLLocalTransformer,
                audioEmbeddings: [[Float]],
                tokenizerDir: URL) {
        self.textEncoder = textEncoder
        self.decoder = decoder
        self.nanoCodec = nanoCodec
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

        let tCodec = CFAbsoluteTimeGetCurrent()
        let codec = try MagpieCoreMLNanoCodec(url: paths.nanocodecCompiled)
        tick("load nanocodec", tCodec)

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
            textEncoder: te, decoder: dec, nanoCodec: codec,
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
            let stepOut = try decoder.step(
                audioEmbedding: audioEmb, position: position,
                encoderOutput: encOut.encoderOutput,
                encoderMask: encOut.encoderMask,
                saK: saK, saV: saV, xaK: xaK, xaV: xaV)
            if debug { totalStep += CFAbsoluteTimeGetCurrent() - t0 }
            saK = stepOut.saK
            saV = stepOut.saV
            position += 1
            let t1 = debug ? CFAbsoluteTimeGetCurrent() : 0
            let hLastFlat = MagpieCoreMLBridge.toFloat32(stepOut.hLast)
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
            let n = frames.count
            print("[MAGPIE-COREML-PROFILE] \(n) frames")
            print(String(format: "  decoder_step: %6.0f ms total  %5.2f ms/frame", totalStep * 1000, totalStep * 1000 / Double(n)))
            print(String(format: "  LT+sample:    %6.0f ms total  %5.2f ms/frame", totalSample * 1000, totalSample * 1000 / Double(n)))
            print(String(format: "  audio_emb:    %6.0f ms total  %5.2f ms/frame", totalEmbed * 1000, totalEmbed * 1000 / Double(n)))
            print(String(format: "  nanocodec:    %6.0f ms (one call)", codecTime * 1000))
        }
        return audio
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
