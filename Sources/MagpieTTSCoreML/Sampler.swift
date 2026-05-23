// Adapted from FluidInference/FluidAudio (Apache-2.0)
// https://github.com/FluidInference/FluidAudio/blob/main/Sources/FluidAudio/TTS/Magpie/LocalTransformer/MagpieSampler.swift
//
// 8-codebook AR sampler driven by the Swift LocalTransformer. Top-k +
// temperature + (optional) classifier-free guidance; weighted choice via
// NumPy-compatible MT19937 so seeded runs reproduce the upstream output.

import Foundation

public final class MagpieCoreMLSamplerRng {
    private let mt: MagpieCoreMLMT19937

    public init(seed: UInt64?) {
        if let seed = seed {
            self.mt = MagpieCoreMLMT19937(seed: UInt32(truncatingIfNeeded: seed))
        } else {
            var bytes: UInt32 = 0
            withUnsafeMutableBytes(of: &bytes) { buf in
                if let base = buf.baseAddress { arc4random_buf(base, buf.count) }
            }
            self.mt = MagpieCoreMLMT19937(seed: bytes)
        }
    }

    public func numpyChoice(probs: [Double]) -> Int {
        mt.numpyChoice(probs: probs)
    }
}

public struct MagpieCoreMLSampler: Sendable {
    private let lt: MagpieCoreMLLocalTransformer
    private let audioEmbeddings: [[Float]]  // 8 × (numCodes × dModel) row-major

    public init(localTransformer: MagpieCoreMLLocalTransformer,
                audioEmbeddings: [[Float]]) {
        self.lt = localTransformer
        self.audioEmbeddings = audioEmbeddings
    }

    /// Sample one frame's 8 codebook tokens given the decoder hidden state.
    ///
    /// - Parameters:
    ///   - decoderHidden: `[dModel]` conditional hidden (from `decoder_step`).
    ///   - uncondDecoderHidden: `[dModel]` unconditional hidden for CFG. Pass
    ///     `nil` (or use cfgScale == 1.0) to disable CFG and skip the second
    ///     LocalTransformer pass per codebook.
    ///   - forbidEos: when `true`, mask `audioEosId` (avoid early termination).
    ///   - params: temperature / topK / cfgScale.
    ///   - rng: seeded RNG (for reproducibility) or unseeded.
    public func sample(
        decoderHidden: [Float],
        uncondDecoderHidden: [Float]? = nil,
        forbidEos: Bool,
        params: MagpieCoreMLParams,
        rng: MagpieCoreMLSamplerRng
    ) -> [Int32] {
        let numCodebooks = lt.weights.numCodebooks
        let D = lt.weights.localDim
        let useCfg = uncondDecoderHidden != nil && params.cfgScale != 1.0

        // Initial LT input = inProj(decoder_hidden).
        var condSeq = lt.projectInput(hidden: decoderHidden)
        var condLen = 1

        var uncondSeq: [Float] = []
        var uncondLen = 0
        if let uh = uncondDecoderHidden {
            uncondSeq = lt.projectInput(hidden: uh)
            uncondLen = 1
        }

        var codes = Swift.Array<Int32>(repeating: 0, count: numCodebooks)
        let forbidden = forbiddenTokens(eosMasked: forbidEos)

        for cb in 0..<numCodebooks {
            let condOut = lt.forward(sequence: condSeq, length: condLen)
            let lastOff = (condLen - 1) * D
            let lastHidden = Swift.Array(condOut[lastOff..<(lastOff + D)])
            var logits = lt.codebookLogits(lastHidden: lastHidden, codebook: cb)

            if useCfg {
                let uOut = lt.forward(sequence: uncondSeq, length: uncondLen)
                let uOff = (uncondLen - 1) * D
                let uLast = Swift.Array(uOut[uOff..<(uOff + D)])
                let uLogits = lt.codebookLogits(lastHidden: uLast, codebook: cb)
                let s = params.cfgScale
                for i in 0..<logits.count {
                    logits[i] = s * logits[i] + (1.0 - s) * uLogits[i]
                }
            }

            for tok in forbidden where Int(tok) < logits.count {
                logits[Int(tok)] = -.infinity
            }

            let sampled = Self.sampleTopK(
                logits: logits, topK: params.topK,
                temperature: params.temperature, rng: rng)
            codes[cb] = Int32(sampled)

            // Embed sampled token → LT input for the next codebook.
            let tokenEmb = audioEmbeddings[cb]
            let row = Int(sampled) * lt.weights.dModel
            let hiddenSlice = Swift.Array(tokenEmb[row..<(row + lt.weights.dModel)])
            let nextInput = lt.projectInput(hidden: hiddenSlice)
            condSeq.append(contentsOf: nextInput)
            condLen += 1
            if useCfg {
                uncondSeq.append(contentsOf: nextInput)
                uncondLen += 1
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

    static func sampleTopK(
        logits: [Float], topK: Int, temperature: Float,
        rng: MagpieCoreMLSamplerRng
    ) -> Int {
        var truncated = logits
        if topK > 0 && topK < truncated.count {
            let threshold = topKThreshold(values: truncated, k: topK)
            var aboveCount = 0
            for v in truncated where v > threshold { aboveCount += 1 }
            var tiesNeeded = topK - aboveCount
            for i in 0..<truncated.count {
                if truncated[i] > threshold { continue }
                if truncated[i] == threshold && tiesNeeded > 0 {
                    tiesNeeded -= 1
                    continue
                }
                truncated[i] = -.infinity
            }
        }
        let t = max(temperature, 1e-8)
        for i in 0..<truncated.count { truncated[i] /= t }
        let maxVal = truncated.max() ?? 0
        var sum: Double = 0
        var probs = [Double](repeating: 0, count: truncated.count)
        for i in 0..<truncated.count {
            let e = Double(expf(truncated[i] - maxVal))
            probs[i] = e
            sum += e
        }
        if sum <= 0 || !sum.isFinite {
            return logits.indices.max(by: { logits[$0] < logits[$1] }) ?? 0
        }
        let inv = 1.0 / sum
        for i in 0..<probs.count { probs[i] *= inv }
        return rng.numpyChoice(probs: probs)
    }

    private static func topKThreshold(values: [Float], k: Int) -> Float {
        var heap = Swift.Array<Float>(repeating: 0, count: k)
        heap.withUnsafeMutableBufferPointer { buf in
            for i in 0..<k {
                buf[i] = values[i]
                var j = i
                while j > 0 {
                    let parent = (j - 1) >> 1
                    if buf[j] < buf[parent] {
                        buf.swapAt(j, parent)
                        j = parent
                    } else { break }
                }
            }
            for i in k..<values.count {
                let v = values[i]
                if v <= buf[0] { continue }
                buf[0] = v
                var j = 0
                while true {
                    let left = 2 * j + 1
                    let right = left + 1
                    var smallest = j
                    if left < k && buf[left] < buf[smallest] { smallest = left }
                    if right < k && buf[right] < buf[smallest] { smallest = right }
                    if smallest == j { break }
                    buf.swapAt(j, smallest)
                    j = smallest
                }
            }
        }
        return heap[0]
    }
}
