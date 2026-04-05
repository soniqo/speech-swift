import Accelerate
import AudioCommon
import CoreML
import Foundation

/// Reusable MLFeatureProvider that avoids dictionary allocation on every CoreML prediction.
class ReusableFeatureProvider: MLFeatureProvider {
    let featureNames: Set<String>
    private var values: [String: MLFeatureValue]

    init(_ dict: [String: MLMultiArray]) {
        self.featureNames = Set(dict.keys)
        self.values = dict.mapValues { MLFeatureValue(multiArray: $0) }
    }

    func featureValue(for name: String) -> MLFeatureValue? { values[name] }

    func update(_ name: String, _ array: MLMultiArray) {
        values[name] = MLFeatureValue(multiArray: array)
    }
}

/// Result of RNNT greedy decoding over a set of encoder frames.
struct RNNTDecodeResult {
    let tokens: [Int]
    let tokenLogProbs: [Float]
    let eouDetected: Bool
}

/// Greedy RNNT decoder for Parakeet EOU streaming ASR.
///
/// Unlike the TDT decoder, RNNT has no duration bins — blank always advances
/// by one encoder frame. EOU is detected when the model emits token ID 1024.
struct RNNTGreedyDecoder {
    let config: ParakeetEOUConfig
    let decoder: MLModel
    let joint: MLModel

    /// Maximum non-blank tokens per encoder frame (safety limit).
    private let maxSymbolsPerStep = 10

    /// Decode encoder output with persistent LSTM state.
    ///
    /// - Parameters:
    ///   - encoded: Encoder output MLMultiArray, shape `[1, encoderHidden, T]` (channels-first)
    ///   - encodedLength: Number of valid encoder frames
    ///   - h: LSTM hidden state (mutated in place)
    ///   - c: LSTM cell state (mutated in place)
    ///   - decoderOutput: Previous decoder output (mutated in place)
    ///   - decoderProvider: Reusable feature provider for decoder
    ///   - jointProvider: Reusable feature provider for joint
    ///   - tokenArray: Pre-allocated token MLMultiArray [1, 1]
    ///   - encSlice: Pre-allocated encoder slice [1, 1, encoderHidden]
    ///   - argmaxBuf: Pre-allocated Float buffer for vDSP argmax
    /// - Returns: Decode result with tokens, log-probs, and EOU flag
    func decode(
        encoded: MLMultiArray,
        encodedLength: Int,
        h: inout MLMultiArray,
        c: inout MLMultiArray,
        decoderOutput: inout MLMultiArray,
        decoderProvider: ReusableFeatureProvider,
        jointProvider: ReusableFeatureProvider,
        tokenArray: MLMultiArray,
        encSlice: MLMultiArray,
        argmaxBuf: UnsafeMutablePointer<Float>
    ) throws -> RNNTDecodeResult {
        var tokens = [Int]()
        var tokenLogProbs = [Float]()
        var eouDetected = false

        let tokenPtr = tokenArray.dataPointer.assumingMemoryBound(to: Int32.self)
        let totalClasses = config.vocabSize + 1  // vocab + blank

        for t in 0..<encodedLength {
            // Extract encoder frame at position t — encoder output is [1, D, T] (channels-first)
            copyEncoderFrame(from: encoded, at: t, to: encSlice)

            for _ in 0..<maxSymbolsPerStep {
                // Joint network: (encoder_slice, decoder_output) → logits
                jointProvider.update("encoder_output", encSlice)
                jointProvider.update("decoder_output", decoderOutput)
                let jointOut = try joint.prediction(from: jointProvider)
                let logits = jointOut.featureValue(for: "logits")!.multiArrayValue!

                let tokenId = argmax(logits, count: totalClasses, floatBuf: argmaxBuf)


                if tokenId == config.blankTokenId {
                    break  // Advance to next encoder frame
                }

                if tokenId == config.eouTokenId {
                    eouDetected = true
                    break
                }

                // Emit token
                tokens.append(tokenId)
                let logProb = logSoftmax(logits, tokenId: tokenId, count: totalClasses, floatBuf: argmaxBuf)
                tokenLogProbs.append(logProb)

                // Update decoder LSTM with emitted token
                tokenPtr.pointee = Int32(tokenId)
                decoderProvider.update("h", h)
                decoderProvider.update("c", c)
                let decOut = try decoder.prediction(from: decoderProvider)
                // Decoder output is [1, D, 1] — transpose to [1, 1, D] for joint
                let rawDecOut = decOut.featureValue(for: "decoder_output")!.multiArrayValue!
                decoderOutput = try transposeDecoder(rawDecOut)
                h = decOut.featureValue(for: "h_out")!.multiArrayValue!
                c = decOut.featureValue(for: "c_out")!.multiArrayValue!
            }

            if eouDetected { break }
        }

        return RNNTDecodeResult(tokens: tokens, tokenLogProbs: tokenLogProbs, eouDetected: eouDetected)
    }

    // MARK: - Array Operations

    /// Copy encoder frame at time `t` from channels-first layout [1, D, T].
    /// Output slice is [1, 1, D] for joint network input.
    private func copyEncoderFrame(from encoded: MLMultiArray, at t: Int, to slice: MLMultiArray) {
        let hidden = config.encoderHidden
        let totalFrames = encoded.shape[2].intValue
        let srcBase = encoded.dataPointer.assumingMemoryBound(to: Float16.self)
        let dst = slice.dataPointer.assumingMemoryBound(to: Float16.self)

        // encoded is [1, D, T] — stride over T dimension for each channel
        for d in 0..<hidden {
            dst[d] = srcBase[d * totalFrames + t]
        }
    }

    private func logSoftmax(_ array: MLMultiArray, tokenId: Int, count: Int, floatBuf: UnsafeMutablePointer<Float>) -> Float {
        let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
        for i in 0..<count { floatBuf[i] = Float(ptr[i]) }

        var maxVal: Float = 0
        var maxIdx: vDSP_Length = 0
        vDSP_maxvi(floatBuf, 1, &maxVal, &maxIdx, vDSP_Length(count))

        var negMax = -maxVal
        vDSP_vsadd(floatBuf, 1, &negMax, floatBuf, 1, vDSP_Length(count))

        var n = Int32(count)
        vvexpf(floatBuf, floatBuf, &n)

        var sumExp: Float = 0
        vDSP_sve(floatBuf, 1, &sumExp, vDSP_Length(count))

        let logSumExp = log(sumExp) + maxVal
        let logit = Float(ptr[tokenId])
        return logit - logSumExp
    }

    /// Transpose decoder output from [1, D, 1] to [1, 1, D].
    private func transposeDecoder(_ array: MLMultiArray) throws -> MLMultiArray {
        let d = array.shape[1].intValue
        // If already [1, 1, D], return as-is
        if array.shape[1].intValue == 1 { return array }
        let result = try MLMultiArray(shape: [1, 1, d as NSNumber], dataType: array.dataType)
        memcpy(result.dataPointer, array.dataPointer, d * MemoryLayout<Float16>.stride)
        return result
    }

    private func argmax(_ array: MLMultiArray, count: Int, floatBuf: UnsafeMutablePointer<Float>) -> Int {
        let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
        for i in 0..<count { floatBuf[i] = Float(ptr[i]) }
        var maxVal: Float = 0
        var maxIdx: vDSP_Length = 0
        vDSP_maxvi(floatBuf, 1, &maxVal, &maxIdx, vDSP_Length(count))
        return Int(maxIdx)
    }
}
