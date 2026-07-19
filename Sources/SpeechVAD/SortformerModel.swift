#if canImport(CoreML)
import CoreML
import Foundation
import AudioCommon

/// CoreML wrapper for the Sortformer streaming diarization model.
///
/// Runs on Neural Engine via CoreML. The model takes a chunk of mel features
/// plus streaming state buffers (spkcache, fifo) and outputs per-frame
/// speaker predictions for up to 4 speakers.
final class SortformerCoreMLModel {

    private let model: MLModel
    let config: SortformerConfig

    /// Input shape constants derived from the CoreML model. `chunkFrames` is
    /// the model's fixed mel-frame input dimension — varies by variant.
    /// chunk: `[1, chunkFrames, 128]`, spkcache: `[1, 188, 512]`, fifo: `[1, fifoLen, 512]`.
    private let chunkFrames: Int
    private let spkcacheFrames: Int
    private let fifoFrames: Int
    private let featureDim: Int

    init(model: MLModel, config: SortformerConfig = .default) {
        self.model = model
        self.config = config
        self.chunkFrames = config.coreMLInputFrames
        self.spkcacheFrames = config.spkcacheLen
        self.fifoFrames = config.fifoLen
        self.featureDim = config.fcDModel
    }

    /// Run one streaming inference step.
    ///
    /// - Parameters:
    ///   - chunk: Mel features for this chunk, flat `[chunkFrames * nMels]` float array.
    ///            If fewer frames available, zero-pad on the right.
    ///   - chunkLength: Actual number of valid mel frames in the chunk
    ///   - spkcache: Speaker cache state, flat `[spkcacheFrames * fcDModel]`
    ///   - spkcacheLength: Number of valid frames in speaker cache
    ///   - fifo: FIFO buffer state, flat `[fifoFrames * fcDModel]`
    ///   - fifoLength: Number of valid frames in FIFO
    /// - Returns: `SortformerOutput` with predictions and updated state
    func predict(
        chunk: [Float],
        chunkLength: Int,
        spkcache: [Float],
        spkcacheLength: Int,
        fifo: [Float],
        fifoLength: Int
    ) throws -> SortformerOutput {
        // Create input arrays
        let chunkArray = try makeMultiArray(
            shape: [1, NSNumber(value: chunkFrames), NSNumber(value: config.nMels)],
            from: chunk)
        let chunkLenArray = try makeScalarInt32Array(value: Int32(chunkLength))

        let spkcacheArray = try makeMultiArray(
            shape: [1, NSNumber(value: spkcacheFrames), NSNumber(value: featureDim)],
            from: spkcache)
        let spkcacheLenArray = try makeScalarInt32Array(value: Int32(spkcacheLength))

        let fifoArray = try makeMultiArray(
            shape: [1, NSNumber(value: fifoFrames), NSNumber(value: featureDim)],
            from: fifo)
        let fifoLenArray = try makeScalarInt32Array(value: Int32(fifoLength))

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "chunk": MLFeatureValue(multiArray: chunkArray),
            "chunk_lengths": MLFeatureValue(multiArray: chunkLenArray),
            "spkcache": MLFeatureValue(multiArray: spkcacheArray),
            "spkcache_lengths": MLFeatureValue(multiArray: spkcacheLenArray),
            "fifo": MLFeatureValue(multiArray: fifoArray),
            "fifo_lengths": MLFeatureValue(multiArray: fifoLenArray),
        ])

        let result = try model.prediction(from: input)

        // Extract outputs
        let predsArray = result.featureValue(for: "speaker_preds_out")!.multiArrayValue!
        let embsArray = result.featureValue(for: "chunk_pre_encoder_embs_out")!.multiArrayValue!
        let embsLenArray = result.featureValue(for: "chunk_pre_encoder_lengths_out")!.multiArrayValue!

        let predsShape = (0..<predsArray.shape.count).map { predsArray.shape[$0].intValue }
        let embsShape = (0..<embsArray.shape.count).map { embsArray.shape[$0].intValue }

        // ANE-backed outputs are not guaranteed to be contiguous Float32:
        // the 8-speaker Ultra head comes back as Float16 with the last
        // dimension padded 8 -> 16 (strides [3872, 16, 1]). Honor the
        // array's declared dtype and strides instead of assuming layout —
        // a raw contiguous read scrambles frames into single-spike noise.
        let preds = Self.denseFloats(from: predsArray)
        let embs = Self.denseFloats(from: embsArray)

        let embsLenPtr = embsLenArray.dataPointer.assumingMemoryBound(to: Int32.self)
        let validEmbFrames = Int(embsLenPtr[0])

        return SortformerOutput(
            speakerPreds: preds,
            predsFrames: predsShape.count >= 2 ? predsShape[predsShape.count - 2] : preds.count / config.maxSpeakers,
            numSpeakers: config.maxSpeakers,
            encoderEmbs: embs,
            encoderEmbFrames: embsShape.count >= 2 ? embsShape[embsShape.count - 2] : validEmbFrames,
            embDim: featureDim,
            validEmbFrames: validEmbFrames
        )
    }

    // MARK: - Helpers

    private func makeMultiArray(shape: [NSNumber], from data: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        let count = min(data.count, array.count)
        for i in 0..<count {
            ptr[i] = data[i]
        }
        // Zero-fill remainder
        for i in count..<array.count {
            ptr[i] = 0
        }
        return array
    }

    /// Copy an MLMultiArray into a dense row-major `[Float]`, honoring the
    /// array's dtype and strides. CoreML chooses both per compute backend.
    static func denseFloats(from array: MLMultiArray) -> [Float] {
        let dims = array.shape.count
        let shape = (0..<dims).map { array.shape[$0].intValue }
        let strides = (0..<dims).map { array.strides[$0].intValue }
        let count = shape.reduce(1, *)
        var out = [Float](repeating: 0, count: count)

        func fill(read: (Int) -> Float) {
            var logical = 0
            var index = [Int](repeating: 0, count: dims)
            while logical < count {
                var offset = 0
                for d in 0..<dims { offset += index[d] * strides[d] }
                out[logical] = read(offset)
                logical += 1
                var d = dims - 1
                while d >= 0 {
                    index[d] += 1
                    if index[d] < shape[d] { break }
                    index[d] = 0
                    d -= 1
                }
            }
        }

        switch array.dataType {
        case .float16:
            let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
            fill { Float(ptr[$0]) }
        case .float32:
            let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
            fill { ptr[$0] }
        case .double:
            let ptr = array.dataPointer.assumingMemoryBound(to: Double.self)
            fill { Float(ptr[$0]) }
        default:
            for i in 0..<count { out[i] = array[i].floatValue }
        }
        return out
    }

    private func makeScalarInt32Array(value: Int32) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1], dataType: .int32)
        let ptr = array.dataPointer.assumingMemoryBound(to: Int32.self)
        ptr[0] = value
        return array
    }
}

/// Output from one Sortformer inference step.
struct SortformerOutput {
    /// Speaker predictions, flat `[predsFrames * numSpeakers]`, sigmoid probabilities
    let speakerPreds: [Float]
    /// Number of prediction frames
    let predsFrames: Int
    /// Number of speaker channels
    let numSpeakers: Int
    /// Pre-encoder embeddings for state update, flat `[encoderEmbFrames * embDim]`
    let encoderEmbs: [Float]
    /// Total encoder embedding frames
    let encoderEmbFrames: Int
    /// Embedding dimension
    let embDim: Int
    /// Number of valid (non-padding) embedding frames
    let validEmbFrames: Int

    /// Get speaker prediction probability at (frame, speaker).
    func pred(frame: Int, speaker: Int) -> Float {
        speakerPreds[frame * numSpeakers + speaker]
    }

    /// Get encoder embedding at (frame, dim).
    func emb(frame: Int, dim: Int) -> Float {
        encoderEmbs[frame * embDim + dim]
    }
}
#endif
