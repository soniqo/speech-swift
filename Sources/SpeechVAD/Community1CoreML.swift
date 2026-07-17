#if canImport(CoreML)
import AudioCommon
import CoreML
import Foundation

/// The two fixed-shape neural stages from the Community-1 bundle.
final class Community1CoreMLModels {
    private let segmentation: MLModel
    private let embedding: MLModel
    private let manifest: Community1BundleManifest

    init(
        directory: URL,
        manifest: Community1BundleManifest,
        computeUnits: MLComputeUnits
    ) throws {
        self.manifest = manifest

        let configuration = MLModelConfiguration()
        configuration.computeUnits = CoreMLComputeUnitsResolver.resolved(default: computeUnits)
        configuration.allowLowPrecisionAccumulationOnGPU = true

        segmentation = try Self.loadModel(
            directory.appendingPathComponent(manifest.segmentation.model, isDirectory: true),
            stage: "segmentation",
            configuration: configuration
        )
        embedding = try Self.loadModel(
            directory.appendingPathComponent(manifest.embedding.model, isDirectory: true),
            stage: "embedding",
            configuration: configuration
        )
    }

    private static func loadModel(
        _ url: URL,
        stage: String,
        configuration: MLModelConfiguration
    ) throws -> MLModel {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw Community1Error.invalidBundle("missing \(stage) model at \(url.path)")
        }
        do {
            return try MLModel(contentsOf: url, configuration: configuration)
        } catch {
            throw Community1Error.invalidBundle(
                "could not load \(stage) model: \(error.localizedDescription)"
            )
        }
    }

    /// Hard powerset decoding from `[1, 589, 7]` logits to three binary tracks.
    func segment(waveform: [Float]) throws -> [[Float]] {
        guard waveform.count == Community1Config.chunkSamples else {
            throw Community1Error.inference(
                stage: "segmentation",
                reason: "expected \(Community1Config.chunkSamples) samples, got \(waveform.count)"
            )
        }

        let input = try Self.multiArray(waveform, shape: [1, 1, Community1Config.chunkSamples])
        let output: MLMultiArray = try autoreleasepool {
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                manifest.segmentation.inputName: MLFeatureValue(multiArray: input)
            ])
            let prediction = try segmentation.prediction(from: provider)
            guard let values = prediction.featureValue(
                for: manifest.segmentation.outputName
            )?.multiArrayValue else {
                throw Community1Error.inference(
                    stage: "segmentation",
                    reason: "missing '\(manifest.segmentation.outputName)' output"
                )
            }
            return values
        }

        let logits = try Self.floatValues(
            output,
            expectedCount: Community1Config.framesPerChunk * 7,
            stage: "segmentation"
        )
        let powerset: [[Int]] = [
            [], [0], [1], [2], [0, 1], [0, 2], [1, 2]
        ]
        var tracks = Array(
            repeating: [Float](repeating: 0, count: Community1Config.localSpeakers),
            count: Community1Config.framesPerChunk
        )
        for frame in 0..<Community1Config.framesPerChunk {
            let offset = frame * 7
            var bestClass = 0
            var bestLogit = logits[offset]
            for candidate in 1..<7 where logits[offset + candidate] > bestLogit {
                bestClass = candidate
                bestLogit = logits[offset + candidate]
            }
            for speaker in powerset[bestClass] {
                tracks[frame][speaker] = 1
            }
        }
        return tracks
    }

    /// Extract the three raw, unnormalized 256-dimensional embeddings for a chunk.
    func embed(waveform: [Float], masks: [[Float]]) throws -> [[Float]] {
        guard waveform.count == Community1Config.chunkSamples else {
            throw Community1Error.inference(
                stage: "embedding",
                reason: "expected \(Community1Config.chunkSamples) samples, got \(waveform.count)"
            )
        }
        guard masks.count == Community1Config.framesPerChunk,
              masks.allSatisfy({ $0.count == Community1Config.localSpeakers }) else {
            throw Community1Error.inference(
                stage: "embedding",
                reason: "expected masks shaped [589, 3]"
            )
        }

        var speakerMajorMasks = [Float]()
        speakerMajorMasks.reserveCapacity(
            Community1Config.localSpeakers * Community1Config.framesPerChunk
        )
        let minimumCleanFrames = 2
        for speaker in 0..<Community1Config.localSpeakers {
            var clean = [Float](repeating: 0, count: Community1Config.framesPerChunk)
            var full = [Float](repeating: 0, count: Community1Config.framesPerChunk)
            var cleanCount = 0
            for frame in 0..<Community1Config.framesPerChunk {
                let active = masks[frame][speaker]
                full[frame] = active
                if active > 0, masks[frame].reduce(0, +) < 2 {
                    clean[frame] = active
                    cleanCount += 1
                }
            }
            // pyannote uses the overlap-inclusive mask when fewer than three
            // clean frames are available for the 400-sample minimum input.
            speakerMajorMasks.append(contentsOf: cleanCount > minimumCleanFrames ? clean : full)
        }

        let waveformInput = try Self.multiArray(
            waveform, shape: [1, 1, Community1Config.chunkSamples]
        )
        let weightsInput = try Self.multiArray(
            speakerMajorMasks,
            shape: [1, Community1Config.localSpeakers, Community1Config.framesPerChunk]
        )
        let output: MLMultiArray = try autoreleasepool {
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                manifest.embedding.waveformInputName: MLFeatureValue(multiArray: waveformInput),
                manifest.embedding.weightsInputName: MLFeatureValue(multiArray: weightsInput),
            ])
            let prediction = try embedding.prediction(from: provider)
            guard let values = prediction.featureValue(
                for: manifest.embedding.outputName
            )?.multiArrayValue else {
                throw Community1Error.inference(
                    stage: "embedding",
                    reason: "missing '\(manifest.embedding.outputName)' output"
                )
            }
            return values
        }

        let flat = try Self.floatValues(
            output,
            expectedCount: Community1Config.localSpeakers * Community1Config.embeddingDimension,
            stage: "embedding"
        )
        return (0..<Community1Config.localSpeakers).map { speaker in
            let start = speaker * Community1Config.embeddingDimension
            return Array(flat[start..<(start + Community1Config.embeddingDimension)])
        }
    }

    private static func multiArray(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: shape.map(NSNumber.init(value:)),
            dataType: .float32
        )
        guard array.count == values.count else {
            throw Community1Error.invalidBundle(
                "Core ML input shape has \(array.count) values; received \(values.count)"
            )
        }
        values.withUnsafeBufferPointer { source in
            guard let base = source.baseAddress else { return }
            array.dataPointer.assumingMemoryBound(to: Float.self).update(
                from: base, count: values.count
            )
        }
        return array
    }

    private static func floatValues(
        _ array: MLMultiArray,
        expectedCount: Int,
        stage: String
    ) throws -> [Float] {
        guard array.count == expectedCount else {
            throw Community1Error.inference(
                stage: stage,
                reason: "expected \(expectedCount) output values, got \(array.count)"
            )
        }
        switch array.dataType {
        case .float32:
            let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
            return Array(UnsafeBufferPointer(start: pointer, count: expectedCount))
        case .float16:
            let pointer = array.dataPointer.assumingMemoryBound(to: Float16.self)
            return (0..<expectedCount).map { Float(pointer[$0]) }
        case .double:
            let pointer = array.dataPointer.assumingMemoryBound(to: Double.self)
            return (0..<expectedCount).map { Float(pointer[$0]) }
        default:
            throw Community1Error.inference(
                stage: stage,
                reason: "unsupported Core ML output type \(array.dataType.rawValue)"
            )
        }
    }
}
#endif
