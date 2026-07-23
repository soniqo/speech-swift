import AudioCommon
import CoreML
import Foundation

protocol LocalVQEAECResidualMasking: AnyObject {
    func reset()
    func warmUp() throws
    func process(
        microphoneSpectrum: [Float],
        referenceSpectrum: [Float]
    ) throws -> [Float]
}

final class LocalVQEAECResidualMask: LocalVQEAECResidualMasking {
    private static let elementCount = 2 * 256
    private let model: MLModel
    private var state: MLState
    private let microphoneInput: MLMultiArray
    private let referenceInput: MLMultiArray

    init(modelURL: URL, computeUnits: MLComputeUnits) throws {
        self.model = try CoreMLLoader.load(
            url: modelURL,
            computeUnits: computeUnits,
            name: "LocalVQE AEC residual mask")
        self.state = model.makeState()
        self.microphoneInput = try MLMultiArray(
            shape: [1, 2, 1, 256], dataType: .float16)
        self.referenceInput = try MLMultiArray(
            shape: [1, 2, 1, 256], dataType: .float16)
    }

    func reset() {
        state = model.makeState()
    }

    func warmUp() throws {
        try autoreleasepool {
            let zeroMicrophone = try MLMultiArray(
                shape: [1, 2, 1, 256], dataType: .float16)
            let zeroReference = try MLMultiArray(
                shape: [1, 2, 1, 256], dataType: .float16)
            let inputs = try MLDictionaryFeatureProvider(dictionary: [
                "mic_spectrum": MLFeatureValue(multiArray: zeroMicrophone),
                "reference_spectrum": MLFeatureValue(multiArray: zeroReference),
            ])
            let temporaryState = model.makeState()
            _ = try model.prediction(from: inputs, using: temporaryState)
        }
    }

    func process(
        microphoneSpectrum: [Float],
        referenceSpectrum: [Float]
    ) throws -> [Float] {
        precondition(microphoneSpectrum.count == Self.elementCount)
        precondition(referenceSpectrum.count == Self.elementCount)
        write(microphoneSpectrum, to: microphoneInput)
        write(referenceSpectrum, to: referenceInput)

        // Streaming capture workers commonly have no run loop to drain
        // autoreleased Core ML objects. Keep the provider, prediction, output,
        // and output copy in one per-frame pool; otherwise IOSurface-backed
        // temporaries accumulate until macOS aborts a long-running process.
        return try autoreleasepool {
            let inputs = try MLDictionaryFeatureProvider(dictionary: [
                "mic_spectrum": MLFeatureValue(multiArray: microphoneInput),
                "reference_spectrum": MLFeatureValue(multiArray: referenceInput),
            ])
            let prediction: MLFeatureProvider
            do {
                prediction = try model.prediction(from: inputs, using: state)
            } catch {
                throw AudioModelError.inferenceFailed(
                    operation: "LocalVQE AEC residual mask",
                    reason: error.localizedDescription)
            }
            guard let output = prediction.featureValue(
                for: "enhanced_spectrum")?.multiArrayValue else {
                throw LocalVQEEchoCancellationError.missingModelOutput(
                    "enhanced_spectrum")
            }
            guard output.count == Self.elementCount else {
                throw LocalVQEEchoCancellationError.invalidModelOutputCount(
                    expected: Self.elementCount, actual: output.count)
            }
            return read(output)
        }
    }

    private func write(_ values: [Float], to array: MLMultiArray) {
        let pointer = array.dataPointer.assumingMemoryBound(to: Float16.self)
        for index in values.indices {
            pointer[index] = Float16(values[index])
        }
    }

    private func read(_ array: MLMultiArray) -> [Float] {
        var result = [Float](repeating: 0, count: array.count)
        switch array.dataType {
        case .float16:
            let pointer = array.dataPointer.assumingMemoryBound(to: Float16.self)
            for index in result.indices {
                result[index] = Float(pointer[index])
            }
        case .float32:
            let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
            let count = result.count
            result.withUnsafeMutableBufferPointer { destination in
                destination.baseAddress?.update(from: pointer, count: count)
            }
        default:
            for index in result.indices {
                result[index] = array[index].floatValue
            }
        }
        return result
    }
}
