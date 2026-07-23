import Foundation
import LocalVQEAECFrontend

final class LocalVQEAECAdaptiveFilter {
    private let handle: OpaquePointer

    init(weights: [Float], enablePrealignment: Bool) throws {
        guard weights.count == Int(localvqe_aec_daf_weight_count()) else {
            throw LocalVQEEchoCancellationError.invalidControllerWeightCount(
                expected: Int(localvqe_aec_daf_weight_count()),
                actual: weights.count)
        }
        let created = weights.withUnsafeBufferPointer { buffer in
            localvqe_aec_daf_create(buffer.baseAddress, buffer.count)
        }
        guard let created else {
            throw LocalVQEEchoCancellationError.frontendInitializationFailed
        }
        self.handle = created
        localvqe_aec_daf_set_prealignment(handle, enablePrealignment)
    }

    deinit {
        localvqe_aec_daf_destroy(handle)
    }

    var currentDelaySamples: Int {
        Int(localvqe_aec_daf_current_delay_samples(handle))
    }

    var delayConfidence: Float {
        localvqe_aec_daf_delay_confidence(handle)
    }

    func reset() {
        localvqe_aec_daf_reset(handle)
    }

    func primeDelay(microphone: [Float], reference: [Float]) throws {
        precondition(microphone.count == reference.count)
        let succeeded = microphone.withUnsafeBufferPointer { microphoneBuffer in
            reference.withUnsafeBufferPointer { referenceBuffer in
                localvqe_aec_daf_prime_delay(
                    handle,
                    microphoneBuffer.baseAddress,
                    referenceBuffer.baseAddress,
                    microphoneBuffer.count)
            }
        }
        guard succeeded else {
            throw LocalVQEEchoCancellationError.frontendDelayEstimationFailed
        }
    }

    func process(
        microphone: [Float],
        reference: [Float]
    ) throws -> (residual: [Float], echoEstimate: [Float]) {
        precondition(microphone.count == reference.count)
        var residual = [Float](repeating: 0, count: microphone.count)
        var echoEstimate = [Float](repeating: 0, count: microphone.count)
        let succeeded = microphone.withUnsafeBufferPointer { microphoneBuffer in
            reference.withUnsafeBufferPointer { referenceBuffer in
                residual.withUnsafeMutableBufferPointer { residualBuffer in
                    echoEstimate.withUnsafeMutableBufferPointer { estimateBuffer in
                        localvqe_aec_daf_process(
                            handle,
                            microphoneBuffer.baseAddress,
                            referenceBuffer.baseAddress,
                            microphoneBuffer.count,
                            residualBuffer.baseAddress,
                            estimateBuffer.baseAddress)
                    }
                }
            }
        }
        guard succeeded else {
            throw LocalVQEEchoCancellationError.frontendProcessingFailed
        }
        return (residual, echoEstimate)
    }
}
