import Accelerate
import Foundation

/// Reference-compatible 512/256 sqrt-Hann streaming codec.
final class LocalVQEAECCodec {
    private static let fftSize = LocalVQEEchoCanceller.fftSize
    private static let hopSize = LocalVQEEchoCanceller.frameSize
    private static let spectrumBins = 256

    private let analysisSetup: OpaquePointer
    private let synthesisSetup: OpaquePointer
    private let window: [Float]

    private var microphoneHistory = [Float](repeating: 0, count: hopSize)
    private var referenceHistory = [Float](repeating: 0, count: hopSize)
    private var overlap = [Float](repeating: 0, count: fftSize)

    init(window: [Float]) throws {
        guard window.count == Self.fftSize else {
            throw LocalVQEEchoCancellationError.invalidFrontendArray(
                name: "analysis_window",
                expected: Self.fftSize,
                actual: window.count)
        }
        guard let analysisSetup = vDSP_DFT_zop_CreateSetup(
            nil, vDSP_Length(Self.fftSize), .FORWARD),
              let synthesisSetup = vDSP_DFT_zop_CreateSetup(
                nil, vDSP_Length(Self.fftSize), .INVERSE)
        else {
            throw LocalVQEEchoCancellationError.fftInitializationFailed
        }
        self.analysisSetup = analysisSetup
        self.synthesisSetup = synthesisSetup
        self.window = window
    }

    deinit {
        vDSP_DFT_DestroySetup(analysisSetup)
        vDSP_DFT_DestroySetup(synthesisSetup)
    }

    func reset() {
        microphoneHistory = [Float](repeating: 0, count: Self.hopSize)
        referenceHistory = [Float](repeating: 0, count: Self.hopSize)
        overlap = [Float](repeating: 0, count: Self.fftSize)
    }

    func analyze(
        residual: [Float],
        echoEstimate: [Float]
    ) -> (microphone: [Float], reference: [Float]) {
        let microphone = analyze(frame: residual, history: &microphoneHistory)
        let reference = analyze(frame: echoEstimate, history: &referenceHistory)
        return (microphone, reference)
    }

    private func analyze(frame: [Float], history: inout [Float]) -> [Float] {
        precondition(frame.count == Self.hopSize)
        var input = history + frame
        for index in input.indices {
            input[index] *= window[index]
        }
        history = frame

        let zeroImaginary = [Float](repeating: 0, count: Self.fftSize)
        var transformedReal = [Float](repeating: 0, count: Self.fftSize)
        var transformedImaginary = [Float](repeating: 0, count: Self.fftSize)
        vDSP_DFT_Execute(
            analysisSetup,
            input,
            zeroImaginary,
            &transformedReal,
            &transformedImaginary)

        // Core ML tensor layout is [1, 2, 1, 256]: all real bins followed by
        // all imaginary bins. DC is omitted; bins 1...256 are retained.
        var spectrum = [Float](repeating: 0, count: Self.spectrumBins * 2)
        for bin in 0..<Self.spectrumBins {
            spectrum[bin] = transformedReal[bin + 1]
            spectrum[Self.spectrumBins + bin] = transformedImaginary[bin + 1]
        }
        return spectrum
    }

    func synthesize(spectrum: [Float]) -> [Float] {
        precondition(spectrum.count == Self.spectrumBins * 2)
        var fullReal = [Float](repeating: 0, count: Self.fftSize)
        var fullImaginary = [Float](repeating: 0, count: Self.fftSize)

        for bin in 0..<(Self.spectrumBins - 1) {
            fullReal[bin + 1] = spectrum[bin]
            fullImaginary[bin + 1] = spectrum[Self.spectrumBins + bin]
        }
        // LocalVQE's packed representation carries half the real-valued
        // Nyquist synthesis coefficient.
        fullReal[Self.spectrumBins] = 2.0 * spectrum[Self.spectrumBins - 1]
        for bin in 1..<Self.spectrumBins {
            fullReal[Self.fftSize - bin] = fullReal[bin]
            fullImaginary[Self.fftSize - bin] = -fullImaginary[bin]
        }

        var inverseReal = [Float](repeating: 0, count: Self.fftSize)
        var inverseImaginary = [Float](repeating: 0, count: Self.fftSize)
        vDSP_DFT_Execute(
            synthesisSetup,
            fullReal,
            fullImaginary,
            &inverseReal,
            &inverseImaginary)

        // vDSP's inverse transform is unnormalized; LocalVQE's inverse FFT
        // divides by N before applying its synthesis window.
        let inverseScale = 1.0 / Float(Self.fftSize)
        for index in 0..<Self.fftSize {
            overlap[index] += inverseReal[index] * inverseScale * window[index]
        }
        let output = Array(overlap.prefix(Self.hopSize))
        let tail = Array(overlap[Self.hopSize..<Self.fftSize])
        overlap.replaceSubrange(
            0..<Self.hopSize,
            with: tail)
        for index in Self.hopSize..<Self.fftSize {
            overlap[index] = 0
        }
        return output
    }
}
