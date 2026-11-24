import Accelerate
import CoreML
import Foundation

/// The v1 checkpoint uses a centered, unnormalized STFT with a periodic Hann window.
final class InworldIsolationDSP {
    static let fftSize = 2_048
    static let hop = 480
    static let bins = fftSize / 2 + 1
    private static let pad = fftSize / 2

    private let forward: vDSP_DFT_Setup
    private let inverse: vDSP_DFT_Setup
    private let window: [Float]

    init() throws {
        let size = vDSP_Length(Self.fftSize)
        guard let forward = vDSP_DFT_zop_CreateSetup(nil, size, .FORWARD) else {
            throw InworldIsolationError.predictionFailed("could not create the forward DFT")
        }
        guard let inverse = vDSP_DFT_zop_CreateSetup(nil, size, .INVERSE) else {
            vDSP_DFT_DestroySetup(forward)
            throw InworldIsolationError.predictionFailed("could not create the inverse DFT")
        }
        self.forward = forward
        self.inverse = inverse
        self.window = (0..<Self.fftSize).map { i in
            0.5 * (1 - cos(2 * .pi * Float(i) / Float(Self.fftSize)))
        }
    }

    deinit {
        vDSP_DFT_DestroySetup(forward)
        vDSP_DFT_DestroySetup(inverse)
    }

    func spectrum(_ audio: ArraySlice<Float>) throws -> MLMultiArray {
        let frames = audio.count / Self.hop + 1
        return try spectrum(padded: Self.centerPad(audio), frames: frames)
    }

    func spectrum(padded: [Float], frames: Int) throws -> MLMultiArray {
        let spectrum = try MLMultiArray(shape: [1, NSNumber(value: frames), NSNumber(value: Self.bins), 2], dataType: .float32)
        var windowed = [Float](repeating: 0, count: Self.fftSize)
        let zero = windowed
        var real = windowed
        var imag = windowed

        spectrum.withUnsafeMutableBufferPointer(ofType: Float.self) { values, strides in
            padded.withUnsafeBufferPointer { input in
                for frame in 0..<frames {
                    let base = input.baseAddress! + frame * Self.hop
                    vDSP_vmul(base, 1, window, 1, &windowed, 1, vDSP_Length(Self.fftSize))
                    vDSP_DFT_Execute(forward, windowed, zero, &real, &imag)
                    for bin in 0..<Self.bins {
                        let offset = frame * strides[1] + bin * strides[2]
                        values[offset] = real[bin]
                        values[offset + strides[3]] = imag[bin]
                    }
                }
            }
        }
        return spectrum
    }

    private static func centerPad(_ audio: ArraySlice<Float>) -> [Float] {
        let count = audio.count
        var padded = [Float](repeating: 0, count: count + 2 * pad)
        padded.replaceSubrange(pad..<(pad + count), with: audio)
        // PyTorch switches to zero padding when reflection would exceed the clip.
        if count > pad {
            for i in 0..<pad {
                padded[i] = audio[audio.startIndex + pad - i]
                padded[pad + count + i] = audio[audio.endIndex - 2 - i]
            }
        }
        return padded
    }

    func synthesize(_ audio: ArraySlice<Float>, spectrum: MLMultiArray, maskResidual: MLMultiArray) -> [Float] {
        let frames = audio.count / Self.hop + 1
        let inverseLength = (frames - 1) * Self.hop + Self.fftSize
        var signal = [Float](repeating: 0, count: inverseLength)
        var weights = signal
        var workspace = SynthesisWorkspace()
        for frame in 0..<frames {
            addCorrection(spectrum, frame: frame, mask: maskResidual, maskFrame: frame, at: frame * Self.hop,
                          signal: &signal, weights: &weights, workspace: &workspace)
        }

        return audio.enumerated().map { index, sample in
            let position = index + Self.pad
            return sample + signal[position] / weights[position]
        }
    }

    struct SynthesisWorkspace {
        var real = [Float](repeating: 0, count: fftSize)
        var imag = [Float](repeating: 0, count: fftSize)
        var time = [Float](repeating: 0, count: fftSize)
        var unused = [Float](repeating: 0, count: fftSize)
    }

    func addCorrection(
        _ spectrum: MLMultiArray, frame: Int, mask: MLMultiArray, maskFrame: Int, at start: Int,
        signal: inout [Float], weights: inout [Float], workspace: inout SynthesisWorkspace
    ) {
        let specStrides = spectrum.strides.map(\.intValue)
        let maskStrides = mask.strides.map(\.intValue)
        let scale: Float = 1 / Float(Self.fftSize)
        spectrum.withUnsafeBufferPointer(ofType: Float.self) { input in
            mask.withUnsafeBufferPointer(ofType: Float.self) { raw in
                for bin in 0..<Self.bins {
                    let inputOffset = frame * specStrides[1] + bin * specStrides[2]
                    let rawOffset = maskFrame * maskStrides[1] + bin * maskStrides[2]
                    let stride = maskStrides[3]
                    let x = input[inputOffset]
                    let y = input[inputOffset + specStrides[3]]
                    let maskR = 2 * tanh(raw[rawOffset])
                    let maskI = 2 * tanh(raw[rawOffset + stride])
                    let residualR = tanh(raw[rawOffset + 2 * stride])
                    let residualI = tanh(raw[rawOffset + 3 * stride])
                    let bound = 0.5 * hypot(x, y) / max(1, hypot(residualR, residualI))
                    workspace.real[bin] = x * maskR - y * maskI + bound * residualR
                    workspace.imag[bin] = x * maskI + y * maskR + bound * residualI
                }
            }
        }
        workspace.imag[0] = 0
        workspace.imag[Self.pad] = 0
        for bin in 1..<Self.pad {
            workspace.real[Self.fftSize - bin] = workspace.real[bin]
            workspace.imag[Self.fftSize - bin] = -workspace.imag[bin]
        }

        vDSP_DFT_Execute(inverse, workspace.real, workspace.imag, &workspace.time, &workspace.unused)
        for i in 0..<Self.fftSize {
            let w = window[i]
            signal[start + i] += workspace.time[i] * w * scale
            weights[start + i] += w * w
        }
    }
}
