import Foundation
import MLX
import MLXFFT
import MLXNN

/// Audio to text through the VoiceChat perception bundle.
///
/// This is the model's transcript channel — its running record of what the user
/// said — emitted by an RNN-Transducer head over a 1024-entry vocabulary. It is
/// separate from the language backbone, and it is the cheapest end-to-end check
/// we have: a mel frontend that disagrees with training, a transposed
/// filterbank or an off-by-one blank index all still produce embeddings, and
/// only a transcript reveals which.
public final class VoiceChatTranscriber {
    public static let sampleRate = 16_000
    private static let nFFT = 512
    private static let winLength = 400        // 25 ms
    private static let hopLength = 160        // 10 ms
    private static let preEmphasis: Float = 0.97
    private static let logGuard = Foundation.pow(Float(2), -24)
    /// The joint scores 1025 classes; blank is the last.
    private static let blank = 1024

    private let perception: VoiceChatPerception
    private let filterbank: MLXArray        // (257, 128)
    private let window: MLXArray           // (512,) centred
    private let rnnt: RNNTHead

    public init(perception: VoiceChatPerception, weights: [String: MLXArray]) throws {
        self.perception = perception

        guard let rawFilters = weights["preprocessor.featurizer.fb"],
              let rawWindow = weights["preprocessor.featurizer.window"]
        else { throw VoiceChatLoadError.unexpectedKeys(["preprocessor.featurizer.*"]) }

        // Use the checkpoint's own filterbank rather than rebuilding a mel
        // scale: a near-miss filterbank still produces a plausible spectrogram
        // and nothing anywhere raises.
        self.filterbank = rawFilters.asType(.float32).squeezed(axis: 0).transposed(1, 0)

        // The stored window is 400 long against a 512-point FFT, so centre it.
        var padded = MLXArray.zeros([Self.nFFT], dtype: .float32)
        let start = (Self.nFFT - Self.winLength) / 2
        padded[start ..< (start + Self.winLength)] = rawWindow.asType(.float32)
        self.window = padded

        self.rnnt = try RNNTHead(weights: weights)
    }

    /// Log-mel features matching NeMo's AudioToMelSpectrogramPreprocessor.
    /// `normalize` is "NA" for this checkpoint, so no mean/variance step.
    public func logMel(_ samples: [Float]) -> MLXArray {
        let signal = MLXArray(samples).asType(.float32)
        let emphasised = MLX.concatenated([
            signal[0 ..< 1],
            signal[1...] - MLXArray(Self.preEmphasis) * signal[..<(signal.dim(0) - 1)],
        ])

        let pad = Self.nFFT / 2
        let padded = MLX.padded(emphasised, widths: [IntOrPair((pad, pad))])
        let frames = max(1, 1 + (padded.dim(0) - Self.nFFT) / Self.hopLength)
        let strided = asStrided(padded, [frames, Self.nFFT],
                                strides: [Self.hopLength, 1], offset: 0)

        let spectrum = rfft(strided * window.expandedDimensions(axis: 0), axis: -1)
        let power = MLX.abs(spectrum).square()
        let mel = MLX.matmul(power, filterbank)
        return MLX.log(mel + MLXArray(Self.logGuard)).expandedDimensions(axis: 0)
    }

    /// Transcribe 16 kHz mono samples.
    public func transcribe(_ samples: [Float]) -> String {
        let mel = logMel(samples)
        let hidden = perception.encoder(mel)     // (1, T, 1024), pre-projection
        eval(hidden)
        return rnnt.decode(hidden[0])
    }
}

/// Greedy RNN-Transducer decode.
///
/// The prediction network is a two-layer LSTM over emitted tokens. The joint
/// adds a projection of the current encoder frame to a projection of the
/// prediction state and scores 1025 classes: blank advances time, anything else
/// emits a token and advances the prediction state. Emissions per frame are
/// capped so a degenerate model cannot loop forever.
final class RNNTHead {
    private struct LSTMLayer {
        let wIH: MLXArray, wHH: MLXArray, bIH: MLXArray, bHH: MLXArray
    }

    private let embed: MLXArray
    private let layers: [LSTMLayer]
    private let encW: MLXArray, encB: MLXArray
    private let predW: MLXArray, predB: MLXArray
    private let outW: MLXArray, outB: MLXArray
    private let hidden = 640
    private let maxSymbolsPerFrame = 10

    init(weights: [String: MLXArray]) throws {
        func need(_ key: String) throws -> MLXArray {
            guard let value = weights[key] else { throw VoiceChatLoadError.unexpectedKeys([key]) }
            return value.asType(.float32)
        }
        // The head is under 9 M parameters, so dequantizing it costs nothing and
        // keeps the decode loop plain matmuls rather than a second quantized
        // path to get wrong.
        func dense(_ key: String) throws -> MLXArray {
            let base = key.replacingOccurrences(of: ".weight", with: "")
            if let scales = weights["\(base).scales"], let biases = weights["\(base).biases"],
               let packed = weights[key] {
                return dequantized(packed, scales: scales, biases: biases,
                                   groupSize: 64, bits: inferBits(packed, scales))
            }
            return try need(key)
        }
        func inferBits(_ packed: MLXArray, _ scales: MLXArray) -> Int {
            // packed stores 32/bits values per uint32 along the last axis.
            let groups = scales.dim(-1)
            let elements = groups * 64
            return max(2, min(8, packed.dim(-1) * 32 / elements))
        }

        embed = try dense("decoder.prediction.embed.weight")
        layers = try (0 ..< 2).map { layer in
            LSTMLayer(
                wIH: try need("decoder.prediction.dec_rnn.lstm.weight_ih_l\(layer)"),
                wHH: try need("decoder.prediction.dec_rnn.lstm.weight_hh_l\(layer)"),
                bIH: try need("decoder.prediction.dec_rnn.lstm.bias_ih_l\(layer)"),
                bHH: try need("decoder.prediction.dec_rnn.lstm.bias_hh_l\(layer)"))
        }
        encW = try dense("joint.enc.weight");  encB = try need("joint.enc.bias")
        predW = try dense("joint.pred.weight"); predB = try need("joint.pred.bias")
        outW = try dense("joint.joint_net.2.weight"); outB = try need("joint.joint_net.2.bias")
    }

    /// One LSTM step. PyTorch packs gates as [input, forget, cell, output].
    private func step(_ x: MLXArray, _ state: [(MLXArray, MLXArray)]) -> (MLXArray, [(MLXArray, MLXArray)]) {
        var input = x
        var next: [(MLXArray, MLXArray)] = []
        for (index, layer) in layers.enumerated() {
            var (h, c) = state[index]
            let gates = MLX.matmul(input, layer.wIH.transposed(1, 0)) + layer.bIH
                + MLX.matmul(h, layer.wHH.transposed(1, 0)) + layer.bHH
            let i = sigmoid(gates[0..., 0 ..< hidden])
            let f = sigmoid(gates[0..., hidden ..< (2 * hidden)])
            let g = MLX.tanh(gates[0..., (2 * hidden) ..< (3 * hidden)])
            let o = sigmoid(gates[0..., (3 * hidden)...])
            c = f * c + i * g
            h = o * MLX.tanh(c)
            next.append((h, c))
            input = h
        }
        return (input, next)
    }

    func decode(_ encoded: MLXArray) -> String {
        let encScores = MLX.matmul(encoded.asType(.float32), encW.transposed(1, 0)) + encB
        var state = (0 ..< 2).map { _ in
            (MLXArray.zeros([1, hidden]), MLXArray.zeros([1, hidden]))
        }
        // blank_as_pad: the prediction network starts from the blank embedding.
        var (pred, newState) = step(embed[VoiceChatTranscriber.blankIndex].expandedDimensions(axis: 0), state)
        state = newState

        var tokens: [Int] = []
        for t in 0 ..< encScores.dim(0) {
            var emitted = 0
            while emitted < maxSymbolsPerFrame {
                let joint = MLX.maximum(
                    encScores[t].expandedDimensions(axis: 0)
                        + MLX.matmul(pred, predW.transposed(1, 0)) + predB,
                    MLXArray(Float(0)))
                let logits = MLX.matmul(joint, outW.transposed(1, 0)) + outB
                let token = argMax(logits[0], axis: -1).item(Int.self)
                if token == VoiceChatTranscriber.blankIndex { break }
                tokens.append(token)
                (pred, newState) = step(embed[token].expandedDimensions(axis: 0), state)
                state = newState
                emitted += 1
            }
        }
        return VoiceChatTranscriber.detokenize(tokens)
    }
}

public extension VoiceChatTranscriber {
    static var blankIndex: Int { blank }

    /// Vocabulary loaded from the bundle, if present. Falls back to token ids.
    nonisolated(unsafe) static var vocabulary: [String] = []

    static func detokenize(_ tokens: [Int]) -> String {
        guard !vocabulary.isEmpty else { return tokens.map(String.init).joined(separator: " ") }
        return tokens.compactMap { $0 < vocabulary.count ? vocabulary[$0] : nil }
            .joined()
            .replacingOccurrences(of: "\u{2581}", with: " ")
            .trimmingCharacters(in: .whitespaces)
    }
}
