import Foundation
import MLX
import MLXFFT
import MLXNN

/// Match NumPy/NeMo's one-dimensional `mode="reflect"` padding.
///
/// MLX Swift currently exposes constant and edge padding, but the VoiceChat
/// perception checkpoint was trained with reflection padding. A general index
/// mapping keeps short diagnostic inputs correct as well as normal live audio
/// windows, including when the padding is wider than the input.
func voiceChatReflectPad(_ values: MLXArray, padding: Int) -> MLXArray {
    guard padding > 0 else { return values }
    let count = values.dim(0)
    precondition(count > 0, "reflection padding requires non-empty input")
    guard count > 1 else {
        return MLX.tiled(values, repetitions: [2 * padding + 1])
    }

    let period = 2 * (count - 1)
    let indexes = (-padding ..< count + padding).map { position -> Int32 in
        var reflected = position % period
        if reflected < 0 { reflected += period }
        if reflected >= count { reflected = period - reflected }
        return Int32(reflected)
    }
    return MLX.take(values, MLXArray(indexes), axis: 0)
}

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

        guard let vocabularyURL = Bundle.module.url(
            forResource: "rnnt_vocab", withExtension: "json") else {
            throw VoiceChatLoadError.unexpectedKeys(["bundled rnnt_vocab.json"])
        }
        let vocabulary = try JSONDecoder().decode(
            [String].self, from: Data(contentsOf: vocabularyURL))

        guard let rawFilters = weights["preprocessor.featurizer.fb"],
              let rawWindow = weights["preprocessor.featurizer.window"]
        else { throw VoiceChatLoadError.unexpectedKeys(["preprocessor.featurizer.*"]) }

        // Use the checkpoint's own filterbank rather than rebuilding a mel
        // scale: a near-miss filterbank still produces a plausible spectrogram
        // and nothing anywhere raises.
        self.filterbank = rawFilters.asType(.float32).squeezed(axis: 0).transposed(1, 0)

        // The stored window is 400 long against a 512-point FFT, so centre it.
        let padded = MLXArray.zeros([Self.nFFT], dtype: .float32)
        let start = (Self.nFFT - Self.winLength) / 2
        padded[start ..< (start + Self.winLength)] = rawWindow.asType(.float32)
        self.window = padded

        self.rnnt = try RNNTHead(weights: weights, vocabulary: vocabulary)
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
        let padded = voiceChatReflectPad(emphasised, padding: pad)
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
        guard !samples.isEmpty else { return "" }
        let mel = logMel(samples)
        let hidden = perception.encoder(mel)     // (1, T, 1024), pre-projection
        eval(hidden)
        return rnnt.decode(hidden[0])
    }

    struct StreamState {
        fileprivate var decoder: RNNTHead.State
        fileprivate var transcript = ""
    }

    struct StreamFrameResult {
        let transcript: String
        /// Whether the RNN-T head's first prediction for this encoder frame was
        /// blank. NVIDIA's realtime wrapper uses this exact signal for EOU/BOU.
        let isBlank: Bool
        /// Whether this frame emitted at least one recognized lexical token.
        /// A short word can be fully decoded in one encoder frame even though
        /// turn-taking otherwise waits for several non-blank frame starts.
        let hasLexicalToken: Bool
    }

    func makeStreamState() -> StreamState {
        StreamState(decoder: rnnt.makeState())
    }

    /// Decode encoder frames already produced by the duplex stream.
    ///
    /// Reusing those frames avoids a second FastConformer pass merely to show
    /// the user transcript. The RNN-T prediction state is carried across calls,
    /// so the returned text is the complete append-only transcript so far.
    func transcribeStreamingFrame(
        _ encoded: MLXArray,
        state: inout StreamState
    ) -> StreamFrameResult {
        let result = rnnt.consume(encoded, state: &state.decoder)
        if !result.emittedTokens.isEmpty {
            state.transcript = rnnt.transcript(state.decoder)
        }
        return StreamFrameResult(
            transcript: state.transcript,
            isBlank: result.firstPredictionWasBlank.last ?? true,
            hasLexicalToken: result.emittedTokens.contains { token in
                rnnt.isLexicalToken(token)
            })
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
    private let vocabulary: [String]
    private let hidden = 640
    private let maxSymbolsPerFrame = 10

    struct State {
        fileprivate var recurrent: [(MLXArray, MLXArray)]
        fileprivate var prediction: MLXArray
        fileprivate var projectedPrediction: MLXArray
        fileprivate var tokens: [Int]
    }

    struct ConsumeResult {
        let emittedTokens: [Int]
        /// One value per encoder frame. This intentionally records the first
        /// prediction, matching NVIDIA's turn-taking implementation even when
        /// the label loop emits more tokens before reaching blank.
        let firstPredictionWasBlank: [Bool]
    }

    init(weights: [String: MLXArray], vocabulary: [String]) throws {
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
        self.vocabulary = vocabulary
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

    func makeState() -> State {
        let recurrent = (0 ..< 2).map { _ in
            (MLXArray.zeros([1, hidden]), MLXArray.zeros([1, hidden]))
        }
        // Keep caption decoding on the same MLX inference stream as the
        // encoder. Moving these small recurrent operations to the CPU forces
        // a device synchronization for every 80 ms frame and prevents the
        // complete duplex pipeline from sustaining real time.
        let (prediction, next) = step(
            embed[VoiceChatTranscriber.blankIndex].expandedDimensions(axis: 0),
            recurrent)
        let projectedPrediction = MLX.matmul(
            prediction, predW.transposed(1, 0)) + predB
        MLX.eval(
            [prediction, projectedPrediction]
                + next.flatMap { [$0.0, $0.1] })
        return State(
            recurrent: next,
            prediction: prediction,
            projectedPrediction: projectedPrediction,
            tokens: [])
    }

    @discardableResult
    func consume(_ encoded: MLXArray, state: inout State) -> ConsumeResult {
        let encScores = MLX.matmul(encoded.asType(.float32), encW.transposed(1, 0)) + encB
        var emittedTokens: [Int] = []
        var firstPredictionWasBlank: [Bool] = []
        firstPredictionWasBlank.reserveCapacity(encScores.dim(0))
        for t in 0 ..< encScores.dim(0) {
            var emitted = 0
            var isFirstPrediction = true
            while emitted < maxSymbolsPerFrame {
                let joint = MLX.maximum(
                    encScores[t].expandedDimensions(axis: 0)
                        + state.projectedPrediction,
                    MLXArray(Float(0)))
                let logits = MLX.matmul(joint, outW.transposed(1, 0)) + outB
                let token = argMax(logits[0], axis: -1).item(Int.self)
                if isFirstPrediction {
                    firstPredictionWasBlank.append(
                        token == VoiceChatTranscriber.blankIndex)
                    isFirstPrediction = false
                }
                if token == VoiceChatTranscriber.blankIndex { break }
                emittedTokens.append(token)
                let (prediction, recurrent) = step(
                    embed[token].expandedDimensions(axis: 0), state.recurrent)
                state.prediction = prediction
                state.recurrent = recurrent
                state.projectedPrediction = MLX.matmul(
                    prediction, predW.transposed(1, 0)) + predB
                emitted += 1
            }
        }
        if !emittedTokens.isEmpty {
            MLX.eval(
                [state.prediction, state.projectedPrediction]
                    + state.recurrent.flatMap { [$0.0, $0.1] })
        }
        state.tokens.append(contentsOf: emittedTokens)
        return ConsumeResult(
            emittedTokens: emittedTokens,
            firstPredictionWasBlank: firstPredictionWasBlank)
    }

    func transcript(_ state: State) -> String {
        detokenize(state.tokens)
    }

    func isLexicalToken(_ tokenID: Int) -> Bool {
        guard tokenID > 0, tokenID < vocabulary.count else { return false }
        return Self.isLexicalVocabularyToken(vocabulary[tokenID])
    }

    static func isLexicalVocabularyToken(_ token: String) -> Bool {
        guard token != "<unk>", token != "\u{2047}" else { return false }
        return token.unicodeScalars.contains {
            CharacterSet.alphanumerics.contains($0)
        }
    }

    func decode(_ encoded: MLXArray) -> String {
        var state = makeState()
        _ = consume(encoded, state: &state)
        return transcript(state)
    }

    private func detokenize(_ tokens: [Int]) -> String {
        tokens.compactMap { $0 < vocabulary.count ? vocabulary[$0] : nil }
            .joined()
            .replacingOccurrences(of: "\u{2581}", with: " ")
            .trimmingCharacters(in: .whitespaces)
    }
}

public extension VoiceChatTranscriber {
    static var blankIndex: Int { blank }
}
