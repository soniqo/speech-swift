import Foundation
import MLX
import MLXCommon
import MLXNN

/// Multi-layer LSTM matching the reference `StackedLSTM`: each layer's full hidden
/// sequence feeds the next; we expose the last layer's final-timestep hidden.
final class StackedLSTM: Module {
    @ModuleInfo(key: "layers") var layers: [LSTM]

    init(inputSize: Int, hiddenSize: Int, numLayers: Int) {
        self._layers.wrappedValue = (0 ..< numLayers).map {
            LSTM(inputSize: $0 == 0 ? inputSize : hiddenSize, hiddenSize: hiddenSize)
        }
    }

    /// x: `[B, T, inputSize]` → last layer's final hidden state `[B, hiddenSize]`.
    func finalHidden(_ x: MLXArray) -> MLXArray {
        var out = x
        for layer in layers { out = layer(out).0 }  // (allHidden, allCell) → allHidden [B,T,H]
        return out[.ellipsis, -1, 0...]
    }
}

/// Resemblyzer-style LSTM speaker encoder (Chatterbox `ve.*`). Produces a
/// 256-d L2-normalised speaker embedding used by T3's conditioning. Port of
/// the reference `VoiceEncoder`; uses native `MLXNN.LSTM` whose `Wx/Wh/bias`
/// param keys match the converted weights exactly.
public final class ChatterboxVoiceEncoder: Module {
    @ModuleInfo(key: "lstm") var lstm: StackedLSTM
    @ModuleInfo(key: "proj") var proj: Linear
    @ParameterInfo(key: "similarity_weight") var similarityWeight: MLXArray
    @ParameterInfo(key: "similarity_bias") var similarityBias: MLXArray

    public static let numMels = 40
    public static let hiddenSize = 256
    public static let embedSize = 256
    public static let partialFrames = 160
    public static let sampleRate = 16000

    /// Mel front-end config for the speaker encoder (40-mel slaney, power, no-log).
    public static let melConfig = SlaneyMelConfig(
        sampleRate: sampleRate, nFft: 400, hop: 160, win: 400, nMels: numMels,
        fmin: 0, fmax: 8000, power: 2.0, logMel: false, centerPad: true)

    public override init() {
        self._lstm.wrappedValue = StackedLSTM(
            inputSize: Self.numMels, hiddenSize: Self.hiddenSize, numLayers: 3)
        self._proj.wrappedValue = Linear(Self.hiddenSize, Self.embedSize)
        self._similarityWeight.wrappedValue = MLXArray([Float(10.0)])
        self._similarityBias.wrappedValue = MLXArray([Float(-5.0)])
        super.init()
    }

    /// Embed a batch of partials `[B, partialFrames, numMels]` → L2-normed `[B, 256]`.
    func embedPartials(_ partials: MLXArray) -> MLXArray {
        let h = lstm.finalHidden(partials)            // [B, 256]
        let e = relu(proj(h))                          // ve_final_relu
        return e / MLX.sqrt(MLX.sum(e * e, axis: 1, keepDims: true))
    }

    /// Full-utterance speaker embedding from a mel `(T, numMels)`. Mirrors
    /// the reference `inference` (rate=1.3, overlap=0.5, min_coverage=0.8): split
    /// into partial windows, embed each, mean, L2-normalise.
    public func embed(
        mel: MLXArray, rate: Float = 1.3, overlap: Float = 0.5, minCoverage: Float = 0.8
    ) -> MLXArray {
        let nFrames = mel.dim(0)
        let step = frameStep(overlap: overlap, rate: rate)
        let (nWins, targetN) = numWins(nFrames: nFrames, step: step, minCoverage: minCoverage)

        var m = mel
        if targetN > nFrames {
            m = concatenated([m, MLXArray.zeros([targetN - nFrames, Self.numMels])], axis: 0)
        }
        var parts: [MLXArray] = []
        for w in 0 ..< nWins {
            let s = w * step
            parts.append(m[s ..< s + Self.partialFrames, 0...])  // [partialFrames, numMels]
        }
        let batch = stacked(parts, axis: 0)            // [nWins, partialFrames, numMels]
        let partEmbeds = embedPartials(batch)          // [nWins, 256]
        let raw = MLX.mean(partEmbeds, axis: 0)        // [256]
        return raw / MLX.sqrt(MLX.sum(raw * raw))      // final L2
    }

    /// Convenience: speaker embedding straight from 16 kHz mono samples.
    public func embed(samples: [Float]) -> MLXArray {
        embed(mel: SlaneyMel.melSpec(samples: samples, config: Self.melConfig))
    }

    // get_frame_step: int(round((sample_rate / rate) / partial_frames)) for rate != nil.
    private func frameStep(overlap: Float, rate: Float) -> Int {
        Int((Float(Self.sampleRate) / rate / Float(Self.partialFrames)).rounded())
    }

    // get_num_wins: number of partial windows + the padded target frame count.
    private func numWins(nFrames: Int, step: Int, minCoverage: Float) -> (Int, Int) {
        let winSize = Self.partialFrames
        let base = max(nFrames - winSize + step, 0)
        var nWins = base / step
        let remainder = base % step
        if nWins == 0 || Float(remainder + (winSize - step)) / Float(winSize) >= minCoverage {
            nWins += 1
        }
        let targetN = winSize + step * (nWins - 1)
        return (nWins, targetN)
    }
}
