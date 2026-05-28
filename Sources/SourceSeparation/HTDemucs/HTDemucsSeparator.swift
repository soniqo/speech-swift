import Foundation
import MLX
import MLXNN

/// Bag-of-models source separator for Hybrid Transformer Demucs (Demucs v4).
///
/// `htdemucs_ft` ships 4 fine-tuned sub-models with diagonal combine-weights, so
/// each sub-model contributes only its own stem. Loads all sub-models from the
/// exported safetensors (keys prefixed `model_{i}.`).
public final class HTDemucsSeparator {
    public let config: HTDemucsConfig
    let models: [HTDemucs]

    init(config: HTDemucsConfig, models: [HTDemucs]) {
        self.config = config
        self.models = models
    }

    /// Load the bag from a local export directory containing
    /// `<model>.safetensors` + `<model>_config.json`. Weights are cast to
    /// float32 (parity-friendly); use a quantized loader for int8.
    public static func fromLocal(directory: URL, modelName: String = "htdemucs_ft") throws -> HTDemucsSeparator {
        let cfg = try HTDemucsConfig.load(
            from: directory.appendingPathComponent("\(modelName)_config.json"))
        let all = try MLX.loadArrays(
            url: directory.appendingPathComponent("\(modelName).safetensors"))

        var models: [HTDemucs] = []
        for i in 0..<cfg.numModels {
            let prefix = "model_\(i)."
            var sub: [String: MLXArray] = [:]
            sub.reserveCapacity(all.count / cfg.numModels + 1)
            for (k, v) in all where k.hasPrefix(prefix) {
                sub[String(k.dropFirst(prefix.count))] = v.asType(.float32)
            }
            let model = HTDemucs(cfg)
            // .all => fail on any unused/missing/shape-mismatched key, i.e. the
            // Swift module tree must exactly mirror the exported names.
            try model.update(parameters: ModuleParameters.unflattened(sub), verify: .all)
            models.append(model)
        }
        return HTDemucsSeparator(config: cfg, models: models)
    }

    /// Triangular cross-fade weight over `length` (ramp up to the middle, down to
    /// the edges), normalized to max 1. Matches demucs' segment transition weight.
    private func triangularWeight(_ length: Int) -> [Float] {
        let half = length / 2
        var w = [Float](repeating: 0, count: length)
        for i in 0..<half { w[i] = Float(i + 1) }          // 1..half
        for i in half..<length { w[i] = Float(length - i) } // (length-half)..1
        let m = w.max() ?? 1
        return w.map { $0 / m }
    }

    /// Apply one sub-model across the full mixture with overlapping
    /// training-length windows + triangular-weighted overlap-add. `mix`:
    /// [1, C, L] → [1, S, C, L]. (shifts=0, deterministic.)
    private func applySplit(_ model: HTDemucs, _ mix: MLXArray, overlap: Float = 0.25) -> MLXArray {
        let length = mix.dim(-1)
        let S = config.sources.count, C = config.audioChannels
        let segLen = config.trainingLength
        let stride = max(1, Int((1 - overlap) * Float(segLen)))
        let wFull = MLXArray(triangularWeight(segLen))

        var out = MLXArray.zeros([1, S, C, length])
        var sumW = MLXArray.zeros([length])
        var offset = 0
        while offset < length {
            let chunkLen = min(segLen, length - offset)
            let chunk = mix[0..., 0..., offset ..< (offset + chunkLen)]   // model pads internally
            let chunkOut = model(chunk)                                   // [1, S, C, chunkLen]
            let w = wFull[0 ..< chunkLen]
            let right = length - offset - chunkLen
            let pads = [IntOrPair((0, 0)), IntOrPair((0, 0)), IntOrPair((0, 0)), IntOrPair((offset, right))]
            out = out + padded(chunkOut * w.reshaped([1, 1, 1, chunkLen]), widths: pads)
            sumW = sumW + padded(w, widths: [IntOrPair((offset, right))])
            eval(out, sumW)
            offset += stride
        }
        return out / sumW    // [length] broadcasts over [1, S, C, length]
    }

    /// Separate a mixture into stems. `mix`: [1, audioChannels, L] @ 44.1 kHz.
    /// Returns source name → [1, audioChannels, L]. Diagonal bag: each stem comes
    /// from its own fine-tuned sub-model.
    public func separate(_ mix: MLXArray) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (s, name) in config.sources.enumerated() {
            let full = applySplit(models[s], mix)    // [1, S, C, L]
            let stem = full[0..., s, 0..., 0...]      // [1, C, L]
            eval(stem)
            result[name] = stem
        }
        return result
    }
}
