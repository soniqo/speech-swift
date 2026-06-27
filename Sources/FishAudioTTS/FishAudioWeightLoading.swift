import Foundation
import MLX
import MLXCommon

public enum FishAudioWeightLoader {
    public static func loadWeights(
        into model: FishAudioDualARModel,
        from directory: URL,
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws {
        progressHandler?(0.05, "Loading Fish Audio weights")
        let raw = try loadModelSafetensors(from: directory)
        let weights = remapFishQwen3OmniKeys(raw)

        CommonWeightLoader.applyEmbeddingWeights(
            to: model.slow.embeddings,
            prefix: "embeddings",
            from: weights)
        CommonWeightLoader.applyEmbeddingWeights(
            to: model.slow.codebookEmbeddings,
            prefix: "codebook_embeddings",
            from: weights)
        CommonWeightLoader.applyRMSNormWeights(to: model.slow.norm, prefix: "norm", from: weights)
        if let output = model.slow.output {
            CommonWeightLoader.applyLinearWeights(to: output, prefix: "output", from: weights)
        }
        loadLayers(model.slow.layers, prefix: "layers", from: weights)
        progressHandler?(0.60, "Loaded Fish Audio slow transformer")

        CommonWeightLoader.applyEmbeddingWeights(
            to: model.fast.embeddings,
            prefix: "fast_embeddings",
            from: weights)
        CommonWeightLoader.applyRMSNormWeights(to: model.fast.norm, prefix: "fast_norm", from: weights)
        CommonWeightLoader.applyLinearWeights(to: model.fast.output, prefix: "fast_output", from: weights)
        loadLayers(model.fast.layers, prefix: "fast_layers", from: weights)

        eval(model)
        progressHandler?(1.0, "Fish Audio weights ready")
    }

    static func remapFishQwen3OmniKeys<Value>(_ weights: [String: Value]) -> [String: Value] {
        guard weights.keys.contains(where: { $0.hasPrefix("text_model.") || $0.hasPrefix("audio_decoder.") }) else {
            return weights
        }

        var remapped: [String: Value] = [:]
        remapped.reserveCapacity(weights.count)
        for (key, value) in weights {
            if key.hasPrefix("text_model.model.") {
                remapped[String(key.dropFirst("text_model.model.".count))] = value
            } else if key.hasPrefix("audio_decoder.") {
                let suffix = String(key.dropFirst("audio_decoder.".count))
                if suffix.hasPrefix("codebook_embeddings.") {
                    remapped[suffix] = value
                } else {
                    remapped["fast_\(suffix)"] = value
                }
            } else {
                remapped[key] = value
            }
        }
        return remapped
    }

    static func modelSafetensorFiles(in directory: URL) throws -> [URL] {
        let fm = FileManager.default
        let single = directory.appendingPathComponent("model.safetensors")
        if fm.fileExists(atPath: single.path) {
            return [single]
        }

        let contents = try fm.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil)
        let files = contents
            .filter { url in
                url.pathExtension == "safetensors"
                    && url.lastPathComponent.hasPrefix("model")
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else {
            throw FishAudioError.missingFile(single)
        }
        return files
    }

    static func loadModelSafetensors(from directory: URL) throws -> [String: MLXArray] {
        var merged: [String: MLXArray] = [:]
        for file in try modelSafetensorFiles(in: directory) {
            let weights = try MLX.loadArrays(url: file)
            merged.merge(weights) { _, new in new }
        }
        return merged
    }

    private static func loadLayers(
        _ layers: [FishAudioLayer],
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        for (index, layer) in layers.enumerated() {
            let layerPrefix = "\(prefix).\(index)"
            CommonWeightLoader.applyRMSNormWeights(
                to: layer.attentionNorm,
                prefix: "\(layerPrefix).attention_norm",
                from: weights)
            CommonWeightLoader.applyRMSNormWeights(
                to: layer.ffnNorm,
                prefix: "\(layerPrefix).ffn_norm",
                from: weights)

            let attention = layer.attention
            CommonWeightLoader.applyLinearWeights(
                to: attention.wqkv,
                prefix: "\(layerPrefix).attention.wqkv",
                from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: attention.wo,
                prefix: "\(layerPrefix).attention.wo",
                from: weights)
            CommonWeightLoader.applyRMSNormWeights(
                to: attention.qNorm,
                prefix: "\(layerPrefix).attention.q_norm",
                from: weights)
            CommonWeightLoader.applyRMSNormWeights(
                to: attention.kNorm,
                prefix: "\(layerPrefix).attention.k_norm",
                from: weights)

            let feedForward = layer.feedForward
            CommonWeightLoader.applyLinearWeights(
                to: feedForward.w1,
                prefix: "\(layerPrefix).feed_forward.w1",
                from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: feedForward.w2,
                prefix: "\(layerPrefix).feed_forward.w2",
                from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: feedForward.w3,
                prefix: "\(layerPrefix).feed_forward.w3",
                from: weights)
        }
    }
}
