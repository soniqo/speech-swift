import Foundation
import MLX
import MLXNN

/// Loads the 4-bundle MagpieTTS MLX layout into our Swift module tree.
///
/// PyTorch tensor names in the safetensors files use NeMo's checkpoint path
/// conventions (e.g. `encoder.layers.0.norm_self.weight`). We strip the
/// container prefix (`encoder.` / `decoder.`) and map LocalTransformer keys
/// out to a sibling module. Quantised bundles store the
/// `weight` tensor as a `(name.q, name.s, name.b)` triplet plus a
/// `quantized_shapes[name]` entry recording the original conv/linear shape;
/// we call `mx.dequantize` and reshape before handing weights to MLX-Swift's
/// `update(parameters:)`.
public enum MagpieWeightLoader {

    // MARK: - Public entry points

    public static func loadTextEncoder(bundleDir: URL,
                                        config: MagpieTextEncoderConfig
    ) throws -> MagpieTextEncoder {
        let module = MagpieTextEncoder(config: config)
        var weights = try loadAndDequantize(
            bundleDir: bundleDir,
            quantization: config.quantization,
            quantizedShapes: config.quantizedShapes)
        weights = remapTextEncoderKeys(weights)
        try applyWeights(module, mapping: weights, label: "text_encoder")
        eval(module.parameters())
        return module
    }

    public static func loadDecoder(bundleDir: URL,
                                    config: MagpieDecoderConfig
    ) throws -> (decoder: MagpieDecoder, localTransformer: MagpieLocalTransformer) {
        let decoder = MagpieDecoder(config: config)
        let lt = MagpieLocalTransformer(
            dModelDec: config.dModel,
            dModel: config.localTransformer.dModel,
            nHeads: config.localTransformer.nHeads,
            numCodebooks: config.numCodebooks,
            vocabPerCodebook: config.vocabPerCodebook)
        let raw = try loadAndDequantize(
            bundleDir: bundleDir,
            quantization: config.quantization,
            quantizedShapes: config.quantizedShapes)
        let (dec, ltw) = splitDecoderKeys(raw)
        try applyWeights(decoder, mapping: dec, label: "decoder")
        try applyWeights(lt, mapping: ltw, label: "local_transformer")
        eval(decoder.parameters())
        eval(lt.parameters())
        return (decoder, lt)
    }

    public static func loadNanoCodec(bundleDir: URL,
                                      config: MagpieNanoCodecConfig
    ) throws -> MagpieNanoCodec {
        let module = MagpieNanoCodec()
        var raw = try loadAndDequantize(
            bundleDir: bundleDir,
            quantization: config.quantization,
            quantizedShapes: config.quantizedShapes)
        // The PyTorch checkpoint paths sit directly at the top level
        // (`pre_conv.weight`, `up_sample_conv_layers.0.weight`, …) and we
        // mount the codec under `decoder.` in MagpieNanoCodec.
        var prefixed: [String: MLXArray] = [:]
        for (k, v) in raw { prefixed["decoder.\(k)"] = v }
        raw = prefixed
        try applyWeights(module, mapping: raw, label: "nano_codec")
        eval(module.parameters())
        return module
    }

    // MARK: - Dequantisation

    /// Read ``model.safetensors`` and rebuild full float tensors from any
    /// `.q/.s/.b` triplets, reshaping each result to its original shape in
    /// `quantized_shapes`. Non-quantised entries pass through.
    private static func loadAndDequantize(
        bundleDir: URL,
        quantization: MagpieQuantization?,
        quantizedShapes: [String: [Int]]?
    ) throws -> [String: MLXArray] {
        let url = bundleDir.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MagpieTTSError.missingFile(url.lastPathComponent)
        }
        let raw: [String: MLXArray]
        do {
            raw = try MLX.loadArrays(url: url)
        } catch {
            throw MagpieTTSError.weightLoadFailed("\(url.path): \(error)")
        }
        guard let qcfg = quantization else {
            return promoteToFloat32(raw)
        }
        let shapes = quantizedShapes ?? [:]
        var triplets: [String: (q: MLXArray?, s: MLXArray?, b: MLXArray?)] = [:]
        var out: [String: MLXArray] = [:]
        for (k, v) in raw {
            if k.hasSuffix(".q") {
                let base = String(k.dropLast(2))
                var t = triplets[base] ?? (nil, nil, nil)
                t.q = v
                triplets[base] = t
            } else if k.hasSuffix(".s") {
                let base = String(k.dropLast(2))
                var t = triplets[base] ?? (nil, nil, nil)
                t.s = v
                triplets[base] = t
            } else if k.hasSuffix(".b") {
                let base = String(k.dropLast(2))
                var t = triplets[base] ?? (nil, nil, nil)
                t.b = v
                triplets[base] = t
            } else {
                out[k] = v.asType(.float32)
            }
        }
        for (base, t) in triplets {
            guard let q = t.q, let s = t.s, let b = t.b else {
                throw MagpieTTSError.weightLoadFailed(
                    "incomplete quantised triplet for \(base)")
            }
            let flat = MLX.dequantized(
                q,
                scales: s.asType(.float32),
                biases: b.asType(.float32),
                groupSize: qcfg.groupSize,
                bits: qcfg.bits)
            // Materialise the dequantised tensor and force a contiguous copy
            // before reshape. Without `.eval` the reshape acts on a lazy
            // dequant graph node whose subsequent fancy-gather (`weight[x]`)
            // returns wrong values (mlx-swift codepath bug).
            eval(flat)
            let dense: MLXArray
            if let orig = shapes[base] {
                dense = flat.reshaped(orig)
            } else {
                dense = flat
            }
            // One more materialisation so the reshaped view is concrete.
            eval(dense)
            out[base] = dense
        }
        return out
    }

    private static func promoteToFloat32(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (k, v) in weights {
            if v.dtype == .int32 {
                out[k] = v
            } else {
                out[k] = v.asType(.float32)
            }
        }
        return out
    }

    // MARK: - Key remapping

    /// Strip ``encoder.`` from text-encoder keys.
    private static func remapTextEncoderKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (k, v) in weights {
            if k.hasPrefix("encoder.") {
                out[String(k.dropFirst("encoder.".count))] = v
            } else {
                out[k] = v
            }
        }
        return out
    }

    /// Split the decoder bundle into (decoder, local_transformer) flat maps.
    /// Drops `encoder.` and `decoder.` prefixes where appropriate, keeps
    /// audio/baked/final_proj/norm_out top-level keys, and rebrands the
    /// LocalTransformer's prefixes to their Swift counterparts.
    private static func splitDecoderKeys(_ weights: [String: MLXArray]
    ) -> (dec: [String: MLXArray], lt: [String: MLXArray]) {
        var dec: [String: MLXArray] = [:]
        var lt: [String: MLXArray] = [:]
        for (k, v) in weights {
            if k.hasPrefix("audio_embeddings.") || k == "baked_context_embedding.weight"
               || k == "baked_context_embedding_len" || k.hasPrefix("final_proj.") {
                dec[k] = v
            } else if k == "decoder.norm_out.weight" {
                dec["norm_out.weight"] = v
            } else if k == "decoder.position_embeddings.weight" {
                dec["position_embeddings.weight"] = v
            } else if k.hasPrefix("decoder.layers.") {
                dec[String(k.dropFirst("decoder.".count))] = v
            } else if k.hasPrefix("local_transformer_in_projection.") {
                let suffix = k.replacingOccurrences(of: "local_transformer_in_projection.",
                                                      with: "in_projection.")
                lt[suffix] = v
            } else if k.hasPrefix("local_transformer.layers.") {
                lt[String(k.dropFirst("local_transformer.".count))] = v
            } else if k == "local_transformer.position_embeddings.weight" {
                lt["position_embeddings.weight"] = v
            } else if k.hasPrefix("local_transformer_out_projections.") {
                let suffix = k.replacingOccurrences(
                    of: "local_transformer_out_projections.",
                    with: "out_projections.")
                lt[suffix] = v
            }
            // silently drop anything else
        }
        return (dec, lt)
    }

    private static func applyWeights(_ module: Module,
                                      mapping: [String: MLXArray],
                                      label: String) throws {
        let params = ModuleParameters.unflattened(mapping)
        do {
            try module.update(parameters: params, verify: .all)
        } catch {
            throw MagpieTTSError.weightLoadFailed("\(label): \(error)")
        }
    }
}
