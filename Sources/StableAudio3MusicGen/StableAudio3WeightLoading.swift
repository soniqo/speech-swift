import Foundation
import MLX
import MLXNN
import AudioCommon

/// Errors raised by the SA3 weight loader.
public enum StableAudio3Error: Error, LocalizedError {
    case missingFile(String)
    case missingTensor(String)
    case weightLoadFailed(String)
    case tokenizerLoadFailed(String)
    case unsupportedFamily(String)

    public var errorDescription: String? {
        switch self {
        case .missingFile(let f): return "Stable Audio 3: missing file \(f)"
        case .missingTensor(let k): return "Stable Audio 3: missing tensor \(k)"
        case .weightLoadFailed(let m): return "Stable Audio 3: weight load failed: \(m)"
        case .tokenizerLoadFailed(let m): return "Stable Audio 3: tokenizer load failed: \(m)"
        case .unsupportedFamily(let m): return "Stable Audio 3: unsupported family: \(m)"
        }
    }
}

/// Internal: read one `model.safetensors` into a `[String: MLXArray]`.
func sa3LoadBundleComponent(_ dir: URL) throws -> [String: MLXArray] {
    let url = dir.appendingPathComponent("model.safetensors")
    guard FileManager.default.fileExists(atPath: url.path) else {
        throw StableAudio3Error.missingFile(url.path)
    }
    do {
        return try MLX.loadArrays(url: url)
    } catch {
        throw StableAudio3Error.weightLoadFailed("\(url.lastPathComponent): \(error)")
    }
}

/// Apply a flat `[String: MLXArray]` to an MLX module, verifying every
/// parameter shape matches. Mirrors VoxCPM2 / MAGNeT pattern.
func sa3ApplyFlatWeights(into module: Module, mapping: [String: MLXArray]) throws {
    let params = ModuleParameters.unflattened(mapping)
    try module.update(parameters: params, verify: .shapeMismatch)
}

/// Rewrite checkpoint keys for the DiT (medium) so that the numeric-string
/// indices `.0.` / `.2.` (which MLX-Swift would deserialize as a 3-element
/// list with a `none` at index 1) become named children `.inProj.` / `.outProj.`
/// for the conditioner MLPs and `.glu.` / `.out.` for the GeGLU feed-forward.
/// This lets `update(parameters:)` consume a `Module` with named ModuleInfo
/// children instead of a `[Module?]` array.
func sa3RewriteDiTMediumKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    let twoLinearOwners: [String] = [
        "to_cond_embed", "to_global_embed", "to_timestep_embed",
        "transformer.global_cond_embedder",
        // per-layer paths handled below
    ]
    func rewrite(_ k: String) -> String {
        var out = k
        for owner in twoLinearOwners {
            if out.hasPrefix(owner + ".0.") {
                out = owner + ".inProj." + out.dropFirst((owner + ".0.").count)
            } else if out.hasPrefix(owner + ".2.") {
                out = owner + ".outProj." + out.dropFirst((owner + ".2.").count)
            }
        }
        // Per-layer keys: transformer.layers.N.{ff.ff,to_local_embed.seq}.{0,2}.*
        if let layerStart = out.range(of: "transformer.layers.") {
            let tail = out[layerStart.upperBound...]
            if let dotIdx = tail.firstIndex(of: ".") {
                let n = String(tail[..<dotIdx])
                let body = String(tail[tail.index(after: dotIdx)...])
                let base = "transformer.layers.\(n)."
                func mapList(_ owner: String, _ a: String, _ b: String) -> String? {
                    if body.hasPrefix(owner + ".0.") {
                        return base + owner + "." + a + "." + String(body.dropFirst((owner + ".0.").count))
                    }
                    if body.hasPrefix(owner + ".2.") {
                        return base + owner + "." + b + "." + String(body.dropFirst((owner + ".2.").count))
                    }
                    return nil
                }
                if let rewritten = mapList("to_local_embed.seq", "inProj", "outProj") {
                    out = rewritten
                } else if let rewritten = mapList("ff.ff", "glu", "out") {
                    out = rewritten
                }
            }
        }
        return out
    }
    var renamed: [String: MLXArray] = [:]
    for (k, v) in weights {
        renamed[rewrite(k)] = v
    }
    return renamed
}

/// Strip a prefix from a weight dict's keys, returning only the matching subset.
func sa3StripPrefix(_ weights: [String: MLXArray], prefix: String) -> [String: MLXArray] {
    var out: [String: MLXArray] = [:]
    let pfx = prefix.hasSuffix(".") ? prefix : prefix + "."
    for (k, v) in weights {
        if k.hasPrefix(pfx) {
            out[String(k.dropFirst(pfx.count))] = v
        }
    }
    return out
}

/// Load + finalize the DiT (medium INT8/INT4). Returns `(model, conditioner)`
/// where the conditioner is the SecondsTotalEmbedder + learned padding
/// embedding bundled into the DiT safetensors under the `cond.` prefix.
func sa3LoadDiTMedium(dir: URL, tLat: Int, bits: Int) throws
    -> (model: DiTMedium, padding: MLXArray, secondsEmbedder: SecondsTotalEmbedder) {
    var raw = try sa3LoadBundleComponent(dir)

    // Extract the conditioner triplet before we hand the rest to load.
    let padKey = "cond.padding_embedding"
    let secsWKey = "cond.seconds_total_weight"
    let secsBKey = "cond.seconds_total_bias"
    guard let padding = raw.removeValue(forKey: padKey) else {
        throw StableAudio3Error.missingTensor(padKey)
    }
    guard let secsW = raw.removeValue(forKey: secsWKey) else {
        throw StableAudio3Error.missingTensor(secsWKey)
    }
    guard let secsB = raw.removeValue(forKey: secsBKey) else {
        throw StableAudio3Error.missingTensor(secsBKey)
    }
    let secondsEmbedder = SecondsTotalEmbedder(
        weight: secsW.asType(.float32), bias: secsB.asType(.float32))

    let model = DiTMedium(tLat: tLat, bits: bits)
    let rewritten = sa3RewriteDiTMediumKeys(raw)
    try sa3ApplyFlatWeights(into: model, mapping: rewritten)
    eval(model.parameters())
    return (model, padding.asType(.float32), secondsEmbedder)
}

/// Load SAME-L decoder. Reshapes the `mapping.weight` from PyTorch Conv1d
/// `(out, in, 1)` to Linear `(out, in)` if present in 3-D form.
func sa3LoadSAMELDecoder(dir: URL) throws -> SAMELDecoder {
    var raw = try sa3LoadBundleComponent(dir)
    if let w = raw["mapping.weight"], w.ndim == 3 && w.dim(2) == 1 {
        raw["mapping.weight"] = w.reshaped([w.dim(0), w.dim(1)])
    }
    let model = SAMELDecoder()
    try sa3ApplyFlatWeights(into: model, mapping: raw)
    eval(model.parameters())
    return model
}

/// Load the T5Gemma encoder + bundled SentencePiece tokenizer.
func sa3LoadT5Gemma(dir: URL) throws -> T5GemmaText {
    var raw = try sa3LoadBundleComponent(dir)

    // Tokenizer model bytes are baked into the safetensors under "TOKENIZER_MODEL"
    // as a uint8 byte array. META holds the JSON config (we ignore it since we
    // hard-code the dims; the bundle has fixed dims).
    guard let tokBlobArr = raw.removeValue(forKey: "TOKENIZER_MODEL") else {
        throw StableAudio3Error.missingTensor("TOKENIZER_MODEL")
    }
    _ = raw.removeValue(forKey: "META")
    _ = raw.removeValue(forKey: "rope_inv_freq")  // we compute RoPE on the fly

    // tokBlobArr is uint8 (N,). Parse SentencePiece proto then wrap in Unigram.
    let tokenizer: UnigramTokenizer
    do {
        let bytes = tokBlobArr.asType(.uint8).asArray(UInt8.self)
        let spModel = try SentencePieceModel(data: Data(bytes))
        tokenizer = UnigramTokenizer(model: spModel)
    } catch {
        throw StableAudio3Error.tokenizerLoadFailed("\(error)")
    }

    let model = T5GemmaEncoderModel()
    try sa3ApplyFlatWeights(into: model, mapping: raw)
    eval(model.parameters())
    return T5GemmaText(encoder: model, tokenizer: tokenizer)
}
