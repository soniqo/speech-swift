import Foundation
import MLX
import MLXNN

/// The speech-understanding front end of NemotronLabs VoiceChat 11B:
/// streaming FastConformer encoder plus the single Linear that bridges its
/// output into the language model's hidden size.
///
/// The NeMo config calls the modality adapter an `IdentityConnector`, so this
/// projection is the whole bridge — encoder output goes straight in as input
/// embeddings for the Nemotron-H backbone.
public final class VoiceChatPerception: Module {
    @ModuleInfo(key: "encoder") public var encoder: VoiceChatEncoder
    @ModuleInfo(key: "modality_proj") public var modalityProj: Linear

    public let config: VoiceChatPerceptionConfig

    public init(_ config: VoiceChatPerceptionConfig) {
        self.config = config
        self._encoder.wrappedValue = VoiceChatEncoder(config.encoder)
        self._modalityProj.wrappedValue = Linear(
            config.modalityProj.inFeatures, config.modalityProj.outFeatures)
    }

    /// - Parameter logMel: (B, T, 128) log-mel at 16 kHz
    /// - Returns: (B, T/8, 4480) — already in the language model's embedding space
    public func callAsFunction(_ logMel: MLXArray) -> MLXArray {
        modalityProj(encoder(logMel))
    }

    /// Per-stage timings for one encode, in milliseconds.
    ///
    /// Split this way because the two halves scale differently: subsampling is
    /// convolutional and roughly linear in mel frames, while the conformer
    /// stack carries the attention term. Knowing which dominates at a given
    /// utterance length is what tells you where optimisation is worth spending.
    public struct StageTimings: Sendable {
        public let subsamplingMs: Double
        public let conformerMs: Double
        public let projectionMs: Double
        public var totalMs: Double { subsamplingMs + conformerMs + projectionMs }
    }

    /// Each stage is timed directly over `iterations` runs after a warmup, so
    /// the numbers add up to the measured total instead of being inferred by
    /// subtracting one noisy measurement from another.
    public func profile(_ logMel: MLXArray, iterations: Int = 8, warmup: Int = 2) -> StageTimings {
        func time(_ body: () -> MLXArray) -> Double {
            for _ in 0 ..< warmup { eval(body()) }
            var samples: [Double] = []
            for _ in 0 ..< iterations {
                let start = DispatchTime.now().uptimeNanoseconds
                eval(body())
                samples.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000)
            }
            // Median, not mean: a single scheduling hiccup should not move the
            // reported cost of a stage.
            return samples.sorted()[samples.count / 2]
        }

        let subsampled = encoder.preEncode(logMel)
        eval(subsampled)
        let hidden = encoder(logMel)
        eval(hidden)

        return StageTimings(
            subsamplingMs: time { self.encoder.preEncode(logMel) },
            conformerMs: time { self.encoder.conformerStack(subsampled) },
            projectionMs: time { self.modalityProj(hidden) })
    }

    /// Frames the encoder emits for a given mel length. Nominally one per 80 ms,
    /// running one frame long because each causal stage is `floor(n/2) + 1`.
    public func outputFrames(melFrames: Int) -> Int {
        CausalSubsampling.outputFrames(melFrames: melFrames)
    }
}

// MARK: - Loading

public enum VoiceChatLoadError: Error, CustomStringConvertible {
    case missingWeights(URL)
    case unexpectedKeys([String])

    public var description: String {
        switch self {
        case .missingWeights(let url):
            return "no model.safetensors at \(url.path)"
        case .unexpectedKeys(let keys):
            return "bundle contains keys the module tree has no slot for: \(keys.prefix(5).joined(separator: ", "))"
        }
    }
}

/// The subsampling stack is a flat list in the checkpoint (`pre_encode.conv.0`,
/// `.2`, `.3`, `.5`, `.6`) because that is how NeMo and the Python export store
/// it. Swift's `@ModuleInfo` treats a dotted key as a single literal name rather
/// than a path, so the index is folded into the key instead:
///
///     pre_encode.conv.0.conv.weight  ->  pre_encode.conv0.conv.weight
///     pre_encode.conv.3.weight       ->  pre_encode.conv3.weight
///
/// Indices 0, 2 and 5 keep their inner `.conv` because those entries are
/// `CausalConv2D` wrappers around a `Conv2d`; 3 and 6 are plain convolutions.
private func rename(_ key: String) -> String {
    guard let range = key.range(of: "pre_encode.conv.") else { return key }
    let tail = key[range.upperBound...]
    guard let dot = tail.firstIndex(of: "."),
          Int(tail[tail.startIndex ..< dot]) != nil
    else { return key }
    let index = tail[tail.startIndex ..< dot]
    return key[key.startIndex ..< range.lowerBound]
        + "pre_encode.conv\(index)"
        + tail[dot...]
}

public extension VoiceChatPerception {
    /// Load an exported `encoder/` bundle.
    ///
    /// The bundle also carries the RNNT transcript head (`decoder.*`, `joint.*`)
    /// and the mel filterbank (`preprocessor.*`). Neither belongs to this module
    /// tree, so both are filtered out rather than rejected — the duplex runtime
    /// picks them up separately.
    static func load(from directory: URL) throws -> VoiceChatPerception {
        let config = try VoiceChatPerceptionConfig.load(from: directory)
        let weightsURL = directory.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw VoiceChatLoadError.missingWeights(weightsURL)
        }

        let model = VoiceChatPerception(config)
        var weights = try MLX.loadArrays(url: weightsURL)
        weights = weights.filter { key, _ in
            key.hasPrefix("encoder.") || key.hasPrefix("modality_proj.")
        }
        weights = Dictionary(uniqueKeysWithValues: weights.map { (rename($0.key), $0.value) })

        if let quantization = config.quantization {
            // Only Linear layers were quantized on the export side; the
            // relative-position biases pos_bias_u/v are raw parameters and stay
            // dense even though they are two-dimensional.
            quantize(model: model, groupSize: quantization.groupSize, bits: quantization.bits) {
                path, module in
                guard module is Linear else { return false }
                return weights["\(path).scales"] != nil
            }
        }

        let parameters = ModuleParameters.unflattened(weights)
        // Diff the module tree against the bundle before updating: a structural
        // mismatch here crashes rather than throwing, so report it ourselves.
        let expected = Set(model.parameters().flattened().map { $0.0 })
        let provided = Set(weights.keys)
        let missing = expected.subtracting(provided).sorted()
        let extra = provided.subtracting(expected).sorted()
        if !missing.isEmpty || !extra.isEmpty {
            FileHandle.standardError.write(
                "[load] missing \(missing.count): \(missing.prefix(6))\n[load] extra \(extra.count): \(extra.prefix(6))\n"
                    .data(using: .utf8)!)
            throw VoiceChatLoadError.unexpectedKeys(missing + extra)
        }
        var shapeMismatch: [String] = []
        for (key, array) in model.parameters().flattened() {
            if let incoming = weights[key], incoming.shape != array.shape {
                shapeMismatch.append("\(key): tree \(array.shape) vs bundle \(incoming.shape)")
            }
        }
        if !shapeMismatch.isEmpty {
            FileHandle.standardError.write(
                "[load] shape mismatches \(shapeMismatch.count):\n  \(shapeMismatch.prefix(8).joined(separator: "\n  "))\n"
                    .data(using: .utf8)!)
            throw VoiceChatLoadError.unexpectedKeys(shapeMismatch)
        }
        model.update(parameters: parameters)
        eval(model)
        return model
    }
}
