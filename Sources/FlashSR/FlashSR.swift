import Foundation
import MLX
import MLXNN
import MLXRandom
import AudioCommon

// MARK: - Public errors

public enum FlashSRError: Error, LocalizedError {
    case missingFile(String)
    case weightLoadFailed(String)
    case unsupportedSampleRate(Int)
    case audioTooShort(Int)

    public var errorDescription: String? {
        switch self {
        case .missingFile(let f): return "FlashSR: missing file \(f)"
        case .weightLoadFailed(let m): return "FlashSR: weight load failed: \(m)"
        case .unsupportedSampleRate(let r): return "FlashSR: unsupported sample rate \(r). Resample to 48 kHz first."
        case .audioTooShort(let n): return "FlashSR: audio too short (\(n) samples). Minimum is 1 sample (will be padded to 5.12s window)."
        }
    }
}

// MARK: - Parameters

public struct FlashSRParams: Sendable {
    public var timestep: Int
    public var seed: UInt64?

    public init(timestep: Int = 999, seed: UInt64? = nil) {
        self.timestep = timestep
        self.seed = seed
    }
}

// MARK: - Public entry

/// One-step versatile audio super-resolution (FlashSR, ICASSP 2025).
///
/// Operates on 5.12-second windows of 48 kHz mono audio. Longer inputs are
/// processed window-by-window (concatenated without overlap-add).
///
/// Pipeline per window:
///   1. Normalize: mean-center + max-abs scale to ±0.5.
///   2. Log-mel spectrogram (n_fft=2048, hop=480, 256 mels).
///   3. VAE encode → 16-channel latent (scaled by 0.3342).
///   4. Single-step DPM-Solver (v-prediction, cosine α̅): one UNet forward.
///   5. VAE decode → reconstructed mel.
///   6. SR Vocoder (mel + LR audio condition) → 48 kHz waveform.
///   7. Denormalize.
public final class FlashSR {
    public let config: FlashSRConfig
    internal let vae: FlashSRAutoencoderKL
    internal let unet: FlashSRAudioSRUnet
    internal let vocoder: FlashSRSRVocoder

    public static let sampleRate = 48000
    public static let frameSamples = 245760
    private static let numTrainSteps = 1000

    public var sampleRate: Int { Self.sampleRate }
    public var frameSamples: Int { Self.frameSamples }

    private init(
        config: FlashSRConfig,
        vae: FlashSRAutoencoderKL,
        unet: FlashSRAudioSRUnet,
        vocoder: FlashSRSRVocoder
    ) {
        self.config = config
        self.vae = vae
        self.unet = unet
        self.vocoder = vocoder
    }

    // MARK: - Loading

    public static func fromPretrained(
        variant: FlashSRVariant = .int4,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws -> FlashSR {
        let paths = try await FlashSRDownloader.ensureDownloaded(
            variant: variant, progressHandler: progressHandler)
        let cfg = try FlashSRConfig.load(from: paths.bundleDir.appendingPathComponent("config.json"))

        // Load the safetensors bundle, then dequantize per `quantized_shapes`.
        let raw: [String: MLXArray]
        do {
            raw = try MLX.loadArrays(url: paths.bundleDir.appendingPathComponent("model.safetensors"))
        } catch {
            throw FlashSRError.weightLoadFailed("model.safetensors: \(error)")
        }
        let dequant = Self.dequantizeBundle(raw, config: cfg)

        // Slice into 3 sub-model dicts on `vae.`, `ldm.`, `voc.` prefixes
        // (PyTorch checkpoint had no conv weight permutation applied; MLX-side
        // permutation was done at export time and the keys are already in NHWC
        // / mlx layout).
        var vaeW: [String: MLXArray] = [:]
        var ldmW: [String: MLXArray] = [:]
        var vocW: [String: MLXArray] = [:]
        for (k, v) in dequant {
            if k.hasPrefix("vae.") { vaeW[String(k.dropFirst("vae.".count))] = v }
            else if k.hasPrefix("ldm.") { ldmW[String(k.dropFirst("ldm.".count))] = v }
            else if k.hasPrefix("voc.") { vocW[String(k.dropFirst("voc.".count))] = v }
        }

        // LDM + Vocoder key remaps: every nn.Sequential / ModuleList-of-Sequential
        // node maps to a FlashSRSeqLayers (a Swift class whose @ModuleInfo
        // property is `layers: [Module]`), so the saved-key path must gain a
        // `.layers.` segment between the parent and the slot index.
        ldmW = Self.remapTSeqKeys(ldmW)
        vocW = Self.remapTSeqKeys(vocW)

        let vae = FlashSRAutoencoderKL(cfg: cfg.vae)
        let unet = FlashSRAudioSRUnet(cfg: cfg.ldm)
        let vocoder = FlashSRSRVocoder()
        try loadWeights(into: vae, mapping: vaeW, label: "vae")
        try loadWeights(into: unet, mapping: ldmW, label: "ldm")
        try loadWeights(into: vocoder, mapping: vocW, label: "voc")

        eval(vae.parameters(), unet.parameters(), vocoder.parameters())
        return FlashSR(config: cfg, vae: vae, unet: unet, vocoder: vocoder)
    }

    /// Apply `mx.dequantize` to every entry in `quantized_shapes`, then reshape
    /// the dequantised tensor back to its original conv/linear shape. Drops
    /// the `.scales` / `.biases` companions. Non-quantised entries pass through.
    private static func dequantizeBundle(
        _ weights: [String: MLXArray], config: FlashSRConfig
    ) -> [String: MLXArray] {
        guard let qcfg = config.quantization, let shapes = config.quantizedShapes else {
            return weights
        }
        let bits = qcfg.bits
        let groupSize = qcfg.groupSize
        var out: [String: MLXArray] = [:]
        for (k, v) in weights {
            if let originalShape = shapes[k] {
                guard let s = weights["\(k).scales"], let b = weights["\(k).biases"] else {
                    continue
                }
                let dq = MLX.dequantized(v, scales: s, biases: b,
                                         groupSize: groupSize, bits: bits)
                out[k] = dq.reshaped(originalShape)
            } else if k.hasSuffix(".scales") || k.hasSuffix(".biases") {
                continue
            } else {
                out[k] = v
            }
        }
        return out
    }

    /// Insert `.layers.` between the FlashSRSeqLayers parent key and the slot
    /// index, for every keyword path that PyTorch saved as a nn.Sequential
    /// (or ModuleList-of-Sequential). Bundle paths look like
    /// `input_blocks.<i>.<j>.<rest>` — our Swift modules need
    /// `input_blocks.<i>.layers.<j>.<rest>` so mlx-swift's unflatten lines up.
    ///
    /// Specifically we handle (for LDM UNet):
    ///   input_blocks.<i>.<j>.*         → input_blocks.<i>.layers.<j>.*
    ///   output_blocks.<i>.<j>.*        → output_blocks.<i>.layers.<j>.*
    ///   middle_block.<j>.*             → middle_block.layers.<j>.*
    ///   time_embed.<j>.*               → time_embed.layers.<j>.*
    ///   out.<j>.*                      → out.layers.<j>.*
    ///   ...in_layers.<j>.*             → ...in_layers.layers.<j>.*
    ///   ...emb_layers.<j>.*            → ...emb_layers.layers.<j>.*
    ///   ...out_layers.<j>.*            → ...out_layers.layers.<j>.*
    ///   ...to_out.<j>.*                → ...to_out.layers.<j>.*
    ///   ...net.<j>.*                   → ...net.layers.<j>.*   (FeedForward inner)
    /// All of these target FlashSRSeqLayers instances.
    private static func remapTSeqKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        // Parent suffixes that wrap FlashSRSeqLayers. We insert `.layers.`
        // between the parent and the next numeric component.
        // Parents whose IMMEDIATE child index is the FlashSRSeqLayers slot
        // (single numeric segment after the parent name).
        let parents: Set<String> = [
            "middle_block", "time_embed", "out",
            "in_layers", "emb_layers", "out_layers", "to_out", "net",
        ]
        // Array-of-FlashSRSeqLayers parents: their child index is an array
        // index, and the NEXT numeric is the slot.
        let arrayParents: Set<String> = [
            "input_blocks", "output_blocks", "downsamples", "ups",
        ]
        var out: [String: MLXArray] = [:]
        for (k, v) in weights {
            out[insertLayersSegment(in: k, parents: parents, arrayParents: arrayParents)] = v
        }
        return out
    }

    /// Walk dot-separated key components, inserting "layers" between:
    ///   - `parents.<int>` patterns (immediate FlashSRSeqLayers child)
    ///   - `arrayParents.<i>.<j>` patterns (array of FlashSRSeqLayers — keep
    ///     the array index, insert "layers" before the slot)
    private static func insertLayersSegment(
        in key: String,
        parents: Set<String>,
        arrayParents: Set<String>
    ) -> String {
        let parts = key.split(separator: ".", omittingEmptySubsequences: false).map(String.init)
        var rewritten: [String] = []
        var i = 0
        while i < parts.count {
            let token = parts[i]
            rewritten.append(token)
            if arrayParents.contains(token),
               i + 2 < parts.count,
               Int(parts[i + 1]) != nil,
               Int(parts[i + 2]) != nil {
                rewritten.append(parts[i + 1])
                rewritten.append("layers")
                i += 2
                continue
            }
            if parents.contains(token),
               i + 1 < parts.count,
               Int(parts[i + 1]) != nil {
                rewritten.append("layers")
            }
            i += 1
        }
        return rewritten.joined(separator: ".")
    }

    private static func loadWeights(
        into module: Module, mapping: [String: MLXArray], label: String
    ) throws {
        let params = ModuleParameters.unflattened(mapping)
        do {
            try module.update(parameters: params, verify: .shapeMismatch)
        } catch {
            throw FlashSRError.weightLoadFailed("\(label): \(error)")
        }
    }

    // MARK: - Inference

    /// Up-sample a single 5.12-second window of 48 kHz mono audio.
    /// `samples.count` must be ≤ frameSamples; shorter inputs are zero-padded.
    public func upsampleWindow(samples: [Float], params: FlashSRParams = FlashSRParams()) -> [Float] {
        if let seed = params.seed { MLXRandom.seed(seed) }
        var window = samples
        if window.count < Self.frameSamples {
            window.append(contentsOf: [Float](repeating: 0, count: Self.frameSamples - window.count))
        } else if window.count > Self.frameSamples {
            window = Array(window.prefix(Self.frameSamples))
        }
        let audio = MLXArray(window).expandedDimensions(axis: 0)                        // (1, T)

        // 1. Normalize
        let (norm, mean, vScale) = normalizeWav(audio)

        // 2. Mel
        let mel = FlashSRMelPreprocessor.melSpec(audio: norm, config: config.mel)         // (1, n_mels, T_mel)
        // VAE wants NHWC where H=T_mel, W=n_mels, C=1.
        let melNHWC = mel.transposed(0, 2, 1).expandedDimensions(axis: -1)               // (1, T_mel, n_mels, 1)

        // 3. VAE encode → cond_z (scaled mean of the posterior)
        let posterior = vae.encode(melNHWC)
        let scale = config.vae.scaleFactorZ
        let condZ = posterior.mean * scale                                                // (1, T_mel/8, n_mels/8, 16)

        // 4. Single-step DPM-Solver (v_prediction, cosine schedule)
        let noise = MLXRandom.normal(condZ.shape)
        let tArr = MLXArray([Int32(params.timestep)])
        let vPred = unet(noise, cond: condZ, timesteps: tArr)
        let alpha = Self.cosineAlphaBar(params.timestep)
        let sa = Float(alpha.squareRoot())
        let sb = Float(max(1.0 - alpha, 1e-12).squareRoot())
        let z0 = MLXArray(sa) * noise - MLXArray(sb) * vPred

        // 5. VAE decode
        let melRec = vae.decode(z0 / scale)                                               // (1, T_mel, n_mels, 1)
        let melForVoc = melRec.squeezed(axis: -1)                                         // (1, T_mel, n_mels)

        // 6. SR Vocoder. mel: (B, T_mel, n_mels); lr_audio: (B, T_wav).
        let hrNorm = vocoder(melForVoc, lrAudio: norm)                                     // (1, T_wav_hr)

        // 7. Denormalize
        let hr = denormalizeWav(hrNorm, mean: mean, scale: vScale)
        eval(hr)
        return Array(hr[0].asArray(Float.self).prefix(Self.frameSamples))
    }

    /// Up-sample arbitrary-length 48 kHz mono audio by processing
    /// non-overlapping 5.12-second windows. Inputs shorter than one window are
    /// padded; the output is trimmed to the original input length.
    public func upsample(audio: [Float], sampleRate: Int) throws -> [Float] {
        if sampleRate != Self.sampleRate {
            throw FlashSRError.unsupportedSampleRate(sampleRate)
        }
        if audio.isEmpty { throw FlashSRError.audioTooShort(0) }
        let win = Self.frameSamples
        let nWindows = (audio.count + win - 1) / win
        var out = [Float](); out.reserveCapacity(nWindows * win)
        for w in 0..<nWindows {
            let start = w * win
            let end = min(start + win, audio.count)
            let slice = Array(audio[start..<end])
            out.append(contentsOf: upsampleWindow(samples: slice))
        }
        // Trim to original input length (we processed in fixed windows).
        return Array(out.prefix(audio.count))
    }

    // MARK: - Math helpers

    private static func cosineAlphaBar(_ t: Int, total: Int = numTrainSteps) -> Double {
        let f = (Double(t) / Double(total) + 0.008) / 1.008
        let c = cos(f * .pi / 2.0)
        return c * c
    }

    private func normalizeWav(_ audio: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let mean = MLX.mean(audio, axis: -1, keepDims: true)
        let centered = audio - mean
        let vMax = MLX.max(abs(centered), axis: -1, keepDims: true)
        let scaled = centered / (vMax + MLXArray(Float(1e-8)))
        return (scaled * MLXArray(Float(0.5)), mean, vMax)
    }

    private func denormalizeWav(_ audio: MLXArray, mean: MLXArray, scale: MLXArray) -> MLXArray {
        return audio * Float(2.0) * (scale + MLXArray(Float(1e-8))) + mean
    }
}
