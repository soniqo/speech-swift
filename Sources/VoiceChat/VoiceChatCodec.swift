import Foundation
import MLX
import MLXFFT
import MLXNN

/// Decoder half of VoiceChat's 22.05 kHz neural audio codec.
///
/// One frame contains 31 RVQ codes. Dequantization sums one 512-dimensional
/// mean from each codebook, then the decoder upsamples by 9 × 7 × 7 and a
/// 16-point ISTFT produces exactly 1,764 samples (80 ms).
public final class VoiceChatCodec {
    public static let sampleRate = 22_050
    public static let samplesPerFrame = 1_764
    public static let quantizers = 31
    public static let codebookSize = 1_024
    public static let latentSize = 512
    public static let silenceRMSLimit: Float = 1e-5

    public struct SilenceMetrics: Sendable {
        public let rms: Float
        public let peak: Float
        public var passed: Bool { rms < VoiceChatCodec.silenceRMSLimit }
    }

    private let weights: [String: MLXArray]
    private let codebooks: [MLXArray]
    private var compiledLiveDecode: (@Sendable (MLXArray) -> MLXArray)?
    public let silenceCodes: MLXArray

    public init(weights: [String: MLXArray]) throws {
        let prefix = "tts_model.audio_codec."
        // The converter stores codec parameters densely at fp16 to keep the
        // bundle compact. NeMo and the verified Python decoder both perform
        // the codec math in fp32, so promote once at load time rather than
        // relying on mixed-dtype convolution promotion rules.
        let promoted = Dictionary(uniqueKeysWithValues: weights.lazy
            .filter { $0.key.hasPrefix(prefix) }
            .map { ($0.key, $0.value.asType(.float32)) })
        guard promoted.count == 214 else {
            throw VoiceChatLoadError.unexpectedKeys([
                "expected 214 tts_model.audio_codec tensors, found \(promoted.count)",
            ])
        }
        let codebooks = try (0 ..< Self.quantizers).map { index in
            guard let value = promoted["tts_model.audio_codec.prvq.mus_list.\(index)"] else {
                throw VoiceChatLoadError.unexpectedKeys(["audio_codec.prvq.mus_list.\(index)"])
            }
            guard value.shape == [Self.codebookSize, Self.latentSize] else {
                throw VoiceChatLoadError.unexpectedKeys([
                    "audio_codec.prvq.mus_list.\(index) shape \(value.shape)",
                ])
            }
            return value
        }
        guard let silence = weights["tts_model.codec_silence_tokens"] else {
            throw VoiceChatLoadError.unexpectedKeys(["tts_model.codec_silence_tokens"])
        }
        guard silence.shape == [Self.quantizers] else {
            throw VoiceChatLoadError.unexpectedKeys([
                "tts_model.codec_silence_tokens shape \(silence.shape)",
            ])
        }

        let directLayers = Set([0, 4, 8, 12])
        let residualSuffixes = [
            "dwconv.weight", "dwconv.bias", "norm.weight", "norm.bias",
            "pwconv1.weight", "pwconv1.bias", "pwconv2.weight", "pwconv2.bias",
        ]
        var missingDecoderWeights: [String] = []
        for index in 0 ..< 13 {
            let layer = "tts_model.audio_codec.decoder.layers.\(index)"
            let required = directLayers.contains(index)
                ? ["weight"]
                : residualSuffixes
            missingDecoderWeights += required.compactMap {
                promoted["\(layer).\($0)"] == nil ? "\(layer).\($0)" : nil
            }
        }
        guard missingDecoderWeights.isEmpty else {
            throw VoiceChatLoadError.unexpectedKeys(missingDecoderWeights)
        }

        self.weights = promoted
        self.codebooks = codebooks
        self.silenceCodes = silence.asType(.int32)
    }

    /// Load from a complete bundle root, its `tts/` directory, or the original
    /// checkpoint directory. The converter deliberately keeps codec weights
    /// dense fp16 in every quantized variant.
    public static func load(from directory: URL) throws -> VoiceChatCodec {
        let fileManager = FileManager.default
        let bundled = directory.appendingPathComponent("tts/model.safetensors")
        let direct = directory.appendingPathComponent("model.safetensors")
        let weightsURL = fileManager.fileExists(atPath: bundled.path) ? bundled : direct
        guard fileManager.fileExists(atPath: weightsURL.path) else {
            throw VoiceChatLoadError.missingWeights(weightsURL)
        }
        return try VoiceChatCodec(weights: MLX.loadArrays(url: weightsURL))
    }

    /// `(B, T, 31)` integer codes → `(B, T, 512)` codec latents.
    public func dequantize(_ codes: MLXArray) -> MLXArray {
        precondition(codes.shape.count == 3 && codes.dim(-1) == Self.quantizers)
        let ids = MLX.minimum(
            MLX.maximum(codes.asType(.int32), MLXArray(Int32(0))),
            MLXArray(Int32(Self.codebookSize - 1)))
        var latent = MLXArray.zeros(
            [codes.dim(0), codes.dim(1), Self.latentSize], dtype: .float32)
        for index in 0 ..< Self.quantizers {
            latent = latent + codebooks[index][ids[0..., 0..., index]]
        }
        return latent
    }

    /// Decode `(B, T, 31)` codes directly to `(B, T * 1764)` waveform samples.
    public func decode(codes: MLXArray) -> MLXArray {
        decode(latents: dequantize(codes))
    }

    /// Decode `(B, T, 512)` dequantized latents.
    public func decode(latents: MLXArray) -> MLXArray {
        if latents.dim(1) == VoiceChatSession.codecContextFrames,
           let compiledLiveDecode {
            return compiledLiveDecode(latents.asType(.float32))
        }
        return decodeUncompiled(latents: latents)
    }

    /// Compile the steady-state eight-frame live window once. Short startup
    /// windows and arbitrary offline lengths keep the general eager path.
    func warmUpLiveDecoding() {
        if compiledLiveDecode == nil {
            compiledLiveDecode = compile(shapeless: false) { [unowned self] in
                self.decodeUncompiled(latents: $0)
            }
        }
        let input = MLXArray.zeros(
            [1, VoiceChatSession.codecContextFrames, Self.latentSize],
            dtype: .float32)
        eval(compiledLiveDecode!(input))
    }

    private func decodeUncompiled(latents: MLXArray) -> MLXArray {
        var x = latents.asType(.float32)
        for index in 0 ..< 13 {
            x = decoderLayer(index, x)
        }
        return Self.inverseSTFT(x)
    }

    /// The cheapest codec regression gate: the checkpoint's canonical silence
    /// frame must remain below 1e-5 RMS. The historical magnitude/phase bug
    /// produced RMS 1.60 and is caught here in one frame.
    public func verifySilence() -> SilenceMetrics {
        let codes = silenceCodes.reshaped([1, 1, Self.quantizers])
        let audio = decode(codes: codes)
        eval(audio)
        let rms = MLX.sqrt(MLX.mean(audio.square())).item(Float.self)
        let peak = MLX.max(MLX.abs(audio)).item(Float.self)
        return SilenceMetrics(rms: rms, peak: peak)
    }

    private func decoderLayer(_ index: Int, _ x: MLXArray) -> MLXArray {
        let prefix = "tts_model.audio_codec.decoder.layers.\(index)"
        if let stride = [0: 9, 4: 7, 8: 7, 12: 1][index],
           let weight = weights["\(prefix).weight"] {
            if stride == 1 {
                // NeMo Conv1d [out, in, kernel] → MLX [out, kernel, in].
                return MLX.conv1d(
                    x, weight.transposed(0, 2, 1), stride: 1,
                    padding: (weight.dim(2) - 1) / 2)
            }
            // NeMo ConvTranspose1d [in, out, kernel] → MLX [out, kernel, in].
            return MLX.convTransposed1d(
                x, weight.transposed(1, 2, 0), stride: stride,
                padding: (weight.dim(2) - stride) / 2)
        }

        let channels = weights["\(prefix).dwconv.weight"]!.dim(0)
        var h = MLX.padded(
            x, widths: [IntOrPair(0), IntOrPair((6, 0)), IntOrPair(0)])
        h = MLX.conv1d(
            h, weights["\(prefix).dwconv.weight"]!.transposed(0, 2, 1),
            groups: channels)
        h = h + weights["\(prefix).dwconv.bias"]!
        h = layerNorm(
            h, weight: weights["\(prefix).norm.weight"]!,
            bias: weights["\(prefix).norm.bias"]!)
        h = MLX.conv1d(
            h, weights["\(prefix).pwconv1.weight"]!.transposed(0, 2, 1))
        h = gelu(h + weights["\(prefix).pwconv1.bias"]!)
        h = MLX.conv1d(
            h, weights["\(prefix).pwconv2.weight"]!.transposed(0, 2, 1))
        h = h + weights["\(prefix).pwconv2.bias"]!
        return x + h
    }

    private func layerNorm(_ x: MLXArray, weight: MLXArray, bias: MLXArray) -> MLXArray {
        let mean = MLX.mean(x, axis: -1, keepDims: true)
        let variance = MLX.mean((x - mean).square(), axis: -1, keepDims: true)
        return (x - mean) * MLX.rsqrt(variance + MLXArray(Float(1e-5))) * weight + bias
    }

    /// Magnitude/phase ISTFT used by the checkpoint. This is intentionally
    /// internal so no-weight unit tests can hold the DSP contract directly.
    static func inverseSTFT(_ channels: MLXArray) -> MLXArray {
        precondition(channels.dim(-1) == 18)
        let bins = 9
        let magnitudeRaw = channels[0..., 0..., ..<bins]
        let phase = channels[0..., 0..., bins...]
        let magnitude = MLXArray(Float(100)) * MLX.exp(
            -softplus(-magnitudeRaw + MLXArray(Float(Foundation.log(100.0)))))

        let real = magnitude * MLX.cos(phase)
        var imag = magnitude * MLX.sin(phase)
        imag = MLX.concatenated([
            MLXArray.zeros(like: imag[0..., 0..., ..<1]),
            imag[0..., 0..., 1 ..< (bins - 1)],
            MLXArray.zeros(like: imag[0..., 0..., (bins - 1)...]),
        ], axis: -1)

        let window = periodicHann(16)
        var frames = MLXFFT.irfft(real + imag.asImaginary(), n: 16, axis: -1)
        let bounded = MLX.minimum(MLX.maximum(frames, -window), window)
        frames = bounded * window

        let batch = frames.dim(0)
        let frameCount = frames.dim(1)
        let segments = frames.reshaped([batch, frameCount, 4, 4])
        var accumulated: MLXArray?
        for offset in 0 ..< 4 {
            let part = segments[0..., 0..., offset, 0...]
            let padded = MLX.padded(
                part,
                widths: [IntOrPair(0), IntOrPair((offset, 3 - offset)), IntOrPair(0)])
            accumulated = accumulated.map { $0 + padded } ?? padded
        }
        let combined = accumulated!.reshaped([batch, (frameCount + 3) * 4])

        let squared = (window * window).reshaped([4, 4])
        var envelope: MLXArray?
        for offset in 0 ..< 4 {
            let row = MLX.broadcast(
                squared[offset, 0...].reshaped([1, 4]), to: [frameCount, 4])
            let padded = MLX.padded(
                row, widths: [IntOrPair((offset, 3 - offset)), IntOrPair(0)])
            envelope = envelope.map { $0 + padded } ?? padded
        }
        let norm = envelope!.reshaped([(frameCount + 3) * 4])
        let normalized = combined / MLX.maximum(norm, MLXArray(Float(1e-11)))
        return normalized[0..., 6 ..< (normalized.dim(1) - 6)]
    }

    static func periodicHann(_ count: Int) -> MLXArray {
        MLXArray((0 ..< count).map { index in
            Float(0.5 - 0.5 * Foundation.cos(2 * .pi * Double(index) / Double(count)))
        })
    }
}
