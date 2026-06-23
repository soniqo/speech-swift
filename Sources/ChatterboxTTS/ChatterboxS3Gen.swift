import Foundation
import MLX
import MLXCommon
import MLXNN

// MARK: - ChatterboxS3Gen — the speech-token → 24 kHz waveform stage
//
// Orchestrates the S3Gen components that turn the T3 speech tokens (plus a
// reference clip) into audio:
//
//   reference clip
//     → CAMPPlus x-vector (192-d, 16 kHz fbank)               [conditioning]
//     → S3TokenizerV2 prompt-token codes (16 kHz)             [prompt tokens]
//     → 24 kHz 80-mel prompt feature                          [prompt mel / cond]
//
//   speech tokens + reference
//     → flow encoder (Conformer) → encoder_proj → mu
//     → MatchaCFM ODE solve (mu, mask, projected spk, prompt-mel cond) → mel
//     → HiFTGenerator vocoder → waveform
//     → 20 ms fade-in trim
//
// This mirrors `CausalMaskedDiffWithXvec.inference` (flow) and
// `S3Token2Wav` (embed_ref / flow_inference / hift_inference / trim_fade).

/// The reference conditioning produced by `embedRef`: the speaker x-vector, the
/// reference prompt-token codes, and the 24 kHz prompt mel — together they form
/// the `ref_dict` the flow consumes.
public struct ChatterboxS3GenRef {
    /// Raw CAMPPlus x-vector `[1, 192]` (not yet L2-normalized / affine-projected).
    public let xVector: MLXArray
    /// Reference prompt speech-token codes (length-aligned to `promptFeat`).
    public let promptToken: [Int]
    /// 24 kHz 80-mel prompt feature `[1, T_mel, 80]` (used both to align the
    /// token length and as the CFM `cond`).
    public let promptFeat: MLXArray
}

/// The S3Gen waveform-synthesis stage of Chatterbox.
public final class ChatterboxS3Gen {
    // Components (loaded from the bundle by `ChatterboxTTSModel`).
    public let speakerEncoder: CAMPPlus
    public let tokenizer: S3TokenizerV2
    public let flow: S3GenConformer
    public let cfm: MatchaCFM
    public let vocoder: S3GenVocoder

    /// 24 kHz S3Gen sample rate.
    public static let sampleRate = 24000
    /// 16 kHz tokenizer / speaker-encoder sample rate.
    public static let tokenSampleRate = 16000
    /// `token_mel_ratio` — each speech token maps to 2 mel frames.
    public static let tokenMelRatio = 2

    public init(
        speakerEncoder: CAMPPlus,
        tokenizer: S3TokenizerV2,
        flow: S3GenConformer,
        cfm: MatchaCFM,
        vocoder: S3GenVocoder
    ) {
        self.speakerEncoder = speakerEncoder
        self.tokenizer = tokenizer
        self.flow = flow
        self.cfm = cfm
        self.vocoder = vocoder
    }

    // MARK: - 24 kHz prompt-feature mel front-end
    //
    // Matches the reference `mel_spectrogram(n_fft=1920, hop=480, win=1920,
    // fmin=0, fmax=8000, center=False)`: it reflect-pads by `(n_fft-hop)/2 = 720`
    // samples (NOT n_fft/2), runs a symmetric-Hann STFT, takes the *magnitude*
    // (power=1), projects through a slaney/slaney mel filterbank, and applies
    // `log(max(mel, 1e-5))`. Distinct from the speaker-mel and tokenizer-mel
    // front-ends — note the 720 reflect pad and `center=False`.

    /// 24 kHz prompt mel `[T, 80]` from 24 kHz mono samples.
    static func promptFeatMel(samples24k: [Float]) -> MLXArray {
        let nFft = 1920
        let hop = 480
        let padAmount = (nFft - hop) / 2  // 720

        // numpy `mode='reflect'` (excludes the boundary sample), matching the
        // reference `_reflect_pad_2d`. SlaneyMel then frames with centerPad=false.
        let padded = reflectPad(samples24k, pad: padAmount)

        let config = SlaneyMelConfig(
            sampleRate: Self.sampleRate, nFft: nFft, hop: hop, win: nFft, nMels: 80,
            fmin: 0, fmax: 8000, power: 1.0,
            logMel: true, logFloor: 1e-5, centerPad: false)
        return SlaneyMel.melSpec(samples: padded, config: config)  // [T, 80]
    }

    /// numpy `mode='reflect'` padding (excludes the boundary sample), as the
    /// reference uses before the `center=False` STFT.
    private static func reflectPad(_ row: [Float], pad: Int) -> [Float] {
        guard pad > 0, row.count > pad else { return row }
        var out = [Float](repeating: 0, count: pad + row.count + pad)
        for i in 0 ..< row.count { out[pad + i] = row[i] }
        for i in 0 ..< pad { out[i] = row[pad - i] }
        let last = row.count - 1
        for i in 0 ..< pad { out[pad + row.count + i] = row[last - 1 - i] }
        return out
    }

    // MARK: - embed_ref

    /// Build the reference conditioning from a reference clip.
    ///
    /// Mirrors `S3Token2Wav.embed_ref`: the 24 kHz mel becomes `promptFeat`, the
    /// 16 kHz audio drives both CAMPPlus (x-vector) and S3TokenizerV2
    /// (prompt-token codes), and the mel / token lengths are forced consistent so
    /// `mel_frames == 2 * token_len`.
    ///
    /// - Parameters:
    ///   - refWav24k: reference samples at 24 kHz (mono).
    ///   - refWav16k: reference samples at 16 kHz (mono) — the 24 kHz clip
    ///     resampled to 16 kHz, used for the tokenizer and CAMPPlus.
    public func embedRef(refWav24k: [Float], refWav16k: [Float]) -> ChatterboxS3GenRef {
        // 24 kHz prompt mel: [T, 80] -> [1, T, 80].
        var promptFeat = Self.promptFeatMel(samples24k: refWav24k)
            .expandedDimensions(axis: 0)
        eval(promptFeat)

        // 16 kHz x-vector (raw, L2 + affine happen later inside the flow).
        let xVector = speakerEncoder.inference(refWav16k)  // [1, 192]

        // 16 kHz reference codes.
        var promptToken = tokenizer.encode(refWav16k)

        // Force mel_len == 2 * token_len (truncate whichever is longer).
        var actualTokenLen = promptToken.count
        let expectedTokenLen = promptFeat.dim(1) / Self.tokenMelRatio
        if actualTokenLen != expectedTokenLen {
            if actualTokenLen < expectedTokenLen {
                let expectedMelLen = Self.tokenMelRatio * actualTokenLen
                promptFeat = promptFeat[0..., 0 ..< expectedMelLen, 0...]
            } else {
                promptToken = Array(promptToken.prefix(expectedTokenLen))
                actualTokenLen = expectedTokenLen
            }
        }
        eval(promptFeat)
        return ChatterboxS3GenRef(xVector: xVector, promptToken: promptToken, promptFeat: promptFeat)
    }

    // MARK: - flow + vocoder

    /// Run the flow encoder + CFM to produce the generated mel `[1, 80, T_gen]`.
    ///
    /// Mirrors `CausalMaskedDiffWithXvec.inference` with `finalize=True`:
    ///   - prepend `promptToken` to `speechTokens` (axis 1), embed, encode,
    ///     project → `mu`;
    ///   - L2-normalize the x-vector then affine → `spks`;
    ///   - place `promptFeat` in the first `mel_len1` frames of a zero `cond`;
    ///   - solve the CFM, then drop the prompt mel region.
    public func flowMel(speechTokens: [Int], ref: ChatterboxS3GenRef) -> MLXArray {
        let dtype = flow.encoderProj.weight.dtype

        // token = concat(prompt_token, token) on axis 1.
        let allTokens = ref.promptToken + speechTokens
        let tokenArr = MLXArray(allTokens.map { Int32($0) }).reshaped([1, allTokens.count])

        // input_embedding(token) — single unpadded sequence, so the mask is all 1s.
        let tokenEmb = flow.inputEmbedding(tokenArr).asType(dtype)  // [1, T, 512]
        let h = flow.encode(tokenEmbeddings: tokenEmb)              // [1, T_mel, 80] (mu)

        let melLen1 = ref.promptFeat.dim(1)
        let melLen2 = h.dim(1) - melLen1
        let totalLen = melLen1 + melLen2

        // spk: L2-normalize (with epsilon) then affine — matches flow.inference.
        let spks = flow.projectSpeaker(ref.xVector.asType(dtype))  // [1, 80]

        // cond: zeros [1, total, 80] with prompt mel in the first mel_len1 frames,
        // transposed to [1, 80, total].
        var cond = MLXArray.zeros([1, totalLen, flow.config.melDim]).asType(dtype)
        let promptFeatD = ref.promptFeat.asType(dtype)
        cond[0..., 0 ..< melLen1, 0...] = promptFeatD
        cond = cond.transposed(0, 2, 1)  // [1, 80, total]

        // mu/mask in NCL; decoder mask is all-ones over the full length.
        let mu = h.transposed(0, 2, 1)                              // [1, 80, total]
        let mask = MLXArray.ones([1, 1, totalLen]).asType(dtype)

        var mel = cfm.solve(mu: mu, mask: mask, spks: spks, cond: cond)  // [1, 80, total]

        // Drop the prompt mel region — keep only the generated frames.
        mel = mel[0..., 0..., melLen1...]                            // [1, 80, mel_len2]
        eval(mel)
        return mel
    }

    /// Full synthesis: speech tokens + reference → 24 kHz waveform `[Float]`.
    ///
    /// Mirrors `S3Token2Wav.__call__`: flow mel → HiFTGenerator → 20 ms fade-in.
    public func synthesize(speechTokens: [Int], ref: ChatterboxS3GenRef) -> [Float] {
        let mel = flowMel(speechTokens: speechTokens, ref: ref)     // [1, 80, T]
        var wav = vocoder.inference(mel: mel.asType(.float32))      // [1, samples]

        // trim_fade: zero the first 20 ms, raised-cosine fade-in over the next
        // 20 ms. n_trim = sr/50 = 480; window length = 960.
        let nTrim = Self.sampleRate / 50
        let fadeLen = 2 * nTrim
        if wav.dim(1) >= fadeLen {
            let fade = Self.trimFade(nTrim: nTrim).asType(wav.dtype)  // [960]
            let head = wav[0..., 0 ..< fadeLen] * fade.reshaped([1, fadeLen])
            let tail = wav[0..., fadeLen...]
            wav = concatenated([head, tail], axis: 1)
        }
        eval(wav)
        let flat = wav.reshaped([wav.size]).asType(.float32).asArray(Float.self)
        return flat
    }

    /// The reference `trim_fade` window: `n_trim` zeros followed by an
    /// `n_trim`-sample raised-cosine ramp rising 0 → 1.
    private static func trimFade(nTrim: Int) -> MLXArray {
        // (cos(linspace(pi, 0, n_trim)) + 1) / 2
        var ramp = [Float](repeating: 0, count: nTrim)
        for i in 0 ..< nTrim {
            let t = Float.pi - Float.pi * Float(i) / Float(max(nTrim - 1, 1))
            ramp[i] = (cos(t) + 1.0) / 2.0
        }
        let zeros = [Float](repeating: 0, count: nTrim)
        return MLXArray(zeros + ramp)
    }
}
