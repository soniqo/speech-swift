import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - RepeatInterleaveUpsampler

/// Upsamples token embeddings from token rate (25 Hz) to mel rate (50 Hz) via repeat-interleave.
///
/// This is a pure function with no learnable parameters. Each input frame is repeated `ratio`
/// times along the time axis, matching Python's `torch.repeat_interleave(ratio, dim=1)`.
///
/// For `ratio=2`: `[a, b, c]` → `[a, a, b, b, c, c]`
public enum RepeatInterleaveUpsampler {

    /// Upsample by repeating each frame `ratio` times along the time axis.
    ///
    /// - Parameters:
    ///   - x: `[B, T, D]` input tensor
    ///   - ratio: integer repeat factor (e.g. 2 for 25 Hz -> 50 Hz)
    /// - Returns: `[B, T*ratio, D]` upsampled tensor
    public static func upsample(_ x: MLXArray, ratio: Int) -> MLXArray {
        guard ratio > 1 else { return x }

        // [B, T, D] → [B, T, 1, D] → repeat along axis 2 → [B, T, ratio, D] → [B, T*ratio, D]
        let expanded = x.expandedDimensions(axis: 2)             // [B, T, 1, D]
        let rep = repeated(expanded, count: ratio, axis: 2)      // [B, T, ratio, D]
        var shape = x.shape
        shape[1] *= ratio
        return rep.reshaped(shape)                                // [B, T*ratio, D]
    }
}

// MARK: - ConditionalFlowMatching

/// ODE solver with classifier-free guidance for flow matching.
///
/// Uses the Euler method to integrate the velocity field predicted by the DiT decoder,
/// applying classifier-free guidance (CFG) to improve sample quality. The ODE evolves
/// from pure noise (`t=0`) to the target distribution (`t=1`) over `nTimesteps` steps.
public class ConditionalFlowMatching: Module {
    public let config: CosyVoiceFlowConfig

    @ModuleInfo var decoder: DiT

    /// Compiled DiT forward passes for kernel fusion. Uses shapeless=false
    /// since all shapes are constant across the 10 ODE steps within a
    /// generation. Two variants: nil spks/cond (plain TTS) and full
    /// conditioning (zero-shot cloning: spks + cond present).
    private var compiledDiTForward: (([MLXArray]) -> [MLXArray])?
    private var compiledDiTForwardFull: (([MLXArray]) -> [MLXArray])?

    /// Fixed initial ODE noise `[1, 80, N]`. Upstream CausalConditionalCFM
    /// never samples fresh noise — it slices a fixed torch-seed-0 buffer
    /// (`rand_noise = randn([1, 80, 50 * 300])`), an implicitly validated
    /// draw for the 10-step Euler solver. Set from a bundle-provided buffer
    /// (see `loadFixedNoise`); when nil, a keyed draw provides the same
    /// fixed-noise property without depending on the global RNG.
    public var fixedNoise: MLXArray?

    public init(config: CosyVoiceFlowConfig) {
        self.config = config
        self._decoder.wrappedValue = DiT(config: config.dit)
        super.init()
    }

    /// Load a raw little-endian float32 noise buffer and reshape to
    /// `[1, 80, N]` (the upstream `rand_noise` layout).
    public static func loadFixedNoise(path: String) -> MLXArray? {
        guard let data = FileManager.default.contents(atPath: path) else { return nil }
        let count = data.count / MemoryLayout<Float>.size
        guard count >= 80, count % 80 == 0 else { return nil }
        let floats = data.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self).prefix(count))
        }
        return MLXArray(floats, [1, 80, count / 80])
    }

    /// Set up compiled DiT forward passes for Metal kernel fusion.
    ///
    /// With shapeless=false, the compiled graph is traced once per input shape
    /// and reused across all 10 ODE steps (shapes are identical each step).
    /// Fuses ~330 Metal kernel dispatches (22 DiT layers) per forward pass.
    public func setupCompilation() {
        let decoderRef = decoder

        compiledDiTForward = compile(
            inputs: [decoderRef], outputs: [decoderRef], shapeless: false
        ) { inputs in
            let velocity = decoderRef(
                inputs[0], mask: inputs[1], mu: inputs[2], t: inputs[3],
                spks: nil, cond: nil)
            return [velocity]
        }

        compiledDiTForwardFull = compile(
            inputs: [decoderRef], outputs: [decoderRef], shapeless: false
        ) { inputs in
            let velocity = decoderRef(
                inputs[0], mask: inputs[1], mu: inputs[2], t: inputs[3],
                spks: inputs[4], cond: inputs[5])
            return [velocity]
        }
    }

    /// Execute DiT forward pass (compiled when available, falls back to
    /// uncompiled for the mixed spks-without-cond case).
    private func executeDiTForward(
        x: MLXArray, mask: MLXArray, mu: MLXArray, t: MLXArray,
        spks: MLXArray?, cond: MLXArray?
    ) -> MLXArray {
        if let compiled = compiledDiTForward, spks == nil, cond == nil {
            return compiled([x, mask, mu, t])[0]
        }
        if let compiled = compiledDiTForwardFull, let spks, let cond {
            return compiled([x, mask, mu, t, spks, cond])[0]
        }
        return decoder(x, mask: mask, mu: mu, t: t, spks: spks, cond: cond)
    }

    /// Warm up the compiled DiT with small dummy forward passes.
    /// Traces both compiled graphs and pre-compiles Metal shaders so the
    /// first real generation pays zero compilation cost. Dummies use the
    /// loaded weight dtype so the traces match runtime call signatures.
    public func warmUp() {
        guard compiledDiTForward != nil else { return }

        let dtype = decoder.projOut.weight.dtype
        let melDim = config.dit.melDim
        let dummyX = MLXArray.zeros([2, melDim, 4]).asType(dtype)
        let dummyMask = MLXArray.ones([2, 1, 4]).asType(dtype)
        let dummyMu = MLXArray.zeros([2, melDim, 4]).asType(dtype)
        let dummyT = MLXArray.zeros([2]).asType(dtype)

        let plain = executeDiTForward(
            x: dummyX, mask: dummyMask, mu: dummyMu, t: dummyT,
            spks: nil, cond: nil)
        eval(plain)

        let dummySpks = MLXArray.zeros([2, config.dit.spkDim]).asType(dtype)
        let dummyCond = MLXArray.zeros([2, melDim, 4]).asType(dtype)
        let full = executeDiTForward(
            x: dummyX, mask: dummyMask, mu: dummyMu, t: dummyT,
            spks: dummySpks, cond: dummyCond)
        eval(full)
    }

    /// Initial noise for the ODE solver, always float32.
    /// Slices the fixed buffer when present (batch 1 and long enough),
    /// otherwise draws with a fixed key so repeated calls are identical.
    func initialNoise(batch: Int, timeSteps: Int, temperature: Float) -> MLXArray {
        var z: MLXArray
        if let buf = fixedNoise, batch == 1, buf.dim(2) >= timeSteps {
            z = buf[0..., 0..., 0 ..< timeSteps].asType(.float32)
        } else {
            z = MLXRandom.normal(
                [batch, config.outputSize, timeSteps], key: MLXRandom.key(0))
        }
        if temperature != 1.0 {
            z = z * temperature
        }
        return z
    }

    /// Solve the flow matching ODE to generate a mel spectrogram.
    ///
    /// The solver starts from Gaussian noise scaled by `temperature` and integrates
    /// using the Euler method with classifier-free guidance. At each timestep, the
    /// DiT is called with a doubled batch (conditioned + unconditioned) and the
    /// velocity is blended using `cfgRate`.
    ///
    /// - Parameters:
    ///   - mu: `[B, 80, T]` conditioning mel from the encoder
    ///   - mask: `[B, 1, T]` validity mask (1 = valid, 0 = padding)
    ///   - nTimesteps: number of ODE integration steps (default 10)
    ///   - temperature: noise scaling factor (default 1.0)
    ///   - spks: `[B, 80]` projected speaker embedding, or nil
    ///   - cond: `[B, 80, T]` additional conditioning, or nil
    /// - Returns: `[B, 80, T]` generated mel spectrogram
    public func forward(
        mu: MLXArray,
        mask: MLXArray,
        nTimesteps: Int = 10,
        temperature: Float = 1.0,
        spks: MLXArray? = nil,
        cond: MLXArray? = nil
    ) -> MLXArray {
        // 1. Initial noise, float32. The solver state stays float32 for the
        //    whole integration — upstream solves in float32 and returns
        //    `.float()`; accumulating in the half-precision weight dtype
        //    adds an audible spectral noise floor over the 10 Euler steps.
        var x = initialNoise(
            batch: mu.dim(0), timeSteps: mu.dim(2), temperature: temperature)

        // 2. Create time schedule with cosine mapping
        // Python: t_span = torch.linspace(0, 1, n_timesteps + 1)
        //         t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        let tSchedule: [Float] = (0 ... nTimesteps).map { i in
            let t = Float(i) / Float(nTimesteps)
            return 1.0 - cos(t * 0.5 * .pi)
        }

        // 3. Euler solver with classifier-free guidance. The DiT runs in the
        //    loaded weight dtype; the CFG combine and Euler update run fp32.
        let ditDtype = mu.dtype
        let cfgRate = config.cfgRate

        // Pre-build unconditioned inputs (zeros) for CFG — reused across all ODE steps
        let muZeros = MLXArray.zeros(mu.shape, dtype: mu.dtype)
        let spksZeros: MLXArray? = spks.map { MLXArray.zeros($0.shape, dtype: ditDtype) }
        let condZeros: MLXArray? = cond.map { MLXArray.zeros($0.shape, dtype: ditDtype) }

        for i in 0 ..< nTimesteps {
            let t = tSchedule[i]
            let dt = tSchedule[i + 1] - tSchedule[i]

            // Batch doubling for CFG: run conditioned + unconditioned in one forward pass
            let batchSize = x.dim(0)
            let xIn = concatenated([x, x], axis: 0).asType(ditDtype)      // [2B, 80, T]
            let maskIn = concatenated([mask, mask], axis: 0)              // [2B, 1, T]
            let muIn = concatenated([mu, muZeros], axis: 0)               // [2B, 80, T]
            let tArr = MLXArray([Float](repeating: t, count: batchSize * 2)).asType(ditDtype)

            let spksIn: MLXArray? = spks.flatMap { s in
                spksZeros.map { z in concatenated([s.asType(ditDtype), z], axis: 0) }
            }
            let condIn: MLXArray? = cond.flatMap { c in
                condZeros.map { z in concatenated([c.asType(ditDtype), z], axis: 0) }
            }

            // Single forward pass through DiT with doubled batch
            let velocity = executeDiTForward(
                x: xIn, mask: maskIn, mu: muIn, t: tArr, spks: spksIn, cond: condIn)

            // Split conditioned and unconditioned predictions, combine in fp32:
            //   v = (1 + cfg_rate) * v_cond - cfg_rate * v_uncond
            let vCond = velocity[0 ..< batchSize].asType(.float32)
            let vUncond = velocity[batchSize...].asType(.float32)
            let v = (1.0 + cfgRate) * vCond - cfgRate * vUncond

            // Euler step: x_{t+dt} = x_t + dt * v
            x = x + dt * v

            // Evaluate to avoid building too large a computation graph
            eval(x)
        }

        return x
    }
}

// MARK: - PreLookaheadLayer

/// Causal convolution encoder before DiT.
/// conv1(80→1024, k=4, look-ahead) → leaky_relu(0.01) → conv2(1024→80, k=3,
/// causal) → residual add with the input. The residual and the leaky slope
/// match upstream `upsample_encoder.PreLookaheadLayer` — the DiT was trained
/// with `mu = f(tokens) + tokens`, so dropping either shifts the content
/// conditioning off-distribution on every frame.
public class PreLookaheadLayer: Module {
    @ModuleInfo var conv1: CausalDilatedConv1d
    @ModuleInfo var conv2: CausalDilatedConv1d

    public init(inputDim: Int = 80, hiddenDim: Int = 1024) {
        // conv1: right-padding (look-ahead), kernel_size=4
        // Python: CausalConv1d(input_dim, hidden_dim, kernel_size, causal_type='right')
        self._conv1.wrappedValue = CausalDilatedConv1d(
            inputChannels: inputDim, outputChannels: hiddenDim, kernelSize: 4, causalType: .right)
        // conv2: left-padding (causal), kernel_size=3
        // Python: CausalConv1d(hidden_dim, input_dim, kernel_size - 1, causal_type='left')
        self._conv2.wrappedValue = CausalDilatedConv1d(
            inputChannels: hiddenDim, outputChannels: inputDim, kernelSize: 3)
        super.init()
    }

    /// Input: [B, C, T] (NCL) → Output: [B, C, T] (NCL)
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(x)
        h = leakyRelu(h, negativeSlope: 0.01)
        h = conv2(h)
        return h + x
    }
}

// MARK: - CosyVoiceFlowModel

/// Complete flow matching module for CosyVoice3.
///
/// Combines the speech token encoder (embedding → pre-lookahead → upsample) with
/// the conditional flow matching decoder (DiT + ODE solver) to produce mel spectrograms.
///
/// Pipeline:
/// 1. Embed speech tokens: `[B, T]` → `[B, T, 80]`
/// 2. Pre-lookahead conv encoder: `[B, 80, T]` → `[B, 80, T]`
/// 3. Upsample to mel rate: `[B, T, 80]` → `[B, T*2, 80]` (25 Hz → 50 Hz)
/// 4. Run flow matching ODE with DiT: `[B, 80, T*2]` → `[B, 80, T*2]`
public class CosyVoiceFlowModel: Module {
    public let config: CosyVoiceFlowConfig

    @ModuleInfo(key: "input_embedding") var inputEmbedding: Embedding
    @ModuleInfo(key: "spk_embed_affine_layer") var spkEmbedAffineLayer: Linear
    @ModuleInfo(key: "pre_lookahead_layer") var preLookaheadLayer: PreLookaheadLayer
    @ModuleInfo var decoder: ConditionalFlowMatching

    public init(config: CosyVoiceFlowConfig) {
        self.config = config

        // FSQ vocabulary embedding: 6561 tokens → 80 dims (mel dim directly)
        self._inputEmbedding.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.outputSize)

        // Speaker embedding projection: 192 → 80
        self._spkEmbedAffineLayer.wrappedValue = Linear(
            config.spkEmbedDim, config.outputSize, bias: true)

        // Pre-lookahead causal conv encoder: 80 → 1024 → 80
        self._preLookaheadLayer.wrappedValue = PreLookaheadLayer(
            inputDim: config.outputSize, hiddenDim: 1024)

        // Flow matching decoder (contains DiT)
        self._decoder.wrappedValue = ConditionalFlowMatching(config: config)

        super.init()
    }

    /// Generate a mel spectrogram from speech tokens, optionally conditioned on a
    /// reference clip via prepended prompt tokens + frame-aligned prompt mel.
    ///
    /// When `promptToken` and `promptFeat` are supplied (the upstream zero-shot
    /// cloning path), they're concatenated to `tokens` / written into the DiT's
    /// `cond` slot so the flow has a per-frame anchor of the reference voice. The
    /// returned mel still spans the **full** sequence (prompt + generated); the
    /// caller is responsible for slicing off the first `promptFeat.dim(2)`
    /// frames before passing to HiFi-GAN.
    ///
    /// - Parameters:
    ///   - tokens: `[B, T]` speech token IDs (FSQ codes 0-6560) for the *new*
    ///     content to synthesize.
    ///   - spkEmbedding: `[B, 192]` raw CAM++ speaker embedding, or nil.
    ///   - promptToken: `[B, T_prompt]` FSQ codes of the reference audio (from
    ///     `SpeechTokenizerModel.encode`), or nil for the non-cloning path.
    ///   - promptFeat: `[B, 80, T_prompt_mel]` Matcha-style log-mel of the
    ///     reference (from `FlowMelExtractor.extract`), or nil. **Must** satisfy
    ///     `T_prompt_mel == T_prompt * tokenMelRatio` so the cond region aligns
    ///     with the upsampled mu region — the caller's only invariant.
    ///   - nTimesteps: ODE solver steps (default from config, typically 10).
    ///   - temperature: noise temperature for sampling.
    /// - Returns: `[B, 80, T_mel]` mel spectrogram, T_mel = (T + T_prompt) * 2.
    public func callAsFunction(
        tokens: MLXArray,
        spkEmbedding: MLXArray? = nil,
        promptToken: MLXArray? = nil,
        promptFeat: MLXArray? = nil,
        nTimesteps: Int? = nil,
        temperature: Float = 1.0
    ) -> MLXArray {
        let steps = nTimesteps ?? config.nTimesteps

        // 1. Prepend prompt_token to the generated tokens (upstream:
        //    `text = torch.concat([prompt_text, text], dim=1)` in CausalMaskedDiffWithDiT).
        let combinedTokens: MLXArray
        if let pt = promptToken {
            precondition(pt.dim(0) == tokens.dim(0),
                         "promptToken batch must match tokens batch")
            combinedTokens = concatenated([pt, tokens], axis: 1)
        } else {
            combinedTokens = tokens
        }

        // 2. Embed combined tokens: [B, T_total] → [B, T_total, 80]
        var mu = inputEmbedding(combinedTokens)

        // 3. Pre-lookahead conv encoder: [B, T, 80] → NCL → [B, 80, T] → NLC → [B, T, 80]
        mu = preLookaheadLayer(mu.transposed(0, 2, 1)).transposed(0, 2, 1)

        // 4. Upsample 25 Hz → 50 Hz (repeat-interleave by tokenMelRatio).
        let muUpsampled = RepeatInterleaveUpsampler.upsample(mu, ratio: config.tokenMelRatio)
        let melLen = muUpsampled.dim(1)

        // 5. NLC → NCL for the DiT.
        let muTransposed = muUpsampled.transposed(0, 2, 1)

        // 6. Validity mask (no padding inside generation — all ones).
        let batchSize = combinedTokens.dim(0)
        let mask = MLXArray.ones([batchSize, 1, melLen]).asType(muTransposed.dtype)

        // 7. Speaker conditioning (L2-norm then affine 192 → 80).
        let spks: MLXArray? = spkEmbedding.map { emb in
            let norm = sqrt(sum(emb * emb, axis: -1, keepDims: true)) + 1e-8
            let normalized = emb / norm
            return spkEmbedAffineLayer(normalized)
        }

        // 8. Build the DiT cond signal. Upstream layout:
        //      conds[:, :, :prompt_mel_len] = prompt_feat
        //      conds[:, :, prompt_mel_len:] = 0
        //    This is what gives the flow a per-frame timbre anchor — without it
        //    the LLM's emotion-loaded tokens dominate via the spks path alone.
        let cond: MLXArray?
        if let pf = promptFeat {
            precondition(pf.dim(0) == batchSize,
                         "promptFeat batch must match tokens batch")
            precondition(pf.dim(1) == config.outputSize,
                         "promptFeat mel dim must be \(config.outputSize), got \(pf.dim(1))")
            let promptMelLen = pf.dim(2)
            precondition(promptMelLen <= melLen,
                         "promptFeat (\(promptMelLen) frames) longer than total mel (\(melLen))")
            let genMelLen = melLen - promptMelLen
            if genMelLen > 0 {
                let zerosAfter = MLXArray.zeros(
                    [batchSize, config.outputSize, genMelLen]
                ).asType(pf.dtype)
                cond = concatenated([pf, zerosAfter], axis: 2)
            } else {
                cond = pf
            }
        } else {
            cond = nil
        }

        // 9. Run the flow matching ODE solver.
        let mel = decoder.forward(
            mu: muTransposed,
            mask: mask,
            nTimesteps: steps,
            temperature: temperature,
            spks: spks,
            cond: cond
        )

        return mel  // [B, 80, T_mel] — caller slices off [:, :, :promptMelLen]
    }
}
