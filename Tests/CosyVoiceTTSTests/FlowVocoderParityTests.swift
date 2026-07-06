import XCTest
import MLX
import MLXNN
@testable import CosyVoiceTTS

/// Regression tests pinning the upstream-parity fixes for the CosyVoice3
/// quality issue (reverby/smeared output). Each test would fail on the
/// pre-fix implementation.
final class FlowVocoderParityTests: XCTestCase {

    // MARK: - PreLookaheadLayer

    /// Upstream `PreLookaheadLayer` ends with `outputs = outputs + inputs`.
    /// With all conv weights zeroed the layer must reduce to the identity —
    /// the pre-fix implementation (no residual) returns all zeros instead.
    func testPreLookaheadLayerHasResidualConnection() {
        let layer = PreLookaheadLayer(inputDim: 4, hiddenDim: 8)
        zeroConvParameters(layer.conv1)
        zeroConvParameters(layer.conv2)

        let x = MLXRandom.normal([1, 4, 6])
        let y = layer(x)
        eval(y)

        XCTAssertEqual(y.shape, x.shape)
        let maxDiff = MLX.abs(y - x).max().item(Float.self)
        XCTAssertEqual(maxDiff, 0, accuracy: 1e-6,
                       "Zeroed convs must reduce PreLookaheadLayer to identity (residual)")
    }

    /// The activation between the convs is leaky_relu(0.01), not ReLU:
    /// negative conv1 outputs must leak through at slope 0.01. With conv1
    /// fixed to output a negative constant and conv2 an identity-summing
    /// kernel, ReLU yields exactly x while leaky_relu yields x - 0.01·c·k.
    func testPreLookaheadLayerUsesLeakyRelu() {
        let layer = PreLookaheadLayer(inputDim: 2, hiddenDim: 3)

        // conv1: zero weights, bias -1 → pre-activation is -1 everywhere.
        zeroConvParameters(layer.conv1, bias: -1.0)
        // conv2: kernel of ones, zero bias → sums the (leaked) activations.
        onesConvParameters(layer.conv2)

        let x = MLXArray.zeros([1, 2, 5])
        let y = layer(x)
        eval(y)

        // leaky_relu(-1) = -0.01; conv2 sums hidden(3) × kernel(3) of them,
        // residual adds x = 0. ReLU would give exactly 0.
        let expected: Float = -0.01 * 3 * 3
        let got = y[0, 0, 4].item(Float.self)
        XCTAssertEqual(got, expected, accuracy: 1e-4,
                       "conv1 activation must be leaky_relu(0.01), not ReLU")
    }

    // MARK: - Flow ODE solver

    /// The solver state must stay float32 (upstream returns `.float()`), and
    /// the initial noise must not depend on the global RNG — two runs under
    /// different global seeds must match exactly.
    func testFlowSolverIsFloat32AndDeterministic() {
        var config = CosyVoiceConfig.default.flow
        config.dit.dim = 64
        config.dit.depth = 1
        config.dit.heads = 2
        config.dit.dimHead = 32
        let cfm = ConditionalFlowMatching(config: config)

        let mu = MLXRandom.normal([1, 80, 6]).asType(.float16)
        let mask = MLXArray.ones([1, 1, 6]).asType(.float16)

        MLXRandom.seed(1)
        let a = cfm.forward(mu: mu, mask: mask, nTimesteps: 2)
        MLXRandom.seed(2)
        let b = cfm.forward(mu: mu, mask: mask, nTimesteps: 2)
        eval(a, b)

        XCTAssertEqual(a.dtype, .float32, "solver output must be float32")
        XCTAssertEqual(a.shape, [1, 80, 6])
        let diff = MLX.abs(a - b).max().item(Float.self)
        XCTAssertEqual(diff, 0, "flow noise must not depend on the global RNG")
    }

    /// A bundle-provided fixed noise buffer is sliced to the mel length and
    /// promoted to float32 (never truncated to the weight dtype).
    func testFixedNoiseBufferSlicing() {
        var config = CosyVoiceConfig.default.flow
        config.dit.dim = 64
        config.dit.depth = 1
        config.dit.heads = 2
        config.dit.dimHead = 32
        let cfm = ConditionalFlowMatching(config: config)

        let values = (0 ..< 80 * 10).map { Float($0) }
        cfm.fixedNoise = MLXArray(values, [1, 80, 10])

        let z = cfm.initialNoise(batch: 1, timeSteps: 4, temperature: 1.0)
        eval(z)
        XCTAssertEqual(z.shape, [1, 80, 4])
        XCTAssertEqual(z.dtype, .float32)
        // Row-major [1, 80, 10]: element [0, c, t] = c*10 + t.
        XCTAssertEqual(z[0, 1, 2].item(Float.self), 12)

        // Too-short buffer falls back to the keyed draw (still fp32, right shape).
        let zLong = cfm.initialNoise(batch: 1, timeSteps: 32, temperature: 1.0)
        eval(zLong)
        XCTAssertEqual(zLong.shape, [1, 80, 32])
        XCTAssertEqual(zLong.dtype, .float32)
    }

    // MARK: - NSF sine source (SineGen2-causal semantics)

    /// Upstream SineGen2 upsamples the frame-rate phase with NEAREST, so the
    /// harmonic source is constant within each upsample_scale block and
    /// changes across blocks. The pre-fix per-sample cumsum produced a
    /// smooth sine that varied within blocks.
    func testSineSourceIsFrameStaircase() {
        let gen = SineGenerator(
            sampleRate: 24_000, upsampleScale: 4, harmonicNum: 8,
            sineAmp: 0.1, noiseStd: 0, voicedThreshold: 10)

        let f0 = MLXArray.full([1, 3], values: MLXArray(Float(200)))
        let (sines, uv) = gen(f0)
        eval(sines, uv)

        XCTAssertEqual(sines.shape, [1, 12, 9])
        XCTAssertEqual(uv.shape, [1, 12, 1])
        XCTAssertEqual(uv.min().item(Float.self), 1.0, "200 Hz must be voiced")

        // Constant within each block of 4 samples...
        for block in 0 ..< 3 {
            let first = sines[0, block * 4, 0].item(Float.self)
            for s in 1 ..< 4 {
                XCTAssertEqual(sines[0, block * 4 + s, 0].item(Float.self), first,
                               accuracy: 1e-7, "staircase: constant within a frame block")
            }
        }
        // ...and progressing across blocks.
        XCTAssertNotEqual(sines[0, 0, 0].item(Float.self),
                          sines[0, 4, 0].item(Float.self),
                          "staircase: phase must advance between frame blocks")
    }

    /// Unvoiced excitation amplitude is sine_amp/3 · U[0,1) (upstream), not
    /// noise_std · gaussian. With sine_amp 0.1 the values must reach well
    /// past what a 0.003-std gaussian produces, and never exceed 0.1/3.
    func testUnvoicedExcitationAmplitude() {
        let gen = SineGenerator(
            sampleRate: 24_000, upsampleScale: 4, harmonicNum: 8,
            sineAmp: 0.1, noiseStd: 0.003, voicedThreshold: 10)

        let f0 = MLXArray.zeros([1, 8])
        let (sines, uv) = gen(f0)
        eval(sines, uv)

        XCTAssertEqual(uv.max().item(Float.self), 0.0, "f0=0 must be unvoiced")
        let maxAbs = MLX.abs(sines).max().item(Float.self)
        XCTAssertLessThanOrEqual(maxAbs, 0.1 / 3 + 1e-6,
                                 "unvoiced noise is bounded by sine_amp/3 (uniform)")
        XCTAssertGreaterThan(maxAbs, 0.02,
                             "unvoiced amplitude must be sine_amp/3-scale, not noise_std-scale")
    }

    /// The source must be deterministic call-to-call (upstream draws its
    /// noise buffers once at init). The pre-fix implementation drew fresh
    /// gaussian noise and a fresh initial phase per render.
    func testSourceModuleIsDeterministic() {
        let source = SourceModuleHnNSF(
            sampleRate: 24_000, upsampleScale: 4, harmonicNum: 8,
            sineAmp: 0.1, noiseStd: 0.003, voicedThreshold: 10)

        var f0Values = [Float](repeating: 200, count: 6)
        f0Values.append(contentsOf: [Float](repeating: 0, count: 6))
        let f0 = MLXArray(f0Values, [1, 12])

        MLXRandom.seed(1)
        let a = source(f0)
        MLXRandom.seed(2)
        let b = source(f0)
        eval(a, b)

        XCTAssertEqual(a.shape, [1, 48, 1])
        let diff = MLX.abs(a - b).max().item(Float.self)
        XCTAssertEqual(diff, 0, "source excitation must not depend on the global RNG")
    }

    private func zeroConvParameters(_ conv: CausalDilatedConv1d, bias: Float = 0) {
        let inner = conv.conv
        var params: [String: NestedItem<String, MLXArray>] = [:]
        params["weight"] = .value(MLXArray.zeros(inner.weight.shape, dtype: inner.weight.dtype))
        if let b = inner.bias {
            params["bias"] = .value(MLXArray.full(b.shape, values: MLXArray(bias)))
        }
        inner.update(parameters: ModuleParameters(values: params))
    }

    private func onesConvParameters(_ conv: CausalDilatedConv1d) {
        let inner = conv.conv
        var params: [String: NestedItem<String, MLXArray>] = [:]
        params["weight"] = .value(MLXArray.ones(inner.weight.shape, dtype: inner.weight.dtype))
        if let b = inner.bias {
            params["bias"] = .value(MLXArray.zeros(b.shape, dtype: b.dtype))
        }
        inner.update(parameters: ModuleParameters(values: params))
    }
}
