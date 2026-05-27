import Foundation
import MLX
import MLXNN

/// SwiGLU MLP — shared by ASR text decoder, TTS Talker, and Code Predictor.
/// Linear layers are quantized when `bits > 0`, plain Linear otherwise (bf16/fp32 path).
public class QuantizedMLP: Module {
    @ModuleInfo public var gateProj: Linear
    @ModuleInfo public var upProj: Linear
    @ModuleInfo public var downProj: Linear

    public init(hiddenSize: Int, intermediateSize: Int, groupSize: Int = 64, bits: Int = 4) {
        self._gateProj.wrappedValue = makeMaybeQuantizedLinear(
            hiddenSize, intermediateSize, bias: false,
            groupSize: groupSize, bits: bits)
        self._upProj.wrappedValue = makeMaybeQuantizedLinear(
            hiddenSize, intermediateSize, bias: false,
            groupSize: groupSize, bits: bits)
        self._downProj.wrappedValue = makeMaybeQuantizedLinear(
            intermediateSize, hiddenSize, bias: false,
            groupSize: groupSize, bits: bits)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // SwiGLU: down(silu(gate(x)) * up(x))
        let gate = silu(gateProj(x))
        let up = upProj(x)
        return downProj(gate * up)
    }
}

/// SwiGLU MLP with `Linear` projections — used by modules that need to
/// support both bf16/fp16 and quantized bundles. The runtime swaps
/// Linear → QuantizedLinear in place via `quantize(model:filter:)` when
/// the loaded weights carry `.scales` for these paths.
public class MLP: Module {
    @ModuleInfo public var gateProj: Linear
    @ModuleInfo public var upProj: Linear
    @ModuleInfo public var downProj: Linear

    public init(hiddenSize: Int, intermediateSize: Int) {
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = silu(gateProj(x))
        let up = upProj(x)
        return downProj(gate * up)
    }
}
