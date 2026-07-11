import Foundation
import MLX
import MLXNN

// MARK: - LSTM cell matching mlx-community EnCodec weight layout

/// Single-layer LSTM cell that consumes weight tensors stored under the keys
/// `Wx`, `Wh`, `bias` — the exact layout used by the `mlx-community` EnCodec
/// safetensors (24 kHz, 32 kHz, 48 kHz variants).
///
/// Runs the full input sequence in one call and returns the per-step hidden
/// state stack `[B, T, hiddenSize]`. There is no streaming variant — EnCodec
/// is used on bounded chunks so a full-sequence pass is fine and keeps the
/// gate math straightforward.
///
/// Gate ordering matches Apple's mlx-examples Encodec port (i, f, g, o).
public final class EncodecLSTMCell: Module {
    @ParameterInfo public var Wx: MLXArray
    @ParameterInfo public var Wh: MLXArray
    @ParameterInfo public var bias: MLXArray
    public let hiddenSize: Int

    public init(inputSize: Int, hiddenSize: Int) {
        self.hiddenSize = hiddenSize
        self._Wx = ParameterInfo(wrappedValue: MLXArray.zeros([4 * hiddenSize, inputSize]))
        self._Wh = ParameterInfo(wrappedValue: MLXArray.zeros([4 * hiddenSize, hiddenSize]))
        self._bias = ParameterInfo(wrappedValue: MLXArray.zeros([4 * hiddenSize]))
        super.init()
    }

    /// `x: [B, T, inputSize]` → `[B, T, hiddenSize]`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Precompute Wx·x + b over all time steps.
        let xT = matmul(x, Wx.T) + bias    // [B, T, 4*H]
        let B = x.dim(0)
        let T = x.dim(1)
        var h = MLXArray.zeros([B, hiddenSize], dtype: x.dtype)
        var c = MLXArray.zeros([B, hiddenSize], dtype: x.dtype)
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(T)

        let H = hiddenSize
        for t in 0..<T {
            let xt = xT[0..., t, 0...]                    // [B, 4*H]
            let gates = xt + matmul(h, Wh.T)              // [B, 4*H]
            let iGate = sigmoid(gates[0..., 0..<H])
            let fGate = sigmoid(gates[0..., H..<(2 * H)])
            let gGate = tanh(gates[0..., (2 * H)..<(3 * H)])
            let oGate = sigmoid(gates[0..., (3 * H)..<(4 * H)])
            c = fGate * c + iGate * gGate
            h = oGate * tanh(c)
            outputs.append(h)
        }
        return stacked(outputs, axis: 1)
    }
}

// MARK: - Stacked LSTM with residual

/// EnCodec's LSTM block: `numLayers` of `EncodecLSTMCell` stacked, with a
/// residual add of the original input to the stack's final hidden output.
/// Weight key is `lstm.<i>.{Wx,Wh,bias}` per layer.
public final class EncodecLSTM: Module {
    @ModuleInfo public var lstm: [EncodecLSTMCell]

    public init(dimension: Int, numLayers: Int) {
        self._lstm = ModuleInfo(wrappedValue: (0..<numLayers).map { _ in
            EncodecLSTMCell(inputSize: dimension, hiddenSize: dimension)
        })
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for cell in lstm { h = cell(h) }
        return h + x
    }
}
