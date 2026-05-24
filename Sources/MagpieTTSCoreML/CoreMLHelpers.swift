import CoreML
import Foundation

/// FP32 / Int32 marshalling between Swift buffers and the
/// `MLMultiArray` instances our `.mlmodelc` bundles consume. Our bundle is
/// FP32 at every IO boundary (precision drops to FP16 inside the graph),
/// so we don't need the FP16 conversion routines a FluidInference-style
/// bundle would.
enum MagpieCoreMLBridge {

    static func loadCompiled(at url: URL, label: String) throws -> MLModel {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MagpieCoreMLError.missingFile(url.lastPathComponent)
        }
        let config = MLModelConfiguration()
        // `.cpuAndNeuralEngine` by default — ANE is 7x faster than
        // ANE+GPU+CPU mix (`.all`) and even beats MLX (RTF 0.14 vs
        // 0.27). Per-frame steady-state drops from ~15 ms to ~2 ms.
        //
        // **Important caveat**: ANE runs internal BF16 (vs FP16 on
        // GPU), which adds tiny precision drift to the attention
        // scores. With greedy sampling (`temperature=0`) this drift
        // can flip the argmax and the AR loop diverges (ASR drops to
        // just "Hello." for the first word). Stochastic sampling
        // (the default temperature=0.6 + top-k=80) absorbs the drift
        // into Gumbel noise so the output is statistically equivalent
        // to the GPU/CPU path. The CLI default is stochastic, so this
        // works out of the box.
        //
        // Override with `MAGPIE_COREML_COMPUTE=all|cpuAndGPU|cpuOnly`
        // for bisecting or to force greedy-safe output.
        switch ProcessInfo.processInfo.environment["MAGPIE_COREML_COMPUTE"] {
        case "all":         config.computeUnits = .all
        case "cpuAndGPU":   config.computeUnits = .cpuAndGPU
        case "cpuOnly":     config.computeUnits = .cpuOnly
        default:            config.computeUnits = .cpuAndNeuralEngine
        }
        do {
            return try MLModel(contentsOf: url, configuration: config)
        } catch {
            throw MagpieCoreMLError.modelLoadFailed(
                name: label, underlying: String(describing: error))
        }
    }

    static func makeFp32(_ values: [Float], shape: [NSNumber], label: String) throws
        -> MLMultiArray
    {
        let array: MLMultiArray
        do {
            array = try MLMultiArray(shape: shape, dataType: .float32)
        } catch {
            throw MagpieCoreMLError.inferenceFailed(
                stage: label, underlying: "alloc fp32 \(shape): \(error)")
        }
        precondition(array.count == values.count,
                     "\(label): expected \(array.count) for shape \(shape), got \(values.count)")
        let buf = array.dataPointer.bindMemory(to: Float.self, capacity: values.count)
        for i in 0..<values.count { buf[i] = values[i] }
        return array
    }

    static func makeInt32(_ values: [Int32], shape: [NSNumber], label: String) throws
        -> MLMultiArray
    {
        let array: MLMultiArray
        do {
            array = try MLMultiArray(shape: shape, dataType: .int32)
        } catch {
            throw MagpieCoreMLError.inferenceFailed(
                stage: label, underlying: "alloc int32 \(shape): \(error)")
        }
        precondition(array.count == values.count)
        let buf = array.dataPointer.bindMemory(to: Int32.self, capacity: values.count)
        for i in 0..<values.count { buf[i] = values[i] }
        return array
    }

    static func makeScalarInt32(_ value: Int32, label: String) throws -> MLMultiArray {
        try makeInt32([value], shape: [1], label: label)
    }

    /// Read an MLMultiArray (any dtype) into a flat Float32 buffer. Used to
    /// extract `h_last`, `logits`, and `audio` outputs.
    static func toFloat32(_ array: MLMultiArray) -> [Float] {
        let count = array.count
        var out = [Float](repeating: 0, count: count)
        switch array.dataType {
        case .float32:
            let p = array.dataPointer.bindMemory(to: Float.self, capacity: count)
            for i in 0..<count { out[i] = p[i] }
        case .float16:
            let p = array.dataPointer.bindMemory(to: UInt16.self, capacity: count)
            for i in 0..<count { out[i] = Self.fp16BitsToFloat(p[i]) }
        case .int32:
            let p = array.dataPointer.bindMemory(to: Int32.self, capacity: count)
            for i in 0..<count { out[i] = Float(p[i]) }
        case .double:
            let p = array.dataPointer.bindMemory(to: Double.self, capacity: count)
            for i in 0..<count { out[i] = Float(p[i]) }
        case .int8:
            let p = array.dataPointer.bindMemory(to: Int8.self, capacity: count)
            for i in 0..<count { out[i] = Float(p[i]) }
        @unknown default:
            return out
        }
        return out
    }

    @inline(__always)
    static func fp16BitsToFloat(_ bits: UInt16) -> Float {
        let sign = UInt32(bits & 0x8000) << 16
        let exp  = UInt32((bits & 0x7C00) >> 10)
        let mant = UInt32(bits & 0x03FF)
        var result: UInt32
        if exp == 0 {
            if mant == 0 { result = sign }
            else {
                var e: UInt32 = 127 - 15 + 1
                var m = mant
                while (m & 0x0400) == 0 { m <<= 1; e -= 1 }
                m &= 0x03FF
                result = sign | (e << 23) | (m << 13)
            }
        } else if exp == 0x1F {
            result = sign | 0x7F80_0000 | (mant << 13)
        } else {
            let newExp = UInt32(Int(exp) - 15 + 127)
            result = sign | (newExp << 23) | (mant << 13)
        }
        return Float(bitPattern: result)
    }
}
