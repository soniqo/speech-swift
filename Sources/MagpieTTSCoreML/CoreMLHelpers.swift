import CoreML
import Foundation

/// Small helpers for marshalling between Swift `[Float]` / `[Int32]` and the
/// FP16 / FP32 / Int32 `MLMultiArray` instances that the FluidInference
/// `.mlmodelc` bundles consume.
enum MagpieCoreMLBridge {

    static func loadCompiled(at url: URL, label: String) throws -> MLModel {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MagpieCoreMLError.missingFile(url.lastPathComponent)
        }
        let config = MLModelConfiguration()
        // The decoder is the hot loop — let CoreML pick ANE or GPU per layer.
        config.computeUnits = .all
        do {
            return try MLModel(contentsOf: url, configuration: config)
        } catch {
            throw MagpieCoreMLError.modelLoadFailed(
                name: label, underlying: String(describing: error))
        }
    }

    // MARK: - Float32 → FP16 MLMultiArray

    /// Build a `MLMultiArray` with the given shape from a Float32 buffer,
    /// converting to FP16 storage in-place. Use for the encoder hidden / mask
    /// inputs the decoder consumes.
    static func makeFp16(_ values: [Float], shape: [NSNumber], label: String) throws
        -> MLMultiArray
    {
        let array: MLMultiArray
        do {
            array = try MLMultiArray(shape: shape, dataType: .float16)
        } catch {
            throw MagpieCoreMLError.inferenceFailed(
                stage: label, underlying: "alloc fp16 \(shape): \(error)")
        }
        let count = values.count
        precondition(array.count == count,
                     "\(label): expected \(array.count) elements for shape \(shape), got \(count)")
        let buf = array.dataPointer.bindMemory(to: UInt16.self, capacity: count)
        for i in 0..<count {
            buf[i] = float32ToFloat16Bits(values[i])
        }
        return array
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
        precondition(array.count == values.count)
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

    /// Bool mask backed by `.bool` if the model expects it, or `.fp16` if it
    /// expects a 0/1 mask. The decoder_step.mlmodelc inputs `encoder_mask`
    /// as `bool`; decoder_prefill expects `fp16`. Caller picks.
    static func makeBoolMask(_ flags: [Bool], shape: [NSNumber], label: String) throws
        -> MLMultiArray
    {
        let array: MLMultiArray
        do {
            array = try MLMultiArray(shape: shape, dataType: .float32)
        } catch {
            throw MagpieCoreMLError.inferenceFailed(
                stage: label, underlying: "alloc bool mask \(shape): \(error)")
        }
        precondition(array.count == flags.count)
        let buf = array.dataPointer.bindMemory(to: Float.self, capacity: flags.count)
        for i in 0..<flags.count { buf[i] = flags[i] ? 1.0 : 0.0 }
        return array
    }

    // MARK: - MLMultiArray → Float32

    /// Read out an `MLMultiArray` (any dtype) as a flat Float32 buffer.
    static func toFloat32(_ array: MLMultiArray) -> [Float] {
        let count = array.count
        var out = [Float](repeating: 0, count: count)
        switch array.dataType {
        case .float32:
            let p = array.dataPointer.bindMemory(to: Float.self, capacity: count)
            for i in 0..<count { out[i] = p[i] }
        case .float16:
            let p = array.dataPointer.bindMemory(to: UInt16.self, capacity: count)
            for i in 0..<count { out[i] = NpyReader.float16ToFloat32(bits: p[i]) }
        case .double:
            let p = array.dataPointer.bindMemory(to: Double.self, capacity: count)
            for i in 0..<count { out[i] = Float(p[i]) }
        case .int32:
            let p = array.dataPointer.bindMemory(to: Int32.self, capacity: count)
            for i in 0..<count { out[i] = Float(p[i]) }
        @unknown default:
            return out  // give up gracefully — caller will see zeros
        }
        return out
    }

    /// Snapshot the raw `MLMultiArray` contents into a freshly-allocated
    /// FP16-backed `MLMultiArray` we own (so it survives past the next
    /// prediction call, which may reuse internal buffers). Used between
    /// decoder_prefill / decoder_step to carry KV cache slices.
    static func cloneFp16(_ array: MLMultiArray, label: String) throws -> MLMultiArray {
        guard array.dataType == .float16 else {
            // Convert through float32 if we're handed a different storage type.
            let f32 = toFloat32(array)
            let shape = array.shape
            return try makeFp16(f32, shape: shape, label: "\(label)/clone")
        }
        let copy: MLMultiArray
        do {
            copy = try MLMultiArray(shape: array.shape, dataType: .float16)
        } catch {
            throw MagpieCoreMLError.inferenceFailed(
                stage: label, underlying: "clone fp16 \(array.shape): \(error)")
        }
        let count = array.count
        let src = array.dataPointer.bindMemory(to: UInt16.self, capacity: count)
        let dst = copy.dataPointer.bindMemory(to: UInt16.self, capacity: count)
        memcpy(dst, src, count * 2)
        return copy
    }

    static func cloneScalarFp16(_ array: MLMultiArray, label: String) throws -> MLMultiArray {
        try cloneFp16(array, label: label)
    }

    // MARK: - Slice helper for the rank-5 prefill cache

    /// `decoder_prefill` outputs each layer's KV cache as a single FP16 tensor
    /// of shape `(2, 1, 512, 12, 64)` (axis 0 = K vs V). `decoder_step` takes
    /// them split: 12× `cache_k*` and 12× `cache_v*`, each `(1, 512, 12, 64)`.
    /// This helper slices a packed prefill output into the two `(1,512,12,64)`
    /// halves the step model expects.
    static func splitKVCache(_ packed: MLMultiArray, label: String) throws -> (k: MLMultiArray, v: MLMultiArray) {
        let shape = packed.shape.map { $0.intValue }
        guard shape.count == 5, shape[0] == 2 else {
            throw MagpieCoreMLError.inferenceFailed(
                stage: label,
                underlying: "expected packed KV shape (2, …), got \(shape)")
        }
        let stride = shape[1] * shape[2] * shape[3] * shape[4]  // = 1*512*12*64 = 393216
        let outShape: [NSNumber] = shape[1...].map { NSNumber(value: $0) }
        let kArr = try MLMultiArray(shape: outShape, dataType: .float16)
        let vArr = try MLMultiArray(shape: outShape, dataType: .float16)
        let src = packed.dataPointer.bindMemory(to: UInt16.self, capacity: 2 * stride)
        let kDst = kArr.dataPointer.bindMemory(to: UInt16.self, capacity: stride)
        let vDst = vArr.dataPointer.bindMemory(to: UInt16.self, capacity: stride)
        memcpy(kDst, src, stride * 2)
        memcpy(vDst, src.advanced(by: stride), stride * 2)
        return (kArr, vArr)
    }

    static func makeScalarInt32(_ value: Int32, label: String) throws -> MLMultiArray {
        let arr: MLMultiArray
        do {
            arr = try MLMultiArray(shape: [], dataType: .int32)
        } catch {
            throw MagpieCoreMLError.inferenceFailed(
                stage: label, underlying: "alloc int32 scalar: \(error)")
        }
        arr.dataPointer.bindMemory(to: Int32.self, capacity: 1).pointee = value
        return arr
    }

    static func makeScalarFp16(_ value: Float, label: String) throws -> MLMultiArray {
        try makeFp16([value], shape: [1], label: label)
    }

    /// Convert a scalar int32 to a 1-element FP16 multiarray. Used to bridge
    /// `decoder_prefill`'s int32 `position_*` outputs into `decoder_step`'s
    /// fp16 `position*` inputs.
    static func scalarInt32ToFp16(_ array: MLMultiArray, label: String) throws -> MLMultiArray {
        let v: Int32
        switch array.dataType {
        case .int32:
            v = array.dataPointer.bindMemory(to: Int32.self, capacity: 1).pointee
        case .float32:
            v = Int32(array.dataPointer.bindMemory(to: Float.self, capacity: 1).pointee)
        case .float16:
            let bits = array.dataPointer.bindMemory(to: UInt16.self, capacity: 1).pointee
            v = Int32(NpyReader.float16ToFloat32(bits: bits))
        case .double:
            v = Int32(array.dataPointer.bindMemory(to: Double.self, capacity: 1).pointee)
        @unknown default:
            v = 0
        }
        return try makeScalarFp16(Float(v), label: label)
    }
}

@inline(__always)
private func float32ToFloat16Bits(_ value: Float) -> UInt16 {
    let bits = value.bitPattern
    let sign = UInt16((bits >> 16) & 0x8000)
    var exp = Int32((bits >> 23) & 0xFF) - 127 + 15
    var mant = bits & 0x7F_FFFF

    if exp <= 0 {
        if exp < -10 { return sign }
        // Subnormal: add implicit leading 1, then shift.
        mant |= 0x80_0000
        let shift = 14 - exp
        let result = sign | UInt16((mant >> shift) & 0x3FF)
        return result
    } else if exp >= 31 {
        if mant != 0 { return sign | 0x7E00 }   // NaN
        return sign | 0x7C00                    // Inf
    } else {
        let expBits = UInt16(exp & 0x1F) << 10
        let mantBits = UInt16((mant >> 13) & 0x3FF)
        return sign | expBits | mantBits
    }
}
