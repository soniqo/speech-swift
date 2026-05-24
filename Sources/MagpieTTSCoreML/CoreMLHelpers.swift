import CoreML
import Foundation

/// FP32 / Int32 marshalling between Swift buffers and the
/// `MLMultiArray` instances our `.mlmodelc` bundles consume. Our bundle is
/// FP32 at every IO boundary (precision drops to FP16 inside the graph),
/// so we don't need the FP16 conversion routines a FluidInference-style
/// bundle would.
enum MagpieCoreMLBridge {

    /// What kind of model we're loading — drives the compute-unit
    /// choice. Kept as an explicit hint even though both decoder and
    /// codec now run cleanly on ANE: the override env vars
    /// (`MAGPIE_COREML_COMPUTE_DECODER` / `_CODEC`) target each kind
    /// independently for benchmarking.
    enum Kind {
        case decoder        // text_encoder, decoder_prefill, decoder_step
        case codec          // nanocodec_decoder (HiFi-GAN vocoder)
    }

    static func loadCompiled(at url: URL, label: String, kind: Kind) throws -> MLModel {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MagpieCoreMLError.missingFile(url.lastPathComponent)
        }
        let config = MLModelConfiguration()
        // All four .mlmodelc bundles default to `.cpuAndNeuralEngine`:
        //
        // - **Decoder/encoder graphs**: ANE is ~7× faster than the
        //   ANE+GPU+CPU mix (`.all`). Per-frame decoder_step drops
        //   from ~15 ms to ~2 ms. The BF16 precision drift on
        //   attention scores is absorbed into Gumbel noise by the
        //   default stochastic sampler. With greedy
        //   (`--magpie-temperature 0`), tiny drift can flip the
        //   argmax and the AR loop diverges — opt out via
        //   `MAGPIE_COREML_COMPUTE_DECODER=all` if you need greedy.
        //
        // - **Codec**: the nanocodec is a HiFi-GAN vocoder; the
        //   PRIOR FP16 export produced audible high-frequency hiss
        //   on ANE. The current bundle ships an ANE-friendly
        //   re-export (Kokoro recipe: `ct.convert(...,
        //   compute_units=CPU_AND_NE, compute_precision=FLOAT32)`)
        //   that lowers cleanly to ANE with the same audio quality
        //   as the GPU path. Force GPU instead via
        //   `MAGPIE_COREML_COMPUTE_CODEC=cpuAndGPU` if regressions
        //   surface on a future macOS.
        let env = ProcessInfo.processInfo.environment
        let envKey = kind == .decoder ? "MAGPIE_COREML_COMPUTE_DECODER" : "MAGPIE_COREML_COMPUTE_CODEC"
        // Backwards-compat: `MAGPIE_COREML_COMPUTE` overrides both kinds.
        let chosen = env[envKey] ?? env["MAGPIE_COREML_COMPUTE"]
        switch chosen {
        case "all":             config.computeUnits = .all
        case "cpuAndGPU":       config.computeUnits = .cpuAndGPU
        case "cpuOnly":         config.computeUnits = .cpuOnly
        case "ane", nil:        config.computeUnits = .cpuAndNeuralEngine
        default:                config.computeUnits = .all
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
