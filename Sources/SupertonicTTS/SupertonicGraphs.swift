#if canImport(CoreML)
import CoreML
import Foundation
import AudioCommon

// MARK: - MLMultiArray marshalling

enum SupertonicBridge {
    static func fp32(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        precondition(arr.count == values.count, "fp32 shape \(shape) != \(values.count)")
        let p = arr.dataPointer.bindMemory(to: Float.self, capacity: values.count)
        for i in 0..<values.count { p[i] = values[i] }
        return arr
    }

    static func int32(_ values: [Int32], shape: [Int]) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
        precondition(arr.count == values.count, "int32 shape \(shape) != \(values.count)")
        let p = arr.dataPointer.bindMemory(to: Int32.self, capacity: values.count)
        for i in 0..<values.count { p[i] = values[i] }
        return arr
    }

    static func toFloat32(_ arr: MLMultiArray) -> [Float] {
        let n = arr.count
        var out = [Float](repeating: 0, count: n)
        switch arr.dataType {
        case .float32:
            let p = arr.dataPointer.bindMemory(to: Float.self, capacity: n)
            for i in 0..<n { out[i] = p[i] }
        case .float16:
            let p = arr.dataPointer.bindMemory(to: UInt16.self, capacity: n)
            for i in 0..<n { out[i] = float16BitsToFloat(p[i]) }
        case .double:
            let p = arr.dataPointer.bindMemory(to: Double.self, capacity: n)
            for i in 0..<n { out[i] = Float(p[i]) }
        case .int32:
            let p = arr.dataPointer.bindMemory(to: Int32.self, capacity: n)
            for i in 0..<n { out[i] = Float(p[i]) }
        @unknown default: break
        }
        return out
    }

    @inline(__always)
    static func float16BitsToFloat(_ bits: UInt16) -> Float {
        let sign = Float((bits >> 15) & 1)
        let exp = Int((bits >> 10) & 0x1F)
        let frac = Float(bits & 0x3FF)
        let s: Float = sign == 1 ? -1 : 1
        if exp == 0 { return s * (frac / 1024) * powf(2, -14) }
        if exp == 0x1F { return frac == 0 ? s * .infinity : .nan }
        return s * (1 + frac / 1024) * powf(2, Float(exp - 15))
    }

    /// Resolve a graph dir name to a compiled `.mlmodelc`: pass `.mlmodelc` through; compile a
    /// `.mlpackage` once into `<cacheDir>/compiled/`. CoreML's `MLModel(contentsOf:)` needs a
    /// compiled model — the Supertonic HF repo ships `.mlpackage`.
    static func compiledURL(name: String, in dir: URL, cacheDir: URL) throws -> URL {
        let fm = FileManager.default
        let mlmodelc = dir.appendingPathComponent("\(name).mlmodelc", isDirectory: true)
        if fm.fileExists(atPath: mlmodelc.path) { return mlmodelc }

        let pkg = dir.appendingPathComponent("\(name).mlpackage", isDirectory: true)
        guard fm.fileExists(atPath: pkg.path) else { throw SupertonicError.missingFile("\(name).mlpackage") }

        let compiledDir = cacheDir.appendingPathComponent("compiled", isDirectory: true)
        try fm.createDirectory(at: compiledDir, withIntermediateDirectories: true)
        let dest = compiledDir.appendingPathComponent("\(name).mlmodelc", isDirectory: true)
        if fm.fileExists(atPath: dest.path) { return dest }

        let tmp = try MLModel.compileModel(at: pkg)
        if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
        try fm.copyItem(at: tmp, to: dest)   // copy (tmp may be on a different volume), then load
        return dest
    }

    static func load(_ url: URL, computeUnits: MLComputeUnits) throws -> MLModel {
        let config = MLModelConfiguration()
        config.computeUnits = CoreMLComputeUnitsResolver.resolved(default: computeUnits)
        return try MLModel(contentsOf: url, configuration: config)
    }
}

// MARK: - the four flow-matching graphs

/// Owns the four CoreML graphs and runs them by feature name (CoreML binds by name, so there is no
/// positional-order trap). The host drives the flow-matching ODE; the graphs contain no control flow.
final class SupertonicGraphs {
    private let duration: MLModel
    private let encoder: MLModel
    private let vector: MLModel
    private let vocoder: MLModel

    init(dir: URL, cacheDir: URL, computeUnits: MLComputeUnits) throws {
        duration = try SupertonicBridge.load(
            SupertonicBridge.compiledURL(name: "DurationPredictor", in: dir, cacheDir: cacheDir), computeUnits: computeUnits)
        encoder = try SupertonicBridge.load(
            SupertonicBridge.compiledURL(name: "TextEncoder", in: dir, cacheDir: cacheDir), computeUnits: computeUnits)
        vector = try SupertonicBridge.load(
            SupertonicBridge.compiledURL(name: "VectorEstimator", in: dir, cacheDir: cacheDir), computeUnits: computeUnits)
        vocoder = try SupertonicBridge.load(
            SupertonicBridge.compiledURL(name: "Vocoder", in: dir, cacheDir: cacheDir), computeUnits: computeUnits)
    }

    private func predict(_ model: MLModel, _ inputs: [String: MLMultiArray], output: String) throws -> MLMultiArray {
        let provider = try MLDictionaryFeatureProvider(
            dictionary: inputs.mapValues { MLFeatureValue(multiArray: $0) })
        let out = try model.prediction(from: provider)
        guard let arr = out.featureValue(for: output)?.multiArrayValue else {
            throw SupertonicError.inference("missing output '\(output)'")
        }
        return arr
    }

    /// duration_predictor → predicted seconds.
    func predictDuration(textIds: MLMultiArray, styleDp: MLMultiArray, textMask: MLMultiArray) throws -> Float {
        let arr = try predict(duration, ["text_ids": textIds, "style_dp": styleDp, "text_mask": textMask],
                              output: "duration")
        return SupertonicBridge.toFloat32(arr).first ?? 0
    }

    /// text_encoder → text_emb [1,256,T] (flat row-major).
    func encodeText(textIds: MLMultiArray, styleTtl: MLMultiArray, textMask: MLMultiArray) throws -> MLMultiArray {
        try predict(encoder, ["text_ids": textIds, "style_ttl": styleTtl, "text_mask": textMask],
                    output: "text_emb")
    }

    /// One vector_estimator ODE step → denoised_latent [1,144,L] (flat).
    func vectorStep(noisy: MLMultiArray, textEmb: MLMultiArray, styleTtl: MLMultiArray,
                    latentMask: MLMultiArray, textMask: MLMultiArray,
                    currentStep: MLMultiArray, totalStep: MLMultiArray) throws -> MLMultiArray {
        try predict(vector, [
            "noisy_latent": noisy, "text_emb": textEmb, "style_ttl": styleTtl,
            "latent_mask": latentMask, "text_mask": textMask,
            "current_step": currentStep, "total_step": totalStep,
        ], output: "denoised_latent")
    }

    /// vocoder → wav [1, 3072*L].
    func vocode(latent: MLMultiArray) throws -> [Float] {
        SupertonicBridge.toFloat32(try predict(vocoder, ["latent": latent], output: "wav"))
    }
}
#endif
