import Foundation
import CoreML
import AudioCommon

/// CoreML wrapper for the Sidon predictor (LoRA-merged w2v-BERT 2.0, 8 layers).
///
/// `input_features[1, T, 160]` → `features[1, T, 1024]` (the cleansed
/// `last_hidden_state`). `T` is fixed at export time (`SidonConfig.frames`).
final class SidonPredictorModel {
    private let model: MLModel

    init(modelURL: URL, computeUnits: MLComputeUnits = .all) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        self.model = try CoreMLLoader.load(
            url: modelURL, configuration: config, name: "sidon-predictor")
    }

    /// Run the predictor on one fixed-length window.
    /// - Parameter inputFeatures: `[1, frames, 160]` FP32.
    /// - Returns: `features` `[1, frames, 1024]`.
    func predict(inputFeatures: MLMultiArray) throws -> MLMultiArray {
        let input = try MLDictionaryFeatureProvider(
            dictionary: ["input_features": MLFeatureValue(multiArray: inputFeatures)])
        let output = try model.prediction(from: input)
        guard let features = output.featureValue(for: "features")?.multiArrayValue else {
            throw SidonModelError.predictionFailed("predictor: missing 'features' output")
        }
        return features
    }
}

/// CoreML wrapper for the Sidon vocoder (DAC decoder, `rates=[8,5,4,3,2]`).
///
/// `features[1, T, 1024]` → `audio[1, M]` at 48 kHz. The `transpose(1,2)` that
/// the upstream pipeline applies before the decoder is baked into the exported
/// graph, so the runtime feeds `features` as-is.
final class SidonVocoderModel {
    private let model: MLModel

    init(modelURL: URL, computeUnits: MLComputeUnits = .all) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        self.model = try CoreMLLoader.load(
            url: modelURL, configuration: config, name: "sidon-vocoder")
    }

    /// Decode one window of predictor features into 48 kHz audio.
    /// - Parameter features: `[1, frames, 1024]` FP32.
    /// - Returns: `audio` `[1, M]`.
    func predict(features: MLMultiArray) throws -> MLMultiArray {
        let input = try MLDictionaryFeatureProvider(
            dictionary: ["features": MLFeatureValue(multiArray: features)])
        let output = try model.prediction(from: input)
        guard let audio = output.featureValue(for: "audio")?.multiArrayValue else {
            throw SidonModelError.predictionFailed("vocoder: missing 'audio' output")
        }
        return audio
    }
}

enum SidonModelError: Error, LocalizedError {
    case predictionFailed(String)
    case modelNotFound(String)
    case compileFailed(String)

    var errorDescription: String? {
        switch self {
        case .predictionFailed(let m): return "Sidon CoreML prediction failed: \(m)"
        case .modelNotFound(let m): return "Sidon model not found: \(m)"
        case .compileFailed(let m): return "Sidon CoreML compile failed: \(m)"
        }
    }
}

// MARK: - Compiled-model resolution

enum SidonModelResolver {
    /// Resolve a usable model URL for `baseName`, preferring a pre-compiled
    /// `.mlmodelc` and falling back to compiling a `.mlpackage` on device.
    ///
    /// The other CoreML modules in this repo ship `.mlmodelc` only, because
    /// on-device `MLModel.compileModel` from a `.mlpackage` is known to drift
    /// per runtime (Mac vs simulator vs iPhone). Sidon's provisional artifacts
    /// are `.mlpackage`, so we support both: when the published repo gains the
    /// compiled bundles, the `.mlmodelc` branch is taken automatically and the
    /// on-device compile is never reached.
    static func resolve(
        directory: URL, compiledName: String, packageName: String
    ) throws -> URL {
        let fm = FileManager.default
        let compiled = directory.appendingPathComponent(compiledName, isDirectory: true)
        if fm.fileExists(atPath: compiled.path) {
            return compiled
        }
        let package = directory.appendingPathComponent(packageName, isDirectory: true)
        guard fm.fileExists(atPath: package.path) else {
            throw SidonModelError.modelNotFound(
                "neither \(compiledName) nor \(packageName) in \(directory.path)")
        }
        // Compile the .mlpackage to a sibling .mlmodelc and cache it so the
        // (slow) compile only happens once per install.
        return try compileIfNeeded(package: package, into: compiled)
    }

    private static func compileIfNeeded(package: URL, into cached: URL) throws -> URL {
        let fm = FileManager.default
        if fm.fileExists(atPath: cached.path) { return cached }
        do {
            let tmpCompiled = try MLModel.compileModel(at: package)
            // `compileModel` writes to a temp location; move it next to the
            // package for reuse across launches.
            if fm.fileExists(atPath: cached.path) { try? fm.removeItem(at: cached) }
            do {
                try fm.moveItem(at: tmpCompiled, to: cached)
                return cached
            } catch {
                // If the move fails (e.g. read-only cache), fall back to the
                // temp compiled URL for this process.
                return tmpCompiled
            }
        } catch {
            throw SidonModelError.compileFailed("\(package.lastPathComponent): \(error)")
        }
    }
}
