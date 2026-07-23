import AudioCommon
import Foundation

struct LocalVQEAECArtifacts {
    static let modelFileName = "LocalVQEAECResidualMask.mlmodelc"
    static let frontendFileName = "LocalVQEAECFrontend.npz"

    let modelURL: URL
    let controllerWeights: [Float]
    let analysisWindow: [Float]

    static func load(from directory: URL) throws -> LocalVQEAECArtifacts {
        let modelURL = directory.appendingPathComponent(modelFileName)
        let frontendURL = directory.appendingPathComponent(frontendFileName)
        let fileManager = FileManager.default

        guard fileManager.fileExists(atPath: modelURL.path) else {
            throw AudioModelError.modelLoadFailed(
                modelId: "LocalVQE AEC",
                reason: "Missing compiled model at \(modelURL.path)")
        }
        guard fileManager.fileExists(atPath: frontendURL.path) else {
            throw AudioModelError.weightLoadingFailed(path: frontendURL.path)
        }

        let arrays: [String: [Float]]
        do {
            arrays = try NpzReader.read(url: frontendURL)
        } catch {
            throw AudioModelError.weightLoadingFailed(
                path: frontendURL.path, underlying: error)
        }

        let controllerLayout: [(String, Int)] = [
            ("daf.glob.norm.ln.weight", 6),
            ("daf.glob.norm.ln.bias", 6),
            ("daf.glob.gru.weight_ih", 24 * 10),
            ("daf.glob.gru.weight_hh", 24 * 8),
            ("daf.glob.gru.bias_ih", 24),
            ("daf.glob.gru.bias_hh", 24),
            ("daf.bins.norm.ln.weight", 6),
            ("daf.bins.norm.ln.bias", 6),
            ("daf.bins.gru.weight_ih", 48 * 18),
            ("daf.bins.gru.weight_hh", 48 * 16),
            ("daf.bins.gru.bias_ih", 48),
            ("daf.bins.gru.bias_hh", 48),
            ("daf.part.ln.weight", 2),
            ("daf.part.ln.bias", 2),
            ("daf.part.gru.weight_ih", 24 * 10),
            ("daf.part.gru.weight_hh", 24 * 8),
            ("daf.part.gru.bias_ih", 24),
            ("daf.part.gru.bias_hh", 24),
            ("daf.head.weight", 16),
            ("daf.head.bias", 1),
            ("daf.part.head.weight", 8),
            ("daf.part.head.bias", 1),
        ]

        var controllerWeights = [Float]()
        controllerWeights.reserveCapacity(2_742)
        for (name, expectedCount) in controllerLayout {
            guard let values = arrays[name] else {
                throw AudioModelError.weightLoadingFailed(
                    path: frontendURL.path,
                    underlying: LocalVQEEchoCancellationError.missingFrontendArray(name))
            }
            guard values.count == expectedCount else {
                throw AudioModelError.weightLoadingFailed(
                    path: frontendURL.path,
                    underlying: LocalVQEEchoCancellationError.invalidFrontendArray(
                        name: name, expected: expectedCount, actual: values.count))
            }
            guard values.allSatisfy(\.isFinite) else {
                throw AudioModelError.weightLoadingFailed(
                    path: frontendURL.path,
                    underlying: LocalVQEEchoCancellationError.nonFiniteFrontendArray(name))
            }
            controllerWeights.append(contentsOf: values)
        }

        guard let analysisWindow = arrays["analysis_window"] else {
            throw AudioModelError.weightLoadingFailed(
                path: frontendURL.path,
                underlying: LocalVQEEchoCancellationError.missingFrontendArray(
                    "analysis_window"))
        }
        guard analysisWindow.count == LocalVQEEchoCanceller.fftSize else {
            throw AudioModelError.weightLoadingFailed(
                path: frontendURL.path,
                underlying: LocalVQEEchoCancellationError.invalidFrontendArray(
                    name: "analysis_window",
                    expected: LocalVQEEchoCanceller.fftSize,
                    actual: analysisWindow.count))
        }
        guard analysisWindow.allSatisfy(\.isFinite) else {
            throw AudioModelError.weightLoadingFailed(
                path: frontendURL.path,
                underlying: LocalVQEEchoCancellationError.nonFiniteFrontendArray(
                    "analysis_window"))
        }

        return LocalVQEAECArtifacts(
            modelURL: modelURL,
            controllerWeights: controllerWeights,
            analysisWindow: analysisWindow)
    }
}
