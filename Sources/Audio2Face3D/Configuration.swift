import Foundation

public enum Audio2Face3DBackend: String, Codable, Sendable {
    /// Full NVIDIA Audio2Face-3D v2.3 graph exported into MLX tensors. The
    /// export script lands the weights; graph execution is a hand-written MLX
    /// port and must match the ONNX parity fixtures.
    case mlx
}

public struct Audio2Face3DConfiguration: Codable, Equatable, Sendable {
    /// Published MLX bundles, one per NVIDIA identity. Coefficients are
    /// identity-specific (Mark: 272 skin, Claire/James: 140 skin), so
    /// renderers need the matching rig or retarget projection.
    public static let markModelId = "aufklarer/Audio2Face-3D-v2.3-Mark-MLX"
    public static let claireModelId = "aufklarer/Audio2Face-3D-v2.3.1-Claire-MLX"
    public static let jamesModelId = "aufklarer/Audio2Face-3D-v2.3.1-James-MLX"
    public static let defaultModelId = jamesModelId

    public var modelId: String
    public var inputSampleRate: Int
    public var bufferLength: Int
    public var hopLength: Int
    public var framesPerSecond: Double
    public var implicitEmotionCount: Int
    public var explicitEmotionCount: Int
    public var coefficientLayout: Audio2Face3DCoefficientLayout
    public var inputStrength: Float

    public init(
        modelId: String = Self.defaultModelId,
        inputSampleRate: Int = 16_000,
        bufferLength: Int = 8_320,
        hopLength: Int = 4_160,
        framesPerSecond: Double = 30,
        implicitEmotionCount: Int = 16,
        explicitEmotionCount: Int = 10,
        coefficientLayout: Audio2Face3DCoefficientLayout = .nvidiaV23Mark,
        inputStrength: Float = 1.3
    ) {
        self.modelId = modelId
        self.inputSampleRate = inputSampleRate
        self.bufferLength = bufferLength
        self.hopLength = hopLength
        self.framesPerSecond = framesPerSecond
        self.implicitEmotionCount = implicitEmotionCount
        self.explicitEmotionCount = explicitEmotionCount
        self.coefficientLayout = coefficientLayout
        self.inputStrength = inputStrength
    }

    public var emotionVectorLength: Int {
        explicitEmotionCount + implicitEmotionCount
    }

    public var frameSampleCount: Int {
        max(1, Int((Double(inputSampleRate) / framesPerSecond).rounded()))
    }

    public var outputCoefficientCount: Int {
        coefficientLayout.coefficientCount
    }
}

public enum Audio2Face3DError: Error, LocalizedError, Equatable {
    case invalidSampleRate(Int)
    case invalidAudioWindow(expected: Int, got: Int)
    case invalidEmotionVector(expected: Int, got: Int)
    case invalidModelOutput(expected: Int, got: Int)
    case missingTensor(String)
    case unsupportedBackend(Audio2Face3DBackend)
    case missingExportedWeights(String)

    public var errorDescription: String? {
        switch self {
        case .invalidSampleRate(let sampleRate):
            return "Invalid sample rate: \(sampleRate)"
        case .invalidAudioWindow(let expected, let got):
            return "Invalid Audio2Face3D audio window: expected \(expected) samples, got \(got)"
        case .invalidEmotionVector(let expected, let got):
            return "Invalid Audio2Face3D emotion vector: expected \(expected) values, got \(got)"
        case .invalidModelOutput(let expected, let got):
            return "Invalid Audio2Face3D model output: expected \(expected) coefficients, got \(got)"
        case .missingTensor(let key):
            return "Missing Audio2Face3D tensor '\(key)' in exported MLX weights"
        case .unsupportedBackend(let backend):
            return "Unsupported Audio2Face3D backend '\(backend.rawValue)'"
        case .missingExportedWeights(let path):
            return "Missing exported Audio2Face3D MLX weights at \(path)"
        }
    }
}
