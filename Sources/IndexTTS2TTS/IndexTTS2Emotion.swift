import Foundation

public enum IndexTTS2EmotionControlError: Error, LocalizedError, Equatable {
    case unknownPreset(String)
    case invalidVectorCount(Int)
    case invalidVectorValue(Float)
    case invalidVectorSum(Float)
    case invalidWeight(Float)

    public var errorDescription: String? {
        switch self {
        case .unknownPreset(let value):
            return "Unknown IndexTTS2 emotion preset '\(value)'."
        case .invalidVectorCount(let count):
            return "IndexTTS2 emotion vector must contain exactly 8 values; got \(count)."
        case .invalidVectorValue(let value):
            return "IndexTTS2 emotion vector values must be finite values between 0.0 and 1.0; got \(value)."
        case .invalidVectorSum(let sum):
            return "IndexTTS2 emotion vector sum must be <= 0.8; got \(sum)."
        case .invalidWeight(let value):
            return "IndexTTS2 emotion weight must be finite and between 0.0 and 1.0; got \(value)."
        }
    }
}

public enum IndexTTS2EmotionPreset: String, CaseIterable, Sendable {
    case happy
    case eager
    case excited
    case angry
    case sad
    case afraid
    case disgusted
    case melancholic
    case surprised
    case calm

    public static let emotionOrder = [
        "happy",
        "angry",
        "sad",
        "afraid",
        "disgusted",
        "melancholic",
        "surprised",
        "calm",
    ]

    public init?(named name: String) {
        switch name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "happy":
            self = .happy
        case "eager":
            self = .eager
        case "excited":
            self = .excited
        case "angry":
            self = .angry
        case "sad":
            self = .sad
        case "afraid", "fear", "fearful":
            self = .afraid
        case "disgusted", "disgust":
            self = .disgusted
        case "melancholic", "melancholy":
            self = .melancholic
        case "surprised", "surprise":
            self = .surprised
        case "calm", "neutral":
            self = .calm
        default:
            return nil
        }
    }

    public var vector: [Float] {
        switch self {
        case .happy:
            return [0.8, 0, 0, 0, 0, 0, 0, 0]
        case .eager, .excited:
            return [0.65, 0, 0, 0, 0, 0, 0.15, 0]
        case .angry:
            return [0, 0.8, 0, 0, 0, 0, 0, 0]
        case .sad:
            return [0, 0, 0.8, 0, 0, 0, 0, 0]
        case .afraid:
            return [0, 0, 0, 0.8, 0, 0, 0, 0]
        case .disgusted:
            return [0, 0, 0, 0, 0.8, 0, 0, 0]
        case .melancholic:
            return [0, 0, 0, 0, 0, 0.8, 0, 0]
        case .surprised:
            return [0, 0, 0, 0, 0, 0, 0.8, 0]
        case .calm:
            return [0, 0, 0, 0, 0, 0, 0, 0.8]
        }
    }
}

public struct IndexTTS2EmotionControl: Equatable, Sendable {
    public let vector: [Float]
    public let weight: Float

    public init(preset: IndexTTS2EmotionPreset, weight: Float = 1.0) throws {
        try self.init(vector: preset.vector, weight: weight)
    }

    public init(vector: [Float], weight: Float = 1.0) throws {
        guard vector.count == 8 else {
            throw IndexTTS2EmotionControlError.invalidVectorCount(vector.count)
        }
        for value in vector {
            guard value.isFinite, value >= 0, value <= 1 else {
                throw IndexTTS2EmotionControlError.invalidVectorValue(value)
            }
        }
        let sum = vector.reduce(Float(0), +)
        guard sum <= 0.800_001 else {
            throw IndexTTS2EmotionControlError.invalidVectorSum(sum)
        }
        guard weight.isFinite, weight >= 0, weight <= 1 else {
            throw IndexTTS2EmotionControlError.invalidWeight(weight)
        }
        self.vector = vector
        self.weight = weight
    }

    public var scaledVector: [Float] {
        vector.map { $0 * weight }
    }

    public var scaledVectorSum: Float {
        scaledVector.reduce(Float(0), +)
    }
}
