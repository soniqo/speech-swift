#if canImport(CoreML)
import Foundation

/// Runtime settings encoded by the published Community-1 Core ML bundle.
///
/// Community-1 is a complete diarization pipeline rather than a single neural
/// network. The segmentation and masked speaker-embedding stages run through
/// Core ML; PLDA, VBx clustering, and timeline reconstruction run as native
/// Swift host code.
public struct Community1Config: Sendable {
    public static let sampleRate = 16_000
    public static let chunkSamples = 160_000
    public static let chunkDuration: Float = 10.0
    public static let chunkStep: Float = 1.0
    public static let framesPerChunk = 589
    public static let localSpeakers = 3
    public static let embeddingDimension = 256
    public static let pldaDimension = 128

    /// Receptive-field geometry reported by the original PyanNet model.
    public static let frameDuration: Float = 0.0619375
    public static let frameStep: Float = 0.016875

    /// Euclidean cut applied to centroid linkage over L2-normalized embeddings.
    public var clusteringThreshold: Float
    /// VBx sufficient-statistics scale.
    public var fa: Float
    /// VBx speaker regularization coefficient.
    public var fb: Float
    /// Initial posterior smoothing applied to the AHC labels.
    public var initialSmoothing: Float
    /// Maximum VBx iterations.
    public var maxIterations: Int
    /// VBx convergence threshold.
    public var convergenceEpsilon: Float
    /// Minimum ratio of clean, single-speaker frames used to train clustering.
    public var minimumActiveRatio: Float
    /// Speaker-prior cutoff below which a VBx component is discarded.
    public var speakerPriorCutoff: Float

    public init(
        clusteringThreshold: Float = 0.6,
        fa: Float = 0.07,
        fb: Float = 0.8,
        initialSmoothing: Float = 7.0,
        maxIterations: Int = 20,
        convergenceEpsilon: Float = 1e-4,
        minimumActiveRatio: Float = 0.2,
        speakerPriorCutoff: Float = 1e-7
    ) {
        self.clusteringThreshold = clusteringThreshold
        self.fa = fa
        self.fb = fb
        self.initialSmoothing = initialSmoothing
        self.maxIterations = maxIterations
        self.convergenceEpsilon = convergenceEpsilon
        self.minimumActiveRatio = minimumActiveRatio
        self.speakerPriorCutoff = speakerPriorCutoff
    }

    public static let `default` = Community1Config()
}

/// Optional bounds for Community-1 speaker-count inference.
public struct Community1SpeakerBounds: Sendable {
    public var exact: Int?
    public var minimum: Int
    public var maximum: Int?

    public init(exact: Int? = nil, minimum: Int = 1, maximum: Int? = nil) {
        self.exact = exact
        self.minimum = minimum
        self.maximum = maximum
    }

    public static let inferred = Community1SpeakerBounds()
}

struct Community1BundleManifest: Decodable {
    struct Segmentation: Decodable {
        let model: String
        let inputName: String
        let outputName: String

        enum CodingKeys: String, CodingKey {
            case model
            case inputName = "input_name"
            case outputName = "output_name"
        }
    }

    struct Embedding: Decodable {
        let model: String
        let waveformInputName: String
        let weightsInputName: String
        let outputName: String

        enum CodingKeys: String, CodingKey {
            case model
            case waveformInputName = "waveform_input_name"
            case weightsInputName = "weights_input_name"
            case outputName = "output_name"
        }
    }

    struct PLDA: Decodable {
        let dimension: Int
        let weights: String
    }

    let modelType: String
    let sampleRate: Int
    let segmentation: Segmentation
    let embedding: Embedding
    let plda: PLDA

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case sampleRate = "sample_rate"
        case segmentation
        case embedding
        case plda
    }

    static func load(from directory: URL) throws -> Community1BundleManifest {
        let url = directory.appendingPathComponent("config.json")
        do {
            let manifest = try JSONDecoder().decode(
                Community1BundleManifest.self,
                from: Data(contentsOf: url)
            )
            guard manifest.modelType == "pyannote-community-1-coreml" else {
                throw Community1Error.invalidBundle(
                    "config.json has model_type '\(manifest.modelType)'"
                )
            }
            guard manifest.sampleRate == Community1Config.sampleRate else {
                throw Community1Error.invalidBundle(
                    "expected \(Community1Config.sampleRate) Hz, got \(manifest.sampleRate) Hz"
                )
            }
            guard manifest.plda.dimension == Community1Config.pldaDimension else {
                throw Community1Error.invalidBundle(
                    "expected \(Community1Config.pldaDimension)-dimensional PLDA"
                )
            }
            return manifest
        } catch let error as Community1Error {
            throw error
        } catch {
            throw Community1Error.invalidBundle(
                "could not decode \(url.path): \(error.localizedDescription)"
            )
        }
    }
}

/// Errors raised by the Community-1 native runtime.
public enum Community1Error: Error, LocalizedError {
    case invalidBundle(String)
    case invalidSpeakerBounds(String)
    case inference(stage: String, reason: String)

    public var errorDescription: String? {
        switch self {
        case .invalidBundle(let reason):
            return "Invalid Community-1 bundle: \(reason)"
        case .invalidSpeakerBounds(let reason):
            return "Invalid Community-1 speaker bounds: \(reason)"
        case .inference(let stage, let reason):
            return "Community-1 \(stage) inference failed: \(reason)"
        }
    }
}
#endif
