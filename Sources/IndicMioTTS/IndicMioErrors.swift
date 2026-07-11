import Foundation

public enum IndicMioError: LocalizedError, Equatable {
    case missingFile(URL)
    case invalidConfig(String)
    case tokenizerLoadFailed(String)
    case noSpeechTokensGenerated
    case codecDecodeNotImplemented(String)
    case speakerEmbeddingNotImplemented(String)
    case unsupportedMarker(String)

    public var errorDescription: String? {
        switch self {
        case .missingFile(let url):
            return "Missing Indic-Mio file: \(url.path)"
        case .invalidConfig(let reason):
            return "Invalid Indic-Mio config: \(reason)"
        case .tokenizerLoadFailed(let reason):
            return "Failed to load Indic-Mio tokenizer: \(reason)"
        case .noSpeechTokensGenerated:
            return "Indic-Mio generated no MioCodec speech tokens"
        case .codecDecodeNotImplemented(let reason):
            return "Indic-Mio codec decode is not fully implemented: \(reason)"
        case .speakerEmbeddingNotImplemented(let reason):
            return "Indic-Mio speaker embedding is not implemented: \(reason)"
        case .unsupportedMarker(let marker):
            return "Unsupported Indic-Mio emotion marker: \(marker)"
        }
    }
}
