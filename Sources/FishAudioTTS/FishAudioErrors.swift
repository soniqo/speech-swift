import Foundation

public enum FishAudioError: Error, Equatable, CustomStringConvertible {
    case missingFile(URL)
    case invalidConfig(String)
    case malformedTokenizerJSON
    case tokenizerLoadFailed(String)
    case missingToken(String)
    case invalidCodebookShape(String)

    public var description: String {
        switch self {
        case .missingFile(let url):
            return "Missing Fish Audio file: \(url.path)"
        case .invalidConfig(let reason):
            return "Invalid Fish Audio config: \(reason)"
        case .malformedTokenizerJSON:
            return "Malformed Fish Audio tokenizer.json"
        case .tokenizerLoadFailed(let reason):
            return "Fish Audio tokenizer load failed: \(reason)"
        case .missingToken(let token):
            return "Fish Audio tokenizer is missing required token \(token)"
        case .invalidCodebookShape(let reason):
            return "Invalid Fish Audio codebook shape: \(reason)"
        }
    }
}
