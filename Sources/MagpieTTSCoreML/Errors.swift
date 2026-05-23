import Foundation

/// Errors thrown by the CoreML Magpie-TTS pipeline. Mirrors the MLX module's
/// ``MagpieTTSError`` shape so callers can write a single catch path when they
/// switch backends.
public enum MagpieCoreMLError: Error, LocalizedError {
    case missingFile(String)
    case invalidConstants(String)
    case invalidNpyFile(path: String, reason: String)
    case modelLoadFailed(name: String, underlying: String)
    case inferenceFailed(stage: String, underlying: String)
    case unsupportedLanguage(String)
    case textTooLong(tokens: Int, max: Int)
    case audioTooLong(frames: Int, max: Int)

    public var errorDescription: String? {
        switch self {
        case .missingFile(let f):
            return "MagpieCoreML: missing file \(f)"
        case .invalidConstants(let m):
            return "MagpieCoreML: invalid constants: \(m)"
        case .invalidNpyFile(let p, let r):
            return "MagpieCoreML: invalid .npy at \(p): \(r)"
        case .modelLoadFailed(let n, let u):
            return "MagpieCoreML: failed to load \(n): \(u)"
        case .inferenceFailed(let s, let u):
            return "MagpieCoreML: \(s) inference failed: \(u)"
        case .unsupportedLanguage(let l):
            return "MagpieCoreML: language \(l) not supported by the CoreML bundle (use --engine magpie for Japanese)"
        case .textTooLong(let t, let m):
            return "MagpieCoreML: text produced \(t) tokens; bundle max is \(m)"
        case .audioTooLong(let f, let m):
            return "MagpieCoreML: generated \(f) codec frames; bundle nano-codec max is \(m) (≈\(Double(m) / 21.5)s)"
        }
    }
}
