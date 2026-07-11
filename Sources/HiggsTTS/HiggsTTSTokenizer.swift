import Foundation
import Hub
import Tokenizers

/// Higgs protocol special tokens (added tokens in the upstream tokenizer).
public enum HiggsTTSToken {
    public static let tts = "<|tts|>"
    public static let refAudio = "<|ref_audio|>"
    public static let refText = "<|ref_text|>"
    public static let text = "<|text|>"
    public static let textEnd = "<|text_end|>"
    public static let audio = "<|audio|>"
    public static let audioEnd = "<|audio_end|>"
    public static let endOfText = "<|endoftext|>"
}

/// Resolved ids for the protocol tokens the prompt builder needs.
public struct HiggsTTSSpecialTokens: Equatable, Sendable {
    public let tts: Int32
    public let refAudio: Int32
    public let refText: Int32?
    public let text: Int32
    public let audio: Int32

    public init(tts: Int32, refAudio: Int32, refText: Int32?, text: Int32, audio: Int32) {
        self.tts = tts
        self.refAudio = refAudio
        self.refText = refText
        self.text = text
        self.audio = audio
    }
}

/// Qwen tokenizer wrapper for Higgs TTS 3, loading the upstream
/// `tokenizer.json` (+ `tokenizer_config.json`) via swift-transformers,
/// following the FishAudioTTS pattern.
public final class HiggsTTSTokenizer: @unchecked Sendable {
    private let tokenizer: Tokenizer
    public let specials: HiggsTTSSpecialTokens

    public init(tokenizer: Tokenizer) throws {
        self.tokenizer = tokenizer
        func id(_ token: String) throws -> Int32 {
            guard let value = tokenizer.convertTokenToId(token) else {
                throw HiggsTTSError.missingRequiredFile("tokenizer special token \(token)")
            }
            return Int32(value)
        }
        self.specials = HiggsTTSSpecialTokens(
            tts: try id(HiggsTTSToken.tts),
            refAudio: try id(HiggsTTSToken.refAudio),
            refText: tokenizer.convertTokenToId(HiggsTTSToken.refText).map(Int32.init),
            text: try id(HiggsTTSToken.text),
            audio: try id(HiggsTTSToken.audio))
    }

    public static func load(from directory: URL) async throws -> HiggsTTSTokenizer {
        let dataURL = directory.appendingPathComponent("tokenizer.json")
        let configURL = directory.appendingPathComponent("tokenizer_config.json")
        guard FileManager.default.fileExists(atPath: dataURL.path) else {
            throw HiggsTTSError.missingRequiredFile("tokenizer.json")
        }
        let hubApi = HubApi()
        let tokenizerData = try hubApi.configuration(fileURL: dataURL)
        let tokenizerConfig: Config
        if FileManager.default.fileExists(atPath: configURL.path) {
            tokenizerConfig = try hubApi.configuration(fileURL: configURL)
        } else {
            tokenizerConfig = Config([String: Config]())
        }
        let tokenizer = try AutoTokenizer.from(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: tokenizerData,
            strict: false)
        return try HiggsTTSTokenizer(tokenizer: tokenizer)
    }

    public func encode(_ text: String) -> [Int32] {
        tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
    }

    public func tokenId(_ token: String) -> Int32? {
        tokenizer.convertTokenToId(token).map(Int32.init)
    }

    public func decode(_ ids: [Int32], skipSpecialTokens: Bool = false) -> String {
        tokenizer.decode(tokens: ids.map(Int.init), skipSpecialTokens: skipSpecialTokens)
    }
}
