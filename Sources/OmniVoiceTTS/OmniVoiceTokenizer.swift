import Foundation
import Tokenizers
import Hub

/// Qwen3 BPE tokenizer wrapper for OmniVoice's text front-end.
///
/// Thin adapter over swift-transformers' `Tokenizer` (the published bundle ships
/// a `Qwen2Tokenizer`-class `tokenizer.json` + `tokenizer_config.json`). The
/// seven OmniVoice control tokens (`<|denoise|>`, `<|lang_start|>`, `<|lang_end|>`,
/// `<|instruct_start|>`, `<|instruct_end|>`, `<|text_start|>`, `<|text_end|>`)
/// live in the bundle's added-tokens table; the underlying tokenizer splits them
/// out automatically, so a literal-string encode of the assembled prompt yields
/// the control id followed by the BPE ids of the surrounding text — matching the
/// HF `text_tokenizer(...)` call in the reference.
public final class OmniVoiceTokenizer: @unchecked Sendable {
    private let tokenizer: Tokenizer

    /// Non-verbal tags tokenized standalone (mirrors `_NONVERBAL_PATTERN`), so
    /// their ids are stable regardless of surrounding-language context.
    static let nonverbalPattern = "\\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|surprise-yo|dissatisfaction-hnn)\\]"

    public init(tokenizer: Tokenizer) {
        self.tokenizer = tokenizer
    }

    /// Load from a model directory holding `tokenizer.json` (+ optional
    /// `tokenizer_config.json`).
    public static func load(from directory: URL) async throws -> OmniVoiceTokenizer {
        let hubApi = HubApi()
        let configURL = directory.appendingPathComponent("tokenizer_config.json")
        let dataURL = directory.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: dataURL.path) else {
            // Falls back to the folder loader (resolves config + data itself).
            return OmniVoiceTokenizer(
                tokenizer: try await AutoTokenizer.from(modelFolder: directory, strict: false))
        }
        let tokenizerData = try hubApi.configuration(fileURL: dataURL)
        let tokenizerConfig: Config
        if FileManager.default.fileExists(atPath: configURL.path) {
            tokenizerConfig = try hubApi.configuration(fileURL: configURL)
        } else {
            tokenizerConfig = Config([String: Config]())
        }
        let tk = try AutoTokenizer.from(
            tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, strict: false)
        return OmniVoiceTokenizer(tokenizer: tk)
    }

    /// Id for an exact token string (resolves added/special tokens), or `nil`.
    public func tokenId(_ token: String) -> Int? { tokenizer.convertTokenToId(token) }

    /// Encode `text` to ids. Added/special tokens embedded in the string are split
    /// out by the underlying tokenizer; no BOS/EOS is prepended (Qwen2 default).
    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }

    public func decode(_ ids: [Int]) -> String {
        tokenizer.decode(tokens: ids, skipSpecialTokens: false)
    }

    /// Encode text that may carry non-verbal tags (`[laughter]`, `[sigh]`, …),
    /// tokenizing each tag standalone so its ids don't depend on the surrounding
    /// language. For plain text this is identical to `encode`. Mirrors
    /// `_tokenize_with_nonverbal_tags`.
    public func encodeWithNonverbalTags(_ text: String) -> [Int] {
        guard let regex = try? NSRegularExpression(pattern: Self.nonverbalPattern) else {
            return encode(text)
        }
        let ns = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: ns.length))
        if matches.isEmpty { return encode(text) }

        var out: [Int] = []
        var lastEnd = 0
        for m in matches {
            if m.range.location > lastEnd {
                let seg = ns.substring(with: NSRange(location: lastEnd, length: m.range.location - lastEnd))
                let ids = encode(seg)
                if !ids.isEmpty { out.append(contentsOf: ids) }
            }
            let tag = ns.substring(with: m.range)
            let tagIds = encode(tag)
            if !tagIds.isEmpty { out.append(contentsOf: tagIds) }
            lastEnd = m.range.location + m.range.length
        }
        if lastEnd < ns.length {
            let seg = ns.substring(with: NSRange(location: lastEnd, length: ns.length - lastEnd))
            let ids = encode(seg)
            if !ids.isEmpty { out.append(contentsOf: ids) }
        }
        return out
    }
}
