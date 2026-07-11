import Foundation
import NaturalLanguage

/// Per-language tokenizer config dumped from NeMo's MagpieTTS multilingual
/// vocabulary. Each language's JSON ships the SAME shared 2360-entry vocab
/// (so the cross-lingual decoder sees a single token-ID space); the
/// per-language difference is *how* input text is converted to that
/// vocab's symbols.
public struct MagpieTokenizerConfig: Codable, Sendable {
    public let language: String
    public let tokenizerName: String
    public let type: String
    public let vocabSize: Int
    public let tokens: [String]
    public let bosId: Int?
    public let eosId: Int?
    public let padId: Int?
    public let config: MagpieTokenizerSubConfig?

    enum CodingKeys: String, CodingKey {
        case language
        case tokenizerName = "tokenizer_name"
        case type
        case vocabSize = "vocab_size"
        case tokens
        case bosId = "bos_id"
        case eosId = "eos_id"
        case padId = "pad_id"
        case config
    }

    public struct MagpieTokenizerSubConfig: Codable, Sendable {
        public let punct: Bool?
        public let apostrophe: Bool?
        public let padWithSpace: Bool?
        public let pretrainedModel: String?
        enum CodingKeys: String, CodingKey {
            case punct, apostrophe
            case padWithSpace = "pad_with_space"
            case pretrainedModel = "pretrained_model"
        }
    }
}

/// Maps language-specific text to a sequence of vocab IDs. Loads the
/// shared vocab JSON at construction time.
///
/// **Current scope.** The initial port implements:
///
/// * Symbol-level char tokenization against the bundle's vocab list. Tokens
///   not present in the vocab map to the `<oov>` slot (or are silently
///   dropped if `<oov>` is absent — older bundles).
/// * `pad_with_space` handling per the per-language NeMo config (Mandarin and
///   Hindi prepend / append a space).
/// * For Japanese (no shipped tokenizer JSON) we transliterate the input via
///   Apple's `CFStringTokenizer` → katakana before falling back to the EN
///   tokenizer's vocab. This is the same trick KokoroTTS uses.
///
/// **What is intentionally simplified.** Each language ships a NeMo G2P /
/// transliteration model on the Python side (eSpeak-style rule sets for
/// en/es/de, pypinyin for zh, BPMF for hi, byT5 byte tokenization for
/// fr/it/vi). Replicating each of those in pure Swift is a separate, large
/// effort. For now we feed the user's raw text through with light
/// normalisation. The TTS still produces audio — the diction is approximate
/// when the input is not pre-phonemised. Callers who already have IPA can
/// pass it as-is (or use the `--magpie-prephonemized` CLI flag).
public final class MagpieTokenizer {
    public let language: MagpieLanguage
    public let config: MagpieTokenizerConfig
    public let tokenToId: [String: Int]
    public let oovId: Int?
    public let padId: Int?

    public init(language: MagpieLanguage, config: MagpieTokenizerConfig) {
        self.language = language
        self.config = config
        // Build a **per-language** char → aggregated-id map by slicing the
        // bundle's full 2360-entry vocab to this language's sub-tokenizer
        // section `[offset, offset + size)`. NeMo's BaseTokenizer builds the
        // lookup via `{l: i for i, l in enumerate(tokens)}` — a Python dict
        // comprehension where **later occurrences overwrite earlier ones**.
        // Some tokenizers (notably `HindiCharsTokenizer`) emit duplicate
        // entries within their own vocab (the IPA punct list overlaps with
        // the Devanagari character set), and the model was trained using the
        // last-occurrence id. We mirror that here.
        let offset = MagpieSubVocab.offset(for: language)
        let size = MagpieSubVocab.size(for: language)
        let end = min(offset + size, config.tokens.count)
        var map: [String: Int] = [:]
        for i in offset..<end {
            map[config.tokens[i]] = i  // overwrite; last wins
        }
        self.tokenToId = map
        self.oovId = map["<oov>"]
        self.padId = map["<pad>"] ?? config.padId
    }

    public static func load(from url: URL, language: MagpieLanguage) throws -> MagpieTokenizer {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MagpieTTSError.missingFile(url.lastPathComponent)
        }
        let data = try Data(contentsOf: url)
        let cfg = try JSONDecoder().decode(MagpieTokenizerConfig.self, from: data)
        return MagpieTokenizer(language: language, config: cfg)
    }

    /// Magpie multilingual text-embedding adds BOS + EOS rows past the vocab:
    /// `text_embedding.size = vocab_size + 2`. Round-trip inference requires
    /// the text token sequence to be terminated with `eos_id` so the decoder
    /// can attend to a proper sentence boundary. (See
    /// `magpietts.py`: `self.eos_id = num_tokens - 1` with
    /// `num_tokens = vocab_size + 2`.)
    public var eosId: Int { config.vocabSize + 1 }
    public var bosId: Int { config.vocabSize }

    /// Phonemise + char-tokenise. Returns the integer ID sequence consumed by
    /// the text encoder. The trailing `eos_id = vocab_size + 1` is **always**
    /// appended (mirroring NeMo's round-trip test and the model's training
    /// regime); without it the AR loop often fails to terminate.
    ///
    /// Dispatch per language (matches `model_config.yaml`):
    /// - en / es / de — IPA dict G2P, char-tokenise IPA stream
    /// - fr / it / vi — `byT5` UTF-8 byte encoder (`byte + 1979`)
    /// - hi          — char-level (Devanagari graphemes are in the vocab)
    /// - zh / ja     — char-level fallback **for now** (needs jieba+pypinyin
    ///                 and Apple morpheme analyzer, tracked as follow-up)
    public func tokenize(_ text: String, prephonemized: Bool = false) -> [Int] {
        var content: [Int]
        if prephonemized {
            content = charLookup(text)
        } else {
            content = encodePerLanguage(text)
        }

        // `pad_with_space` per the bundle's tokenizer config, with one
        // override: Japanese reuses the EN tokenizer JSON (no JA file
        // ships), so we hardcode `pad_with_space=true` for JA to match
        // `model_config.yaml`.
        let padWithSpace: Bool
        if language == .japanese {
            padWithSpace = true
        } else {
            padWithSpace = config.config?.padWithSpace ?? false
        }
        let space = tokenToId[" "]
        if padWithSpace, let s = space {
            content.insert(s, at: 0)
            content.append(s)
        }

        var clean = content.filter { $0 >= 0 }
        clean.append(eosId)
        return clean
    }

    /// Map raw text through the per-language preprocessing + phoneme
    /// pipeline. Always returns Magpie shared-vocab IDs.
    private func encodePerLanguage(_ text: String) -> [Int] {
        switch language {
        case .english:
            // NeMo `english_text_preprocessing(text, lower=False)`: NFD
            // normalise + strip combining marks. We keep case to feed the
            // dictionary, which is uppercased on lookup.
            let normalised = text
                .precomposedStringWithCompatibilityMapping  // close-enough NFC-ish
            return charLookup(MagpieDictG2P.english.phonemize(normalised))

        case .spanish:
            // `spanish_text_preprocessing(text) = text.lower()`.
            return charLookup(MagpieDictG2P.spanish.phonemize(text.lowercased()))

        case .german:
            // `any_locale_text_preprocessing` — NFC + smart quote → '.
            let normalised = text
                .precomposedStringWithCompatibilityMapping
                .replacingOccurrences(of: "\u{2019}", with: "'")
            return charLookup(MagpieDictG2P.german.phonemize(normalised))

        case .french:
            // byt5_byte tokeniser. NeMo lowercases before encoding.
            return MagpieByT5Encoder.encode(text.lowercased(), language: .french)

        case .italian:
            return MagpieByT5Encoder.encode(text.lowercased(), language: .italian)

        case .vietnamese:
            return MagpieByT5Encoder.encode(text.lowercased(), language: .vietnamese)

        case .hindi:
            // char_hindi tokeniser: feed Devanagari graphemes verbatim, plus
            // standard ASCII punctuation. The vocab covers all Devanagari
            // chars (and most punctuation) so char-level lookup works.
            return charLookup(text)

        case .chinese:
            // Apple `mandarinToLatin` for Hanzi → tone-marked pinyin, then
            // our bundled pinyin → IPA dict + tone-marker injection
            // (`MagpieChineseG2P`). The phonemizer returns *grouped*
            // phonemes so we can concatenate within-syllable tokens
            // (matching NeMo's encode) and insert a space token between
            // syllables only.
            return lookupGroups(MagpieChineseG2P.shared.phonemize(text))

        case .japanese:
            // Magpie's JA sub-vocab expects per-mora pitch markers `0`/`1`
            // interleaved with katakana (NeMo `JapaneseKatakanaAccentG2p`).
            // `MagpieJapaneseG2P.phonemes` produces a flat token stream
            // (chains contain no inter-word spaces — only explicit
            // whitespace / punctuation makes it into the stream) which
            // mirrors NeMo's `encode_from_g2p` exactly.
            return lookupFlatTokens(MagpieJapaneseG2P.phonemes(text))
        }
    }

    /// Lookup grouped phonemes: tokens within a group are concatenated
    /// adjacent (no inter-token space); groups are separated by a single
    /// space token. This matches NeMo `ChinesePhonemesTokenizer.encode`
    /// behaviour where a phonemized syllable's pieces ride together and a
    /// space marks the next syllable.
    private func lookupGroups(_ groups: [[String]]) -> [Int] {
        var ids: [Int] = []
        let space = tokenToId[" "]
        var first = true
        for group in groups {
            if !first, let sp = space { ids.append(sp) }
            first = false
            for token in group {
                if let id = tokenToId[token] {
                    ids.append(id)
                } else if let oov = oovId {
                    ids.append(oov)
                }
            }
        }
        return ids
    }

    /// Lookup a flat phoneme stream against the per-language vocab. Matches
    /// NeMo's `encode_from_g2p`: pass through tokens, deduplicate adjacent
    /// spaces, drop trailing spaces (the outer `pad_with_space` re-wraps).
    private func lookupFlatTokens(_ tokens: [String]) -> [Int] {
        let space = tokenToId[" "]
        var ids: [Int] = []
        for token in tokens {
            if token == " " {
                // Skip leading / duplicate spaces.
                if let sp = space, ids.last != sp, !ids.isEmpty {
                    ids.append(sp)
                }
                continue
            }
            if let id = tokenToId[token] {
                ids.append(id)
            } else if let oov = oovId {
                ids.append(oov)
            }
        }
        // Trim trailing space (the outer pad_with_space adds one back).
        while let last = ids.last, last == space { ids.removeLast() }
        return ids
    }

    /// Codepoint-by-codepoint vocab lookup for IPA / Devanagari / Latin
    /// strings. We iterate `unicodeScalars` (not `Character`) so that
    /// Devanagari conjuncts like "स्ते" (4 codepoints clustered into one
    /// grapheme by Swift) decompose into individual vocab entries — Magpie
    /// trained on per-codepoint inputs.
    private func charLookup(_ s: String) -> [Int] {
        var ids: [Int] = []
        ids.reserveCapacity(s.unicodeScalars.count)
        for scalar in s.unicodeScalars {
            ids.append(idForChar(String(scalar)))
        }
        return ids
    }

    private func idForChar(_ s: String) -> Int {
        if let id = tokenToId[s] { return id }
        if let oov = oovId { return oov }
        return -1
    }

}
