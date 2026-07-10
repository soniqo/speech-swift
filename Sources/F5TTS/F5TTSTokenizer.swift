import Foundation

public struct F5TTSTokenizedText: Equatable, Sendable {
    public let symbols: [String]
    public let ids: [Int32]

    public init(symbols: [String], ids: [Int32]) {
        self.symbols = symbols
        self.ids = ids
    }
}

public struct F5TTSTokenizer: Sendable {
    public let vocab: [String: Int32]
    public let symbols: [String]
    public let pinyin: F5TTSPinyinConverter?

    public init(vocabURL: URL, pinyinLexiconURL: URL? = nil) throws {
        let text = try String(contentsOf: vocabURL, encoding: .utf8)
        var vocab: [String: Int32] = [:]
        var symbols: [String] = []
        let normalizedNewlines = text
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        var lines = normalizedNewlines
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        if let last = lines.last, last.isEmpty {
            lines.removeLast()
        }
        for (index, line) in lines.enumerated() {
            var symbol = line
            if index == 0, symbol.first == "\u{feff}" {
                symbol.removeFirst()
            }
            vocab[symbol] = Int32(index)
            symbols.append(symbol)
        }
        guard !symbols.isEmpty else {
            throw F5TTSError.unsupportedText("empty vocab.txt")
        }
        self.vocab = vocab
        self.symbols = symbols
        if let pinyinLexiconURL {
            self.pinyin = try F5TTSPinyinConverter(lexiconURL: pinyinLexiconURL)
        } else {
            self.pinyin = nil
        }
    }

    public init(vocab: [String: Int32], pinyin: F5TTSPinyinConverter? = nil) {
        self.vocab = vocab
        self.symbols = vocab.sorted { $0.value < $1.value }.map(\.key)
        self.pinyin = pinyin
    }

    public func tokenize(_ text: String) throws -> F5TTSTokenizedText {
        let normalized = Self.normalize(text)
        let out: [String]
        if Self.containsCJK(normalized) {
            guard let pinyin else {
                throw F5TTSError.unsupportedText(
                    "Mandarin and mixed CJK text require the bundle's pinyin lexicon (pinyin_lexicon.tsv); this bundle predates the pinyin frontend.")
            }
            out = pinyin.convert(normalized)
        } else {
            out = normalized.map(String.init)
        }
        let ids = out.map { vocab[$0] ?? 0 }
        return F5TTSTokenizedText(symbols: out, ids: ids)
    }

    public func encode(_ text: String) throws -> [Int32] {
        try tokenize(text).ids
    }

    public static func normalize(_ text: String) -> String {
        String(text.map { ch in
            switch ch {
            case ";": return ","
            case "“", "”": return "\""
            case "‘", "’": return "'"
            default: return ch
            }
        })
    }

    private static func containsCJK(_ text: String) -> Bool {
        text.unicodeScalars.contains { scalar in
            (0x3100...0x9fff).contains(Int(scalar.value))
        }
    }
}
