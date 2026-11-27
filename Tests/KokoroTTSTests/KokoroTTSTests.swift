import XCTest
@testable import KokoroTTS

final class KokoroTTSTests: XCTestCase {

    func testDefaultModelId() {
        XCTAssertEqual(KokoroTTSModel.defaultModelId, "aufklarer/Kokoro-82M-CoreML")
    }

    func testDefaultConfig() {
        let config = KokoroConfig.default
        XCTAssertEqual(config.sampleRate, 24000)
        XCTAssertEqual(config.maxPhonemeLength, 128)
        XCTAssertEqual(config.styleDim, 256)
        XCTAssertEqual(config.languages.count, 9)
        XCTAssertTrue(config.languages.contains("en"))
        XCTAssertTrue(config.languages.contains("de"))
    }

    func testConfigCodable() throws {
        let config = KokoroConfig.default
        let data = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(KokoroConfig.self, from: data)
        XCTAssertEqual(decoded.sampleRate, config.sampleRate)
        XCTAssertEqual(decoded.maxPhonemeLength, config.maxPhonemeLength)
        XCTAssertEqual(decoded.styleDim, config.styleDim)
    }

    // MARK: - Phonemizer Tests

    func testPhonemizerTokenize() {
        let vocab: [String: Int] = [
            "<pad>": 0, "<bos>": 1, "<eos>": 2,
            "h": 3, "e": 4, "l": 5, "o": 6, " ": 7,
        ]
        let phonemizer = KokoroPhonemizer(vocab: vocab)
        let ids = phonemizer.tokenize("hello")
        XCTAssertEqual(ids.first, 1)
        XCTAssertEqual(ids.last, 2)
        XCTAssertTrue(ids.count >= 3)
    }

    func testPhonemizerPadding() {
        let vocab: [String: Int] = ["<pad>": 0, "<bos>": 1, "<eos>": 2, "a": 3]
        let phonemizer = KokoroPhonemizer(vocab: vocab)
        let ids = phonemizer.tokenize("a")
        let padded = phonemizer.pad(ids, to: 10)
        XCTAssertEqual(padded.count, 10)
        XCTAssertEqual(padded[0], 1)
        XCTAssertEqual(padded.last(where: { $0 != 0 }), 2)
    }

    func testPhonemizerTruncation() {
        let vocab: [String: Int] = ["<pad>": 0, "<bos>": 1, "<eos>": 2, "a": 3]
        let phonemizer = KokoroPhonemizer(vocab: vocab)
        let longText = String(repeating: "a", count: 1000)
        let ids = phonemizer.tokenize(longText, maxLength: 20)
        XCTAssertEqual(ids.count, 20)
        XCTAssertEqual(ids.first, 1)
        XCTAssertEqual(ids.last, 2)
    }

    func testPhonemizerUnknownChars() {
        let vocab: [String: Int] = ["<pad>": 0, "<bos>": 1, "<eos>": 2, "a": 3]
        let phonemizer = KokoroPhonemizer(vocab: vocab)
        let ids = phonemizer.tokenize("axyz")
        XCTAssertEqual(ids, [1, 3, 2])
    }

    /// Regression: tokenization went Character by Character, so a vowel and a
    /// combining mark (the nasal tilde, U+0303) were one lookup that matched
    /// neither entry, and both were dropped. Kokoro looks up each code point.
    func testPhonemizerTokenizesCombiningMarksSeparately() {
        let vocab: [String: Int] = ["b": 3, "ɔ": 4, "\u{0303}": 5, "n": 6, "ˈ": 7]
        let phonemizer = KokoroPhonemizer(vocab: vocab)
        phonemizer.addPronunciations(["bonbon": "bɔ\u{0303}bɔ\u{0303}"])
        XCTAssertEqual(Array(phonemizer.tokenize("bonbon").dropFirst().dropLast()), [3, 4, 5, 3, 4, 5])
        XCTAssertTrue(phonemizer.tokenize("bon", language: "fr").contains(5),
            "French nasal vowels must keep their tilde")
    }
}

// MARK: - Chinese Phonemizer Tests

final class ChinesePhonemeizerTests: XCTestCase {

    // MARK: - Tone Extraction

    func testExtractTone1() {
        let (base, tone) = ChinesePhonemizer.extractTone("nǐ")
        XCTAssertEqual(base, "ni")
        XCTAssertEqual(tone, "3")
    }

    func testExtractTone2() {
        let (base, tone) = ChinesePhonemizer.extractTone("hǎo")
        XCTAssertEqual(base, "hao")
        XCTAssertEqual(tone, "3")
    }

    func testExtractTone4() {
        let (base, tone) = ChinesePhonemizer.extractTone("shì")
        XCTAssertEqual(base, "shi")
        XCTAssertEqual(tone, "4")
    }

    func testExtractToneNeutral() {
        let (base, tone) = ChinesePhonemizer.extractTone("de")
        XCTAssertEqual(base, "de")
        XCTAssertEqual(tone, "5")
    }

    // MARK: - Finals Normalization

    func testNormalizeIU() {
        XCTAssertEqual(ChinesePhonemizer.normalizeFinalsNotation("liu"), "liou")
    }

    func testNormalizeUI() {
        XCTAssertEqual(ChinesePhonemizer.normalizeFinalsNotation("gui"), "guei")
    }

    func testNormalizeUN() {
        XCTAssertEqual(ChinesePhonemizer.normalizeFinalsNotation("gun"), "guen")
    }

    func testNormalizeJQXU() {
        // After j/q/x, u → ü
        XCTAssertEqual(ChinesePhonemizer.normalizeFinalsNotation("ju"), "jü")
        XCTAssertEqual(ChinesePhonemizer.normalizeFinalsNotation("qu"), "qü")
        XCTAssertEqual(ChinesePhonemizer.normalizeFinalsNotation("xu"), "xü")
    }

    // MARK: - Syllable → IPA

    func testSyllableMA() {
        let ipa = ChinesePhonemizer.syllableToIPA("mā")
        XCTAssertTrue(ipa.contains("m"), "Should contain initial 'm': \(ipa)")
        XCTAssertTrue(ipa.contains("a"), "Should contain final 'a': \(ipa)")
    }

    func testSyllableSHI() {
        let ipa = ChinesePhonemizer.syllableToIPA("shì")
        XCTAssertTrue(ipa.hasPrefix("ʂ"), "Should start with retroflex: \(ipa)")
    }

    func testSyllableZHI() {
        // "zhi" → zh + retroflex i
        let ipa = ChinesePhonemizer.syllableToIPA("zhī")
        XCTAssertTrue(ipa.hasPrefix("ʈʂ"), "Should start with ʈʂ: \(ipa)")
    }

    // MARK: - Full Pipeline

    func testChinesePhonemizerProducesOutput() {
        let phonemizer = ChinesePhonemizer()
        let result = phonemizer.phonemize("你好")
        XCTAssertFalse(result.isEmpty, "Should produce phonemes for Chinese text")
    }

    func testChinesePhonemizerPunctuation() {
        let phonemizer = ChinesePhonemizer()
        let result = phonemizer.phonemize("你好，世界。")
        XCTAssertTrue(result.contains(","), "Should convert Chinese comma")
        XCTAssertTrue(result.contains("."), "Should convert Chinese period")
    }

    func testChinesePhonemizerMultipleSyllables() {
        let phonemizer = ChinesePhonemizer()
        let result = phonemizer.phonemize("你好世界")
        // Should have multiple phoneme segments
        XCTAssertGreaterThan(result.count, 4, "Should produce multi-syllable IPA: \(result)")
    }
}

// MARK: - Japanese Phonemizer Tests

final class JapanesePhonemeizerTests: XCTestCase {

    // MARK: - Katakana → Phonemes

    func testSingleKatakana() {
        XCTAssertEqual(JapanesePhonemizer.katakanaToPhonemes("ア"), "a")
        XCTAssertEqual(JapanesePhonemizer.katakanaToPhonemes("カ"), "ka")
        XCTAssertEqual(JapanesePhonemizer.katakanaToPhonemes("サ"), "sa")
    }

    func testDigraphKatakana() {
        XCTAssertEqual(JapanesePhonemizer.katakanaToPhonemes("シャ"), "sha")
        XCTAssertEqual(JapanesePhonemizer.katakanaToPhonemes("チャ"), "cha")
        XCTAssertEqual(JapanesePhonemizer.katakanaToPhonemes("キョ"), "kyo")
    }

    func testSpecialKatakana() {
        XCTAssertEqual(JapanesePhonemizer.katakanaToPhonemes("ッ"), "ʔ")
        XCTAssertEqual(JapanesePhonemizer.katakanaToPhonemes("ン"), "ɴ")
        XCTAssertEqual(JapanesePhonemizer.katakanaToPhonemes("ー"), "ː")
    }

    func testKatakanaSequence() {
        // コンニチハ → ko ɴ ni chi ha
        let result = JapanesePhonemizer.katakanaToPhonemes("コンニチハ")
        XCTAssertEqual(result, "koɴnichiha")
    }

    func testDigraphPriority() {
        // キャ should match as digraph "kya", not キ+ャ → "ki"+"ya"
        let result = JapanesePhonemizer.katakanaToPhonemes("キャ")
        XCTAssertEqual(result, "kya")
    }

    // MARK: - Romaji → Katakana

    func testRomajiToKatakana() {
        let result = JapanesePhonemizer.romajiToKatakana("toukyou")
        // Should produce katakana
        XCTAssertTrue(result.unicodeScalars.allSatisfy {
            (0x30A0...0x30FF).contains($0.value) || $0.value == 0x30FC
        }, "Should be katakana: \(result)")
    }

    // MARK: - Full Pipeline

    func testJapanesePhonemizerProducesOutput() {
        let phonemizer = JapanesePhonemizer()
        let result = phonemizer.phonemize("こんにちは")
        XCTAssertFalse(result.isEmpty, "Should produce phonemes for Japanese text")
    }

    func testJapanesePhonemizerPunctuation() {
        let phonemizer = JapanesePhonemizer()
        let result = phonemizer.phonemize("こんにちは。")
        XCTAssertTrue(result.contains("."), "Should convert Japanese period")
    }

    func testJapanesePhonemizerKanji() {
        let phonemizer = JapanesePhonemizer()
        let result = phonemizer.phonemize("東京")
        XCTAssertFalse(result.isEmpty, "Should phonemize kanji: \(result)")
        // Should produce something like "toukyou" phonemes
        XCTAssertGreaterThan(result.count, 3, "Should produce multi-mora IPA: \(result)")
    }
}


// MARK: - Hindi Phonemizer Tests

final class HindiPhonemizerTests: XCTestCase {

    func testRomanToIPA() {
        let ipa = HindiPhonemizer.romanToIPA("namastē")
        XCTAssertFalse(ipa.isEmpty)
        XCTAssertTrue(ipa.contains("n"), "Should contain 'n': \(ipa)")
    }

    func testHindiPhonemizerProducesOutput() {
        let phonemizer = HindiPhonemizer()
        let result = phonemizer.phonemize("नमस्ते")
        XCTAssertFalse(result.isEmpty, "Should produce phonemes for Hindi: \(result)")
    }

    func testHindiMultipleWords() {
        let phonemizer = HindiPhonemizer()
        let result = phonemizer.phonemize("नमस्ते दुनिया")
        XCTAssertGreaterThan(result.count, 5, "Should produce multi-word IPA: \(result)")
    }
}

// MARK: - Latin Phonemizer Tests (French, Spanish, Portuguese)

final class LatinPhonemizerTests: XCTestCase {

    func testFrenchBasic() {
        let phonemizer = LatinPhonemizer(language: .french)
        let result = phonemizer.phonemize("Bonjour le monde")
        XCTAssertFalse(result.isEmpty)
        XCTAssertTrue(result.contains("ʒ") || result.contains("ʁ"),
            "French should produce French IPA phonemes: \(result)")
    }

    func testSpanishBasic() {
        let phonemizer = LatinPhonemizer(language: .spanish)
        let result = phonemizer.phonemize("Hola mundo")
        XCTAssertFalse(result.isEmpty)
        XCTAssertTrue(result.contains("o") && result.contains("l"),
            "Spanish should produce correct phonemes: \(result)")
    }

    func testPortugueseBasic() {
        let phonemizer = LatinPhonemizer(language: .portuguese)
        let result = phonemizer.phonemize("Olá mundo")
        XCTAssertFalse(result.isEmpty)
        XCTAssertTrue(result.contains("a"),
            "Portuguese should produce correct phonemes: \(result)")
    }

    func testFrenchNasalVowels() {
        let phonemizer = LatinPhonemizer(language: .french)
        let result = phonemizer.phonemize("bonjour")
        XCTAssertTrue(result.contains("ɔ̃") || result.contains("ʒ"),
            "Should handle French nasals/consonants: \(result)")
    }

    func testSpanishContextRules() {
        let phonemizer = LatinPhonemizer(language: .spanish)
        // c before e → θ
        let result = phonemizer.phonemize("cena")
        XCTAssertTrue(result.contains("θ"), "c before e should be θ: \(result)")
    }

    func testPunctuation() {
        let phonemizer = LatinPhonemizer(language: .french)
        let result = phonemizer.phonemize("Bonjour, monde!")
        XCTAssertTrue(result.contains(","), "Should preserve comma")
        XCTAssertTrue(result.contains("!"), "Should preserve exclamation")
    }
}

// MARK: - Multilingual Tokenizer Routing Tests

final class MultilingualTokenizerTests: XCTestCase {

    private func makeVocab() -> [String: Int] {
        ["<pad>": 0, "<bos>": 1, "<eos>": 2,
         "a": 3, "b": 4, "d": 5, "e": 6, "f": 7, "h": 8,
         "i": 9, "k": 10, "l": 11, "m": 12, "n": 13, "o": 14,
         "p": 15, "r": 16, "s": 17, "t": 18, "u": 19, "w": 20,
         "ə": 21, "ɾ": 22, "ɡ": 23, "ʃ": 24, "ʒ": 25, "ʁ": 26,
         "ɲ": 27, "θ": 28, "x": 29, "ɛ": 30, "ɔ": 31, "j": 32,
         " ": 33, ",": 34, ".": 35]
    }

    func testChineseRouting() {
        let phonemizer = KokoroPhonemizer(vocab: makeVocab())
        let ids = phonemizer.tokenize("你好", language: "zh")
        XCTAssertEqual(ids.first, 1, "Should start with BOS")
        XCTAssertEqual(ids.last, 2, "Should end with EOS")
        XCTAssertGreaterThan(ids.count, 2, "Should produce tokens for Chinese")
    }

    func testJapaneseRouting() {
        let phonemizer = KokoroPhonemizer(vocab: makeVocab())
        let ids = phonemizer.tokenize("こんにちは", language: "ja")
        XCTAssertEqual(ids.first, 1)
        XCTAssertEqual(ids.last, 2)
        XCTAssertGreaterThan(ids.count, 2, "Should produce tokens for Japanese")
    }

    func testHindiRouting() {
        let phonemizer = KokoroPhonemizer(vocab: makeVocab())
        let ids = phonemizer.tokenize("नमस्ते", language: "hi")
        XCTAssertEqual(ids.first, 1)
        XCTAssertEqual(ids.last, 2)
        XCTAssertGreaterThan(ids.count, 2, "Should produce tokens for Hindi")
    }

    func testFrenchRouting() {
        let phonemizer = KokoroPhonemizer(vocab: makeVocab())
        let ids = phonemizer.tokenize("bonjour", language: "fr")
        XCTAssertEqual(ids.first, 1)
        XCTAssertEqual(ids.last, 2)
        XCTAssertGreaterThan(ids.count, 2, "Should produce tokens for French")
    }

    func testSpanishRouting() {
        let phonemizer = KokoroPhonemizer(vocab: makeVocab())
        let ids = phonemizer.tokenize("hola", language: "es")
        XCTAssertEqual(ids.first, 1)
        XCTAssertEqual(ids.last, 2)
        XCTAssertGreaterThan(ids.count, 2, "Should produce tokens for Spanish")
    }

    func testPortugueseRouting() {
        let phonemizer = KokoroPhonemizer(vocab: makeVocab())
        let ids = phonemizer.tokenize("olá", language: "pt")
        XCTAssertEqual(ids.first, 1)
        XCTAssertEqual(ids.last, 2)
        XCTAssertGreaterThan(ids.count, 2, "Should produce tokens for Portuguese")
    }

    func testEnglishRoutingDefault() {
        let phonemizer = KokoroPhonemizer(vocab: makeVocab())
        let idsDefault = phonemizer.tokenize("hello")
        let idsEn = phonemizer.tokenize("hello", language: "en")
        XCTAssertEqual(idsDefault, idsEn, "Default should route to English")
    }
}

// MARK: - Custom Pronunciation Tests

/// `addPronunciations` writes into the gold dictionary, which `resolveWord`
/// consults before the silver dict, suffix stemming, and the neural `bartG2P`
/// fallback. These tests pin that ordering without loading the CoreML G2P
/// models: with no models loaded `bartG2P` returns nil, so resolution falls
/// through to the raw-letter last resort, which stands in for it here.
/// `E2EKokoroTests.testCustomPronunciationBeatsNeuralG2P` covers the real
/// neural fallback with downloaded weights.
final class KokoroPronunciationTests: XCTestCase {

    /// "tah-tee-AH-nuh" — a name absent from both shipped dictionaries.
    private let tatianaIPA = "tɑtiˈɑnə"

    /// Covers the IPA symbols used below plus the plain letters the raw-letter
    /// fallback emits, so tokenization round-trips either outcome.
    private func makeVocab() -> [String: Int] {
        ["<pad>": 0, "<bos>": 1, "<eos>": 2,
         "a": 3, "e": 4, "i": 5, "n": 6, "t": 7,
         "ɑ": 8, "ə": 9, "æ": 10, "ˈ": 11, " ": 12]
    }

    private func makePhonemizer() -> KokoroPhonemizer {
        KokoroPhonemizer(vocab: makeVocab())
    }

    /// Writes a minimal `us_gold.json` to a temp directory and loads it.
    private func loadGoldDictionary(
        _ entries: [String: String], into phonemizer: KokoroPhonemizer
    ) throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }

        let data = try JSONSerialization.data(withJSONObject: entries)
        try data.write(to: dir.appendingPathComponent("us_gold.json"))
        try phonemizer.loadDictionaries(from: dir)
    }

    /// The motivating case: a word in neither dictionary reaches the G2P
    /// fallback, and an injected entry pre-empts it.
    func testInjectedEntryPreemptsFallback() {
        let phonemizer = makePhonemizer()
        XCTAssertEqual(phonemizer.textToPhonemes("Tatiana"), "tatiana",
            "With no dictionaries and no G2P models, resolution falls through to raw letters")

        phonemizer.addPronunciations(["tatiana": tatianaIPA])

        XCTAssertEqual(phonemizer.textToPhonemes("Tatiana"), tatianaIPA,
            "Injected entry must be resolved instead of the fallback")
    }

    /// An injected entry outranks a shipped gold entry, so a caller can correct
    /// a dictionary pronunciation rather than only fill a gap.
    func testInjectedEntryOverridesGoldDictionary() throws {
        let phonemizer = makePhonemizer()
        try loadGoldDictionary(["tatiana": "tætiænə"], into: phonemizer)
        XCTAssertEqual(phonemizer.textToPhonemes("Tatiana"), "tætiænə")

        phonemizer.addPronunciations(["tatiana": tatianaIPA])

        XCTAssertEqual(phonemizer.textToPhonemes("Tatiana"), tatianaIPA,
            "Injected entry must outrank the shipped gold entry")
    }

    /// Entries are stored lowercased and `resolveWord` lowercases the word, so
    /// capitalization on either side is irrelevant — proper nouns arrive
    /// capitalized in real text.
    func testLookupIsCaseInsensitive() {
        let phonemizer = makePhonemizer()
        phonemizer.addPronunciations(["TaTiAnA": tatianaIPA])

        for spelling in ["tatiana", "Tatiana", "TATIANA"] {
            XCTAssertEqual(phonemizer.textToPhonemes(spelling), tatianaIPA,
                "\"\(spelling)\" should resolve to the injected pronunciation")
        }
    }

    /// A later call replaces an earlier entry for the same word, and leaves the
    /// other entries alone.
    func testLaterCallReplacesEarlierEntry() {
        let phonemizer = makePhonemizer()
        phonemizer.addPronunciations(["tatiana": "tætiænə", "tania": "ˈtɑniə"])
        phonemizer.addPronunciations(["tatiana": tatianaIPA])

        XCTAssertEqual(phonemizer.textToPhonemes("tatiana"), tatianaIPA)
        XCTAssertEqual(phonemizer.textToPhonemes("tania"), "ˈtɑniə",
            "Unrelated entries must survive a later call")
    }

    /// Behaviour end to end: the injected IPA reaches the token ids the model
    /// is fed, wrapped in BOS/EOS.
    func testInjectedPronunciationSurvivesTokenization() {
        let phonemizer = makePhonemizer()
        let vocab = makeVocab()
        let before = phonemizer.tokenize("Tatiana")

        phonemizer.addPronunciations(["tatiana": tatianaIPA])
        let after = phonemizer.tokenize("Tatiana")

        let expected = [phonemizer.bosId] + tatianaIPA.compactMap { vocab[String($0)] }
            + [phonemizer.eosId]
        XCTAssertEqual(after, expected)
        XCTAssertNotEqual(after, before, "Token ids must change once a pronunciation is injected")
    }

    /// Documented constraint: text is split on whitespace and punctuation before
    /// resolution, so a key spanning more than one token is never looked up.
    func testMultiTokenKeysAreNeverMatched() {
        let phonemizer = makePhonemizer()
        phonemizer.addPronunciations([
            "ana maria": "ˈɑnəməˈɹiə",
            "ana-maria": "ˈɑnəməˈɹiə",
        ])

        XCTAssertEqual(phonemizer.textToPhonemes("Ana Maria"), "ana maria",
            "A key containing a space cannot be reached")
        XCTAssertEqual(phonemizer.textToPhonemes("Ana-Maria"), "ana-maria",
            "A key containing a hyphen cannot be reached")
    }

    /// Documented constraint: `specialCase` resolves before the dictionaries, so
    /// its function words are not overridable.
    func testSpecialCaseWordsAreNotOverridden() {
        let phonemizer = makePhonemizer()
        phonemizer.addPronunciations(["the": "ðiː", "of": "ɒv"])

        XCTAssertEqual(phonemizer.textToPhonemes("the"), "ðə")
        XCTAssertEqual(phonemizer.textToPhonemes("of"), "ʌv")
    }

    /// Documented constraint: `loadDictionaries` assigns the gold dictionary
    /// rather than merging into it, so injections must come afterwards.
    func testLoadingDictionariesDiscardsEarlierInjections() throws {
        let phonemizer = makePhonemizer()
        phonemizer.addPronunciations(["tatiana": tatianaIPA])
        XCTAssertEqual(phonemizer.textToPhonemes("Tatiana"), tatianaIPA)

        try loadGoldDictionary(["hello": "həlˈoʊ"], into: phonemizer)

        XCTAssertEqual(phonemizer.textToPhonemes("Tatiana"), "tatiana",
            "Entries injected before loadDictionaries do not survive it")
    }

    /// IPA symbols outside the model vocabulary are dropped at tokenization
    /// (documented on the API), not rejected — the rest still tokenizes.
    func testSymbolsOutsideVocabularyAreDropped() {
        let phonemizer = makePhonemizer()
        phonemizer.addPronunciations(["tatiana": "tɑ↗ti"])

        XCTAssertEqual(phonemizer.tokenize("Tatiana"),
            [phonemizer.bosId, 7, 8, 7, 5, phonemizer.eosId])
    }

    /// Documented constraint: only the English path consults the dictionaries.
    /// `tokenize` routes every other language to a dedicated phonemizer, so an
    /// injected entry is a no-op there.
    func testInjectionAppliesToEnglishOnly() {
        let others = ["fr", "es", "it", "pt", "hi", "ja", "zh"]
        let phonemizer = makePhonemizer()
        let baseline = others.map { phonemizer.tokenize("Tatiana", language: $0) }

        phonemizer.addPronunciations(["tatiana": tatianaIPA])

        let vocab = makeVocab()
        XCTAssertEqual(phonemizer.tokenize("Tatiana", language: "en"),
            [phonemizer.bosId] + tatianaIPA.compactMap { vocab[String($0)] } + [phonemizer.eosId],
            "English must resolve the injected entry")

        for (language, before) in zip(others, baseline) {
            XCTAssertEqual(phonemizer.tokenize("Tatiana", language: language), before,
                "\"\(language)\" must be unaffected by the injected entry")
        }
    }
}
