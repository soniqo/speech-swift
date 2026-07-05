import Hub
import XCTest

@testable import ChatterboxTTS

/// Numeric gate for the Swift `MTLTokenizer`: token ids must match the
/// reference golden exactly for core languages, and every runtime-supported
/// language frontend must encode without unknown-token fallback.
final class MTLTokenizerTests: XCTestCase {
    private func loadTokenizer() async throws -> MTLTokenizer {
        let folder = try await HubApi().snapshot(
            from: "aufklarer/Chatterbox-Multilingual-MLX-fp16",
            matching: ["tokenizer.json", "Cangjie5_TC.json"]
        )
        return try MTLTokenizer(modelFolder: folder)
    }

    func testEncodeMatchesReferenceGolden() async throws {
        let tok = try await loadTokenizer()

        // Golden ids from the reference tokenizer over aufklarer/…-fp16.
        XCTAssertEqual(
            tok.encode("Hello there, friend.", languageId: "en"),
            [708, 62, 84, 28, 2, 172, 7, 2, 19, 101, 204, 9])
        XCTAssertEqual(
            tok.encode("مرحبا بك", languageId: "ar"),
            [721, 1491, 1471, 1467, 1462, 1456, 2, 1462, 1489])
        XCTAssertEqual(
            tok.encode("नमस्ते दोस्त", languageId: "hi"),
            [722, 1706, 1712, 1720, 1740, 1702, 1734, 2, 1704, 1738, 1720, 1740, 1702])
    }

    func testLanguageTokenPrepended() async throws {
        let tok = try await loadTokenizer()
        // The first id is the [lang] control token: [en]=708, [ar]=721, [hi]=722.
        XCTAssertEqual(tok.encode("a", languageId: "en").first, 708)
        XCTAssertEqual(tok.encode("a", languageId: "ar").first, 721)
        XCTAssertEqual(tok.encode("a", languageId: "hi").first, 722)
    }

    func testLanguageSetsSeparateUpstreamFromRuntimeReadyFrontends() async throws {
        let expected: Set<String> = [
            "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
            "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
            "sw", "tr", "zh",
        ]
        XCTAssertEqual(MTLTokenizer.upstreamLanguages, expected)
        XCTAssertEqual(MTLTokenizer.upstreamLanguages.count, 23)
        XCTAssertFalse(MTLTokenizer.supportedLanguages.contains("he"))
        XCTAssertTrue(MTLTokenizer.supportedLanguages.contains("ja"))
        XCTAssertTrue(MTLTokenizer.supportedLanguages.contains("zh"))

        let tok = try await loadTokenizer()
        for language in MTLTokenizer.supportedLanguages.sorted() {
            let encoded = tok.encode("hello", languageId: language)
            XCTAssertFalse(encoded.isEmpty, "language \(language) should encode")
            XCTAssertEqual(
                tok.decode([encoded[0]]),
                "[\(language)]",
                "language \(language) should prepend its control token")
        }
    }

    func testChineseUsesCangjieFrontendWhenMapIsAvailable() async throws {
        let tok = try await loadTokenizer()
        let encoded = tok.encode("今天", languageId: "zh")

        XCTAssertEqual(encoded.first, 725)
        XCTAssertEqual(
            tok.decode(Array(encoded.dropFirst())),
            "[cj_o][cj_i][cj_n][cj_.][cj_m][cj_k][cj_.]")
        XCTAssertFalse(encoded.contains(1), "Cangjie-mapped Chinese should not fall back to [UNK]")
    }

    func testChineseFrontendSegmentsWordsBeforeCangjie() async throws {
        let tok = try await loadTokenizer()
        let encoded = tok.encode("今天我们测试中文语音", languageId: "zh")
        let decoded = tok.decode(Array(encoded.dropFirst()))

        XCTAssertTrue(decoded.contains(" "), "Chinese frontend should preserve word boundaries")
        XCTAssertTrue(decoded.contains("[cj_"), "Chinese text should be converted to Cangjie tokens")
        XCTAssertFalse(encoded.contains(1), "Segmented Chinese should not fall back to [UNK]")
    }

    func testJapaneseUsesHiraganaFrontendForKanji() async throws {
        let tok = try await loadTokenizer()
        let encoded = tok.encode("今日はカタカナです。", languageId: "ja")

        XCTAssertEqual(encoded.first, 723)
        XCTAssertTrue(
            tok.decode(Array(encoded.dropFirst())).contains("きょうはカタカナです"),
            "Kanji should be normalized to hiragana while katakana stays unchanged"
        )
        XCTAssertFalse(encoded.contains(1), "Japanese frontend should not fall back to [UNK]")
    }

    func testBenchmarkSentencesDoNotEmitUnknownTokens() async throws {
        let tok = try await loadTokenizer()
        let samples = [
            "ar": "اليوم نختبر ما إذا كان الكلام واضحا وسهل الفهم.",
            "da": "I dag tester vi, om talen er klar og let at forstå.",
            "de": "Heute testen wir, ob die Sprache klar und leicht zu verstehen ist.",
            "el": "Σήμερα ελέγχουμε αν η ομιλία είναι καθαρή και εύκολη στην κατανόηση.",
            "en": "Today we test whether the speech is clear and easy to understand.",
            "es": "Hoy probamos si el habla es clara y fácil de entender.",
            "fi": "Tänään testaamme, onko puhe selkeää ja helppoa ymmärtää.",
            "fr": "Aujourd'hui, nous testons si la parole est claire et facile à comprendre.",
            "hi": "आज हम जाँचते हैं कि आवाज़ साफ़ और समझने में आसान है या नहीं।",
            "it": "Oggi testiamo se il parlato è chiaro e facile da capire.",
            "ja": "今日は音声が明瞭で理解しやすいかを確認します。",
            "ko": "오늘 우리는 음성이 명확하고 이해하기 쉬운지 테스트합니다.",
            "ms": "Hari ini kami menguji sama ada pertuturan jelas dan mudah difahami.",
            "nl": "Vandaag testen we of de spraak duidelijk en gemakkelijk te begrijpen is.",
            "no": "I dag tester vi om talen er klar og lett å forstå.",
            "pl": "Dzisiaj sprawdzamy, czy mowa jest wyraźna i łatwa do zrozumienia.",
            "pt": "Hoje testamos se a fala é clara e fácil de entender.",
            "ru": "Сегодня мы проверяем, является ли речь четкой и легкой для понимания.",
            "sv": "I dag testar vi om talet är tydligt och lätt att förstå.",
            "sw": "Leo tunajaribu kama hotuba iko wazi na ni rahisi kuelewa.",
            "tr": "Bugün konuşmanın net ve kolay anlaşılır olup olmadığını test ediyoruz.",
            "zh": "今天我们测试中文语音是否清楚且容易理解。",
        ]

        XCTAssertEqual(Set(samples.keys), MTLTokenizer.supportedLanguages)
        for (language, text) in samples {
            let encoded = tok.encode(text, languageId: language)
            XCTAssertFalse(encoded.contains(1), "\(language) should not emit [UNK]")
        }
    }
}
