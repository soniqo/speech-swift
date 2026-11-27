import XCTest
@testable import KokoroTTS

/// German front-end: normalization, rule-based G2P, dictionary fallbacks and
/// routing. Expected strings follow espeak-ng 1.52 German as post-processed by
/// misaki (the kikiri-tts training front-end) and mirror speech-core's
/// `test_kokoro_german.cpp`, so the two ports stay in lockstep.
final class GermanPhonemizerTests: XCTestCase {

    private let g = GermanPhonemizer()

    func testNormalizeNumbers() {
        XCTAssertEqual(g.normalize("Ich habe 21 Katzen."), "Ich habe einundzwanzig Katzen.")
        XCTAssertEqual(g.normalize("1 Hund"), "eins Hund")
        XCTAssertEqual(g.normalize("3,14 ist Pi"), "drei Komma eins vier ist Pi")
        XCTAssertEqual(g.normalize("1.234.567 Euro"),
                       "eine Million zweihundertvierunddreißigtausendfünfhundertsiebenundsechzig Euro")
        XCTAssertEqual(g.normalize("um 18:20 Uhr"), "um achtzehn Uhr zwanzig")
        XCTAssertEqual(g.normalize("gegen 9:05"), "gegen neun Uhr fünf")
        XCTAssertEqual(g.normalize("am 14.05.2026"), "am vierzehnten Mai zweitausendsechsundzwanzig")
        XCTAssertEqual(g.normalize("Heute ist der 3.10.1990."), "Heute ist der dritte Oktober neunzehnhundertneunzig.")
        XCTAssertEqual(g.normalize("die 2. Schiene"), "die zweite Schiene")
        XCTAssertEqual(g.normalize("am 1. Mai"), "am ersten Mai")
        XCTAssertEqual(g.normalize("seit 1999"), "seit neunzehnhundertneunundneunzig")
        XCTAssertEqual(g.normalize("12,80 €"), "zwölf Euro und achtzig Cent")
        XCTAssertEqual(g.normalize("$ 5"), "fünf Dollar")
        XCTAssertEqual(g.normalize("25 % mehr"), "fünfundzwanzig Prozent mehr")
        XCTAssertEqual(g.normalize("1,5 kg und 500 g"), "eins Komma fünf Kilogramm und fünfhundert Gramm")
        XCTAssertEqual(g.normalize("Ruf 0170 123456 an"), "Ruf null eins sieben null eins zwei drei vier fünf sechs an")
        XCTAssertEqual(g.normalize("Prof. Klein, z.B. heute"), "Professor Klein, zum Beispiel heute")
        XCTAssertEqual(g.normalize("Nr. 7"), "Nummer sieben")
        XCTAssertEqual(g.normalize("„Hallo“ sagte er"), "\"Hallo\" sagte er")
        XCTAssertEqual(g.normalize("Hallo Welt"), "Hallo Welt")
    }

    func testRuleBasedG2P() {
        let cases: [(String, String)] = [
            ("Tag", "tˈɑːk"), ("Wasser", "vˈasɜ"), ("Nase", "nˈɑːzə"), ("Fenster", "fˈɛnstɜ"),
            ("Küche", "kˈyçə"), ("Kuchen", "kˈuːxən"), ("Mädchen", "mˈɛːtçən"), ("Vögel", "fˈøːɡəl"),
            ("Schule", "ʃˈuːlə"), ("Straße", "ʃtɾˈɑːsə"), ("Fahrrad", "fˈɑːrɑːt"),
            ("Zeit", "ʦˈIt"), ("Haus", "hˈWs"), ("Leute", "lˈɔøtə"), ("Häuser", "hˈɔøzɜ"),
            ("sechs", "zˈɛks"), ("Nacht", "nˈaxt"), ("Licht", "lˈɪçt"), ("Chemie", "çeːmˈiː"),
            ("Quelle", "kvˈɛlə"), ("Pfütze", "pfˈyʦə"), ("Sport", "ʃpˈɔɾt"), ("Stein", "ʃtˈIn"),
            ("Vater", "fˈɑːtɜ"), ("Klavier", "klavˈiːɾ"), ("Freund", "frˈɔønt"), ("Brot", "bɾˈoːt"),
            ("Burg", "bˈʊɐk"), ("Kurs", "kˈuːɾs"), ("deutsch", "dˈɔøtʃ"), ("Tschüss", "ʧˈys"),
            ("lebend", "lˈeːbənt"), ("Kindern", "kˈɪndɜn"), ("redete", "rˈeːdətə"),
            ("verstehen", "fɛɾʃtˈeːən"), ("Gebäude", "ɡəbˈɔødə"), ("anfangen", "ˈanfˌaŋən"),
            ("Universität", "ˌuːniːveːɾziːtˈɛːt"), ("Information", "ˌɪnfɔɾmaʦjˈoːn"),
            ("studieren", "ʃtuːdˈiːrən"), ("Wissenschaft", "vˈɪsənʃˌaft"), ("Lehrerin", "lˈeːrərˌɪn"),
            ("Beratung", "bərˈɑːtʊŋ"), ("König", "kˈøːnɪç"), ("Freundlichkeit", "frˈɔøntlɪçkˌIt"),
            ("ausgewählt", "ˈWsɡəvˌɛːlt"), ("bevorstehen", "bəfˈoːɾʃtˌeːən"), ("herunter", "hɛɾˈʊntɜ"),
            ("TAG", "tˈɑːk"),
        ]
        for (word, ipa) in cases {
            XCTAssertEqual(g.g2pWord(word), ipa, word)
        }
        XCTAssertEqual(g.g2pWord(""), "")
    }

    func testDictionaryFallbacks() {
        let dict: [String: String] = [
            "kind": "kˈɪnt", "kinder": "kˈɪndɜ", "tag": "tˈɑːk", "haus": "hˈWs",
            "sagen": "zˈɑːɡən", "kritisieren": "kɾˌiːtiːzˈiːrən", "willig": "vˈɪlɪç",
            "eisen": "ˈIzən", "bahn": "bˈɑːn", "hof": "hˈoːf", "kleider": "klˈIdɜ",
            "ordnung": "ˈɔɾdnʊŋ", "weg": "vɛk", "geringer": "ɡərˈɪŋɜ", "cafe": "kafˈeː",
            "leben": "lˈeːbən",
        ]
        XCTAssertEqual(g.phonemizeWord("Kind", dictionary: dict), "kˈɪnt")
        XCTAssertEqual(g.phonemizeWord("Kindern", dictionary: dict), "kˈɪndɜn")
        XCTAssertEqual(g.phonemizeWord("Tage", dictionary: dict), "tˈɑːɡə")
        XCTAssertEqual(g.phonemizeWord("Tages", dictionary: dict), "tˈɑːɡəs")
        XCTAssertEqual(g.phonemizeWord("sagt", dictionary: dict), "zˈɑːkt")
        XCTAssertEqual(g.phonemizeWord("lebte", dictionary: dict), "lˈeːptə")
        XCTAssertEqual(g.phonemizeWord("kritisierst", dictionary: dict), "kɾˌiːtiːzˈiːɾst")
        XCTAssertEqual(g.phonemizeWord("willigen", dictionary: dict), "vˈɪlɪɡən")
        XCTAssertEqual(g.phonemizeWord("geringere", dictionary: dict), "ɡərˈɪŋərə")
        XCTAssertEqual(g.phonemizeWord("Eisenbahn", dictionary: dict), "ˈIzənbˌɑːn")
        XCTAssertEqual(g.phonemizeWord("Bahnhof", dictionary: dict), "bˈɑːnhoːf")
        XCTAssertEqual(g.phonemizeWord("Kleiderordnung", dictionary: dict), "klˈIdərˌɔɾdnʊŋ")
        XCTAssertEqual(g.phonemizeWord("Kindeshaus", dictionary: dict), "kˈɪndəshˌWs")
        XCTAssertEqual(g.phonemizeWord("Zwerge", dictionary: dict), "ʦvˈɛɾɡə")
        XCTAssertEqual(g.phonemizeWord("Weg", dictionary: dict), "vˈeːk")
        XCTAssertEqual(g.phonemizeWord("weg", dictionary: dict), "vɛk")
        XCTAssertEqual(g.phonemizeWord("Café", dictionary: dict), "kafˈeː")
    }

    func testPhonemizeText() {
        let dict: [String: String] = [
            "guten": "ɡˈuːtən", "tag": "tˈɑːk", "wie": "viː", "geht": "ɡˈeːt", "es": "ɛs",
            "ihnen": "iːnən", "mein": "mIn", "name": "nˈɑːmə", "ist": "ɪst", "um": "ʊm",
            "uhr": "ˈuːɾ", "acht": "ˈaxt",
        ]
        XCTAssertEqual(g.phonemize("Guten Tag, wie geht es Ihnen? Mein Name ist Martin.", dictionary: dict),
                       "ɡˈuːtən tˈɑːk, viː ɡˈeːt ɛs iːnən? mIn nˈɑːmə ɪst mˈaɾtɪn.")
        XCTAssertEqual(g.phonemize("um 8 Uhr", dictionary: dict), "ʊm ˈaxt ˈuːɾ")
        XCTAssertEqual(g.phonemize("E-Mail", dictionary: dict), "ˈeːmˌIl")
        XCTAssertEqual(g.phonemize("", dictionary: dict), "")
    }

    func testTokenizerRoutesGerman() {
        let vocab: [String: Int] = ["$": 0, "ʃ": 1, "t": 2, "ɾ": 3, "ˈ": 4, "ɑ": 5, "ː": 6, "s": 7, "ə": 8, " ": 16]
        let phonemizer = KokoroPhonemizer(vocab: vocab)
        let ids = phonemizer.tokenize("Straße", language: "de")
        // ʃ t ɾ ˈ ɑ ː s ə between BOS/EOS (the pad id, 0)
        XCTAssertEqual(ids, [0, 1, 2, 3, 4, 5, 6, 7, 8, 0])
        XCTAssertEqual(phonemizer.tokenize("Straße", language: "german"), ids)
    }

    /// Optional parity check against a reference TSV (`word<TAB>g2p<TAB>phonemized`)
    /// produced by speech-core's C++ driver; skipped when the files are absent.
    func testParityWithSpeechCore() throws {
        let env = ProcessInfo.processInfo.environment
        guard let tsv = env["KOKORO_GERMAN_PARITY_TSV"], let dictPath = env["KOKORO_GERMAN_DICT"] else {
            throw XCTSkip("set KOKORO_GERMAN_PARITY_TSV and KOKORO_GERMAN_DICT to run the speech-core parity check")
        }
        let dict = try JSONDecoder().decode([String: String].self, from: Data(contentsOf: URL(fileURLWithPath: dictPath)))
        let lines = try String(contentsOfFile: tsv, encoding: .utf8).split(separator: "\n")
        var mismatches = 0
        for line in lines {
            let parts = line.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
            guard parts.count == 3 else { continue }
            if g.g2pWord(parts[0]) != parts[1] || g.phonemizeWord(parts[0], dictionary: dict) != parts[2] {
                mismatches += 1
            }
        }
        XCTAssertEqual(mismatches, 0, "\(mismatches) of \(lines.count) words differ from speech-core")
    }
}
