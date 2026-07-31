import XCTest
import Foundation
@testable import Qwen3Chat

/// Exact token-id parity between the rewritten Gemma 4 BPE merge loop and the implementation it
/// replaced, over the real 262,144-token vocabulary and its 514,906 merge rules.
///
/// The unit suite proves the algorithm on a vocabulary small enough to reason about; this proves it
/// on the one that ships, where a rule can span a space, a merge can outrank a longer word, and the
/// text is interleaved Russian, English and technical terms. Divergence here would change what the
/// model reads with nothing in the pipeline reporting it, so the assertion is identity.
final class E2EGemma4TokenizerParityTests: XCTestCase {
    private static let modelDir: URL = {
        if let p = ProcessInfo.processInfo.environment["GEMMA4_MODEL_DIR"] {
            return URL(fileURLWithPath: p)
        }
        return URL(fileURLWithPath:
            "/Users/ivan/repos/runner-speech-models/speech-models/out/mlx/gemma-4-E2B-it-MLX-4bit")
    }()

    /// Sampled from one real 393-row meeting recording (~55k characters of Russian with English and
    /// technical terms): the rows carrying digits, Latin-in-Cyrillic and the recording's one CJK
    /// character, its four shortest and its longest utterance, and an even spread of the rest.
    /// Point `GEMMA4_PARITY_CORPUS` at a transcript export to compare against all of it instead.
    static let transcriptRows: [String] = [
        "Ну, слушай, а вот про баварский диалекта. В каком кейсе он не мог понять, кто что-то говорит на баварском语?",
        "А. Tom.",
        "Такое R&amp;D.",
        "Vision Lab называется.",
        "more, tipa. Ну типа да, да, в том числе.",
        "Good symbol. Почему? А ты работал в ходаже?",
        "Ну а сколько ты за него платишь вот Plout Node?",
        "Тот же, тот же сегмент B2B. Вот тебе я прям тебе стратегию расписал, что делать надо.",
        "Если я звоню и мне удалось зацепить с первых 15 секунд, когда меня готовы слышать, да, слушать?",
        "Ты рискуешь уйти вот этим, точнее не уйти, а обмануть себя тем, что у тебя будет А, такой бесконечный R&amp;D вы будете все вместе его делать. Вам будет всем это классно, интересно, по кайфу. А, но это будет такой кружок любителей шахмат, да? Вы будете обсуждать ход Е1 на Е12, блядь, и как бы...",
        "no, on Ну он говорит около 40 тысяч человек клиентская база окей ну всех да то есть в принципе вот да то есть вот и и он тоже то есть и он там начинает там вот я вот это начал делать то это Я вот Лого бизнес, мы там разобрались, нет инвестиций, там нет трекшн, есть одни эти как их там. Интервью, интервью очень интересные, да, много вещающих, много пересечений с тем, что я делаю.",
        "in",
        "um",
        "ah",
        "на",
        "может быть, они исчезнут, Если я да расскажу. То есть вот я сейчас делаю вот такую платформу, которая, то есть я по Клауду решению, Слишком много денег нужно на дотации, чтобы зайти в рынок. Соответственно, у меня, то есть локальный инференс, кажется, что решает эту проблему? Это что-то за что, на чем у меня нет колоссальных костов. То есть и соответственно здесь вот вот этот вот запись, транскрипция базовой intelligence, там там deep resarch over Your meeting's date, да, ну вот, вот, вот этих штук. А, то есть плюс, а, сейчас делают кейс Сказочки типа базовый кейс, это Если это типа мы подгружаем адженду митинга, которую ты в Клауд, в Клауде или в чадж пяти себе сделал, то есть ты подготовилась к митингу, сделала какую-то адженду, ты ее туда подгружаешь, и эта штука, она просто бежит и тебе помогает вспоминать то, о чем тебе говорить. Вот, а это все вот так вот такого рода интеллигенсность, эта платформа очень Это платформа очень легко будет расширяемая до кейса агента и поговорить голосом. То есть пока просто вся интеракция агентская такая, да? То есть она будет через текст, вот это такой Ask My Transcripts это все локально, да? Да, то локально, все красиво. А, то сейчас я вот кейс там типа отдела продаж, а, там отдел продаж, они начинают какую-то новую штуку продавать. Мы начинаем и вот. Просто каждому продажнику централизовано в промт по адженте начинаем добавлять там де- десяти, предложить вот это, да? И все, то Предложить вот это, да? И все, то есть и у нас у продажников уже им не нужно думать, у них там появляется какая-то структурная эээ Гайденс, сори про мой рунглиш, вот. И которая начинает их вести. То есть, грубо говоря, Я хочу начать тупо с руководителя отделов продаж. И то есть, ну это будет готово через пару недель на уровне MVP. Мы начнем это все как-то продвигать, и то есть я вот у себя на сайте, у меня куча народу есть, это все Это все в принципе кто-то чем-то пользоваться, собирать инфу. И тут я хочу из этого сделать платформу, на которой будет сейвс коучинг. Это как раз решение. Плюс другие люди, я поговорил, то есть вот там немцы есть, они... Ах, хотят делать коучинг, э, для... Эмигрантов, которые приехали сюда, потеряли работу и теперь их отправили на аусбилдинг пере- переучиваться. А, вот, и там есть куча всяких информационных систем, и кажется, что достаточно тоже большой рынок, ну то есть, ну человек переучивается быть конструктором, ну этим рабочим настройки, да. А, и вот есть ребята, которые пообщались и говорят, что, вообще говоря, очень много там людей, кот- которых... Готовы взять на работу, им нужно сдать аузбильдунг там. Пройти какой-то тест, там какие-то там нормы сдать, что-то еще. Я, я с трудом понимаю, там что, чтоб конкретно там Крытно там они Делают. Но там большая сложность в том, чтобы вот их кочить. То есть там можно навешать несколько разных кейсов. И кажется, что в принципе там достаточно много всего. Но для этого должна быть платформа. И вот в этой истории там инвесторы, то есть это уже дело вторичное просто. А там всё без них заведётся.",
        "No.",
        "April",
        "Mm-hmm.",
        "Вопрос, а",
        "Dawai. Dawai.",
        "Now, i said, ah,",
        "Ну и тоже другое, ну и другое.",
        "Ну вот как-то так. То есть как бы АСРМок?",
        "будет Сегодня будет очередной этот взрыв, как и хит-вейф",
        "Ну в смысле, что я читал что ли? Ну мои знакомые, те кто готовы мне помогать.",
        "Что значит ты, что подразумеваешь, чтоб то, что ты себя не обманул? Что значит готовы комититься?",
        "Ты говоришь, кость. А, у меня есть нечто. Что я разрабатываю, да? Какая-то там начинка, там не важно. значит а У меня Полигон?",
        "Ah. То, что ты говоришь и sales, sale case это все, это все типа компоненты платформы. Они просто на платформу сажаются и если это все работает, будет там несколько серия много, да?",
        "What's in your mind? What's your chance you will? Я вот назад иду. What, Yule? Yoon? Май, апрель, это четыре месяца назад. Четыре месяца назад у меня что-то случилось. Я сидел, вообще не понимал, что с этим делать. Вот вообще просто это какая хуйня какая. Ну что-то работает. Я вот, знаешь, я сейчас, я...",
        "Значит одна из ну опять же там у меня на что я там там много их но я там понятно кто этот шивы все пропанил как бы да значит на что мне зацепил глаз это А у них есть Сирия, это чистая плоская история, правильно? И вот они сейчас разрабатывают, они планируют в конце двадцать седьмого года, но тоже не факт, но они сейчас разрабатывают сбор. Контекста каждого пользователя, чтобы это был сбор через Siri. Ну, то есть Примерно то же, что и я хочу. Да? Но только у Apple есть, ну помимо миллиардов, там триллионов, у Apple еще есть.",
    ]

    /// Reading the whole corpus needs the reference implementation, which is quadratic, so a debug
    /// run stops where that still costs seconds. `GEMMA4_PARITY_FULL=1` compares all of it and times
    /// the reference at every size, which is how the numbers in the change description were taken.
    private static var full: Bool { ProcessInfo.processInfo.environment["GEMMA4_PARITY_FULL"] == "1" }
    private static var referenceCeiling: Int {
        if full { return .max }
#if DEBUG
        return 8_000
#else
        return .max
#endif
    }

    private static func corpus() -> [String] {
        guard let path = ProcessInfo.processInfo.environment["GEMMA4_PARITY_CORPUS"],
              let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let rows = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            return transcriptRows
        }
        return rows.compactMap { $0["text"] as? String }.filter { !$0.isEmpty }
    }

    /// Lowest id of each set of vocabulary entries Swift cannot tell apart.
    ///
    /// Swift compares and hashes strings by canonical equivalence, so the 434 groups of canonically
    /// equal tokens in this vocabulary — `;` U+003B and U+037E GREEK QUESTION MARK among them, 868
    /// ids in all — collapse to one entry when `tokenizer.json` bridges into a `[String: Int]`, and
    /// *which* member survives is not stable from one load to the next. Two independently loaded
    /// tokenizers therefore disagree on those ids on `main` exactly as they do here, so the ids are
    /// compared by group. Nothing in this change can cause such a difference: the pair-keyed merge
    /// table collapses those pieces through the same `[String: Int]` the string-keyed one used.
    private static let canonicalId: [Int: Int] = {
        guard let data = try? Data(contentsOf: modelDir.appendingPathComponent("tokenizer.json")),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = root["model"] as? [String: Any],
              // The bridged Swift dictionary has already dropped the duplicates, so the byte-exact
              // keys have to come off the `NSDictionary`.
              let vocab = model["vocab"] as? NSDictionary else { return [:] }
        var lowest: [String: Int] = [:]
        var tokenById: [Int: String] = [:]
        for (key, value) in vocab {
            guard let token = key as? String, let id = value as? Int else { continue }
            tokenById[id] = token
            lowest[token] = min(lowest[token] ?? id, id)
        }
        return tokenById.mapValues { lowest[$0] ?? -1 }
    }()

    private static func grouped(_ ids: [Int]) -> [Int] { ids.map { canonicalId[$0] ?? $0 } }

    private func loadPair() throws -> (Gemma4Tokenizer, ReferenceGemma4BPE) {
        guard FileManager.default.fileExists(
            atPath: Self.modelDir.appendingPathComponent("tokenizer.json").path) else {
            throw XCTSkip("Gemma 4 tokenizer unavailable: \(Self.modelDir.path)")
        }
        let tokenizer = Gemma4Tokenizer()
        try tokenizer.load(from: Self.modelDir)
        let reference = ReferenceGemma4BPE()
        try reference.load(from: Self.modelDir)
        return (tokenizer, reference)
    }

    private func assertSameTokens(_ produced: [Int], _ expected: [Int], _ what: @autoclosure () -> String,
                                  file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(Self.grouped(produced), Self.grouped(expected), what(), file: file, line: line)
    }

    func testTranscriptRowsTokenizeIdentically() throws {
        let (tokenizer, reference) = try loadPair()
        let rows = Self.corpus()
        var tokens = 0
        for (i, row) in rows.enumerated() {
            let produced = tokenizer.encode(row)
            assertSameTokens(produced, reference.encode(row),
                             "row \(i) diverged: \(row.debugDescription)")
            tokens += produced.count
        }
        print("[gemma4-parity] \(rows.count) rows, \(rows.reduce(0) { $0 + $1.count }) chars, \(tokens) tokens")
    }

    func testConcatenatedTranscriptAndItsPrefixesTokenizeIdentically() throws {
        let (tokenizer, reference) = try loadPair()
        let whole = Self.corpus().joined(separator: " ")
        // Prefixes cut mid-word, mid-space-run and mid-multi-byte-character, which is where a merge
        // loop carrying state across a boundary would show up.
        var lengths = [1, 2, 17, 100, 512, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000]
            .filter { $0 < whole.count }
        lengths.append(whole.count)
        lengths = lengths.filter { $0 <= Self.referenceCeiling }
        for length in lengths {
            let prefix = String(whole.prefix(length))
            assertSameTokens(tokenizer.encode(prefix), reference.encode(prefix),
                             "prefix of \(length) characters diverged")
        }
        print("[gemma4-parity] prefixes compared: \(lengths)")
    }

    func testChatTemplateShapedTextTokenizesIdentically() throws {
        let (tokenizer, reference) = try loadPair()
        // The template puts a special token hard against text on both sides, which is the shape a
        // whitespace-splitting version of this optimization would have had to preserve.
        for row in Self.corpus().prefix(12) {
            for text in ["<|turn>user\n\(row)<turn|>\n",
                         "<bos><|turn>system\n\(row)<turn|>\n<|turn>model\n",
                         "\(row)<|turn>\(row)"] {
                assertSameTokens(tokenizer.encode(text), reference.encode(text),
                                 "template-shaped text diverged: \(text.prefix(60).debugDescription)")
            }
        }
    }

    func testEdgeCasesTokenizeIdenticallyOnTheRealVocabulary() throws {
        let (tokenizer, reference) = try loadPair()
        for text in [
            "", " ", "  ", "   ", "\n", "\t", " \n\t ", "\u{2581}",
            " leading", "trailing ", " both ", "double  space", "many     spaces",
            "> </", "<div> </div>", "a>  </b",
            "<bos>", "<eos>", "<|turn>", "<turn|>", "<|tool_response>", "<|think|>",
            "<bos>hello", "hello<|turn>", "mid<|turn>sentence", "<bos><|turn><turn|>",
            "你好世界", "😊", "👩‍💻", "🇺🇦", "é", "ﬁ", "\u{0}", "\u{7f}", "\u{FFFD}",
            "Привет, мир!", "Привет 你好 hello 😊",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            String(repeating: "ЛЛ", count: 200),
            String(repeating: " ", count: 64),
        ] {
            assertSameTokens(tokenizer.encode(text), reference.encode(text),
                             "edge case diverged: \(text.debugDescription)")
        }
    }

    /// Guards the defect this replaced: tokenization was quadratic in prompt length, so a
    /// 32k-character prompt spent tens of seconds here before the model saw its first token.
    func testTokenizationCostGrowsLinearly() throws {
        let (tokenizer, reference) = try loadPair()
        var filler = Self.corpus().joined(separator: " ")
        while filler.count < 32_000 { filler += " " + filler }

        func fastest(of rounds: Int, _ body: () -> Void) -> Double {
            var best = Double.greatestFiniteMagnitude
            for _ in 0..<rounds {
                let start = DispatchTime.now().uptimeNanoseconds
                body()
                best = min(best, Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9)
            }
            return best
        }

        _ = tokenizer.encode(String(filler.prefix(1_000)))
        // The reference costs minutes at the top sizes; time it there only when asked.
        let referenceLimit = Self.full ? Int.max : 8_000
        var timings: [(size: Int, fast: Double, slow: Double?)] = []
        for size in [2_000, 4_000, 8_000, 16_000, 32_000] {
            let text = String(filler.prefix(size))
            timings.append((size,
                            fastest(of: 3) { _ = tokenizer.encode(text) },
                            size <= referenceLimit ? fastest(of: 1) { _ = reference.encode(text) } : nil))
        }

        for (i, row) in timings.enumerated() {
            let exponent = i == 0 ? nil
                : log(row.fast / timings[i - 1].fast) / log(Double(row.size) / Double(timings[i - 1].size))
            let slow = row.slow.map { String(format: "%.4fs", $0) } ?? "-"
            print("[gemma4-parity] \(row.size) chars"
                  + String(format: "  new %.4fs", row.fast)
                  + "  reference \(slow)"
                  + (exponent.map { String(format: "  exponent %.2f", $0) } ?? ""))
        }

        // Quadratic behaviour put this in the tens of seconds. The bound is loose enough to survive a
        // loaded machine and tight enough that a return to a full rescan cannot pass it.
        XCTAssertLessThan(timings[timings.count - 1].fast, 2.0,
                          "32k characters must not take seconds to tokenize")
    }
}
