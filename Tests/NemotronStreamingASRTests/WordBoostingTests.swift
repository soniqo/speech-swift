import XCTest
@testable import NemotronStreamingASR

final class WordBoostingVocabularyTests: XCTestCase {

    func testEncodeForWordBoostingUsesSentencePieceBoundaries() {
        let vocab = NemotronVocabulary(idToToken: [
            0: "▁n",
            1: "emo",
            2: "tron",
            3: "▁word",
            4: "▁boost",
            5: "ing",
        ])

        XCTAssertEqual(
            vocab.encodeForWordBoosting("nemotron word boosting"),
            [0, 1, 2, 3, 4, 5]
        )
    }

    func testEncodeForWordBoostingReturnsNilForUnsegmentablePhrase() {
        let vocab = NemotronVocabulary(idToToken: [
            0: "▁known",
        ])

        XCTAssertNil(vocab.encodeForWordBoosting("unknown"))
    }

    func testEncodingsForWordBoostingIncludesAlternativeSegmentations() {
        let vocab = NemotronVocabulary(idToToken: [
            0: "▁v",
            1: "o",
            2: "x",
            3: "tra",
            4: "l",
            5: "▁",
            6: "vo",
            7: "tr",
            8: "al",
        ])

        let encodings = vocab.encodingsForWordBoosting("voxtral", maxCount: 4)
        XCTAssertTrue(encodings.contains([0, 1, 2, 3, 4]))
        XCTAssertTrue(encodings.contains([5, 6, 2, 7, 8]))
    }
}

final class WordBoostingTokenizerTests: XCTestCase {

    func testSentencePieceUnigramChoosesHighestScoringPath() {
        let tokenizer = NemotronSentencePieceUnigramTokenizer(rawPieces: [
            (token: "<unk>", score: 0, type: 2),
            (token: "▁", score: -0.1, type: 1),
            (token: "V", score: -1.0, type: 1),
            (token: "▁V", score: -5.0, type: 1),
            (token: "o", score: -1.0, type: 1),
        ])

        XCTAssertEqual(tokenizer.encodeForWordBoosting("Vo"), [1, 2, 4])
    }

    /// Hard-coded parity against the canonical Python `sentencepiece` output
    /// on the Nemotron 3.5 tokenizer.model. Catches normalization drift
    /// (NFKC, whitespace marker, lowercase) when the model bundle changes.
    ///
    /// Regenerate the expected IDs with:
    ///   python scripts/nemotron_spm_parity.py \\
    ///       --tokenizer ~/Library/Caches/qwen3-speech/models/aufklarer/Nemotron-3.5-ASR-Streaming-0.6B-CoreML-INT8/tokenizer.model
    func testSentencePieceTokenizerMatchesCachedNemotronModelWhenAvailable() throws {
        let url = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Caches/qwen3-speech/models/aufklarer/Nemotron-3.5-ASR-Streaming-0.6B-CoreML-INT8/tokenizer.model")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("Cached Nemotron tokenizer.model is not available")
        }

        let tokenizer = try NemotronSentencePieceUnigramTokenizer(modelURL: url)

        XCTAssertEqual(tokenizer.encodeForWordBoosting("Voxtral"), [2, 216, 46, 195, 1071, 66])
        XCTAssertEqual(tokenizer.encodeForWordBoosting("voxtral"), [2, 314, 195, 1071, 66])
        XCTAssertEqual(tokenizer.encodeForWordBoosting("Babble"), [538, 2783, 3012])
        XCTAssertEqual(tokenizer.encodeForWordBoosting("vocab JSON"), [2, 314, 1257, 140, 2, 228, 193, 210, 209])
        XCTAssertEqual(tokenizer.encodeForWordBoosting("should"), [2922])

        // "Babble" segments into 3 pieces for a single word; at 3+ tokens
        // per word the suggestion thresholds rate a phrase hard (see
        // docs/inference/nemotron-asr-streaming.md).
        let suggestion = WordBoostingContext.suggestions(
            for: ["Babble"],
            vocabulary: NemotronVocabulary(idToToken: [:]),
            tokenizer: tokenizer
        )[0]
        XCTAssertEqual(suggestion.difficulty, .hard)
        XCTAssertEqual(suggestion.suggestedBoost, 1.25)
    }

    func testSuggestionTreatsShortSinglePieceTermAsEasy() {
        // "AI" tokenizes to a single in-vocab piece. The previous heuristic
        // routed any phrase with <=4 scalars to .hard / 1.25 regardless of
        // token count — which over-fires for canonical short brand names
        // (AI, EU, iOS, kHz). Single-piece coverage is the canonical .easy
        // case.
        let tokenizer = NemotronSentencePieceUnigramTokenizer(rawPieces: [
            (token: "<unk>", score: 0, type: 2),
            (token: "▁AI", score: -1.0, type: 1),
        ])

        let suggestions = WordBoostingContext.suggestions(
            for: ["AI"],
            vocabulary: NemotronVocabulary(idToToken: [:]),
            tokenizer: tokenizer
        )

        XCTAssertEqual(suggestions.count, 1)
        XCTAssertEqual(suggestions[0].tokenCount, 1)
        XCTAssertEqual(suggestions[0].difficulty, .easy)
        XCTAssertEqual(suggestions[0].suggestedBoost, 0.75)
    }
}

final class WordBoostingTreeTests: XCTestCase {

    func testTreeReturnsDepthScaledScoresForPhrasePrefix() {
        let tree = WordBoostingTree(
            phrases: [(tokens: [1, 2], boost: 1.0)],
            depthScaling: 2.0
        )

        let first = tree.advance(from: 0, token: 1)
        XCTAssertEqual(first.score, 1.0, accuracy: 1e-5)

        let second = tree.advance(from: first.node, token: 2)
        XCTAssertEqual(second.score, 2.0 + log(2.0), accuracy: 1e-5)
    }

    func testTreeFallsBackToOverlappingPhrasePrefix() {
        let tree = WordBoostingTree(
            phrases: [
                (tokens: [1, 2], boost: 1.0),
                (tokens: [2, 3], boost: 1.0),
            ],
            depthScaling: 2.0
        )

        let one = tree.advance(from: 0, token: 1)
        let two = tree.advance(from: one.node, token: 2)
        let three = tree.advance(from: two.node, token: 3)

        XCTAssertGreaterThan(three.score, 0)
    }

    func testTreeRecomputesSharedPrefixScoresForHigherLaterBoost() {
        let tree = WordBoostingTree(
            phrases: [
                (tokens: [1, 2], boost: 0.2),
                (tokens: [1, 2, 3], boost: 1.0),
            ],
            depthScaling: 2.0
        )

        let one = tree.advance(from: 0, token: 1)
        let two = tree.advance(from: one.node, token: 2)
        let three = tree.advance(from: two.node, token: 3)

        XCTAssertEqual(one.score, 1.0, accuracy: 1e-5)
        XCTAssertEqual(two.score, 2.0 + log(2.0), accuracy: 1e-5)
        XCTAssertEqual(three.score, 2.0 + log(3.0), accuracy: 1e-5)
    }
}

final class WordBoostingSelectionTests: XCTestCase {

    func testZeroBoostDisablesWordBoostingContext() {
        let vocab = NemotronVocabulary(idToToken: [
            0: "▁alpha",
        ])

        XCTAssertNil(WordBoostingContext(
            config: WordBoostingConfig(phrases: ["alpha"], boost: 0),
            vocabulary: vocab
        ))
    }

    func testWordBoostingSuggestionsAreBatchComputed() {
        let vocab = NemotronVocabulary(idToToken: [
            0: "▁Babble",
            1: "▁Q",
            2: "w",
            3: "e",
            4: "n",
            5: "▁V",
            6: "o",
            7: "x",
            8: "t",
            9: "r",
            10: "a",
            11: "l",
        ])

        let suggestions = WordBoostingContext.suggestions(
            for: ["Babble", "Qwen", "Voxtral"],
            vocabulary: vocab
        )

        XCTAssertEqual(suggestions.map(\.phrase), ["Babble", "Qwen", "Voxtral"])
        XCTAssertEqual(suggestions[0].difficulty, .easy)
        XCTAssertEqual(suggestions[0].suggestedBoost, 0.75)
        XCTAssertEqual(suggestions[1].difficulty, .hard)
        XCTAssertEqual(suggestions[1].suggestedBoost, 1.25)
        XCTAssertEqual(suggestions[2].difficulty, .hard)
        XCTAssertEqual(suggestions[2].suggestedBoost, 1.25)
    }

    func testWordBoostingSuggestionReportsUnencodablePhrase() {
        let vocab = NemotronVocabulary(idToToken: [
            0: "▁known",
        ])

        let suggestion = WordBoostingContext.suggestions(
            for: ["unknown"],
            vocabulary: vocab
        )[0]

        XCTAssertEqual(suggestion.difficulty, .hard)
        XCTAssertEqual(suggestion.suggestedBoost, 1.25)
        XCTAssertEqual(suggestion.tokenCount, 0)
        XCTAssertTrue(suggestion.tokenizations.isEmpty)
    }

    func testWordBoostingUsesExactCasingForFallback() {
        let vocab = NemotronVocabulary(idToToken: [
            0: "▁v",
            1: "▁V",
            2: "o",
            3: "x",
            4: "tra",
            5: "l",
            6: "▁other",
        ])
        let context = WordBoostingContext(
            config: WordBoostingConfig(phrases: ["Voxtral"], boost: 1.0),
            vocabulary: vocab
        )!
        var state = context.initialState()
        var logits: [Float] = [-0.2, 0.9, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5]
        let count = logits.count

        let selected = logits.withUnsafeMutableBufferPointer { buffer in
            context.selectToken(
                from: buffer.baseAddress!,
                count: count,
                blankTokenId: 7,
                state: &state
            )
        }

        XCTAssertEqual(selected, 1)
    }

    func testFallbackWordBoostingUsesGreedyPathOnly() {
        let vocab = NemotronVocabulary(idToToken: [
            0: "▁v",
            1: "o",
            2: "x",
            3: "tral",
            4: "▁",
            5: "vo",
            6: "xtral",
            7: "▁other",
        ])
        let context = WordBoostingContext(
            config: WordBoostingConfig(phrases: ["voxtral"], boost: 1.0),
            vocabulary: vocab
        )!
        var state = context.initialState()
        var logits: [Float] = [-1.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 1.0, 0.5]
        let count = logits.count

        let selected = logits.withUnsafeMutableBufferPointer { buffer in
            context.selectToken(
                from: buffer.baseAddress!,
                count: count,
                blankTokenId: 8,
                state: &state
            )
        }

        XCTAssertEqual(selected, 7)
    }

    func testWordBoostingSupportsPerPhraseBoosts() {
        let vocab = NemotronVocabulary(idToToken: [
            0: "▁easy",
            1: "▁hard",
            2: "▁other",
        ])
        let context = WordBoostingContext(
            config: WordBoostingConfig(phrases: [
                .init("easy", boost: 0.2),
                .init("hard", boost: 1.0),
            ]),
            vocabulary: vocab
        )!
        var state = context.initialState()
        var logits: [Float] = [0.7, 0.1, 1.0, 0.5]
        let count = logits.count

        let selected = logits.withUnsafeMutableBufferPointer { buffer in
            context.selectToken(
                from: buffer.baseAddress!,
                count: count,
                blankTokenId: 3,
                state: &state
            )
        }

        XCTAssertEqual(selected, 1)
    }

    func testBlankTokenWinsBeforeBoostingIsApplied() {
        let vocab = NemotronVocabulary(idToToken: [
            0: "▁alpha",
            1: "▁beta",
            2: "▁gamma",
        ])
        let context = WordBoostingContext(
            config: WordBoostingConfig(phrases: ["alpha"], boost: 10.0),
            vocabulary: vocab
        )!
        var state = context.initialState()
        var logits: [Float] = [0.0, 0.1, 0.2, 0.3]
        let logitsCount = logits.count

        let selected = logits.withUnsafeMutableBufferPointer { buffer in
            context.selectToken(
                from: buffer.baseAddress!,
                count: logitsCount,
                blankTokenId: 3,
                state: &state
            )
        }

        XCTAssertEqual(selected, 3)
        XCTAssertEqual(state, context.initialState())
    }

    func testBoostedTokenWinsOnlyWhenCloseEnough() {
        let vocab = NemotronVocabulary(idToToken: [
            0: "▁alpha",
            1: "▁beta",
            2: "▁gamma",
        ])

        var lowBoostLogits: [Float] = [0.7, 0.0, 1.0, 0.5]
        let lowBoost = WordBoostingContext(
            config: WordBoostingConfig(phrases: ["alpha"], boost: 0.2),
            vocabulary: vocab
        )!
        var lowState = lowBoost.initialState()
        let lowBoostCount = lowBoostLogits.count
        let lowSelected = lowBoostLogits.withUnsafeMutableBufferPointer { buffer in
            lowBoost.selectToken(
                from: buffer.baseAddress!,
                count: lowBoostCount,
                blankTokenId: 3,
                state: &lowState
            )
        }
        XCTAssertEqual(lowSelected, 2)

        var highBoostLogits: [Float] = [0.7, 0.0, 1.0, 0.5]
        let highBoost = WordBoostingContext(
            config: WordBoostingConfig(phrases: ["alpha"], boost: 0.4),
            vocabulary: vocab
        )!
        var highState = highBoost.initialState()
        let highBoostCount = highBoostLogits.count
        let highSelected = highBoostLogits.withUnsafeMutableBufferPointer { buffer in
            highBoost.selectToken(
                from: buffer.baseAddress!,
                count: highBoostCount,
                blankTokenId: 3,
                state: &highState
            )
        }
        XCTAssertEqual(highSelected, 0)
    }

    func testConfiguredBoostCanOverrideRawLogitGap() {
        let vocab = NemotronVocabulary(idToToken: [
            0: "▁alpha",
            1: "▁beta",
            2: "▁gamma",
        ])
        let context = WordBoostingContext(
            config: WordBoostingConfig(phrases: ["alpha"], boost: 10.0),
            vocabulary: vocab
        )!
        var state = context.initialState()
        // The first-token bonus equals the configured boost, so a boost of
        // 10 overrides a logit gap smaller than 10 (here: 6 points).
        var logits: [Float] = [-5.0, 0.0, 1.0, 0.5]
        let count = logits.count

        let selected = logits.withUnsafeMutableBufferPointer { buffer in
            context.selectToken(
                from: buffer.baseAddress!,
                count: count,
                blankTokenId: 3,
                state: &state
            )
        }

        XCTAssertEqual(selected, 0)
        XCTAssertNotEqual(state, context.initialState())
    }

    func testSelectedBoostedTokenAdvancesPhraseState() {
        let vocab = NemotronVocabulary(idToToken: [
            0: "▁alpha",
            1: "▁beta",
            2: "▁gamma",
        ])
        let context = WordBoostingContext(
            config: WordBoostingConfig(phrases: ["alpha beta"], boost: 0.6),
            vocabulary: vocab
        )!
        var state = context.initialState()

        var firstLogits: [Float] = [0.7, 0.0, 1.0, 0.5]
        let firstCount = firstLogits.count
        let first = firstLogits.withUnsafeMutableBufferPointer { buffer in
            context.selectToken(
                from: buffer.baseAddress!,
                count: firstCount,
                blankTokenId: 3,
                state: &state
            )
        }
        XCTAssertEqual(first, 0)

        var secondLogits: [Float] = [0.0, 0.5, 2.0, 0.4]
        let secondCount = secondLogits.count
        let second = secondLogits.withUnsafeMutableBufferPointer { buffer in
            context.selectToken(
                from: buffer.baseAddress!,
                count: secondCount,
                blankTokenId: 3,
                state: &state
            )
        }
        XCTAssertEqual(second, 1)
    }

    func testDivergentTokenResetsPhraseState() {
        let vocab = NemotronVocabulary(idToToken: [
            0: "▁alpha",
            1: "▁beta",
            2: "▁gamma",
        ])
        let context = WordBoostingContext(
            config: WordBoostingConfig(phrases: ["alpha beta"], boost: 0.1),
            vocabulary: vocab
        )!
        var state = context.initialState()

        var firstLogits: [Float] = [1.0, 0.0, 0.8, 0.5]
        let firstCount = firstLogits.count
        let first = firstLogits.withUnsafeMutableBufferPointer { buffer in
            context.selectToken(
                from: buffer.baseAddress!,
                count: firstCount,
                blankTokenId: 3,
                state: &state
            )
        }
        XCTAssertEqual(first, 0)
        XCTAssertNotEqual(state, context.initialState())

        var secondLogits: [Float] = [0.0, 0.1, 10.0, 0.5]
        let secondCount = secondLogits.count
        let second = secondLogits.withUnsafeMutableBufferPointer { buffer in
            context.selectToken(
                from: buffer.baseAddress!,
                count: secondCount,
                blankTokenId: 3,
                state: &state
            )
        }

        XCTAssertEqual(second, 2)
        XCTAssertEqual(state, context.initialState())
    }
}
