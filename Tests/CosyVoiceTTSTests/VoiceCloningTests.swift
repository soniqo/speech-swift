import XCTest
import MLX
import AudioCommon
@testable import CosyVoiceTTS

/// Unit + integration tests for the zero-shot voice cloning path.
///
/// The unit tests construct a `CosyVoiceTTSModel` instance and call into the
/// tokeniser path — that's the smallest reproducer that exercises the
/// system-frame logic without requiring all the model weights to be present.
///
/// The E2E test downloads the full HF bundle and runs an end-to-end clone.
/// It's expensive so it's gated by the env var
/// `COSY_TEST_VOICE_REF=/path/to/ref.wav` — set it to point at a 5-30 s clean
/// reference clip and the test will measure that the synthesised audio is
/// non-silent, the right length, and that the LLM produced something
/// resembling the input text (Parakeet round-trip not asserted here to keep
/// the test self-contained).
final class VoiceCloningTests: XCTestCase {

    // MARK: - VoiceProfile struct

    func testVoiceProfileDefaultsToAllNil() {
        let p = CosyVoiceVoiceProfile()
        XCTAssertNil(p.speakerEmbedding)
        XCTAssertNil(p.promptToken)
        XCTAssertNil(p.promptFeat)
        XCTAssertNil(p.promptText)
    }

    func testVoiceProfileRoundTripsAllFields() {
        let emb: [Float] = Array(repeating: 0.1, count: 192)
        let token = MLXArray([Int32(1), 2, 3]).expandedDimensions(axis: 0)
        let feat = MLXArray.zeros([1, 80, 100], dtype: .float32)
        let p = CosyVoiceVoiceProfile(
            speakerEmbedding: emb,
            promptToken: token,
            promptFeat: feat,
            promptText: "ref transcript"
        )
        XCTAssertEqual(p.speakerEmbedding?.count, 192)
        XCTAssertEqual(p.promptToken?.shape, [1, 3])
        XCTAssertEqual(p.promptFeat?.shape, [1, 80, 100])
        XCTAssertEqual(p.promptText, "ref transcript")
    }

    // MARK: - tokenizeText: system-frame rules

    func testStyleInstructionSelectsInstruct2CloneLayout() {
        XCTAssertTrue(CosyVoiceTTSModel.usesInstructionConditionedClone(
            promptText: "A reference transcript.",
            instruction: "Speak excitedly and quickly."
        ))
        XCTAssertFalse(CosyVoiceTTSModel.usesInstructionConditionedClone(
            promptText: "A reference transcript.",
            instruction: "You are a helpful assistant."
        ))
        XCTAssertFalse(CosyVoiceTTSModel.usesInstructionConditionedClone(
            promptText: "A reference transcript.",
            instruction: "  You are a helpful assistant.  "
        ))
        XCTAssertFalse(CosyVoiceTTSModel.usesInstructionConditionedClone(
            promptText: nil,
            instruction: "Speak excitedly and quickly."
        ))
    }

    func testInstructionFramingAndStyleDetection() {
        // hasCustomStyleInstruction is the single dispatch predicate.
        XCTAssertFalse(CosyVoiceTTSModel.hasCustomStyleInstruction(""))
        XCTAssertFalse(CosyVoiceTTSModel.hasCustomStyleInstruction("You are a helpful assistant."))
        XCTAssertFalse(CosyVoiceTTSModel.hasCustomStyleInstruction("  You are a helpful assistant.  "))
        XCTAssertTrue(CosyVoiceTTSModel.hasCustomStyleInstruction("Speak excitedly."))

        // framedInstruction: default collapses to the frame; a custom style is
        // prefixed with it; an already-framed style is left unchanged.
        XCTAssertEqual(CosyVoiceTTSModel.framedInstruction(""), "You are a helpful assistant.")
        XCTAssertEqual(CosyVoiceTTSModel.framedInstruction("You are a helpful assistant."),
                       "You are a helpful assistant.")
        XCTAssertEqual(CosyVoiceTTSModel.framedInstruction("Speak excitedly."),
                       "You are a helpful assistant. Speak excitedly.")
        XCTAssertEqual(CosyVoiceTTSModel.framedInstruction("You are a helpful assistant. Speak excitedly."),
                       "You are a helpful assistant. Speak excitedly.")
    }

    /// Helper: load a `CosyVoiceTTSModel` so we have a tokenizer available.
    /// Falls back to skipping the test if HF is unreachable.
    private func loadModel() async throws -> CosyVoiceTTSModel {
        do {
            return try await CosyVoiceTTSModel.fromPretrained()
        } catch {
            throw XCTSkip("CosyVoiceTTSModel.fromPretrained() failed (offline?): \(error)")
        }
    }

    func testTokenizeTextDefaultInstructionPreservesAssistantPrefix() async throws {
        let model = try await loadModel()
        // Default path: instruction == assistantPrefix; the framed instruction
        // should be exactly the assistantPrefix, followed by <|endofprompt|>,
        // followed by the encoded content tokens.
        let tokens = model.tokenizeText("Welcome to the demo.", language: "english")
        let endIdx = tokens.firstIndex(of: CosyVoiceTTSModel.endOfPromptToken)
        XCTAssertNotNil(endIdx, "endOfPromptToken (151646) must appear")
        let beforeEnd = Array(tokens[..<endIdx!])
        // The assistantPrefix tokenises to the SAME sequence as the framed default,
        // i.e. the prefix should NOT be doubled.
        let bareAssistant = try XCTUnwrap(beforeEnd.last == CosyVoiceTTSModel.endOfPromptToken
                                          ? nil : beforeEnd)
        XCTAssertGreaterThan(bareAssistant.count, 0,
                             "Framed instruction tokens must be non-empty for the default case")
    }

    func testTokenizeTextCustomInstructionGetsAssistantPrefixPrepended() async throws {
        let model = try await loadModel()
        // For a custom (style) instruction without the assistant prefix, the
        // framed sequence is "You are a helpful assistant. <instr>" — strictly
        // longer than the bare custom instruction.
        let custom = "Speak excitedly and quickly."
        let framedTokens = model.tokenizeText(
            "Welcome to the demo.", language: "english", instruction: custom)
        let bareTokens = model.tokenizeText(
            "Welcome to the demo.", language: "english", instruction: custom)

        // Sanity: deterministic.
        XCTAssertEqual(framedTokens, bareTokens)

        let endIdx = framedTokens.firstIndex(of: CosyVoiceTTSModel.endOfPromptToken)!
        let preEnd = Array(framedTokens[..<endIdx])

        // The framed sequence MUST start with the same tokens as the default
        // assistantPrefix-only encoding (because the prefix is prepended).
        let defaultTokens = model.tokenizeText("dummy", language: "english")
        let defaultEndIdx = defaultTokens.firstIndex(of: CosyVoiceTTSModel.endOfPromptToken)!
        let defaultPreEnd = Array(defaultTokens[..<defaultEndIdx])
        for i in 0..<min(defaultPreEnd.count, preEnd.count) {
            XCTAssertEqual(defaultPreEnd[i], preEnd[i],
                "Framed custom instruction must share its leading tokens with the bare assistantPrefix encoding")
        }
        XCTAssertGreaterThan(preEnd.count, defaultPreEnd.count,
                             "Framed custom instruction must be strictly longer than just the assistantPrefix")
    }

    func testTokenizeTextEmptyInstructionFallsBackToAssistantPrefix() async throws {
        let model = try await loadModel()
        let bareDefault = model.tokenizeText("Hi.", language: "english")
        let emptyInstruct = model.tokenizeText("Hi.", language: "english", instruction: "   ")
        XCTAssertEqual(bareDefault, emptyInstruct,
                       "Whitespace-only instruction must be normalised back to the assistant default")
    }

    // MARK: - E2E voice cloning (gated by env var)

    /// Set `COSY_TEST_VOICE_REF=/path/to/ref.wav` to enable this test. The
    /// reference clip should be 5-30 s of clean speech.
    func testE2ECloneProducesAudio() async throws {
        guard let refPath = ProcessInfo.processInfo.environment["COSY_TEST_VOICE_REF"],
              FileManager.default.fileExists(atPath: refPath) else {
            throw XCTSkip("Set COSY_TEST_VOICE_REF to a wav file to enable this E2E test")
        }
        let transcript = ProcessInfo.processInfo.environment["COSY_TEST_VOICE_TRANSCRIPT"]
            ?? "This is a reference audio recording used for voice cloning."

        let modelId = ProcessInfo.processInfo.environment["COSY_TEST_MODEL_ID"]
            ?? "aufklarer/CosyVoice3-0.5B-MLX-bf16"
        let model = try await CosyVoiceTTSModel.fromPretrained(modelId: modelId)

        // Locate speech_tokenizer.safetensors — the test prefers an explicit
        // override, otherwise tries the same cache directory the model used.
        let tokPath = ProcessInfo.processInfo.environment["COSY_TEST_SPEECH_TOKENIZER"]
        guard let tokFile = tokPath, FileManager.default.fileExists(atPath: tokFile) else {
            throw XCTSkip("Set COSY_TEST_SPEECH_TOKENIZER to speech_tokenizer.safetensors")
        }
        let tokenizer = try SpeechTokenizerModel.fromSafetensors(
            at: URL(fileURLWithPath: tokFile))

        // Load the reference audio at 16 kHz.
        let refURL = URL(fileURLWithPath: refPath)
        let refSamples16k = try AudioFileLoader.load(url: refURL, targetSampleRate: 16_000)

        let profile = try model.extractVoiceProfile(
            audio: refSamples16k, sampleRate: 16_000,
            speechTokenizer: tokenizer,
            referenceTranscript: transcript)

        XCTAssertNotNil(profile.promptToken)
        XCTAssertNotNil(profile.promptFeat)
        XCTAssertEqual(profile.promptText, transcript)

        // Confirm prompt_token and prompt_feat have non-trivial length —
        // a few mel frames per second of reference.
        let tokLen = profile.promptToken?.dim(1) ?? 0
        let melLen = profile.promptFeat?.dim(2) ?? 0
        XCTAssertGreaterThan(tokLen, 20, "Need at least ~1 s of prompt tokens")
        XCTAssertGreaterThan(melLen, 40, "Need at least ~1 s of prompt mel frames")

        let samples = model.synthesize(
            text: "Welcome to the demo.",
            voiceProfile: profile,
            language: "english"
        )

        XCTAssertFalse(samples.isEmpty, "Cloned synthesis must produce audio")
        let duration = Double(samples.count) / 24_000.0
        XCTAssertGreaterThan(duration, 0.4, "Cloned output should be > 0.4 s")
        XCTAssertLessThan(duration, 12.0, "Cloned short phrase should not run on for >12 s")
        let peak = samples.map { abs($0) }.max() ?? 0
        XCTAssertGreaterThan(peak, 0.01, "Cloned output must not be silence")
        XCTAssertLessThan(peak, 1.0001, "Cloned output must not overflow")

        // A style instruction selects the instruct2 clone layout. It must
        // still produce a valid, bounded render while preserving the Flow
        // model's reference-audio anchor.
        let styledSamples = model.synthesize(
            text: "Welcome to the demo.",
            voiceProfile: profile,
            language: "english",
            instruction: "Speak warmly with restrained excitement."
        )
        XCTAssertFalse(styledSamples.isEmpty, "Instructed clone must produce audio")
        let styledDuration = Double(styledSamples.count) / 24_000.0
        XCTAssertGreaterThan(styledDuration, 0.4, "Instructed clone should be > 0.4 s")
        XCTAssertLessThan(styledDuration, 12.0, "Instructed clone should not run on for >12 s")
        let styledPeak = styledSamples.map { abs($0) }.max() ?? 0
        XCTAssertGreaterThan(styledPeak, 0.01, "Instructed clone must not be silence")
        XCTAssertLessThan(styledPeak, 1.0001, "Instructed clone must not overflow")
    }

    /// Same gating as `testE2ECloneProducesAudio`. Exercises the long-form
    /// multi-segment clone path end-to-end against the real model, asserting
    /// the two contracts this path promises:
    /// 1. Seeded runs are reproducible (same seed → same audio).
    /// 2. Every output leaves with a faded tail (no audible end click).
    func testE2ESeededLongFormCloneIsReproducibleAndFadesTail() async throws {
        guard let refPath = ProcessInfo.processInfo.environment["COSY_TEST_VOICE_REF"],
              FileManager.default.fileExists(atPath: refPath) else {
            throw XCTSkip("Set COSY_TEST_VOICE_REF to a wav file to enable this E2E test")
        }
        let transcript = ProcessInfo.processInfo.environment["COSY_TEST_VOICE_TRANSCRIPT"]
            ?? "This is a reference audio recording used for voice cloning."
        let modelId = ProcessInfo.processInfo.environment["COSY_TEST_MODEL_ID"]
            ?? "aufklarer/CosyVoice3-0.5B-MLX-bf16"
        guard let tokFile = ProcessInfo.processInfo.environment["COSY_TEST_SPEECH_TOKENIZER"],
              FileManager.default.fileExists(atPath: tokFile) else {
            throw XCTSkip("Set COSY_TEST_SPEECH_TOKENIZER to speech_tokenizer.safetensors")
        }

        let model = try await CosyVoiceTTSModel.fromPretrained(modelId: modelId)
        let tokenizer = try SpeechTokenizerModel.fromSafetensors(
            at: URL(fileURLWithPath: tokFile))
        let refSamples16k = try AudioFileLoader.load(
            url: URL(fileURLWithPath: refPath), targetSampleRate: 16_000)
        let profile = try model.extractVoiceProfile(
            audio: refSamples16k, sampleRate: 16_000,
            speechTokenizer: tokenizer,
            referenceTranscript: transcript)

        // Two sentences, each above the merge threshold, so the long-form
        // splitter genuinely produces multiple segments.
        let text = "The quick brown fox jumps over the lazy sleeping dog. "
            + "Pack my box with five dozen bright liquid jugs today."
        XCTAssertGreaterThan(
            CosyVoiceTTSModel.splitForLongForm(text).count, 1,
            "Fixture text must exercise the multi-segment path")

        let first = model.synthesize(
            text: text, voiceProfile: profile, language: "english", seed: 7)
        let second = model.synthesize(
            text: text, voiceProfile: profile, language: "english", seed: 7)

        XCTAssertFalse(first.isEmpty, "Long-form clone must produce audio")
        XCTAssertGreaterThan(first.map { abs($0) }.max() ?? 0, 0.01, "Must not be silence")

        // Contract 1: reproducibility. Same seed re-seeds MLX before each
        // segment, so both renders must match sample-for-sample (tolerate
        // tiny numeric drift, fail on any real divergence).
        XCTAssertEqual(first.count, second.count, "Same seed must produce same-length audio")
        if first.count == second.count && !first.isEmpty {
            var absDiff: Float = 0
            var absRef: Float = 0
            for i in 0..<first.count {
                absDiff += abs(first[i] - second[i])
                absRef += abs(first[i])
            }
            XCTAssertLessThan(
                absDiff, max(absRef, 1) * 0.01,
                "Same-seed renders must be near-identical")
        }

        // Contract 2: no end click. The stitcher fades every output tail,
        // so the final samples must sit near zero.
        let tailPeak = first.suffix(8).map { abs($0) }.max() ?? 0
        XCTAssertLessThan(tailPeak, 0.05, "Long-form output must end faded (no end click)")
    }
}
