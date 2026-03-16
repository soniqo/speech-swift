import XCTest
@testable import KokoroTTS
import CoreML

/// E2E tests that require downloaded CoreML models.
/// Run with: swift test --filter KokoroE2ETests
final class KokoroE2ETests: XCTestCase {

    static let testModelDir = "/tmp/kokoro-coreml-test"

    /// Test loading vocab_index.json from aufklarer/Kokoro-82M-CoreML.
    func testLoadVocabIndex() throws {
        let url = URL(fileURLWithPath: Self.testModelDir + "/vocab_index.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("Models not downloaded — run download script first")
        }
        let phonemizer = try KokoroPhonemizer.loadVocab(from: url)

        // Tokenize a simple IPA string
        let ids = phonemizer.tokenize("hello")
        XCTAssertEqual(ids.first, 1) // BOS
        XCTAssertEqual(ids.last, 2)  // EOS
        XCTAssertTrue(ids.count >= 3)
    }

    /// Test loading pronunciation dictionaries.
    func testLoadDictionaries() throws {
        let dir = URL(fileURLWithPath: Self.testModelDir)
        guard FileManager.default.fileExists(atPath: dir.appendingPathComponent("us_gold.json").path) else {
            throw XCTSkip("Models not downloaded")
        }
        let vocab = URL(fileURLWithPath: Self.testModelDir + "/vocab_index.json")
        let phonemizer = try KokoroPhonemizer.loadVocab(from: vocab)
        try phonemizer.loadDictionaries(from: dir)

        // "hello" should be in the dictionary → produce IPA
        let ids = phonemizer.tokenize("hello")
        XCTAssertTrue(ids.count > 3, "Expected more than BOS+EOS for 'hello'")
    }

    /// Test loading G2P encoder + decoder.
    func testLoadG2PModels() throws {
        let dir = URL(fileURLWithPath: Self.testModelDir)
        let encoderURL = dir.appendingPathComponent("G2PEncoder.mlmodelc")
        let decoderURL = dir.appendingPathComponent("G2PDecoder.mlmodelc")
        let vocabURL = dir.appendingPathComponent("g2p_vocab.json")
        guard FileManager.default.fileExists(atPath: encoderURL.path) else {
            throw XCTSkip("Models not downloaded")
        }

        let mainVocab = URL(fileURLWithPath: Self.testModelDir + "/vocab_index.json")
        let phonemizer = try KokoroPhonemizer.loadVocab(from: mainVocab)
        try phonemizer.loadG2PModels(encoderURL: encoderURL, decoderURL: decoderURL, vocabURL: vocabURL)

        // Try phonemizing an OOV word through the neural G2P
        let ids = phonemizer.tokenize("supercalifragilistic")
        XCTAssertTrue(ids.count > 3, "G2P should produce tokens for OOV word")
    }

    /// Test loading voice embedding JSON.
    func testLoadVoiceEmbedding() throws {
        let voiceURL = URL(fileURLWithPath: Self.testModelDir + "/voices/af_heart.json")
        guard FileManager.default.fileExists(atPath: voiceURL.path) else {
            throw XCTSkip("Models not downloaded")
        }

        let data = try Data(contentsOf: voiceURL)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let embedding = json["embedding"] as! [Double]

        // Voice embedding is 256-dim (matches ref_s input)
        XCTAssertEqual(embedding.count, 256)

        let refS = embedding.map { Float($0) }
        XCTAssertEqual(refS.count, 256)
        XCTAssertFalse(refS.allSatisfy { $0 == 0 }, "Embedding shouldn't be all zeros")
    }

    /// Test loading the CoreML Kokoro model.
    func testLoadKokoroModel() throws {
        let dir = URL(fileURLWithPath: Self.testModelDir)
        let modelURL = dir.appendingPathComponent("kokoro_21_5s.mlmodelc")
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw XCTSkip("Models not downloaded")
        }

        let network = try KokoroNetwork(directory: dir)
        XCTAssertTrue(network.availableBuckets.contains(.v21_5s))
    }

    /// Full E2E: text → phonemes → CoreML inference → audio.
    func testEndToEndSynthesis() throws {
        let dir = URL(fileURLWithPath: Self.testModelDir)
        guard FileManager.default.fileExists(atPath: dir.appendingPathComponent("kokoro_21_5s.mlmodelc").path) else {
            throw XCTSkip("Models not downloaded")
        }

        // Load phonemizer
        let vocab = dir.appendingPathComponent("vocab_index.json")
        let phonemizer = try KokoroPhonemizer.loadVocab(from: vocab)
        try phonemizer.loadDictionaries(from: dir)

        let encoderURL = dir.appendingPathComponent("G2PEncoder.mlmodelc")
        let decoderURL = dir.appendingPathComponent("G2PDecoder.mlmodelc")
        let vocabURL = dir.appendingPathComponent("g2p_vocab.json")
        if FileManager.default.fileExists(atPath: encoderURL.path) {
            try phonemizer.loadG2PModels(encoderURL: encoderURL, decoderURL: decoderURL, vocabURL: vocabURL)
        }

        // Load voice embedding (256-dim)
        let voiceData = try Data(contentsOf: dir.appendingPathComponent("voices/af_heart.json"))
        let voiceJson = try JSONSerialization.jsonObject(with: voiceData) as! [String: Any]
        let embedding = voiceJson["embedding"] as! [Double]
        let styleVector = embedding.map { Float($0) }

        // Load network
        let network = try KokoroNetwork(directory: dir)

        // Create model
        let config = KokoroConfig.default
        let model = KokoroTTSModel(
            config: config,
            network: network,
            phonemizer: phonemizer,
            voiceEmbeddings: ["af_heart": styleVector]
        )

        // Synthesize
        let audio = try model.synthesize(text: "Hello world", voice: "af_heart")

        XCTAssertTrue(audio.count > 0, "Should produce audio samples")
        XCTAssertTrue(audio.count > 1000, "Should produce meaningful audio (got \(audio.count) samples)")

        let duration = Double(audio.count) / 24000.0
        print("E2E synthesis: \(audio.count) samples, \(String(format: "%.2f", duration))s")

        // Audio should have non-zero energy
        let rms = sqrt(audio.map { $0 * $0 }.reduce(0, +) / Float(audio.count))
        XCTAssertGreaterThan(rms, 0.001, "Audio should have non-zero energy")
    }
}
