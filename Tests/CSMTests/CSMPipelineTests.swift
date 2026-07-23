import XCTest
import Foundation
import MLX
import PersonaPlex
@testable import CSM

final class CSMPipelineTests: XCTestCase {
    /// Full text→audio in Swift: tokenizer + Mimi encode(ref) + frame loop +
    /// Mimi decode. Env: CSM_MODEL_DIR (export w/ tokenizer + mimi), CSM_REFAUDIO
    /// (safetensors {audio}), CSM_REFTEXT (transcript), optional CSM_OUT (wav).
    func testTextToAudio() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let dir = env["CSM_MODEL_DIR"], let refPath = env["CSM_REFAUDIO"],
              let refText = env["CSM_REFTEXT"] else {
            throw XCTSkip("set CSM_MODEL_DIR, CSM_REFAUDIO, CSM_REFTEXT")
        }
        let pipeline = try await CSMPipeline(directory: URL(fileURLWithPath: dir))
        let refAudio = try MLX.loadArrays(url: URL(fileURLWithPath: refPath))["audio"]!

        let audio = pipeline.synthesize(
            text: "This is our own runtime, running fully on device.",
            refAudio: refAudio, refText: refText,
            maxFrames: 128, temperature: 0.9, topK: 50)
        eval(audio)

        let n = audio.shape.last ?? 0
        let rms = sqrt((audio.asType(.float32) * audio.asType(.float32)).mean()).item(Float.self)
        print("CSM Swift text→audio: \(n) samples (\(Double(n) / 24000.0)s), RMS \(rms)")
        XCTAssertGreaterThan(n, 24000)          // > 1 s of audio
        XCTAssertGreaterThan(rms, 0.02)         // non-silent speech

        if let out = env["CSM_OUT"] {
            try writeWav(audio.asType(.float32), to: URL(fileURLWithPath: out), sampleRate: 24000)
        }
    }
}
