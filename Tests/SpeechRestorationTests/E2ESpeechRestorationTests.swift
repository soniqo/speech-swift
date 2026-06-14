import XCTest
import AudioCommon
@testable import SpeechRestoration

/// End-to-end test against the real Sidon CoreML artifacts.
///
/// Skipped automatically unless the local artifacts are present. It loads the
/// `.mlpackage` bundles from a local cache (`SIDON_LOCAL_CACHE`, default
/// `/tmp/sidon-local-cache` laid out as `<dir>/fp16/Sidon-{Predictor,Vocoder}.mlpackage`),
/// restores the benchmark reference clip, and compares the waveform to the
/// Python CoreML reference output when available.
final class E2ESpeechRestorationTests: XCTestCase {

    private var cacheDir: URL {
        let path = ProcessInfo.processInfo.environment["SIDON_LOCAL_CACHE"]
            ?? "/tmp/sidon-local-cache"
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    private func skipIfArtifactsMissing() throws {
        let pred = cacheDir.appendingPathComponent("fp16/Sidon-Predictor.mlpackage")
        if !FileManager.default.fileExists(atPath: pred.path) {
            throw XCTSkip("Sidon CoreML artifacts not present at \(pred.path)")
        }
    }

    func testRestoreReferenceClip() async throws {
        try skipIfArtifactsMissing()
        let inputURL = URL(fileURLWithPath: "/tmp/sidon-test/ref_16k.wav")
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw XCTSkip("reference clip /tmp/sidon-test/ref_16k.wav not present")
        }

        let audio = try AudioFileLoader.load(
            url: inputURL, targetSampleRate: SpeechRestorer.inputSampleRate)
        XCTAssertGreaterThan(audio.count, 0)

        let bundleDir = cacheDir.appendingPathComponent("fp16", isDirectory: true)
        let restorer = try SpeechRestorer.fromLocalBundle(directory: bundleDir, variant: .fp16)
        let restored = try restorer.restore(
            audio: audio, sampleRate: SpeechRestorer.inputSampleRate)

        // ~10 s input → ~10 s of 48 kHz output.
        XCTAssertGreaterThan(restored.count, 400_000)
        // Finite, sane amplitude.
        XCTAssertTrue(restored.allSatisfy { $0.isFinite })
        let peak = restored.map { Swift.abs($0) }.max() ?? 0
        XCTAssertGreaterThan(peak, 1e-3, "output should not be silent")
        XCTAssertLessThan(peak, 10.0, "output amplitude should be bounded")

        // Compare to the Python CoreML reference if present (waveform cosine —
        // neural-vocoder phase noise keeps this below 1.0 but it should be high).
        let refURL = URL(fileURLWithPath: "/tmp/sidon-test/restored_coreml_fp16.wav")
        if FileManager.default.fileExists(atPath: refURL.path) {
            let (ref, _) = try AudioFileLoader.loadWAV(url: refURL)
            let n = Swift.min(ref.count, restored.count)
            XCTAssertGreaterThan(n, 400_000)
            var dot = 0.0, na = 0.0, nb = 0.0
            for i in 0..<n {
                let a = Double(ref[i]), b = Double(restored[i])
                dot += a * b; na += a * a; nb += b * b
            }
            let cos = dot / (na.squareRoot() * nb.squareRoot() + 1e-12)
            print("E2E Sidon waveform cosine vs Python CoreML reference: \(cos)")
            XCTAssertGreaterThan(cos, 0.5,
                "Swift pipeline should track the Python CoreML output closely")
        }
    }
}
