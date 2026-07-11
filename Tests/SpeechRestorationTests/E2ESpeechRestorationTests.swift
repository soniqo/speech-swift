import XCTest
import AudioCommon
@testable import SpeechRestoration

/// End-to-end test against the real Sidon CoreML artifacts.
///
/// Runs whenever the local CoreML bundles are present, skipping otherwise. It
/// loads the predictor + vocoder bundles directly from disk (no HuggingFace
/// fetch), restores the benchmark reference clip, and compares the waveform to
/// the Python CoreML reference output when available.
///
/// Artifact discovery, first match wins:
///   1. `$SIDON_LOCAL_CACHE/<variant>/Sidon-{Predictor,Vocoder}.{mlmodelc,mlpackage}`
///      (default cache: `/tmp/sidon-local-cache`).
///   2. The canonical export directories produced by the converter:
///      fp16 → `/tmp/sidon-art/sidon-coreml`, int8 → `/tmp/sidon-art-int8p/sidon-coreml`.
final class E2ESpeechRestorationTests: XCTestCase {

    private var cacheDir: URL {
        let path = ProcessInfo.processInfo.environment["SIDON_LOCAL_CACHE"]
            ?? "/tmp/sidon-local-cache"
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    /// Canonical converter output directory for a variant (fallback when the
    /// per-variant cache subfolder is not laid out).
    private func canonicalArtifactDir(_ variant: SidonVariant) -> URL {
        switch variant {
        case .fp16: return URL(fileURLWithPath: "/tmp/sidon-art/sidon-coreml", isDirectory: true)
        case .int8: return URL(fileURLWithPath: "/tmp/sidon-art-int8p/sidon-coreml", isDirectory: true)
        }
    }

    /// Returns a directory that actually contains the variant's predictor bundle
    /// (compiled or package), or nil if neither location has it.
    private func bundleDir(for variant: SidonVariant) -> URL? {
        let fm = FileManager.default
        func hasPredictor(_ dir: URL) -> Bool {
            fm.fileExists(atPath: dir.appendingPathComponent(SidonConfig.predictorCompiledName).path)
                || fm.fileExists(atPath: dir.appendingPathComponent(SidonConfig.predictorPackageName).path)
        }
        let cacheSub = cacheDir.appendingPathComponent(variant.subfolder, isDirectory: true)
        if hasPredictor(cacheSub) { return cacheSub }
        let canonical = canonicalArtifactDir(variant)
        if hasPredictor(canonical) { return canonical }
        return nil
    }

    private var referenceClipURL: URL { URL(fileURLWithPath: "/tmp/sidon-test/ref_16k.wav") }

    /// The exact output length the engine produces for `inputSamples` of 16 kHz
    /// audio, via the engine's own window planner.
    private func expectedRestoredLength(inputSamples: Int) -> Int {
        SpeechRestorer.windowPlan(inputSamples: inputSamples).outputLength
    }

    /// Shared body: load the bundle, restore the clip, assert output sanity, and
    /// (when a Python reference is present) assert high waveform cosine.
    private func runRestoreCheck(
        variant: SidonVariant, referenceWAV: String
    ) throws {
        guard let bundle = bundleDir(for: variant) else {
            throw XCTSkip("Sidon \(variant.rawValue) CoreML artifacts not present")
        }
        guard FileManager.default.fileExists(atPath: referenceClipURL.path) else {
            throw XCTSkip("reference clip \(referenceClipURL.path) not present")
        }

        let audio = try AudioFileLoader.load(
            url: referenceClipURL, targetSampleRate: SpeechRestorer.inputSampleRate)
        XCTAssertGreaterThan(audio.count, 0)

        let restorer = try SpeechRestorer.fromLocalBundle(directory: bundle, variant: variant)
        let restored = try restorer.restore(
            audio: audio, sampleRate: SpeechRestorer.inputSampleRate)

        // ~10 s input → ~10 s of 48 kHz output. The engine emits a fixed
        // `outputSamplesPerWindow` per fixed window, trimmed down to at most the
        // input duration rescaled onto the 48 kHz timeline. It never pads up, so
        // the result is bounded above by the rescaled target and equals the raw
        // vocoder length when that's already shorter.
        XCTAssertEqual(restored.count, expectedRestoredLength(inputSamples: audio.count),
            "output length should match the engine's window/trim contract")
        XCTAssertGreaterThan(restored.count, 400_000)
        // Finite, sane amplitude.
        XCTAssertTrue(restored.allSatisfy { $0.isFinite })
        let peak = restored.map { Swift.abs($0) }.max() ?? 0
        XCTAssertGreaterThan(peak, 1e-3, "output should not be silent")
        XCTAssertLessThan(peak, 10.0, "output amplitude should be bounded")

        // Compare to the Python CoreML reference if present (waveform cosine —
        // neural-vocoder phase noise keeps this below 1.0 but it should be high).
        let refURL = URL(fileURLWithPath: referenceWAV)
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
            print("E2E Sidon (\(variant.rawValue)) waveform cosine vs Python CoreML reference: \(cos)")
            XCTAssertGreaterThan(cos, 0.95,
                "Swift pipeline should track the Python CoreML output very closely")
        }
    }

    func testRestoreReferenceClip() throws {
        try runRestoreCheck(
            variant: .fp16,
            referenceWAV: "/tmp/sidon-test/restored_coreml_fp16.wav")
    }

    func testRestoreReferenceClipInt8() throws {
        try runRestoreCheck(
            variant: .int8,
            referenceWAV: "/tmp/sidon-test/restored_coreml_int8.wav")
    }

    /// The chunked path (multiple fixed windows) must produce a >window-length
    /// output that stays finite and bounded — exercises window stitching against
    /// the real graphs for audio longer than a single 10 s window.
    func testChunkedRestorePreservesLength() throws {
        guard let bundle = bundleDir(for: .fp16) else {
            throw XCTSkip("Sidon fp16 CoreML artifacts not present")
        }
        guard FileManager.default.fileExists(atPath: referenceClipURL.path) else {
            throw XCTSkip("reference clip \(referenceClipURL.path) not present")
        }
        let clip = try AudioFileLoader.load(
            url: referenceClipURL, targetSampleRate: SpeechRestorer.inputSampleRate)
        // Concatenate the ~10 s clip with itself to force at least two windows.
        let win = SidonConfig.default.windowSamples
        var long = clip
        while long.count <= win { long.append(contentsOf: clip) }

        let restorer = try SpeechRestorer.fromLocalBundle(directory: bundle, variant: .fp16)
        let restored = try restorer.restore(
            audio: long, sampleRate: SpeechRestorer.inputSampleRate)

        // More than one window's worth of output, matching the window/trim contract.
        XCTAssertGreaterThan(restored.count, SidonConfig.default.outputSamplesPerWindow)
        XCTAssertEqual(restored.count, expectedRestoredLength(inputSamples: long.count))
        XCTAssertTrue(restored.allSatisfy { $0.isFinite })
        let peak = restored.map { Swift.abs($0) }.max() ?? 0
        XCTAssertGreaterThan(peak, 1e-3)
        XCTAssertLessThan(peak, 10.0)
    }
}
