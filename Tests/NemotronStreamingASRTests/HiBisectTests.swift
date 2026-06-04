import XCTest
import AudioCommon
@testable import NemotronStreamingASR
import CoreML

/// Diagnostic: dump the raw float32 audio Swift sees for the first N Hindi
/// FLEURS samples, plus the mel + the Swift hypothesis. A sibling Python
/// script reads the same WAVs via soundfile and the same Python mel pipeline,
/// then we compare element-wise to find where the Hindi divergence starts.
final class E2EHiBisect: XCTestCase {

    func testDumpHindiAudioAndMel() async throws {
        let bundlePath = ProcessInfo.processInfo.environment["NEMOTRON_35_LOCAL_BUNDLE"]
            ?? "/tmp/Nemotron-3.5-CoreML-320ms"
        let bundleURL = URL(fileURLWithPath: bundlePath)
        guard FileManager.default.fileExists(atPath: bundleURL.path) else {
            throw XCTSkip("local bundle missing")
        }

        let fleurs = URL(fileURLWithPath: "/Users/ivan/repos/speech-models/benchmarks/fleurs/hi_in")
        let tsvLines = try String(contentsOf: fleurs.appendingPathComponent("test.tsv"))
            .components(separatedBy: .newlines)
            .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

        var picks: [(name: String, url: URL, ref: String)] = []
        for line in tsvLines {
            let cols = line.split(separator: "\t", omittingEmptySubsequences: false)
            guard cols.count >= 4 else { continue }
            let wav = fleurs.appendingPathComponent("audio/test/" + String(cols[1]))
            if FileManager.default.fileExists(atPath: wav.path) {
                picks.append((String(cols[1]), wav, String(cols[3])))
                if picks.count >= 50 { break }
            }
        }

        let outDir = URL(fileURLWithPath: "/tmp/nem35-hi-bisect")
        try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

        let model = try await NemotronStreamingASRModel.fromLocal(bundleDir: bundleURL)
        let pre = StreamingMelPreprocessor(config: model.config)

        var manifest: [[String: Any]] = []
        for (i, p) in picks.enumerated() {
            try autoreleasepool { () -> Void in
            let audio = try AudioFileLoader.load(url: p.url, targetSampleRate: 16000)
            // Dump raw float32 audio (host-endian little-endian on Apple Silicon).
            let audioURL = outDir.appendingPathComponent("\(i)_audio.f32")
            audio.withUnsafeBufferPointer {
                let d = Data(buffer: $0)
                try? d.write(to: audioURL)
            }
            // Dump mel that Swift would feed to the encoder (offline-style:
            // one shot over the whole audio, like compute_mel_chunk would).
            let (melMA, _) = try pre.extractRaw(audio)
            let melCount = melMA.count
            let melURL = outDir.appendingPathComponent("\(i)_mel.f32")
            let melPtr = melMA.dataPointer.assumingMemoryBound(to: Float.self)
            let melBuf = UnsafeBufferPointer(start: melPtr, count: melCount)
            let melData = Data(buffer: melBuf)
            try melData.write(to: melURL)

            // Swift hypothesis (no silence pad, language=hi-IN — matches bench)
            let hyp = try model.transcribeAudio(audio, sampleRate: 16000, language: "hi-IN", padSilence: false)

            manifest.append([
                "i": i, "name": p.name,
                "ref": p.ref,
                "audio_samples": audio.count,
                "audio_path": audioURL.path,
                "mel_shape": [
                    melMA.shape[0].intValue, melMA.shape[1].intValue, melMA.shape[2].intValue
                ],
                "mel_path": melURL.path,
                "swift_hyp": hyp,
            ])
            if i < 3 || i % 10 == 0 {
                print("[\(i)] \(p.name)  audio=\(audio.count)  hyp=\(hyp.prefix(80))")
            }
            }
        }

        let manifestData = try JSONSerialization.data(
            withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
        try manifestData.write(to: outDir.appendingPathComponent("manifest.json"))
        print("dumps → \(outDir.path)")
    }
}
