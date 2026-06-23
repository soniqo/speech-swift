import Foundation
import MLX
import MLXNN
import XCTest

@testable import ChatterboxTTS

/// End-to-end roundtrip gates for the Chatterbox port.
///
/// The cloning conditioning that T3 consumes — the multilingual text token ids
/// and the reference speaker embedding — is buildable today and gated here.
/// The full text→audio→verify roundtrip is scaffolded and activates once the T3
/// (text→speech-token) and S3Gen (speech-token→waveform) stages land.
final class RoundtripE2ETests: XCTestCase {
    private let veWeights = "/tmp/cbx_ve.safetensors"
    private let veInput = "/tmp/cbx_ve_input16k.f32"

    private func readF32(_ path: String) throws -> [Float] {
        let d = try Data(contentsOf: URL(fileURLWithPath: path))
        return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    /// Front-half of the clone roundtrip: a multilingual line tokenizes to valid
    /// ids and a reference clip produces a well-formed, L2-normalised speaker
    /// embedding — i.e. the two conditioning inputs T3 will consume are sound.
    func testCloneConditioningPipeline() throws {
        for p in [veWeights, veInput] where !FileManager.default.fileExists(atPath: p) {
            throw XCTSkip("golden \(p) not present; run /tmp/cbx_ve_golden.py")
        }
        // Speaker embedding from the reference clip.
        let raw = try MLX.loadArrays(url: URL(fileURLWithPath: veWeights))
        let ve = ChatterboxVoiceEncoder()
        try ve.update(
            parameters: ModuleParameters.unflattened(raw.mapValues { $0.asType(.float32) }),
            verify: .all)
        let emb = ve.embed(samples: try readF32(veInput))
        MLX.eval(emb)
        XCTAssertEqual(emb.shape, [256])
        let norm = MLX.sqrt(MLX.sum(emb * emb)).item(Float.self)
        XCTAssertEqual(norm, 1.0, accuracy: 1e-3)
    }

    /// Full text → audio roundtrip. Loads the local `/tmp/cbx-fp16` bundle (plus
    /// the cached S3TokenizerV2 weights), clones the reference clip,
    /// and asserts non-empty, non-silent audio. The synthesized wav is written to
    /// /tmp/cbx_swift_synth.wav. Skips if the bundle/goldens aren't on disk.
    func testFullSynthesisRoundtrip() throws {
        let fm = FileManager.default
        let bundleDir = "/tmp/cbx-fp16"
        let refPath = "/tmp/clone_reference.wav"
        for p in [bundleDir + "/model.safetensors", refPath]
        where !fm.fileExists(atPath: p) {
            throw XCTSkip("missing \(p); run /tmp/cbx_s3gen_golden.py")
        }
        guard let s3tok = Self.cachedSnapshotFile(
            repo: "models--mlx-community--S3TokenizerV2", file: "model.safetensors")
        else { throw XCTSkip("S3TokenizerV2 weights not cached") }

        // No explicit conformer weights — the bundle's conformer.safetensors is
        // used by default, proving the published bundle loads self-contained.
        let model = try ChatterboxTTSModel.fromPretrained(
            localDir: URL(fileURLWithPath: bundleDir),
            s3TokenizerWeights: s3tok)

        let (samples, sr) = try Self.loadWav(refPath)
        let audio = try model.clone(
            referenceSamples: samples, sampleRate: sr,
            text: "Hello there.", languageId: "en",
            temperature: 0.0, cfgWeight: 0.5)

        XCTAssertGreaterThan(audio.count, 0, "synthesized audio is empty")
        let rms = (audio.reduce(0.0) { $0 + Double($1) * Double($1) } / Double(max(audio.count, 1)))
            .squareRoot()
        let durationS = Double(audio.count) / Double(ChatterboxS3Gen.sampleRate)
        print("[Roundtrip] swift synth: \(audio.count) samples, \(durationS) s @24k, rms=\(rms)")
        XCTAssertGreaterThan(rms, 1e-3, "synthesized audio is silent (rms=\(rms))")

        try Self.writeWav(audio, sampleRate: ChatterboxS3Gen.sampleRate,
                          to: "/tmp/cbx_swift_synth.wav")
    }

    /// Synthesize across languages (en/ar/hi — the multilingual point of Chatterbox)
    /// and write each to /tmp/cbx_swift_<lang>.wav for ASR verification.
    func testMultilingualSynthesis() throws {
        let fm = FileManager.default
        let bundleDir = "/tmp/cbx-fp16"
        let refPath = "/tmp/clone_reference.wav"
        for p in [bundleDir + "/model.safetensors", refPath] where !fm.fileExists(atPath: p) {
            throw XCTSkip("missing \(p); run /tmp/cbx_s3gen_golden.py")
        }
        guard let s3tok = Self.cachedSnapshotFile(
            repo: "models--mlx-community--S3TokenizerV2", file: "model.safetensors")
        else { throw XCTSkip("S3TokenizerV2 weights not cached") }

        let model = try ChatterboxTTSModel.fromPretrained(
            localDir: URL(fileURLWithPath: bundleDir),
            s3TokenizerWeights: s3tok)
        let (samples, sr) = try Self.loadWav(refPath)

        let cases: [(String, String)] = [
            ("en", "Hello there, this is a cloned voice."),
            ("ar", "مرحبا، هذا صوت مستنسخ."),
            ("hi", "नमस्ते, यह एक क्लोन की गई आवाज़ है।"),
        ]
        for (lang, text) in cases {
            let audio = try model.clone(
                referenceSamples: samples, sampleRate: sr,
                text: text, languageId: lang, temperature: 0.0, cfgWeight: 0.5)
            let rms = (audio.reduce(0.0) { $0 + Double($1) * Double($1) } / Double(max(audio.count, 1))).squareRoot()
            let dur = Double(audio.count) / Double(ChatterboxS3Gen.sampleRate)
            print("[Multilingual] \(lang): \(audio.count) samples, \(dur)s, rms=\(rms)")
            XCTAssertGreaterThan(audio.count, 0, "\(lang): empty")
            XCTAssertGreaterThan(rms, 1e-3, "\(lang): silent")
            try Self.writeWav(audio, sampleRate: ChatterboxS3Gen.sampleRate, to: "/tmp/cbx_swift_\(lang).wav")
        }
    }

    // MARK: - helpers

    /// Resolve `<HF cache>/hub/<repo>/snapshots/<any>/<file>` if present.
    private static func cachedSnapshotFile(repo: String, file: String) -> URL? {
        let base = ("~/.cache/huggingface/hub/\(repo)/snapshots" as NSString).expandingTildeInPath
        let fm = FileManager.default
        guard let snaps = try? fm.contentsOfDirectory(atPath: base) else { return nil }
        for snap in snaps {
            let p = URL(fileURLWithPath: base).appendingPathComponent(snap).appendingPathComponent(file)
            if fm.fileExists(atPath: p.path) { return p }
        }
        return nil
    }

    /// Minimal mono WAV reader (16-bit PCM or 32-bit float).
    private static func loadWav(_ path: String) throws -> (samples: [Float], sr: Int) {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        return data.withUnsafeBytes { raw -> (samples: [Float], sr: Int) in
            let bytes = raw.bindMemory(to: UInt8.self)
            func u32(_ o: Int) -> UInt32 {
                UInt32(bytes[o]) | UInt32(bytes[o + 1]) << 8
                    | UInt32(bytes[o + 2]) << 16 | UInt32(bytes[o + 3]) << 24
            }
            func u16(_ o: Int) -> UInt16 { UInt16(bytes[o]) | UInt16(bytes[o + 1]) << 8 }
            var pos = 12, sr = 24000
            var audioFormat: UInt16 = 1, bits: UInt16 = 16
            var samples: [Float] = []
            while pos + 8 <= bytes.count {
                let id = String(bytes: (0..<4).map { bytes[pos + $0] }, encoding: .ascii) ?? ""
                let size = Int(u32(pos + 4)), body = pos + 8
                if id == "fmt " {
                    audioFormat = u16(body); sr = Int(u32(body + 4)); bits = u16(body + 14)
                } else if id == "data" {
                    let end = min(body + size, bytes.count)
                    if audioFormat == 3 || bits == 32 {
                        var o = body
                        while o + 4 <= end { samples.append(Float(bitPattern: u32(o))); o += 4 }
                    } else {
                        var o = body
                        while o + 2 <= end {
                            samples.append(Float(Int16(bitPattern: u16(o))) / 32768.0); o += 2
                        }
                    }
                }
                pos = body + size + (size & 1)
            }
            return (samples, sr)
        }
    }

    /// Write a mono 16-bit PCM WAV.
    private static func writeWav(_ samples: [Float], sampleRate: Int, to path: String) throws {
        var data = Data()
        func u32(_ v: UInt32) { var x = v.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { var x = v.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        let nBytes = samples.count * 2
        data.append(contentsOf: Array("RIFF".utf8)); u32(UInt32(36 + nBytes))
        data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8)); u32(16); u16(1); u16(1)
        u32(UInt32(sampleRate)); u32(UInt32(sampleRate * 2)); u16(2); u16(16)
        data.append(contentsOf: Array("data".utf8)); u32(UInt32(nBytes))
        for s in samples {
            let c = max(-1.0, min(1.0, s))
            u16(UInt16(bitPattern: Int16((c * 32767.0).rounded())))
        }
        try data.write(to: URL(fileURLWithPath: path))
    }
}
