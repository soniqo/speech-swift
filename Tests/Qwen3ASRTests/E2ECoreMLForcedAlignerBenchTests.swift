#if canImport(CoreML)
import XCTest
import Foundation
import Darwin
import AudioCommon
@testable import Qwen3ASR

/// Local copy of ``AsrBenchmark.currentRSSBytes`` so this test target doesn't
/// take a dependency on the benchmark module.
private func currentRSSBytes() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(
        MemoryLayout<mach_task_basic_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kerr = withUnsafeMutablePointer(to: &info) { ptr -> kern_return_t in
        ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { p in
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), p, &count)
        }
    }
    guard kerr == KERN_SUCCESS else { return 0 }
    return UInt64(info.resident_size)
}

/// Micro-benchmark for the CoreML aligner. Runs alignment 3 times, reports
/// RTF (alignment time / audio duration) and peak RSS observed during the
/// run. Skips unless ``ALIGNER_COREML_LOCAL_DIR`` points at a converter
/// output. Pair with the same env on the MLX path for an apples-to-apples
/// comparison.
final class E2ECoreMLForcedAlignerBenchTests: XCTestCase {

    func testBenchmarkLocalAligner() throws {
        guard let path = ProcessInfo.processInfo.environment["ALIGNER_COREML_LOCAL_DIR"],
              !path.isEmpty else {
            throw XCTSkip("Set ALIGNER_COREML_LOCAL_DIR to a converter output directory")
        }
        let dir = URL(fileURLWithPath: path, isDirectory: true)
        guard FileManager.default.fileExists(atPath: dir.path) else {
            throw XCTSkip("ALIGNER_COREML_LOCAL_DIR=\(path) does not exist")
        }

        guard let wavURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else {
            throw XCTSkip("test_audio.wav not found in Qwen3ASRTests resources")
        }

        let baseRSS = currentRSSBytes()
        let aligner = try CoreMLForcedAligner.fromDirectory(dir)
        let loadedRSS = currentRSSBytes()

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: wavURL)
        let targetSampleRate = 16000
        let audio: [Float] = sampleRate == targetSampleRate
            ? samples
            : AudioFileLoader.resample(samples, from: sampleRate, to: targetSampleRate)
        let audioSeconds = Float(audio.count) / Float(targetSampleRate)

        let referenceText =
            "Can you guarantee that the replacement part will be shipped tomorrow?"

        // Warm-up + 3 timed runs.
        _ = try aligner.align(
            audio: audio, text: referenceText, sampleRate: targetSampleRate, language: "English")

        var rtfs: [Double] = []
        var peakRSS = loadedRSS
        for _ in 0..<3 {
            let start = CFAbsoluteTimeGetCurrent()
            _ = try aligner.align(
                audio: audio, text: referenceText, sampleRate: targetSampleRate, language: "English")
            let elapsedMs = (CFAbsoluteTimeGetCurrent() - start) * 1000
            let rtf = elapsedMs / Double(audioSeconds * 1000)
            rtfs.append(rtf)
            peakRSS = max(peakRSS, currentRSSBytes())
        }
        let median = rtfs.sorted()[1]
        let loadDeltaMB = Double(loadedRSS - baseRSS) / (1024 * 1024)
        let peakMB = Double(peakRSS) / (1024 * 1024)

        print(String(format: "[COREML-ALIGN-BENCH] dir=%@", dir.path))
        print(String(format: "[COREML-ALIGN-BENCH] audio=%.1fs runs=%d rtfs=%@",
                     audioSeconds, rtfs.count,
                     rtfs.map { String(format: "%.3f", $0) }.joined(separator: ",")))
        print(String(format: "[COREML-ALIGN-BENCH] median RTF=%.3f (%.1fx faster than real-time)",
                     median, 1.0 / median))
        print(String(format: "[COREML-ALIGN-BENCH] RSS: base=%.0fMB after-load=%.0fMB peak=%.0fMB Δload=%.0fMB",
                     Double(baseRSS) / (1024 * 1024),
                     Double(loadedRSS) / (1024 * 1024),
                     peakMB, loadDeltaMB))
    }
}
#endif
