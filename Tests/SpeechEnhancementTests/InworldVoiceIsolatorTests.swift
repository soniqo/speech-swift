import CoreML
import XCTest
@testable import SpeechEnhancement

final class InworldVoiceIsolatorTests: XCTestCase {
    func testPreferredPushSamplesAlignsWithAnalysisAndCrossfadeBoundaries() {
        XCTAssertEqual(InworldStreamingLayout.preferredPushSamples(analyzed: 0, received: 0, blockFrames: 8), 4_384)
        XCTAssertEqual(InworldStreamingLayout.preferredPushSamples(analyzed: 8, received: 4_384, blockFrames: 8), 3_840)
        XCTAssertEqual(InworldStreamingLayout.preferredPushSamples(analyzed: 398, received: 192_000, blockFrames: 8), 544)
        XCTAssertEqual(InworldStreamingLayout.preferredPushSamples(analyzed: 400, received: 192_544, blockFrames: 8), 3_840)
        XCTAssertEqual(InworldStreamingLayout.preferredPushSamples(analyzed: 598, received: 288_000, blockFrames: 8), 544)
        XCTAssertEqual(InworldStreamingLayout.preferredPushSamples(analyzed: 600, received: 288_544, blockFrames: 8), 3_840)
        XCTAssertEqual(InworldStreamingLayout.preferredPushSamples(analyzed: 0, received: 0, blockFrames: 2), 1_504)
    }

    func testStreamingCachePagesCoverTheLastReservedSlot() {
        XCTAssertEqual(InworldStreamingAttention.cachePageCount(for: 0), 0)
        XCTAssertEqual(InworldStreamingAttention.cachePageCount(for: 1), 1)
        XCTAssertEqual(InworldStreamingAttention.cachePageCount(for: 125), 1)
        XCTAssertEqual(InworldStreamingAttention.cachePageCount(for: 126), 2)
        XCTAssertEqual(InworldStreamingAttention.cachePageCount(for: 1_000), 8)
        XCTAssertEqual(InworldStreamingAttention.pagedSlot(for: 124), 124)
        XCTAssertEqual(InworldStreamingAttention.pagedSlot(for: 125), 128)
        XCTAssertEqual(InworldStreamingAttention.pagedSlot(for: 999), 1_020)
    }

    func testTiledBundleRejectsIncorrectRowLimits() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let manifest = #"{"format":"speech-swift-inworld-voice-isolator-tiled-v1","time_rows":0,"band_rows":96}"#
        try Data(manifest.utf8).write(to: directory.appendingPathComponent("manifest.json"))
        XCTAssertThrowsError(try InworldVoiceIsolator(modelDirectory: directory)) { error in
            XCTAssertTrue(error.localizedDescription.contains("incorrect tiled row limits"))
        }
    }

    func testStreamingRejectsUntrainedLookaheadBeforeLoadingStages() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let bundles = [("stateful", 8), ("padded-attention", 2), ("specialized-attention", 2),
                       ("metal-decoder", 2), ("metal-decoder", 4), ("metal-decoder", 8)]
        for (format, frames) in bundles {
            let cacheFields = format == "metal-decoder" && frames == 8 ? #","kv_cache_bits":12,"key_cache_bits":10"# : ""
            let manifest = """
            {"format":"inworld-voice-isolator-realtime-coreml-\(format)-v1","sample_rate":48000,
             "block_frames":\(frames),"window_frames":1200,"stateful_cache_frames":800,"lookahead_frames_choices":[0,2,4]\(cacheFields)}
            """
            try Data(manifest.utf8).write(to: directory.appendingPathComponent("manifest.json"))
            XCTAssertThrowsError(try InworldVoiceIsolatorStream(modelDirectory: directory, lookaheadFrames: 3)) { error in
                XCTAssertTrue(error.localizedDescription.contains("look-ahead must be 0, 2, or 4 frames"))
            }
        }
    }

    func testStreamingDecoderMapsEachChannelToItsBand() throws {
        let ends = (0..<46).map { UInt32($0 * 80) } + [4_100]
        let bands = try InworldBandDecoder.channelBands(in: decoderBoundaryData(ends))
        XCTAssertEqual(bands.count, 4_100)
        for (band, (start, end)) in zip(ends, ends.dropFirst()).enumerated() {
            XCTAssertTrue(bands[Int(start)..<Int(end)].allSatisfy { $0 == UInt8(band) })
        }
    }

    func testStreamingDecoderRejectsInvalidBoundaries() {
        let ends = (0..<46).map { UInt32($0 * 80) } + [4_100]
        for invalidEnd in [UInt32(4_096), UInt32.max] {
            var invalid = ends
            invalid[46] = invalidEnd
            XCTAssertThrowsError(try InworldBandDecoder.channelBands(in: decoderBoundaryData(invalid)))
        }
        for invalidStart in [UInt32(0), 81, 165] {
            var invalid = ends
            invalid[1] = invalidStart
            XCTAssertThrowsError(try InworldBandDecoder.channelBands(in: decoderBoundaryData(invalid)))
        }
        XCTAssertThrowsError(try InworldBandDecoder.channelBands(in: decoderBoundaryData(Array(ends.dropLast()))))
    }

    private func decoderBoundaryData(_ values: [UInt32]) -> Data {
        values.map(\.littleEndian).withUnsafeBytes { Data($0) }
    }

    func testStreamingRejectsUnsupportedBlockSizeBeforeLoadingStages() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        for (format, frames) in [("padded-attention", 4), ("specialized-attention", 8), ("metal-decoder", 3)] {
            let manifest = """
            {"format":"inworld-voice-isolator-realtime-coreml-\(format)-v1","sample_rate":48000,
             "block_frames":\(frames),"window_frames":1200,"stateful_cache_frames":800,"lookahead_frames_choices":[0,2,4]}
            """
            try Data(manifest.utf8).write(to: directory.appendingPathComponent("manifest.json"))
            XCTAssertThrowsError(try InworldVoiceIsolatorStream(modelDirectory: directory)) { error in
                XCTAssertTrue(error.localizedDescription.contains("expected a v1 realtime manifest"))
            }
        }
    }

    func testFastStreamRejectsUnsupportedCacheEncoding() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let bundles = [
            ("specialized-attention", 2, #""kv_cache_bits":8"#),
            ("stateful", 8, #""kv_cache_bits":10"#),
            ("metal-decoder", 2, #""kv_cache_bits":12,"key_cache_bits":10"#),
        ]
        for (format, frames, cacheFields) in bundles {
            let manifest = """
            {"format":"inworld-voice-isolator-realtime-coreml-\(format)-v1","sample_rate":48000,
             "block_frames":\(frames),"window_frames":1200,"stateful_cache_frames":800,"lookahead_frames_choices":[0,2,4],
             \(cacheFields)}
            """
            try Data(manifest.utf8).write(to: directory.appendingPathComponent("manifest.json"))
            XCTAssertThrowsError(try InworldVoiceIsolatorStream(modelDirectory: directory)) { error in
                XCTAssertTrue(error.localizedDescription.contains("expected a v1 realtime manifest"))
            }
        }
    }

    func testCenteredSTFTOfSingleSample() throws {
        let dsp = try InworldIsolationDSP()
        let spectrum = try dsp.spectrum([1.25][...])
        XCTAssertEqual(spectrum.shape.map(\.intValue), [1, 1, 1025, 2])
        XCTAssertEqual(spectrum[[0, 0, 0, 0]].floatValue, 1.25, accuracy: 1e-6)
        XCTAssertEqual(spectrum[[0, 0, 1, 0]].floatValue, -1.25, accuracy: 1e-6)
        XCTAssertEqual(spectrum[[0, 0, 1, 1]].floatValue, 0, accuracy: 1e-6)
    }

    func testComplexMaskRoundTripKeepsLengthAndAlignment() throws {
        let dsp = try InworldIsolationDSP()
        let signal = (0..<3_150).map { i in Float(sin(Double(i) * 0.07) * 0.2) }
        for audio in [signal[11..<12], signal[11..<1_035], signal[11...]] {
            let spectrum = try dsp.spectrum(audio)
            let frames = spectrum.shape[1]
            let raw = try MLMultiArray(shape: [1, frames, 1025, 4], dataType: .float32)
            raw.withUnsafeMutableBufferPointer(ofType: Float.self) { values, strides in
                for frame in 0..<frames.intValue {
                    for bin in 0..<1_025 {
                        let offset = frame * strides[1] + bin * strides[2]
                        // This real mask delta adds one copy of the input spectrum.
                        values[offset] = atanh(Float(0.5))
                        for i in 1...3 { values[offset + i * strides[3]] = 0 }
                    }
                }
            }
            let output = dsp.synthesize(audio, spectrum: spectrum, maskResidual: raw)
            XCTAssertEqual(output.count, audio.count)
            let maxError = zip(output, audio).map { abs($0 - 2 * $1) }.max() ?? .infinity
            XCTAssertLessThan(maxError, 1e-6)
        }
    }

    func testLastWindowUsesFullContextAndBlendsAtPreviousWindowEnd() throws {
        let window = try InworldIsolationWindow(seconds: 0.01, overlapMs: 1)
        var starts: [Int] = []
        let output = window.render(Array(repeating: 0, count: 700)) { chunk in
            starts.append(chunk.startIndex)
            XCTAssertEqual(chunk.count, 480)
            return Array(repeating: Float(starts.count - 1), count: chunk.count)
        }
        XCTAssertEqual(starts, [0, 220])
        XCTAssertEqual(output.count, 700)
        XCTAssertEqual(output[431], 0)
        XCTAssertEqual(output[432], 0)
        XCTAssertEqual(output[455], 23.0 / 47.0, accuracy: 1e-6)
        XCTAssertEqual(output[479], 1)
        XCTAssertEqual(output[699], 1)
    }

    func testZeroOverlapAndEmptyInput() throws {
        let window = try InworldIsolationWindow(seconds: 0.01, overlapMs: 0)
        let audio = (0..<1_001).map(Float.init)
        var starts: [Int] = []
        XCTAssertEqual(window.render(audio) { chunk in
            starts.append(chunk.startIndex)
            return Array(chunk)
        }, audio)
        XCTAssertEqual(starts, [0, 480, 521])
        XCTAssertEqual(window.render([]) { _ in
            XCTFail("Empty audio reached inference")
            return []
        }, [])
    }

    func testPairedWindowsPreserveCrossfadesAndEndAnchoring() throws {
        let window = try InworldIsolationWindow(seconds: 0.01, overlapMs: 2)
        let audio = (0..<1_200).map(Float.init)
        let process: (ArraySlice<Float>) -> [Float] = { slice in
            slice.map { $0 + Float(slice.startIndex) }
        }
        var batches: [[Int]] = []
        let paired = window.render(audio, pairWindows: true) { slices in
            batches.append(slices.map(\.startIndex))
            return slices.map(process)
        }

        XCTAssertEqual(batches, [[0, 384], [720]])
        XCTAssertEqual(paired, window.render(audio, process: process))
    }

    func testInvalidWindowSettings() throws {
        for seconds in [0, 8.1, .nan, .infinity] {
            XCTAssertThrowsError(try InworldIsolationWindow(seconds: seconds, overlapMs: 0))
        }
        XCTAssertThrowsError(try InworldIsolationWindow(seconds: 8, overlapMs: -1))
        XCTAssertEqual(try InworldIsolationWindow(seconds: 0.01, overlapMs: Int.max).overlap, 240)
    }
}
