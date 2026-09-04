#if canImport(CoreML)
import CoreML
import XCTest

@testable import Qwen3ASR

/// Regression tests for the CoreML audio-embedding extraction path.
///
/// The bug these pin: `audioEmbeddingFromMultiArray` read the encoder's
/// `audio_embeddings` output with `assumingMemoryBound(to: Float.self)`.
/// That output is **Float16** `[1, 390, 1024]`, so the read walked the
/// buffer at twice the real element stride — every value was two Float16s
/// reinterpreted as one Float32, and row 195 of 390 was the first read
/// entirely past the end of the allocation (~15 s of audio, since the
/// encoder emits 13 tokens per 100 mel frames). Under 15 s it returned
/// garbage silently; over 15 s it segfaulted.
///
/// No model download and no GPU: every case builds its own MLMultiArray.
final class CoreMLEmbeddingExtractionTests: XCTestCase {

    private let hidden = 8

    // MARK: - Helpers

    /// Contiguous `[1, rows, hidden]` array whose value at (row, col) is
    /// `Float(row * hidden + col)` — distinct per slot so a misread is
    /// visible, and small enough to be exactly representable in Float16
    /// (integers are exact only up to 2048) so these assert on the
    /// extraction rather than on half-precision rounding.
    private func makeArray(rows: Int, dataType: MLMultiArrayDataType) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: [1, rows as NSNumber, hidden as NSNumber],
                                 dataType: dataType)
        for r in 0..<rows {
            for c in 0..<hidden {
                a[r * hidden + c] = NSNumber(value: Float(r * hidden + c))
            }
        }
        return a
    }

    private func expected(row: Int) -> [Float] {
        (0..<hidden).map { Float(row * hidden + $0) }
    }

    private func values(of array: MLMultiArray) -> [Float] {
        (0..<array.count).map { array[$0].floatValue }
    }

    // MARK: - Float16 source (the shipped encoder's dtype)

    /// The core regression: a Float16 source must be *converted*, not
    /// reinterpreted. Pre-fix this returned garbage for every row.
    func testFloat16SourceConvertsRatherThanReinterprets() throws {
        let rows = 6
        let embeddings = try makeArray(rows: rows, dataType: .float16)

        for row in 0..<rows {
            let slice = try CoreMLTextDecoder.extractRow(
                from: embeddings, at: row, hidden: hidden)
            XCTAssertEqual(values(of: slice), expected(row: row),
                           "row \(row) misread from a Float16 buffer")
        }
    }

    /// Reading a Float16 buffer as Float32 exhausts it at the halfway row
    /// (here row 32 of 64), which is what segfaulted on device. The last row
    /// must read correctly and in bounds.
    func testFloat16LastRowStaysInBounds() throws {
        let rows = 64
        let embeddings = try makeArray(rows: rows, dataType: .float16)

        let slice = try CoreMLTextDecoder.extractRow(
            from: embeddings, at: rows - 1, hidden: hidden)
        XCTAssertEqual(values(of: slice), expected(row: rows - 1))
    }

    // MARK: - Float32 source (unchanged behaviour)

    func testFloat32SourceUnchanged() throws {
        let rows = 6
        let embeddings = try makeArray(rows: rows, dataType: .float32)

        for row in 0..<rows {
            let slice = try CoreMLTextDecoder.extractRow(
                from: embeddings, at: row, hidden: hidden)
            XCTAssertEqual(values(of: slice), expected(row: row))
        }
    }

    // MARK: - Padded strides (the other ANE output shape)

    /// ANE can hand back rows padded to an alignment boundary. Reading with
    /// `index * hidden` would skew progressively further into the padding.
    func testPaddedRowStrideIsHonoured() throws {
        let rows = 5
        let paddedStride = hidden + 3
        var storage = [Float](repeating: -1, count: rows * paddedStride)
        for r in 0..<rows {
            for c in 0..<hidden {
                storage[r * paddedStride + c] = Float(r * hidden + c)
            }
        }

        try storage.withUnsafeMutableBufferPointer { buf in
            let embeddings = try MLMultiArray(
                dataPointer: buf.baseAddress!,
                shape: [1, rows as NSNumber, hidden as NSNumber],
                dataType: .float32,
                strides: [(rows * paddedStride) as NSNumber, paddedStride as NSNumber, 1],
                deallocator: nil)

            for row in 0..<rows {
                let slice = try CoreMLTextDecoder.extractRow(
                    from: embeddings, at: row, hidden: hidden)
                XCTAssertEqual(values(of: slice), expected(row: row),
                               "row \(row) misread from a padded-stride buffer")
            }
        }
    }

    // MARK: - Bounds

    func testRowCountBeyondBufferIsRejected() throws {
        let embeddings = try makeArray(rows: 4, dataType: .float16)

        XCTAssertThrowsError(
            try CoreMLTextDecoder.validateEmbeddingSource(
                embeddings, hidden: hidden, requiredRows: 5),
            "a length claim past the tensor extent must throw, not read past the buffer")
        XCTAssertNoThrow(
            try CoreMLTextDecoder.validateEmbeddingSource(
                embeddings, hidden: hidden, requiredRows: 4))
    }

    func testWidthMismatchIsRejected() throws {
        let embeddings = try makeArray(rows: 4, dataType: .float16)

        XCTAssertThrowsError(
            try CoreMLTextDecoder.validateEmbeddingSource(
                embeddings, hidden: hidden + 1, requiredRows: 1),
            "a hidden-size disagreement must throw rather than stride wrongly")
    }

    func testNegativeIndexIsRejected() throws {
        let embeddings = try makeArray(rows: 4, dataType: .float16)

        XCTAssertThrowsError(
            try CoreMLTextDecoder.extractRow(
                from: embeddings, at: -1, hidden: hidden))
    }

    func testIndexPastLastRowIsRejected() throws {
        let embeddings = try makeArray(rows: 4, dataType: .float16)

        XCTAssertThrowsError(
            try CoreMLTextDecoder.extractRow(
                from: embeddings, at: 4, hidden: hidden))
    }

    // MARK: - Bulk extraction

    /// `transcribeWithoutMLX` now bulk-extracts every audio row in one call.
    /// It must produce the same values, row-major, that the per-row helper
    /// returns — otherwise the batched prefill feeds the decoder garbage.
    func testBulkExtractionMatchesPerRowExtraction() throws {
        let rows = 10
        let embeddings = try makeArray(rows: rows, dataType: .float16)

        let flat = try CoreMLTextDecoder.extractRows(
            from: embeddings, count: rows, hidden: hidden)
        XCTAssertEqual(flat.count, rows * hidden)

        for row in 0..<rows {
            let start = row * hidden
            XCTAssertEqual(Array(flat[start..<(start + hidden)]), expected(row: row),
                           "bulk row \(row) disagrees with the per-row read")
        }
    }

    func testBulkExtractionOfZeroRowsIsEmpty() throws {
        let embeddings = try makeArray(rows: 4, dataType: .float16)

        let flat = try CoreMLTextDecoder.extractRows(
            from: embeddings, count: 0, hidden: hidden)
        XCTAssertTrue(flat.isEmpty)
    }

    func testBulkExtractionPastBufferIsRejected() throws {
        let embeddings = try makeArray(rows: 4, dataType: .float16)

        XCTAssertThrowsError(
            try CoreMLTextDecoder.extractRows(
                from: embeddings, count: 5, hidden: hidden))
    }
}
#endif
