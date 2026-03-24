import XCTest
@testable import AudioCommon

final class AudioSampleRingBufferTests: XCTestCase {

    // MARK: - Basic Operations

    func testInitialState() {
        let rb = AudioSampleRingBuffer(capacity: 1000)
        XCTAssertEqual(rb.availableToRead, 0)
        XCTAssertEqual(rb.availableToWrite, 999) // capacity - 1
    }

    func testWriteAndRead() {
        let rb = AudioSampleRingBuffer(capacity: 100)
        let input: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        let written = rb.write(input)
        XCTAssertEqual(written, 5)
        XCTAssertEqual(rb.availableToRead, 5)

        var output = [Float](repeating: 0, count: 5)
        let read = output.withUnsafeMutableBufferPointer { buf in
            rb.read(into: buf.baseAddress!, count: 5)
        }
        XCTAssertEqual(read, 5)
        XCTAssertEqual(output, input)
        XCTAssertEqual(rb.availableToRead, 0)
    }

    func testPartialRead() {
        let rb = AudioSampleRingBuffer(capacity: 100)
        rb.write([1.0, 2.0, 3.0, 4.0, 5.0])

        var output = [Float](repeating: 0, count: 3)
        let read = output.withUnsafeMutableBufferPointer { buf in
            rb.read(into: buf.baseAddress!, count: 3)
        }
        XCTAssertEqual(read, 3)
        XCTAssertEqual(output, [1.0, 2.0, 3.0])
        XCTAssertEqual(rb.availableToRead, 2) // 4.0 and 5.0 still in buffer
    }

    func testReadMoreThanAvailable() {
        let rb = AudioSampleRingBuffer(capacity: 100)
        rb.write([1.0, 2.0])

        var output = [Float](repeating: -1, count: 10)
        let read = output.withUnsafeMutableBufferPointer { buf in
            rb.read(into: buf.baseAddress!, count: 10)
        }
        XCTAssertEqual(read, 2) // Only 2 available
        XCTAssertEqual(output[0], 1.0)
        XCTAssertEqual(output[1], 2.0)
    }

    func testReadFromEmpty() {
        let rb = AudioSampleRingBuffer(capacity: 100)
        var output = [Float](repeating: -1, count: 5)
        let read = output.withUnsafeMutableBufferPointer { buf in
            rb.read(into: buf.baseAddress!, count: 5)
        }
        XCTAssertEqual(read, 0)
    }

    // MARK: - Wrap-around

    func testWrapAround() {
        let rb = AudioSampleRingBuffer(capacity: 10)

        // Fill most of the buffer
        rb.write([1, 2, 3, 4, 5, 6, 7])
        XCTAssertEqual(rb.availableToRead, 7)

        // Read 5 — advances read pointer near end
        var output = [Float](repeating: 0, count: 5)
        output.withUnsafeMutableBufferPointer { buf in
            rb.read(into: buf.baseAddress!, count: 5)
        }
        XCTAssertEqual(output, [1, 2, 3, 4, 5])

        // Write 5 more — wraps around
        rb.write([8, 9, 10, 11, 12])
        XCTAssertEqual(rb.availableToRead, 7) // 6,7 + 8,9,10,11,12

        // Read all — should cross the wrap boundary
        var output2 = [Float](repeating: 0, count: 7)
        output2.withUnsafeMutableBufferPointer { buf in
            rb.read(into: buf.baseAddress!, count: 7)
        }
        XCTAssertEqual(output2, [6, 7, 8, 9, 10, 11, 12])
    }

    // MARK: - Full Buffer

    func testWriteToFullBuffer() {
        let rb = AudioSampleRingBuffer(capacity: 5)
        let written1 = rb.write([1, 2, 3, 4]) // capacity-1 = 4 max
        XCTAssertEqual(written1, 4)
        XCTAssertEqual(rb.availableToWrite, 0)

        // Try to write more — should return 0
        let written2 = rb.write([5, 6])
        XCTAssertEqual(written2, 0)
    }

    func testWritePartialWhenAlmostFull() {
        let rb = AudioSampleRingBuffer(capacity: 5)
        rb.write([1, 2, 3])
        let written = rb.write([4, 5, 6]) // Only 1 slot free
        XCTAssertEqual(written, 1)
        XCTAssertEqual(rb.availableToRead, 4)
    }

    // MARK: - Reset

    func testReset() {
        let rb = AudioSampleRingBuffer(capacity: 100)
        rb.write([1, 2, 3, 4, 5])
        XCTAssertEqual(rb.availableToRead, 5)

        rb.reset()
        XCTAssertEqual(rb.availableToRead, 0)
        XCTAssertEqual(rb.availableToWrite, 99)
    }

    // MARK: - Stress

    func testManySmallWritesAndReads() {
        let rb = AudioSampleRingBuffer(capacity: 256)
        var totalWritten = 0
        var totalRead = 0

        for i in 0..<1000 {
            let chunk: [Float] = [Float(i), Float(i + 1)]
            totalWritten += rb.write(chunk)

            if rb.availableToRead >= 10 {
                var out = [Float](repeating: 0, count: 10)
                totalRead += out.withUnsafeMutableBufferPointer { buf in
                    rb.read(into: buf.baseAddress!, count: 10)
                }
            }
        }

        XCTAssertGreaterThan(totalWritten, 0)
        XCTAssertGreaterThan(totalRead, 0)
    }

    func testConcurrentWriteAndRead() {
        let rb = AudioSampleRingBuffer(capacity: 48000) // 1s at 48kHz
        let writeExpect = XCTestExpectation(description: "writes done")
        let readExpect = XCTestExpectation(description: "reads done")

        var totalWritten = 0
        var totalRead = 0

        // Producer
        DispatchQueue.global(qos: .userInitiated).async {
            for i in 0..<100 {
                let chunk = [Float](repeating: Float(i) * 0.01, count: 480)
                totalWritten += rb.write(chunk)
                usleep(1000) // 1ms
            }
            writeExpect.fulfill()
        }

        // Consumer
        DispatchQueue.global(qos: .userInitiated).async {
            var output = [Float](repeating: 0, count: 256)
            for _ in 0..<200 {
                totalRead += output.withUnsafeMutableBufferPointer { buf in
                    rb.read(into: buf.baseAddress!, count: 256)
                }
                usleep(500)
            }
            readExpect.fulfill()
        }

        wait(for: [writeExpect, readExpect], timeout: 5.0)
        XCTAssertGreaterThan(totalWritten, 0)
        XCTAssertGreaterThan(totalRead, 0)
        // Written >= Read (some may still be in buffer)
        XCTAssertGreaterThanOrEqual(totalWritten, totalRead)
    }
}
