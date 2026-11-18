import XCTest
@testable import ParakeetASR

final class TDTFrameCursorTests: XCTestCase {
    func testNonBlankZeroDurationKeepsTheFrame() {
        var cursor = TDTFrameCursor()
        cursor.advance(isBlank: false, duration: 0)
        XCTAssertEqual(cursor.frame, 0)
        XCTAssertEqual(cursor.symbolsOnFrame, 1)
        cursor.advance(isBlank: false, duration: 0)
        XCTAssertEqual(cursor.frame, 0)
        XCTAssertEqual(cursor.symbolsOnFrame, 2)
        cursor.advance(isBlank: false, duration: 3)
        XCTAssertEqual(cursor.frame, 3)
        XCTAssertEqual(cursor.symbolsOnFrame, 0)
    }

    func testBlankAlwaysMovesAndHonorsPredictedDuration() {
        var cursor = TDTFrameCursor()
        cursor.advance(isBlank: true, duration: 0)
        XCTAssertEqual(cursor.frame, 1)
        cursor.advance(isBlank: true, duration: 4)
        XCTAssertEqual(cursor.frame, 5)
    }

    func testTenSymbolsForceProgressAndResetTheLimit() {
        var cursor = TDTFrameCursor()
        for _ in 0..<9 { cursor.advance(isBlank: false, duration: 0) }
        XCTAssertEqual(cursor.frame, 0)
        XCTAssertEqual(cursor.symbolsOnFrame, 9)
        cursor.advance(isBlank: false, duration: 0)
        XCTAssertEqual(cursor.frame, 1)
        XCTAssertEqual(cursor.symbolsOnFrame, 0)
        cursor.advance(isBlank: false, duration: 0)
        XCTAssertEqual(cursor.frame, 1)
        XCTAssertEqual(cursor.symbolsOnFrame, 1)
    }

    func testPositiveDurationResetsTheSymbolCount() {
        var cursor = TDTFrameCursor()
        for _ in 0..<9 { cursor.advance(isBlank: false, duration: 0) }
        cursor.advance(isBlank: false, duration: 2)
        XCTAssertEqual(cursor.frame, 2)
        cursor.advance(isBlank: false, duration: 0)
        XCTAssertEqual(cursor.frame, 2)
        XCTAssertEqual(cursor.symbolsOnFrame, 1)
    }

    func testBlankResetsTheSymbolCount() {
        var cursor = TDTFrameCursor()
        for _ in 0..<9 { cursor.advance(isBlank: false, duration: 0) }
        cursor.advance(isBlank: true, duration: 0)
        XCTAssertEqual(cursor.frame, 1)
        XCTAssertEqual(cursor.symbolsOnFrame, 0)
    }
}
