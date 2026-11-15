@testable import F5TTS
import Foundation
import XCTest

final class F5TTSTokenizerTests: XCTestCase {
    func testVocabularyLineEndingsPreserveTokenIDsBeforeCombiningMarks() throws {
        let symbols = [" ", "!", "a", "\u{0301}", "\u{0305}", "С", "е", "+"]
        for newline in ["\n", "\r\n", "\r"] {
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("f5-vocab-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let url = directory.appendingPathComponent("vocab.txt")
            try (symbols.joined(separator: newline) + newline).write(to: url, atomically: true, encoding: .utf8)

            let tokenizer = try F5TTSTokenizer(vocabURL: url)

            XCTAssertEqual(tokenizer.symbols, symbols, "Line endings must not add vocabulary entries")
            XCTAssertEqual(tokenizer.vocab["\u{0301}"], 3)
            XCTAssertEqual(tokenizer.vocab["\u{0305}"], 4)
            XCTAssertEqual(try tokenizer.encode("a Се+"), [2, 0, 5, 6, 7])
        }
    }
}
