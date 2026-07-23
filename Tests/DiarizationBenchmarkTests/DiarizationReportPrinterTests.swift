@testable import DiarizationBenchmark
@testable import BenchmarkSupport
import Foundation
import XCTest

final class DiarizationReportPrinterTests: XCTestCase {
    func testTablePrintsSpeakerCountBreakdownSeparately() {
        let report = makeReport()

        let table = DiarizationReportPrinter.table(report)

        XCTAssertTrue(table.contains("DER by reference speaker count"))
        XCTAssertTrue(table.contains("RefSpk"))
        XCTAssertTrue(table.contains("sortformer-session"))
        let breakdown = table.components(
            separatedBy: "DER by reference speaker count (pooled scored speech)").last ?? ""
        let rows = breakdown.split(separator: "\n")
            .map { $0.split(whereSeparator: \.isWhitespace).map(String.init) }
            .filter { $0.first == "sortformer-session" }
        XCTAssertEqual(rows.map { Array($0.prefix(4)) }, [
            ["sortformer-session", "2", "148", "6.57"],
            ["sortformer-session", "3", "74", "10.05"],
            ["sortformer-session", "4", "20", "12.44"],
        ])
    }

    func testJSONIncludesSpeakerCountBreakdown() throws {
        let data = try JSONEncoder().encode(makeReport())
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        let results = try XCTUnwrap(object["results"] as? [[String: Any]])
        let first = try XCTUnwrap(results.first)
        let groups = try XCTUnwrap(
            first["byReferenceSpeakerCount"] as? [[String: Any]])

        XCTAssertEqual(groups.compactMap { $0["referenceSpeakerCount"] as? Int }, [2, 3, 4])
        XCTAssertEqual(groups.compactMap { $0["derPercent"] as? Double }, [6.57, 10.05, 12.44])
    }

    private func makeReport() -> DiarizationReport {
        let groups = [
            group(speakers: 2, files: 148, der: 6.57),
            group(speakers: 3, files: 74, der: 10.05),
            group(speakers: 4, files: 20, der: 12.44),
        ]
        let result = DiarizationEngineResult(
            engine: "sortformer-session",
            files: 242,
            failures: [],
            derPercent: 8.1,
            derMeanPercent: 8.1,
            derMedianPercent: 7.8,
            jerMeanPercent: 10,
            jerMedianPercent: 9,
            missPercent: 3,
            falseAlarmPercent: 2,
            speakerErrorPercent: 3.1,
            totalSpeechSeconds: 1_000,
            missedSpeechSeconds: 30,
            falseAlarmSeconds: 20,
            confusionSeconds: 31,
            speakerCountAccuracyPercent: 90,
            rtfMean: 0.02,
            rtfMedian: 0.02,
            throughputMeanXRT: 50,
            throughputMedianXRT: 50,
            throughputOverallXRT: 50,
            loadElapsedSeconds: 1,
            audioSecondsTotal: 1_500,
            elapsedSecondsTotal: 30,
            peakRSSBytes: 1_000_000,
            rssDeltaBytes: 500_000,
            byReferenceSpeakerCount: groups)
        return DiarizationReport(
            machine: "test",
            manifestPath: "callhome.tsv",
            files: 242,
            collarSeconds: 0.25,
            resolutionSeconds: 0.01,
            results: [result])
    }

    private func group(
        speakers: Int,
        files: Int,
        der: Double
    ) -> DiarizationSpeakerCountResult {
        DiarizationSpeakerCountResult(
            referenceSpeakerCount: speakers,
            files: files,
            derPercent: der,
            missPercent: 2,
            falseAlarmPercent: 1,
            speakerErrorPercent: der - 3,
            totalSpeechSeconds: 100,
            missedSpeechSeconds: 2,
            falseAlarmSeconds: 1,
            confusionSeconds: der - 3,
            speakerCountAccuracyPercent: 90)
    }
}
