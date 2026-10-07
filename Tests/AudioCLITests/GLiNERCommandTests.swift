import ArgumentParser
import XCTest
@testable import AudioCLILib
import GLiNER

final class GLiNERCommandTests: XCTestCase {
    private func parse(_ args: [String]) throws -> ParsableCommand {
        try AudioCLI.parseAsRoot(["gliner"] + args)
    }

    private func assertRejected(_ args: [String], file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try parse(args), file: file, line: line) { error in
            XCTAssertEqual(AudioCLI.exitCode(for: error), .validationFailure, "\(args)", file: file, line: line)
        }
    }

    func testClassifyDefaults() throws {
        let command = try XCTUnwrap(try parse([
            "classify", "Remind me to call Dad at six PM.",
            "--model-dir", "/models/gliner", "--labels", "create_reminder, send_message ,other",
        ]) as? GLiNERClassifyCommand)
        XCTAssertEqual(command.schema.text, "Remind me to call Dad at six PM.")
        XCTAssertEqual(command.schema.modelDir, "/models/gliner")
        XCTAssertEqual(command.schema.parsedLabels, ["create_reminder", "send_message", "other"])
        XCTAssertEqual(command.schema.parsedDescriptions, [:])
        XCTAssertEqual(command.task, "action")
        XCTAssertFalse(command.schema.json)
        XCTAssertFalse(command.schema.evaluateLayers)
    }

    func testExtractOptions() throws {
        let command = try XCTUnwrap(try parse([
            "extract", "Call Alice at six PM.",
            "--model-dir", "m", "--labels", "person,time",
            "--description", "person=Person name = or relative",
            "--description", "time=Time of day",
            "--threshold", "0.3", "--json", "--evaluate-layers",
        ]) as? GLiNERExtractCommand)
        XCTAssertEqual(command.schema.parsedLabels, ["person", "time"])
        // Only the first "=" separates label from description.
        XCTAssertEqual(command.schema.parsedDescriptions,
                       ["person": "Person name = or relative", "time": "Time of day"])
        XCTAssertEqual(command.threshold, 0.3)
        XCTAssertTrue(command.schema.json)
        XCTAssertTrue(command.schema.evaluateLayers)
    }

    func testTextIsOptionalForStdin() throws {
        let command = try XCTUnwrap(try parse([
            "classify", "--model-dir", "m", "--labels", "a,b",
        ]) as? GLiNERClassifyCommand)
        XCTAssertNil(command.schema.text)
    }

    func testDefaultsToPublishedVariant() throws {
        let command = try XCTUnwrap(try parse(["classify", "x", "--labels", "a,b"]) as? GLiNERClassifyCommand)
        XCTAssertNil(command.schema.modelDir)
        XCTAssertNil(command.schema.model)
        XCTAssertEqual(command.schema.variant, "int8")
        let fp32 = try XCTUnwrap(try parse(["extract", "x", "--labels", "time", "--variant", "fp32",
                                            "--model", "org/custom"]) as? GLiNERExtractCommand)
        XCTAssertEqual(fp32.schema.variant, "fp32")
        XCTAssertEqual(fp32.schema.model, "org/custom")
    }

    func testRejectsMissingLabelsUnknownVariantAndConflictingSources() {
        XCTAssertThrowsError(try parse(["extract", "x", "--model-dir", "m"]))
        assertRejected(["classify", "x", "--labels", "a,b", "--variant", "int4"])
        assertRejected(["classify", "x", "--labels", "a,b", "--model", "org/m", "--model-dir", "m"])
    }

    func testRejectsInvalidLabels() {
        for labels in ["a,a", "a,,b", "a, ,b", ",", ""] {
            assertRejected(["classify", "x", "--model-dir", "m", "--labels", labels])
        }
        let tooMany = (0...255).map { "l\($0)" }.joined(separator: ",")
        assertRejected(["extract", "x", "--model-dir", "m", "--labels", tooMany])
    }

    func testRejectsInvalidDescriptions() {
        let base = ["extract", "x", "--model-dir", "m", "--labels", "person,time"]
        assertRejected(base + ["--description", "person"])
        assertRejected(base + ["--description", "=Name"])
        assertRejected(base + ["--description", "person="])
        assertRejected(base + ["--description", "place=Location"])
        assertRejected(base + ["--description", "person=A", "--description", "person=B"])
    }

    func testRejectsInvalidThresholdAndTask() {
        let extract = ["extract", "x", "--model-dir", "m", "--labels", "person"]
        assertRejected(extract + ["--threshold", "1.5"])
        assertRejected(extract + ["--threshold", "-0.1"])
        assertRejected(extract + ["--threshold", "nan"])
        assertRejected(["classify", "x", "--model-dir", "m", "--labels", "a,b", "--task", " "])
    }

    /// The JSON field names are a documented contract for scripts and agents.
    func testJSONOutputFieldNames() throws {
        let choices = try JSONDecoder().decode([GLiNERChoice].self, from: Data(
            #"[{"label":"a","probability":0.25},{"label":"b","probability":0.75}]"#.utf8))
        let spans = try JSONDecoder().decode([GLiNERSpan].self, from: Data(
            #"[{"text":"six PM","score":0.9,"start":25,"end":31}]"#.utf8))
        let timings = GLiNERTimings(loadMs: 1, inferenceMs: 2)
        func keys<T: Encodable>(_ value: T) throws -> Set<String> {
            let object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
            return Set(try XCTUnwrap(object as? [String: Any]).keys)
        }
        XCTAssertEqual(try keys(GLiNERClassifyOutput(
            text: "x", task: "action", label: "b", probability: 0.75, choices: choices, metrics: timings)),
            ["text", "task", "label", "probability", "choices", "metrics"])
        XCTAssertEqual(try keys(GLiNERExtractOutput(
            text: "x", threshold: 0.5, entities: ["time": spans], metrics: timings)),
            ["text", "threshold", "offset_units", "entities", "metrics"])
        XCTAssertEqual(try keys(timings), ["load_ms", "inference_ms"])
        XCTAssertEqual(try keys(spans[0]), ["text", "score", "start", "end"])
    }
}

final class E2EGLiNERCommandTests: XCTestCase {
    func testCommandsRunExportedCheckpoint() async throws {
        guard let directory = ProcessInfo.processInfo.environment["GLINER_MODEL_DIR"] else {
            throw XCTSkip("Set GLINER_MODEL_DIR to an exported GLiNER checkpoint")
        }
        let text = "Remind me to call Dad at six PM."
        let classify = try XCTUnwrap(try AudioCLI.parseAsRoot([
            "gliner", "classify", text, "--model-dir", directory,
            "--labels", "create_reminder,create_calendar_event,send_message,search_notes,set_timer,other",
        ]) as? GLiNERClassifyCommand)
        let (model, loadMs) = try await classify.schema.loadModel()
        XCTAssertGreaterThan(loadMs, 0)
        let decision = try classify.classify(model: model, text: text, loadMs: loadMs)
        XCTAssertEqual(decision.label, "create_reminder")
        XCTAssertEqual(decision.choices.map(\.label), classify.schema.parsedLabels)
        XCTAssertEqual(decision.choices.reduce(0) { $0 + $1.probability }, 1, accuracy: 1e-4)

        let extract = try XCTUnwrap(try AudioCLI.parseAsRoot([
            "gliner", "extract", text, "--model-dir", directory, "--labels", "person,time",
            "--description", "person=Person name or family member mentioned in the command",
            "--description", "time=Time of day or duration mentioned in the command",
        ]) as? GLiNERExtractCommand)
        let spans = try extract.extract(model: model, text: text, loadMs: loadMs)
        XCTAssertEqual(spans.entities["person"]?.map(\.text), ["Dad"])
        XCTAssertEqual(spans.entities["time"]?.map(\.text), ["six PM"])
        let time = try XCTUnwrap(spans.entities["time"]?.first)
        XCTAssertEqual((text as NSString).substring(with: NSRange(location: time.start, length: time.end - time.start)), "six PM")
    }
}
