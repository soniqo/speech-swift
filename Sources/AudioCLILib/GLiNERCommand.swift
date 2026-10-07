import ArgumentParser
import Foundation
import GLiNER

public struct GLiNERCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "gliner",
        abstract: "Schema-conditioned text classification and entity spans (GLiNER2, MLX)",
        discussion: """
            Runs fastino/GLiNER2.5-Decide natively in MLX. The published INT8 \
            conversion downloads on first use; --variant selects fp16 or fp32 and \
            --model-dir loads a local bundle. Scores are model confidences, not \
            guarantees; entity spans are mentions, not normalized values.
            """,
        subcommands: [GLiNERClassifyCommand.self, GLiNERExtractCommand.self]
    )

    public init() {}
}

/// Options shared by `gliner classify` and `gliner extract`.
public struct GLiNERSchemaOptions: ParsableArguments {
    @Argument(help: "Input text. Omit to read from stdin.")
    public var text: String?

    @Option(name: .long, help: "Published variant to download: int8 (default), fp16 or fp32.")
    public var variant: String = GLiNER.Variant.int8.rawValue

    @Option(name: .long, help: "Hugging Face repository override for the chosen variant's file layout.")
    public var model: String?

    @Option(name: .long, help: "Local GLiNER export directory; skips downloading.")
    public var modelDir: String?

    @Option(name: .long, help: "Comma-separated candidate labels, e.g. create_reminder,send_message,other.")
    public var labels: String

    @Option(name: .customLong("description"), help: "Label description as label=text. Repeatable; the label must be listed in --labels.")
    public var labelDescriptions: [String] = []

    @Flag(name: .long, help: "Materialize each encoder layer before building the next (lower peak allocations, may be slower).")
    public var evaluateLayers: Bool = false

    @Flag(name: .long, help: "Output JSON with results and timings.")
    public var json: Bool = false

    public init() {}

    /// Labels in the order supplied, whitespace-trimmed.
    public var parsedLabels: [String] {
        labels.split(separator: ",", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces) }
    }

    /// Descriptions keyed by label. Call `validate()` first.
    public var parsedDescriptions: [String: String] {
        var result = [String: String]()
        for entry in labelDescriptions {
            guard let (label, text) = Self.splitDescription(entry) else { continue }
            result[label] = text
        }
        return result
    }

    static func splitDescription(_ entry: String) -> (String, String)? {
        guard let eq = entry.firstIndex(of: "=") else { return nil }
        let label = entry[..<eq].trimmingCharacters(in: .whitespaces)
        let text = entry[entry.index(after: eq)...].trimmingCharacters(in: .whitespaces)
        guard !label.isEmpty, !text.isEmpty else { return nil }
        return (label, text)
    }

    public func validate() throws {
        guard GLiNER.Variant(rawValue: variant) != nil else {
            throw ValidationError("--variant must be fp32, fp16 or int8")
        }
        guard model == nil || modelDir == nil else {
            throw ValidationError("Use either --model or --model-dir, not both")
        }
        let labels = parsedLabels
        guard !labels.contains(where: \.isEmpty) else {
            throw ValidationError("--labels must not contain empty entries")
        }
        guard labels.count <= 255 else {
            throw ValidationError("--labels accepts at most 255 labels")
        }
        guard Set(labels).count == labels.count else {
            throw ValidationError("--labels must be distinct")
        }
        var described = Set<String>()
        for entry in labelDescriptions {
            guard let (label, _) = Self.splitDescription(entry) else {
                throw ValidationError("--description must be label=text, got '\(entry)'")
            }
            guard labels.contains(label) else {
                throw ValidationError("--description refers to '\(label)', which is not in --labels")
            }
            guard described.insert(label).inserted else {
                throw ValidationError("--description given twice for '\(label)'")
            }
        }
    }

    func resolveText() throws -> String {
        let value: String
        if let text { value = text } else {
            let data = FileHandle.standardInput.readDataToEndOfFile()
            value = String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        }
        guard !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ValidationError("No input text. Pass as argument or pipe via stdin.")
        }
        return value
    }

    func loadModel() async throws -> (GLiNER, Double) {
        let start = ContinuousClock.now
        let loaded: GLiNER
        if let modelDir {
            let directory = URL(fileURLWithPath: (modelDir as NSString).expandingTildeInPath)
            guard FileManager.default.fileExists(
                atPath: directory.appendingPathComponent("weights.safetensors").path) else {
                throw ValidationError("No GLiNER export at \(directory.path) (weights.safetensors missing).")
            }
            if !json { FileHandle.standardError.write("Loading GLiNER from \(directory.path)...\n".data(using: .utf8)!) }
            loaded = try await GLiNER.load(from: directory, evaluateLayers: evaluateLayers)
        } else {
            let selected = GLiNER.Variant(rawValue: variant) ?? .int8
            loaded = try await GLiNER.fromPretrained(
                variant: selected, modelID: self.model, evaluateLayers: evaluateLayers,
                progressHandler: json ? nil : { progress, status in
                    FileHandle.standardError.write("\r  \(status) \(Int(progress * 100))%   ".data(using: .utf8)!)
                })
            if !json { FileHandle.standardError.write("\n".data(using: .utf8)!) }
        }
        return (loaded, glinerMilliseconds(since: start))
    }
}

func glinerMilliseconds(since start: ContinuousClock.Instant) -> Double {
    let d = start.duration(to: .now).components
    return Double(d.seconds) * 1000 + Double(d.attoseconds) / 1e15
}

struct GLiNERTimings: Encodable {
    let loadMs: Double
    let inferenceMs: Double
    enum CodingKeys: String, CodingKey { case loadMs = "load_ms", inferenceMs = "inference_ms" }
}

struct GLiNERClassifyOutput: Encodable {
    let text: String
    let task: String
    /// Highest-probability label; the caller decides whether to accept it.
    let label: String
    let probability: Float
    /// All candidates in the order supplied on the command line.
    let choices: [GLiNERChoice]
    let metrics: GLiNERTimings
}

struct GLiNERExtractOutput: Encodable {
    let text: String
    let threshold: Float
    let offsetUnits = "utf16"
    let entities: [String: [GLiNERSpan]]
    let metrics: GLiNERTimings
    enum CodingKeys: String, CodingKey {
        case text, threshold, entities, metrics
        case offsetUnits = "offset_units"
    }
}

func printGLiNERJSON<T: Encodable>(_ value: T) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    print(String(decoding: try encoder.encode(value), as: UTF8.self))
}

public struct GLiNERClassifyCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "classify",
        abstract: "Pick one label for the text and report every label's probability"
    )

    @OptionGroup public var schema: GLiNERSchemaOptions

    @Option(name: .long, help: "Task name placed in the schema prompt.")
    public var task: String = "action"

    public init() {}

    public func validate() throws {
        guard !task.trimmingCharacters(in: .whitespaces).isEmpty else {
            throw ValidationError("--task must not be empty")
        }
    }

    public func run() throws {
        try runAsync {
            let text = try schema.resolveText()
            let (model, loadMs) = try await schema.loadModel()
            let output = try classify(model: model, text: text, loadMs: loadMs)
            if schema.json {
                try printGLiNERJSON(output)
            } else {
                for choice in output.choices.sorted(by: { $0.probability > $1.probability }) {
                    print(String(format: "%.4f  %@", choice.probability, choice.label))
                }
            }
        }
    }

    func classify(model: GLiNER, text: String, loadMs: Double) throws -> GLiNERClassifyOutput {
        let start = ContinuousClock.now
        let choices = try model.classify(
            text, task: task, labels: schema.parsedLabels,
            descriptions: schema.parsedDescriptions)
        let inferenceMs = glinerMilliseconds(since: start)
        guard let best = choices.max(by: { $0.probability < $1.probability }) else {
            throw ValidationError("Model returned no choices")
        }
        return GLiNERClassifyOutput(
            text: text, task: task, label: best.label, probability: best.probability,
            choices: choices, metrics: GLiNERTimings(loadMs: loadMs, inferenceMs: inferenceMs))
    }
}

public struct GLiNERExtractCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "extract",
        abstract: "Extract entity mentions for each label, with scores and UTF-16 offsets"
    )

    @OptionGroup public var schema: GLiNERSchemaOptions

    @Option(name: .long, help: "Minimum span score, 0...1.")
    public var threshold: Float = 0.5

    public init() {}

    public func validate() throws {
        guard threshold.isFinite, (0...1).contains(threshold) else {
            throw ValidationError("--threshold must be between 0 and 1")
        }
    }

    public func run() throws {
        try runAsync {
            let text = try schema.resolveText()
            let (model, loadMs) = try await schema.loadModel()
            let output = try extract(model: model, text: text, loadMs: loadMs)
            if schema.json {
                try printGLiNERJSON(output)
            } else {
                for label in schema.parsedLabels {
                    let spans = output.entities[label] ?? []
                    if spans.isEmpty { print("\(label): (none)") }
                    for span in spans {
                        print(String(format: "%@: \"%@\" [%d, %d) %.4f",
                                     label, span.text, span.start, span.end, span.score))
                    }
                }
            }
        }
    }

    func extract(model: GLiNER, text: String, loadMs: Double) throws -> GLiNERExtractOutput {
        let start = ContinuousClock.now
        let entities = try model.extractEntities(
            text, labels: schema.parsedLabels,
            descriptions: schema.parsedDescriptions, threshold: threshold)
        return GLiNERExtractOutput(
            text: text, threshold: threshold, entities: entities,
            metrics: GLiNERTimings(loadMs: loadMs, inferenceMs: glinerMilliseconds(since: start)))
    }
}
