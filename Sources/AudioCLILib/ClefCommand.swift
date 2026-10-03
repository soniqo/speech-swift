import ArgumentParser
import Foundation
import Clef

public struct ClefCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "clef",
        abstract: "Choose from allowed answers with local Clef-flash (MLX)",
        subcommands: [ClefDecideCommand.self])
    public init() {}
}

public struct ClefDecideCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "decide", abstract: "Score every field in a JSON request")

    @Argument(help: "JSON file containing state and an ordered questions array.")
    public var request: String

    @Option(name: .long, help: "Local model directory; skips downloading.")
    public var modelDir: String?

    @Flag(name: .long, help: "Use cached model files without network access.")
    public var offline = false

    @Option(name: .long, help: "Maximum encoded input tokens (1–16384).")
    public var maxTokens = 4096

    public init() {}

    public func validate() throws {
        guard (1...16384).contains(maxTokens) else {
            throw ValidationError("--max-tokens must be in 1...16384")
        }
    }

    public func run() throws {
        let input = try JSONDecoder().decode(ClefRequest.self,
            from: Data(contentsOf: URL(fileURLWithPath: request)))
        let questions = try input.resolvedQuestions()
        try runAsync {
            let loadStart = ContinuousClock.now
            let model: Clef
            if let modelDir {
                model = try Clef.load(from: URL(fileURLWithPath: modelDir))
            } else {
                model = try await Clef.fromPretrained(offlineMode: offline)
            }
            let start = ContinuousClock.now
            let result = try model.decide(state: input.state, questions: questions, maxTokens: maxTokens)
            let end = ContinuousClock.now
            func seconds(_ a: ContinuousClock.Instant, _ b: ContinuousClock.Instant) -> Double {
                let d = a.duration(to: b).components
                return Double(d.seconds) + Double(d.attoseconds) / 1e18
            }
            let decisions: [[String: Any]] = result.decisions.map { d in
                var row: [String: Any] = ["id": d.id, "probabilities": d.probabilities,
                    "selected_option": d.selectedOption, "confidence": d.confidence]
                if let score = d.score { row["score"] = score }
                if let noul = d.noul { row["noul"] = noul }
                return row
            }
            let output: [String: Any] = ["input_tokens": result.inputTokens,
                "load_seconds": seconds(loadStart, start),
                "decision_seconds": seconds(start, end), "decisions": decisions]
            let data = try JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
            print(String(decoding: data, as: UTF8.self))
        }
    }
}
