import Foundation
import Clef

@main
struct ClefCLI {
    static func main() async {
        do {
            let args = Array(CommandLine.arguments.dropFirst())
            guard args.count == 2 else {
                FileHandle.standardError.write(Data("Usage: clef-decide <model-directory> <request.json>\n".utf8))
                Foundation.exit(2)
            }
            let request = try JSONDecoder().decode(ClefRequest.self, from: Data(contentsOf: URL(fileURLWithPath: args[1])))
            let model = try Clef.load(from: URL(fileURLWithPath: args[0]))
            let start = ContinuousClock.now
            let result = try model.decide(state: request.state, questions: request.resolvedQuestions())
            let elapsed = start.duration(to: .now)
            let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
            let fields: [[String: Any]] = result.decisions.map { d in
                var row: [String: Any] = ["id": d.id, "probabilities": d.probabilities,
                                         "selected_option": d.selectedOption, "confidence": d.confidence]
                if let score = d.score { row["score"] = score }
                if let noul = d.noul { row["noul"] = noul }
                return row
            }
            let data = try JSONSerialization.data(withJSONObject: ["input_tokens": result.inputTokens,
                "decision_seconds": seconds, "decisions": fields], options: [.prettyPrinted, .sortedKeys])
            print(String(decoding: data, as: UTF8.self))
        } catch {
            FileHandle.standardError.write(Data("Clef: \(error.localizedDescription)\n".utf8))
            Foundation.exit(1)
        }
    }
}
