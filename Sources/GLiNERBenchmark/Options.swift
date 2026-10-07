import Foundation

struct BenchmarkOptions {
    let model: String
    let cases: String
    let output: String
    let iterations: Int
    let cacheLimitMB: Int?
    let evaluateLayers: Bool
    enum ParseError: Error { case invalidArguments }
    static func parse(_ args: [String]) throws -> Self {
        var values = [String:String]()
        let known: Set<String> = ["--model","--cases","--output","--iterations","--cache-limit-mb","--evaluate-layers"]
        guard args.count % 2 == 0 else { throw ParseError.invalidArguments }
        for i in stride(from:0,to:args.count,by:2) {
            guard known.contains(args[i]), values[args[i]] == nil, !args[i+1].isEmpty else { throw ParseError.invalidArguments }
            values[args[i]] = args[i+1]
        }
        guard let model = values["--model"], let cases = values["--cases"], let output = values["--output"],
              let count = Int(values["--iterations"] ?? "5"), count > 0, count <= 10000 else { throw ParseError.invalidArguments }
        let cache: Int?
        if let value = values["--cache-limit-mb"] {
            guard let parsed = Int(value), parsed >= 0, parsed <= 16384 else { throw ParseError.invalidArguments }
            cache = parsed
        } else { cache = nil }
        let evaluation = values["--evaluate-layers"] ?? "false"
        guard ["true","false"].contains(evaluation) else { throw ParseError.invalidArguments }
        return Self(model:model,cases:cases,output:output,iterations:count,cacheLimitMB:cache,evaluateLayers:evaluation == "true")
    }
}
