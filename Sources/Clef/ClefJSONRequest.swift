import Foundation

/// Input fields use an array so their joint schema order is explicit.
public struct ClefRequest: Decodable {
    struct Field: Decodable {
        let id: String
        let type: String
        let instructions: String?
        let choices: [String: String]?
        let levels: [String]?
        func question() throws -> ClefQuestion {
            let kind: ClefQuestion.Kind
            switch type {
            case "choice":
                guard let choices else { throw ClefError.invalidRequest("choice requires choices") }
                kind = .choice(choices)
            case "score":
                guard let levels else { throw ClefError.invalidRequest("score requires levels") }
                kind = .score(levels)
            case "noul": kind = .noul()
            default: throw ClefError.invalidRequest("Unknown field type: \(type)")
            }
            return .init(id: id, instructions: instructions ?? "", kind: kind)
        }
    }
    public let state: String
    private let questions: [Field]

    public func resolvedQuestions() throws -> [ClefQuestion] {
        try questions.map { try $0.question() }
    }
}
