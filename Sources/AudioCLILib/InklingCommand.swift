import ArgumentParser
import Foundation
import Qwen3Chat

public struct InklingCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(commandName: "inkling",
        abstract: "Generate text with a local Inkling 2 Mini MLX export")
    @Argument(help: "User prompt.") public var prompt: String
    @Option(name: .long, help: "Directory produced by export_inkling_mlx.py.") public var modelDir: String
    @Option(name: .long, help: "System instructions.") public var system = "You are a helpful assistant."
    @Option(name: .long, help: "Optional still-image file attached to the user turn.") public var image: String?
    @Option(name: .long, help: "Maximum generated tokens.") public var maxTokens = 256
    @Option(name: .long, help: "Prompt plus generation token budget.") public var contextLimit = 8192
    @Option(name: .long, help: "Sampling temperature; zero selects greedy decoding.") public var temperature: Float = 1
    @Option(name: .long, help: "Thinking effort in 0...0.99; zero requests a direct answer.") public var reasoningEffort: Float = 0
    public init() {}
    public func validate() throws {
        guard !prompt.isEmpty, maxTokens > 0, contextLimit > maxTokens, contextLimit <= 1048576,
              temperature.isFinite, temperature >= 0,
              reasoningEffort.isFinite, (0...0.99).contains(reasoningEffort) else {
            throw ValidationError("Use a nonempty prompt, positive token budgets, a nonnegative temperature, and effort in 0...0.99.")
        }
    }
    public func run() throws {
        try runAsync {
            let images = try image.map { [try Inkling2Image.load(URL(fileURLWithPath: $0))] } ?? []
            let model = try Inkling2Chat.fromDirectory(URL(fileURLWithPath: modelDir), contextLimit: contextLimit)
            let messages = [ChatMessage(role: .system, content: system), ChatMessage(role: .user, content: prompt)]
            let sampling = ChatSamplingConfig(temperature: temperature, topK: 0, topP: 1,
                                               maxTokens: maxTokens, repetitionPenalty: 1)
            for try await chunk in model.generateStream(messages: messages, reasoningEffort: reasoningEffort,
                                                        sampling: sampling, images: images) {
                FileHandle.standardOutput.write(Data(chunk.utf8))
            }
            FileHandle.standardOutput.write(Data("\n".utf8))
        }
    }
}
