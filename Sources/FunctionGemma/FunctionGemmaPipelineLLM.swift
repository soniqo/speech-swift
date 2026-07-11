import AudioCommon
import Foundation

/// Bridges ``FunctionGemma`` to ``PipelineLLM`` so a ``VoicePipeline`` can use
/// it as its LLM stage. The bridge ignores everything except the **last user
/// message** — FunctionGemma is a single-turn tool-call model, not a
/// general-purpose chat assistant. The pipeline's ASR result is forwarded as
/// the user text; the tool list registered up front via ``setTools(_:)`` is
/// embedded into the prompt; the function-call grammar is emitted by the
/// model and parsed by ``FunctionGemmaParser`` after generation.
@available(iOS 18.0, macOS 15.0, *)
public final class FunctionGemmaPipelineLLM: PipelineLLM, @unchecked Sendable {

    public let model: FunctionGemma
    public let maxNewTokens: Int

    private var tools: [FunctionDeclaration] = []
    private let cancelLock = NSLock()
    private var cancelled: Bool = false

    public init(model: FunctionGemma, maxNewTokens: Int = 64) {
        self.model = model
        self.maxNewTokens = maxNewTokens
    }

    /// Register the tools the LLM may emit. Call once before ``chat(...)``.
    public func setTools(_ tools: [FunctionDeclaration]) {
        self.tools = tools
    }

    public func chat(messages: [(role: MessageRole, content: String)],
                     onToken: @escaping (String, Bool) -> Void) {
        cancelLock.lock(); cancelled = false; cancelLock.unlock()

        // Find the latest user turn; everything else is dropped because
        // FunctionGemma was not trained for multi-turn assistant memory.
        guard let userText = messages.last(where: { $0.role == .user })?.content else {
            onToken("", true)
            return
        }

        let semaphore = DispatchSemaphore(value: 0)
        var emitted = ""
        var emittedError: Error?
        Task {
            do {
                let text = try await model.generate(prompt: userText,
                                                    tools: tools,
                                                    maxNewTokens: maxNewTokens)
                emitted = text
            } catch {
                emittedError = error
            }
            semaphore.signal()
        }
        semaphore.wait()

        cancelLock.lock(); let wasCancelled = cancelled; cancelLock.unlock()
        if wasCancelled {
            onToken("", true)
            return
        }
        if let err = emittedError {
            onToken("[error: \(err.localizedDescription)]", true)
            return
        }
        // Surface the raw generated text as a single token followed by `is_final`.
        // The VoicePipeline forwards it to TTS verbatim; callers that need the
        // parsed tool call can re-run FunctionGemmaParser.parseFunctionCalls.
        onToken(emitted, true)
    }

    public func cancel() {
        cancelLock.lock(); cancelled = true; cancelLock.unlock()
    }

    /// Parse the LLM's last raw output via ``FunctionGemmaParser``.
    /// Convenience for callers that want both the text and the structured
    /// call array without re-implementing the parsing.
    public func parseLastCalls(from text: String) -> [FunctionCall] {
        FunctionGemmaParser.parseFunctionCalls(text)
    }
}
