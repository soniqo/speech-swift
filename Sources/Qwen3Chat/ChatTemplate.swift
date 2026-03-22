import Foundation

/// Qwen3 chat message.
public struct ChatMessage: Sendable {
    public enum Role: String, Sendable {
        case system
        case user
        case assistant
    }

    public let role: Role
    public let content: String

    public init(role: Role, content: String) {
        self.role = role
        self.content = content
    }
}

/// Formats messages into Qwen3 chat template tokens.
///
/// Qwen3 chat format:
/// ```
/// <|im_start|>system
/// {system_message}<|im_end|>
/// <|im_start|>user
/// {user_message}<|im_end|>
/// <|im_start|>assistant
/// ```
enum ChatTemplate {
    // Qwen3 special token IDs
    static let endOfTextId = 151643   // <|endoftext|>
    static let imStartId = 151644
    static let imEndId = 151645
    static let newlineId = 198
    static let thinkStartId = 151667  // <think>
    static let thinkEndId = 151668    // </think>

    /// Strip thinking block from generated tokens.
    ///
    /// Removes tokens from `<think>` through `</think>` (inclusive)
    /// and any trailing newline, returning only the response content.
    static func stripThinking(from tokens: [Int]) -> [Int] {
        guard let startIdx = tokens.firstIndex(of: thinkStartId) else {
            return tokens
        }
        if let endIdx = tokens.firstIndex(of: thinkEndId) {
            // Skip </think> and optional trailing newline
            var afterThink = endIdx + 1
            if afterThink < tokens.count && tokens[afterThink] == newlineId {
                afterThink += 1
            }
            return Array(tokens[0..<startIdx]) + Array(tokens[afterThink...])
        }
        // No </think> found — strip everything from <think> onwards
        return Array(tokens[0..<startIdx])
    }

    /// Encode a conversation into token IDs using Qwen3 chat template.
    ///
    /// - Parameters:
    ///   - enableThinking: If false, injects empty `<think>\n\n</think>\n\n` block
    ///     after the generation prompt so the model skips reasoning and responds directly.
    static func encode(
        messages: [ChatMessage],
        tokenizer: ChatTokenizer,
        addGenerationPrompt: Bool = true,
        enableThinking: Bool = true
    ) -> [Int] {
        var tokens: [Int] = []

        for message in messages {
            // <|im_start|>role\n
            tokens.append(imStartId)
            tokens.append(contentsOf: tokenizer.encode(message.role.rawValue))
            tokens.append(newlineId)

            // content<|im_end|>\n
            tokens.append(contentsOf: tokenizer.encode(message.content))
            tokens.append(imEndId)
            tokens.append(newlineId)
        }

        // Add generation prompt for assistant response
        if addGenerationPrompt {
            tokens.append(imStartId)
            tokens.append(contentsOf: tokenizer.encode("assistant"))
            tokens.append(newlineId)

            // Official Qwen3 enable_thinking=False:
            // Inject empty think block so model starts responding directly.
            if !enableThinking {
                tokens.append(thinkStartId)
                tokens.append(newlineId)
                tokens.append(newlineId)
                tokens.append(thinkEndId)
                tokens.append(newlineId)
                tokens.append(newlineId)
            }
        }

        return tokens
    }
}
