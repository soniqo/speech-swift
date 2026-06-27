import Foundation
import Hub
import Tokenizers

public final class IndicMioTokenizer: @unchecked Sendable {
    private let tokenizer: Tokenizer

    public init(tokenizer: Tokenizer) {
        self.tokenizer = tokenizer
    }

    public static func load(from directory: URL) async throws -> IndicMioTokenizer {
        let dataURL = directory.appendingPathComponent("tokenizer.json")
        let configURL = directory.appendingPathComponent("tokenizer_config.json")
        guard FileManager.default.fileExists(atPath: dataURL.path) else {
            do {
                return IndicMioTokenizer(
                    tokenizer: try await AutoTokenizer.from(modelFolder: directory, strict: false))
            } catch {
                throw IndicMioError.tokenizerLoadFailed(error.localizedDescription)
            }
        }

        do {
            let hubApi = HubApi()
            let tokenizerData = try hubApi.configuration(fileURL: dataURL)
            let tokenizerConfig: Config
            if FileManager.default.fileExists(atPath: configURL.path) {
                tokenizerConfig = try hubApi.configuration(fileURL: configURL)
            } else {
                tokenizerConfig = Config([String: Config]())
            }
            return IndicMioTokenizer(
                tokenizer: try AutoTokenizer.from(
                    tokenizerConfig: tokenizerConfig,
                    tokenizerData: tokenizerData,
                    strict: false))
        } catch {
            throw IndicMioError.tokenizerLoadFailed(error.localizedDescription)
        }
    }

    public func tokenId(_ token: String) -> Int? {
        tokenizer.convertTokenToId(token)
    }

    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }

    public func decode(_ ids: [Int], skipSpecialTokens: Bool = false) -> String {
        tokenizer.decode(tokens: ids, skipSpecialTokens: skipSpecialTokens)
    }

    public func encodeChatPrompt(text: String) throws -> [Int] {
        encode(try IndicMioPrompt.chatPrompt(for: text))
    }
}
