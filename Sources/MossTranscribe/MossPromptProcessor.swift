import Foundation
import Tokenizers

struct MossProcessorConfiguration: Decodable, Sendable {
    let audioTokensPerSecond: Double
    let audioMergeSize: Int
    let timeMarkerEverySeconds: Int
    let enableTimeMarker: Bool

    enum CodingKeys: String, CodingKey {
        case audioTokensPerSecond = "audio_tokens_per_second"
        case audioMergeSize = "audio_merge_size"
        case timeMarkerEverySeconds = "time_marker_every_seconds"
        case enableTimeMarker = "enable_time_marker"
    }
}

struct MossPreparedPrompt: Sendable, Equatable {
    let inputIDs: [Int]
    let audioPlaceholderCount: Int
    let eosTokenID: Int
}

struct MossPromptProcessor {
    static let audioPadToken = "<|audio_pad|>"
    static let audioStartToken = "<|audio_start|>"
    static let audioEndToken = "<|audio_end|>"
    static let eosToken = "<|im_end|>"

    static let defaultInstruction =
        "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
        + "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
        + "并在段末标注结束时间戳，以清晰标明该段语音范围。"

    let tokenizer: Tokenizer
    let configuration: MossProcessorConfiguration
    let audioTokenID: Int
    let eosTokenID: Int
    private let digitTokenIDs: [Character: Int]

    init(
        tokenizer: Tokenizer,
        configuration: MossProcessorConfiguration
    ) throws {
        self.tokenizer = tokenizer
        self.configuration = configuration

        guard let audioTokenID = tokenizer.convertTokenToId(Self.audioPadToken) else {
            throw MossTranscribeError.missingTokenizerToken(Self.audioPadToken)
        }
        guard let eosTokenID = tokenizer.convertTokenToId(Self.eosToken) else {
            throw MossTranscribeError.missingTokenizerToken(Self.eosToken)
        }
        self.audioTokenID = audioTokenID
        self.eosTokenID = eosTokenID

        var digits: [Character: Int] = [:]
        for digit in "0123456789" {
            let ids = tokenizer.encode(
                text: String(digit),
                addSpecialTokens: false
            )
            guard ids.count == 1, let id = ids.first else {
                throw MossTranscribeError.invalidConfiguration(
                    "digit \(digit) must encode to exactly one tokenizer token"
                )
            }
            digits[digit] = id
        }
        digitTokenIDs = digits
    }

    func prepare(
        audioTokenCount: Int,
        instruction: String = Self.defaultInstruction
    ) throws -> MossPreparedPrompt {
        guard audioTokenCount > 0 else {
            throw MossTranscribeError.invalidAudio(
                "audio must produce at least one encoder token"
            )
        }

        let prompt = Self.renderPrompt(instruction: instruction)
        guard let placeholderRange = prompt.range(of: Self.audioPadToken) else {
            throw MossTranscribeError.invalidConfiguration(
                "rendered prompt is missing \(Self.audioPadToken)"
            )
        }
        let remaining = prompt[placeholderRange.upperBound...]
        guard !remaining.contains(Self.audioPadToken) else {
            throw MossTranscribeError.invalidConfiguration(
                "rendered prompt contains more than one audio placeholder"
            )
        }

        let prefix = String(prompt[..<placeholderRange.lowerBound])
        let suffix = String(prompt[placeholderRange.upperBound...])
        var inputIDs = tokenizer.encode(
            text: prefix,
            addSpecialTokens: false
        )
        let audioSpan = makeAudioSpan(audioTokenCount: audioTokenCount)
        inputIDs.append(contentsOf: audioSpan)
        inputIDs.append(contentsOf: tokenizer.encode(
            text: suffix,
            addSpecialTokens: false
        ))
        return MossPreparedPrompt(
            inputIDs: inputIDs,
            audioPlaceholderCount: audioSpan.lazy.filter {
                $0 == audioTokenID
            }.count,
            eosTokenID: eosTokenID
        )
    }

    static func renderPrompt(
        instruction: String = defaultInstruction
    ) -> String {
        "<|im_start|>system\n"
            + "You are a helpful assistant."
            + "<|im_end|>\n"
            + "<|im_start|>user\n"
            + audioStartToken
            + audioPadToken
            + audioEndToken
            + "\n"
            + instruction
            + "<|im_end|>\n"
            + "<|im_start|>assistant\n"
    }

    func makeAudioSpan(audioTokenCount: Int) -> [Int] {
        Self.makeAudioSpan(
            audioTokenCount: audioTokenCount,
            audioTokenID: audioTokenID,
            digitTokenIDs: digitTokenIDs,
            configuration: configuration
        )
    }

    static func makeAudioSpan(
        audioTokenCount: Int,
        audioTokenID: Int,
        digitTokenIDs: [Character: Int],
        configuration: MossProcessorConfiguration
    ) -> [Int] {
        guard
            configuration.enableTimeMarker,
            audioTokenCount > 0,
            configuration.timeMarkerEverySeconds > 0
        else {
            return [Int](repeating: audioTokenID, count: max(0, audioTokenCount))
        }

        let tokensPerMarker = Int(
            configuration.audioTokensPerSecond
                * Double(configuration.timeMarkerEverySeconds)
        )
        guard tokensPerMarker > 0 else {
            return [Int](repeating: audioTokenID, count: audioTokenCount)
        }

        let duration =
            Double(audioTokenCount) / configuration.audioTokensPerSecond
        var output: [Int] = []
        var consumed = 0
        var seconds = configuration.timeMarkerEverySeconds
        while seconds <= Int(duration) {
            let position =
                (seconds / configuration.timeMarkerEverySeconds)
                * tokensPerMarker
            let segmentLength = position - consumed
            if segmentLength > 0 {
                output.append(
                    contentsOf: repeatElement(
                        audioTokenID,
                        count: segmentLength
                    )
                )
                consumed += segmentLength
            }
            for digit in String(seconds) {
                if let tokenID = digitTokenIDs[digit] {
                    output.append(tokenID)
                }
            }
            seconds += configuration.timeMarkerEverySeconds
        }

        let remainder = audioTokenCount - consumed
        if remainder > 0 {
            output.append(
                contentsOf: repeatElement(
                    audioTokenID,
                    count: remainder
                )
            )
        }
        return output
    }
}
