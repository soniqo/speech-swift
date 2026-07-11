import Foundation

/// One reference voice: delay-patterned codec codes plus an optional
/// transcript of the reference clip.
public struct HiggsTTSReference: Sendable {
    /// Delayed rows `[T + N - 1][N]` from `HiggsTTSDelayPattern.apply`.
    public let delayedCodes: [[Int32]]
    public let text: String?

    public init(delayedCodes: [[Int32]], text: String? = nil) {
        self.delayedCodes = delayedCodes
        self.text = text
    }
}

/// Prompt token stream with audio-code spans that the model embeds via the
/// fused codebook table instead of the text embedding.
public struct HiggsTTSPrompt: Sendable {
    /// Placeholder id marking positions covered by audio segments.
    public static let audioPlaceholderId: Int32 = -100

    public let tokenIds: [Int32]
    /// `(start, delayedCodes)`: rows of `delayedCodes` replace
    /// `tokenIds[start ..< start + rows]`.
    public let audioSegments: [(start: Int, delayedCodes: [[Int32]])]
}

/// Builds the Higgs TTS 3 generation prompt, matching the reference
/// implementations:
/// `<|tts|>` [ `<|ref_text|>` transcript ] `<|ref_audio|>` codes … `<|text|>`
/// target `<|audio|>` → autoregressive audio frames.
/// Control tags (`<|emotion:...|>` etc.) ride inside the target text and are
/// resolved by the tokenizer as ordinary added tokens.
public struct HiggsTTSPromptBuilder: Sendable {
    public let specials: HiggsTTSSpecialTokens
    public let encode: @Sendable (String) -> [Int32]

    public init(
        specials: HiggsTTSSpecialTokens,
        encode: @escaping @Sendable (String) -> [Int32]
    ) {
        self.specials = specials
        self.encode = encode
    }

    public func build(text: String, references: [HiggsTTSReference] = []) -> HiggsTTSPrompt {
        var ids: [Int32] = [specials.tts]
        var segments: [(start: Int, delayedCodes: [[Int32]])] = []

        for reference in references {
            if let transcript = reference.text, let refText = specials.refText {
                ids.append(refText)
                ids.append(contentsOf: encode(transcript))
            }
            ids.append(specials.refAudio)
            let start = ids.count
            ids.append(contentsOf: [Int32](
                repeating: HiggsTTSPrompt.audioPlaceholderId,
                count: reference.delayedCodes.count))
            segments.append((start: start, delayedCodes: reference.delayedCodes))
        }

        ids.append(specials.text)
        ids.append(contentsOf: encode(text))
        ids.append(specials.audio)
        return HiggsTTSPrompt(tokenIds: ids, audioSegments: segments)
    }
}
