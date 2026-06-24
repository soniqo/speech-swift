import Foundation
import MLX
import Tokenizers
import Hub

/// Front-end that turns `(text, ref_text, lang, instruct, reference audio tokens)`
/// into the conditioning tensor the OmniVoice diffusion backbone consumes.
///
/// Mirrors `OmniVoice._prepare_inference_inputs` in the reference
/// (`omnivoice/models/omnivoice.py`). The conditional `cond_input_ids` is the
/// concatenation, along the time axis, of four `[1, 8, ·]` blocks:
///
///   1. **style** — `<|denoise|>` (when cloning) +
///      `<|lang_start|>{lang}<|lang_end|>` +
///      `<|instruct_start|>{instruct}<|instruct_end|>`, tokenized then *repeated*
///      across all 8 codebook rows (text ids are identical on every row).
///   2. **text** — `<|text_start|>{combined}<|text_end|>` where `combined` is
///      `ref_text + " " + text` (cleaned), likewise repeated across 8 rows.
///   3. **reference audio** — the `[8, R]` codec tokens of the reference clip;
///      each codebook row carries *its own* tokens (NOT repeated).
///   4. **target** — `[8, targetLen]` filled with the audio mask id (1024).
///
/// The audio mask is zero over style+text and one over the reference + target
/// region (`cond_audio_start_idx = total - targetLen - R`).
public struct OmniVoiceInputBuilder {
    /// Qwen3 BPE tokenizer (Qwen2Tokenizer class), loaded from `tokenizer.json`
    /// + `tokenizer_config.json`. Resolves the OmniVoice special tokens through
    /// its added-tokens table — IDs are never hardcoded.
    public let tokenizer: OmniVoiceTokenizer
    private let cfg: OmniVoiceConfig

    public init(tokenizer: OmniVoiceTokenizer, config: OmniVoiceConfig = OmniVoiceConfig()) {
        self.tokenizer = tokenizer
        self.cfg = config
    }

    /// Build the conditioning ids + audio mask for one utterance.
    ///
    /// - Parameters:
    ///   - text: target text to synthesize.
    ///   - refText: reference transcript (voice cloning); `nil` for none.
    ///   - lang: language id string (e.g. `"en"`); empty/`nil` → `"None"`.
    ///   - refAudioTokens: `[8, R]` Int32 reference codec tokens, or `nil`.
    ///   - targetLen: number of audio tokens to generate (target region length).
    ///   - denoise: include `<|denoise|>` (only takes effect when cloning).
    ///   - instruct: style instruction; `nil` → `"None"`.
    /// - Returns: `inputIds` `[1, 8, L]` Int32 and `audioMask` `[1, L]` Int32.
    public func buildInputs(
        text: String,
        refText: String?,
        lang: String?,
        refAudioTokens: MLXArray?,
        targetLen: Int,
        denoise: Bool,
        instruct: String?
    ) -> (inputIds: MLXArray, audioMask: MLXArray) {
        let rows = cfg.numAudioCodebook

        // --- style tokens ------------------------------------------------------
        var styleText = ""
        if denoise && refAudioTokens != nil {
            styleText += "<|denoise|>"
        }
        let langStr = (lang?.isEmpty == false) ? lang! : "None"
        let instructStr = (instruct?.isEmpty == false) ? instruct! : "None"
        styleText += "<|lang_start|>\(langStr)<|lang_end|>"
        styleText += "<|instruct_start|>\(instructStr)<|instruct_end|>"
        let styleIds = tokenizer.encode(styleText)

        // --- text tokens -------------------------------------------------------
        let fullText = Self.combineText(text: text, refText: refText)
        let wrapped = "<|text_start|>\(fullText)<|text_end|>"
        let textIds = tokenizer.encodeWithNonverbalTags(wrapped)

        // Text rows are identical across all 8 codebooks (Python `.repeat(C, 1)`).
        let styleN = styleIds.count
        let textN = textIds.count
        var styleBlock = [Int32](repeating: 0, count: rows * styleN)
        var textBlock = [Int32](repeating: 0, count: rows * textN)
        for r in 0 ..< rows {
            for i in 0 ..< styleN { styleBlock[r * styleN + i] = Int32(styleIds[i]) }
            for i in 0 ..< textN { textBlock[r * textN + i] = Int32(textIds[i]) }
        }
        let style = MLXArray(styleBlock, [rows, styleN])
        let textArr = MLXArray(textBlock, [rows, textN])

        // --- reference + target blocks ----------------------------------------
        let refLen = refAudioTokens?.shape.last ?? 0
        let target = MLXArray(
            [Int32](repeating: Int32(cfg.audioMaskId), count: rows * targetLen),
            [rows, targetLen]
        )

        var parts: [MLXArray] = [style, textArr]
        if let ref = refAudioTokens {
            parts.append(ref.asType(.int32).reshaped([rows, refLen]))
        }
        parts.append(target)

        // [8, L] → [1, 8, L]
        let cond = MLX.concatenated(parts, axis: 1).expandedDimensions(axis: 0)
        let totalLen = styleN + textN + refLen + targetLen
        let audioStart = totalLen - targetLen - refLen

        var maskRow = [Int32](repeating: 0, count: totalLen)
        for i in audioStart ..< totalLen { maskRow[i] = 1 }
        let audioMask = MLXArray(maskRow, [1, totalLen])

        return (cond, audioMask)
    }

    /// Estimate the target audio-token length for `text` at `frameRate` (Hz).
    ///
    /// Sums per-character script weights (see `RuleDurationEstimator`) and scales
    /// by the frame rate. The reference's `RuleDurationEstimator` produces a
    /// *duration*; for a self-contained token estimate we map weight → seconds
    /// using a nominal Latin speaking rate, then seconds → tokens via the codec
    /// frame rate. The real `generate()` may instead derive duration from a
    /// reference clip; the golden gate passes `targetLen` explicitly.
    public func estimateTargetLen(text: String, lang: String?, frameRate: Double) -> Int {
        let weight = RuleDurationEstimator.shared.totalWeight(text)
        // Nominal: one Latin character (weight 1.0) ≈ 45 ms of speech.
        let seconds = weight * 0.045
        return max(1, Int((seconds * frameRate).rounded()))
    }

    // MARK: - Text combination (mirrors `_combine_text`)

    /// `ref_text.strip() + " " + text.strip()` when a reference transcript is
    /// present, else `text.strip()`, followed by the reference's regex cleanups:
    /// strip CR/LF, fullwidth parens → ascii, collapse runs of spaces/tabs, and
    /// drop spaces adjacent to CJK characters.
    static func combineText(text: String, refText: String?) -> String {
        var full: String
        if let ref = refText, !ref.isEmpty {
            full = ref.trimmingCharacters(in: .whitespacesAndNewlines)
                + " "
                + text.trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            full = text.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        // Strip CR/LF.
        full = full.replacingOccurrences(of: "\r", with: "")
        full = full.replacingOccurrences(of: "\n", with: "")

        // Fullwidth parens → ascii.
        full = full.replacingOccurrences(of: "\u{FF08}", with: "(")
        full = full.replacingOccurrences(of: "\u{FF09}", with: ")")

        // Collapse consecutive spaces/tabs.
        full = collapseSpaces(full)

        // Remove spaces directly adjacent to a CJK ideograph (U+4E00…U+9FFF).
        full = stripSpacesAroundCJK(full)

        return full
    }

    private static func collapseSpaces(_ s: String) -> String {
        var out = String.UnicodeScalarView()
        var prevWasSpaceOrTab = false
        for scalar in s.unicodeScalars {
            let isSpaceTab = scalar == " " || scalar == "\t"
            if isSpaceTab {
                if !prevWasSpaceOrTab { out.append(" ") }
                prevWasSpaceOrTab = true
            } else {
                out.append(scalar)
                prevWasSpaceOrTab = false
            }
        }
        return String(out)
    }

    private static func isCJK(_ scalar: Unicode.Scalar) -> Bool {
        scalar.value >= 0x4E00 && scalar.value <= 0x9FFF
    }

    private static func stripSpacesAroundCJK(_ s: String) -> String {
        let scalars = Array(s.unicodeScalars)
        var keep = [Bool](repeating: true, count: scalars.count)
        for i in 0 ..< scalars.count {
            let sc = scalars[i]
            guard sc == " " || sc == "\t" || sc.properties.isWhitespace else { continue }
            // Match Python: `(?<=CJK)\s+ | \s+(?=CJK)` — drop whitespace that
            // immediately follows or precedes a CJK character.
            let prevCJK = i > 0 && isCJK(scalars[i - 1])
            let nextCJK = i + 1 < scalars.count && isCJK(scalars[i + 1])
            if prevCJK || nextCJK { keep[i] = false }
        }
        var out = String.UnicodeScalarView()
        for i in 0 ..< scalars.count where keep[i] { out.append(scalars[i]) }
        return String(out)
    }
}
