import Foundation

/// MusicGen-style delay pattern and the delayed multi-codebook sampler state
/// machine, matching the cross-validated reference behavior (mlx-audio,
/// vllm-omni, sglang-omni): codebook `k` is offset by `k` steps, BOC rows
/// lead the upper codebooks, and generation stops `N - 2` steps after
/// codebook 0 emits EOC.
public enum HiggsTTSDelayPattern {
    /// Converts raw codec codes `[T][N]` into delayed rows `[T + N - 1][N]`.
    /// Codebook `k` starts with `k` BOC entries and trails into EOC fill.
    public static func apply(
        _ codes: [[Int32]],
        codebooks: Int,
        bocId: Int32,
        eocId: Int32
    ) throws -> [[Int32]] {
        guard !codes.isEmpty else {
            throw HiggsTTSError.invalidCodes("empty code sequence")
        }
        guard codes.allSatisfy({ $0.count == codebooks }) else {
            throw HiggsTTSError.invalidCodes("expected \(codebooks) codebooks per frame")
        }
        let frames = codes.count
        var out = [[Int32]](
            repeating: [Int32](repeating: eocId, count: codebooks),
            count: frames + codebooks - 1)
        for codebook in 0..<codebooks {
            for row in 0..<codebook {
                out[row][codebook] = bocId
            }
            for t in 0..<frames {
                out[codebook + t][codebook] = codes[t][codebook]
            }
        }
        return out
    }

    /// Converts delayed rows `[L][N]` back into raw codec codes `[L - N + 1][N]`.
    public static func reverse(_ delayed: [[Int32]], codebooks: Int) throws -> [[Int32]] {
        guard delayed.allSatisfy({ $0.count == codebooks }) else {
            throw HiggsTTSError.invalidCodes("expected \(codebooks) codebooks per row")
        }
        let frames = delayed.count - codebooks + 1
        guard frames > 0 else {
            throw HiggsTTSError.invalidCodes(
                "delayed rows have L=\(delayed.count), N=\(codebooks); need L >= N")
        }
        var out = [[Int32]](
            repeating: [Int32](repeating: 0, count: codebooks),
            count: frames)
        for codebook in 0..<codebooks {
            for t in 0..<frames {
                out[t][codebook] = delayed[codebook + t][codebook]
            }
        }
        return out
    }
}

/// Per-sequence sampler state for delayed multi-codebook generation.
///
/// Drive it with the raw per-codebook samples for each autoregressive step;
/// `advance` returns the adjusted row to feed back (and record), and flips
/// `isDone` when the EOC countdown completes.
public struct HiggsTTSSamplerState: Sendable {
    public let codebooks: Int
    public let bocId: Int32
    public let eocId: Int32

    public private(set) var delayCount: Int = 0
    public private(set) var eocCountdown: Int? = nil
    public private(set) var isDone: Bool = false

    public init(codebooks: Int, bocId: Int32, eocId: Int32) {
        self.codebooks = codebooks
        self.bocId = bocId
        self.eocId = eocId
    }

    /// Applies the delay-ramp/EOC state machine to one step of sampled codes.
    public mutating func advance(_ sampled: [Int32]) throws -> [Int32] {
        guard sampled.count == codebooks else {
            throw HiggsTTSError.invalidCodes(
                "expected \(codebooks) sampled codes, got \(sampled.count)")
        }
        guard !isDone else { return sampled }

        var codes = sampled
        if delayCount < codebooks {
            let nextCodebook = delayCount + 1
            if nextCodebook < codebooks {
                for index in nextCodebook..<codebooks {
                    codes[index] = bocId
                }
            }
            delayCount += 1
        } else if let countdown = eocCountdown {
            let remaining = countdown - 1
            eocCountdown = remaining
            if remaining <= 0 {
                isDone = true
            }
        } else if codes[0] == eocId {
            if codebooks <= 2 {
                isDone = true
            } else {
                eocCountdown = codebooks - 2
            }
        }
        return codes
    }
}
