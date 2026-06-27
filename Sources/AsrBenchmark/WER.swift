import Foundation

public struct WERBreakdown: Sendable {
    public var substitutions: Int
    public var insertions: Int
    public var deletions: Int
    public var referenceWords: Int

    public var totalErrors: Int { substitutions + insertions + deletions }
    public var wer: Double {
        referenceWords == 0 ? 0 : Double(totalErrors) / Double(referenceWords)
    }
}

public struct CERBreakdown: Sendable {
    public var editDistance: Int
    public var referenceCharacters: Int

    public var cer: Double {
        referenceCharacters == 0 ? 0 : Double(editDistance) / Double(referenceCharacters)
    }
}

/// Word error rate via Levenshtein on whitespace-tokenized strings.
///
/// Reference and hypothesis are assumed to be already normalized — see
/// `Normalizer.normalize`. Substitutions, insertions, deletions all count as
/// one edit; WER = (S+I+D) / |reference|.
public enum WER {
    public static func compute(reference: String, hypothesis: String) -> WERBreakdown {
        let ref = reference.split(separator: " ").map(String.init)
        let hyp = hypothesis.split(separator: " ").map(String.init)
        let m = ref.count
        let n = hyp.count

        if m == 0 {
            return WERBreakdown(substitutions: 0, insertions: n, deletions: 0, referenceWords: 0)
        }

        // dp[i][j] = edits to align ref[..i] with hyp[..j].
        // op[i][j] = which operation led there: 0 match/sub, 1 insertion (hyp), 2 deletion (ref).
        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)
        var op = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)
        for i in 0...m { dp[i][0] = i; op[i][0] = 2 }
        for j in 0...n { dp[0][j] = j; op[0][j] = 1 }
        op[0][0] = 0

        for i in 1...m {
            for j in 1...n {
                let cost = ref[i-1] == hyp[j-1] ? 0 : 1
                let sub = dp[i-1][j-1] + cost
                let ins = dp[i][j-1] + 1
                let del = dp[i-1][j] + 1
                if sub <= ins && sub <= del {
                    dp[i][j] = sub; op[i][j] = 0
                } else if ins <= del {
                    dp[i][j] = ins; op[i][j] = 1
                } else {
                    dp[i][j] = del; op[i][j] = 2
                }
            }
        }

        var subs = 0, ins = 0, dels = 0
        var i = m, j = n
        while i > 0 || j > 0 {
            switch op[i][j] {
            case 0:
                if i > 0 && j > 0 && ref[i-1] != hyp[j-1] { subs += 1 }
                i -= 1; j -= 1
            case 1:
                ins += 1; j -= 1
            case 2:
                dels += 1; i -= 1
            default:
                i = 0; j = 0
            }
        }
        return WERBreakdown(substitutions: subs, insertions: ins, deletions: dels, referenceWords: m)
    }
}

/// Character error rate using normalized text with whitespace removed.
///
/// This matches the usual ASR benchmark convention for CER: punctuation and
/// casing are already handled by `Normalizer`, then spaces are removed before
/// character-level edit distance is computed.
public enum CER {
    public static func compute(reference: String, hypothesis: String) -> CERBreakdown {
        let ref = Array(reference.replacingOccurrences(of: " ", with: ""))
        let hyp = Array(hypothesis.replacingOccurrences(of: " ", with: ""))
        return CERBreakdown(
            editDistance: editDistance(ref, hyp),
            referenceCharacters: ref.count
        )
    }

    private static func editDistance<T: Equatable>(_ ref: [T], _ hyp: [T]) -> Int {
        let m = ref.count
        let n = hyp.count
        if m == 0 { return n }
        if n == 0 { return m }

        var previous = Array(0...n)
        var current = Array(repeating: 0, count: n + 1)

        for i in 1...m {
            current[0] = i
            for j in 1...n {
                if ref[i - 1] == hyp[j - 1] {
                    current[j] = previous[j - 1]
                } else {
                    current[j] = 1 + min(previous[j], min(current[j - 1], previous[j - 1]))
                }
            }
            swap(&previous, &current)
        }

        return previous[n]
    }
}
