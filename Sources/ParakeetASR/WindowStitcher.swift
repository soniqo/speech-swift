import AudioCommon
import Foundation

enum ParakeetWindowPlan {
    /// Retain the original cuts and insert context windows halfway between
    /// them. Short inputs keep a single pass, and every start is hop-aligned.
    static func starts(validFrames: Int, maxWindow: Int, overlap: Bool) -> [Int] {
        precondition(validFrames >= 0 && maxWindow > 0)
        var result: [Int] = []
        for start in stride(from: 0, to: validFrames, by: maxWindow) {
            result.append(start)
            if overlap && validFrames > maxWindow && maxWindow > 1 {
                let context = start + maxWindow / 2
                if context < validFrames { result.append(context) }
            }
        }
        return result
    }

    /// Wider statistics for an empty primary window, without an unbounded
    /// full-recording feature allocation.
    static func normalizationContext(start: Int, length: Int, total: Int, limit: Int) -> Range<Int> {
        precondition(start >= 0 && length > 0 && start + length <= total && limit >= length)
        let count = min(total, limit)
        let first = max(0, min(start - (count - length) / 2, total - count))
        return first..<(first + count)
    }
}

/// Join overlapping hypotheses through matching words near the same time.
/// Empty windows leave the previous hypothesis and its coverage intact.
struct ParakeetWindowStitcher {
    private(set) var words: [ParakeetDecodedWord] = []
    private(set) var coveredEnd: Double?
    private var previousWasPrimary: Bool?
    private let edgeContext: Double
    private static let timeTolerance = 0.6

    init(edgeContext: Double = 0.32) {
        self.edgeContext = edgeContext
    }

    private enum Step { case none, up, left, match }
    private struct Cell {
        var weight = 0.0
        var count = 0
        var error = 0.0
        var step = Step.none
    }
    private struct AnchorWord {
        let index: Int
        let key: String
        let midpoint: Double
    }
    private enum PrimaryPreference { case previous, incoming, neither }

    mutating func append(_ incoming: [ParakeetDecodedWord], start: Double, end: Double, primary: Bool? = nil) {
        guard !incoming.isEmpty else { return }
        guard !words.isEmpty, let previousEnd = coveredEnd else {
            words = incoming
            coveredEnd = end
            previousWasPrimary = primary
            return
        }
        // A shorter suffix adds no acoustic context after a successful window
        // that already reaches the end. Keep the complete hypothesis.
        guard end > previousEnd else { return }
        var boundary = (previousEnd + start) / 2
        let preference: PrimaryPreference
        if previousWasPrimary == true && primary == false {
            boundary = min(previousEnd - edgeContext, words.last!.value.endTime ?? words.last!.midpoint)
            preference = .previous
        } else if previousWasPrimary == false && primary == true {
            boundary = max(start, incoming[0].value.startTime ?? incoming[0].midpoint)
            preference = .incoming
        } else {
            preference = .neither
        }
        let old = words.enumerated().compactMap { index, word -> AnchorWord? in
            guard word.midpoint >= start - Self.timeTolerance else { return nil }
            return AnchorWord(index: index, key: Self.anchorKey(word.value.word), midpoint: word.midpoint)
        }
        let new = incoming.enumerated().compactMap { index, word -> AnchorWord? in
            guard word.midpoint <= previousEnd + Self.timeTolerance else { return nil }
            return AnchorWord(index: index, key: Self.anchorKey(word.value.word), midpoint: word.midpoint)
        }
        guard !old.isEmpty && !new.isEmpty else {
            words += incoming
            coveredEnd = end
            previousWasPrimary = primary
            return
        }

        let width = new.count + 1
        var cells = [Cell](repeating: Cell(), count: (old.count + 1) * width)
        for i in 1...old.count {
            for j in 1...new.count {
                var best = cells[(i - 1) * width + j]
                best.step = .up
                var left = cells[i * width + j - 1]
                left.step = .left
                if Self.better(left, than: best) { best = left }
                let a = old[i - 1], b = new[j - 1]
                let distance = abs(a.midpoint - b.midpoint)
                if !a.key.isEmpty && a.key == b.key && distance < Self.timeTolerance {
                    var match = cells[(i - 1) * width + j - 1]
                    match.weight += 1 - distance / Self.timeTolerance
                    match.count += 1
                    match.error += distance
                    match.step = .match
                    if Self.better(match, than: best) { best = match }
                }
                cells[i * width + j] = best
            }
        }

        var i = old.count, j = new.count
        var anchors: [(old: Int, new: Int)] = []
        while i > 0 && j > 0 {
            switch cells[i * width + j].step {
            case .match:
                anchors.append((old[i - 1].index, new[j - 1].index))
                i -= 1; j -= 1
            case .up: i -= 1
            case .left: j -= 1
            case .none: i = 0
            }
        }
        anchors.reverse()
        let quality = cells[old.count * width + new.count].weight
        let reliable = quality >= 0.75 && (anchors.count >= 2 ||
            (anchors.count == 1 && Self.anchorKey(words[anchors[0].old].value.word).count >= 5))
        // A shared word aligns the hypotheses at that point; it does not
        // establish that their preceding or following phrases agree. Protect
        // a more confident primary phrase from a conflicting context window.
        let eligible = reliable ? anchors.filter {
            preservesPrimaryWords(at: $0, anchors: anchors, incoming: incoming,
                boundary: boundary, preference: preference, overlapStart: old[0].index)
        } : []
        if reliable, let anchor = eligible.min(by: {
            abs((words[$0.old].midpoint + incoming[$0.new].midpoint) / 2 - boundary) <
                abs((words[$1.old].midpoint + incoming[$1.new].midpoint) / 2 - boundary)
        }) {
            words = Array(words[...anchor.old]) + incoming.dropFirst(anchor.new + 1)
        } else {
            let kept = words.filter { $0.midpoint < boundary }
            if preference == .previous, let last = kept.last,
                incoming.first(where: { $0.midpoint >= boundary }).map({ last.value.confidence >= $0.value.confidence }) ?? true {
                // Do not keep half of a primary word and then add a second
                // hypothesis for the same time span on the other side.
                boundary = max(boundary, min(previousEnd, last.value.endTime ?? last.midpoint))
            }
            words = kept + incoming.filter { $0.midpoint >= boundary }
        }
        coveredEnd = end
        previousWasPrimary = primary
    }

    private func preservesPrimaryWords(
        at anchor: (old: Int, new: Int), anchors: [(old: Int, new: Int)],
        incoming: [ParakeetDecodedWord], boundary: Double, preference: PrimaryPreference, overlapStart: Int
    ) -> Bool {
        switch preference {
        case .incoming:
            let primary = incoming[...anchor.new]
            let context = words[overlapStart...anchor.old].filter { $0.midpoint >= boundary }
            guard Self.confidence(primary) >= Self.confidence(context) else { return true }
            let matched = anchors.filter { $0.old <= anchor.old && $0.new <= anchor.new }
            let oldMatches = Set(matched.map(\.old)), newMatches = Set(matched.map(\.new))
            return incoming[..<anchor.new].indices.allSatisfy { newMatches.contains($0) }
                && words[overlapStart...anchor.old].indices.allSatisfy {
                    words[$0].midpoint < boundary || oldMatches.contains($0)
                }
        case .previous:
            let primary = words[anchor.old...].filter { $0.midpoint < boundary }
            let context = incoming[anchor.new...].filter { $0.midpoint < boundary }
            guard Self.confidence(primary) >= Self.confidence(context) else { return true }
            let matched = anchors.filter { $0.old > anchor.old && $0.new > anchor.new }
            let oldMatches = Set(matched.map(\.old)), newMatches = Set(matched.map(\.new))
            return words[(anchor.old + 1)...].indices.allSatisfy {
                words[$0].midpoint >= boundary || oldMatches.contains($0)
            } && incoming[(anchor.new + 1)...].indices.allSatisfy {
                incoming[$0].midpoint >= boundary || newMatches.contains($0)
            }
        case .neither:
            return true
        }
    }

    private static func confidence<C: Collection>(_ values: C) -> Double where C.Element == ParakeetDecodedWord {
        guard !values.isEmpty else { return -.infinity }
        let mean = values.reduce(0.0) { $0 + log(max(Double($1.value.confidence), 1e-30)) } / Double(values.count)
        // Inputs are Float32 confidences. Keep ties independent of how many
        // words contributed to the sum, below their available precision.
        return (mean * 1e10).rounded(.toNearestOrEven)
    }

    /// Different windows can place the same word on slightly different frames.
    /// Keep output starts monotonic and all reported times within the audio.
    func finalized(audioDuration: Double) -> [ParakeetDecodedWord] {
        var previousStart = 0.0
        return words.map { word in
            let start = min(audioDuration, max(previousStart, word.value.startTime ?? 0))
            let end = min(audioDuration, max(start, word.value.endTime ?? start))
            previousStart = start
            return ParakeetDecodedWord(
                value: WordConfidence(word: word.value.word, confidence: word.value.confidence,
                    startTime: start, endTime: end),
                tokenLogProbs: word.tokenLogProbs)
        }
    }

    private static func better(_ candidate: Cell, than current: Cell) -> Bool {
        let a = (candidate.weight * 1e10).rounded(.toNearestOrEven)
        let b = (current.weight * 1e10).rounded(.toNearestOrEven)
        if a != b { return a > b }
        if candidate.count != current.count { return candidate.count > current.count }
        return candidate.error < current.error
    }

    private static func anchorKey(_ text: String) -> String {
        var key = text.lowercased().precomposedStringWithCanonicalMapping
        for (source, target) in [("’", "'"), ("‘", "'"), ("–", "-"), ("—", "-")] {
            key = key.replacingOccurrences(of: source, with: target)
        }
        return key.trimmingCharacters(in: .punctuationCharacters).trimmingCharacters(in: .whitespaces)
    }
}
