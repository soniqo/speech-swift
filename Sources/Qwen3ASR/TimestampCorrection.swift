import Foundation

/// Monotonicity correction for forced alignment timestamps.
public enum TimestampCorrection {

    /// Enforce nondecreasing timestamps while preserving equal word boundaries.
    ///
    /// 1. Find a longest nondecreasing subsequence (O(n log n)), choosing the
    ///    earliest predecessor and endpoint when lengths tie.
    /// 2. For positions outside that subsequence:
    ///    - Small gaps (<=2): nearest-neighbor correction
    ///    - Larger gaps: linear interpolation between LIS anchors
    ///
    /// - Parameter rawIndices: Raw timestamp class indices from argmax
    /// - Returns: Corrected nondecreasing indices
    public static func enforceMonotonicity(_ rawIndices: [Int]) -> [Int] {
        guard rawIndices.count > 1 else { return rawIndices }

        return correct(rawIndices, anchorPositions: longestNondecreasingSubsequencePositions(rawIndices))
    }

    /// Predictions beyond the audio extent must not compete as anchors.
    /// Clamping them first would turn an invalid run into a valid repeated plateau.
    static func enforceMonotonicity(_ rawIndices: [Int], maxIndex: Int) -> [Int] {
        let inRangePositions = rawIndices.indices.filter { rawIndices[$0] <= maxIndex }
        guard !inRangePositions.isEmpty else {
            return rawIndices.map { min($0, maxIndex) }
        }
        let inRangeValues = inRangePositions.map { rawIndices[$0] }
        let anchors = longestNondecreasingSubsequencePositions(inRangeValues)
            .map { inRangePositions[$0] }
        return correct(rawIndices, anchorPositions: anchors)
    }

    private static func correct(_ rawIndices: [Int], anchorPositions: [Int]) -> [Int] {
        let anchorSet = Set(anchorPositions)

        // Build anchor points: (position_in_array, value)
        var anchors: [(pos: Int, val: Int)] = []
        for pos in anchorPositions {
            anchors.append((pos, rawIndices[pos]))
        }

        // If every position is an anchor, the input is already nondecreasing.
        if anchors.count == rawIndices.count {
            return rawIndices
        }

        var corrected = rawIndices

        // Fill gaps between anchors
        var anchorIdx = 0
        var i = 0
        while i < corrected.count {
            if anchorSet.contains(i) {
                // This position is an anchor, keep it
                anchorIdx = anchors.firstIndex(where: { $0.pos == i }) ?? anchorIdx
                i += 1
                continue
            }

            // Find surrounding anchors
            let prevAnchor: (pos: Int, val: Int)?
            let nextAnchor: (pos: Int, val: Int)?

            if anchorIdx < anchors.count && anchors[anchorIdx].pos < i {
                prevAnchor = anchors[anchorIdx]
            } else if anchorIdx > 0 {
                prevAnchor = anchors[anchorIdx - 1]
            } else {
                prevAnchor = nil
            }

            // Find next anchor after position i
            var nextIdx = anchorIdx
            while nextIdx < anchors.count && anchors[nextIdx].pos <= i {
                nextIdx += 1
            }
            nextAnchor = nextIdx < anchors.count ? anchors[nextIdx] : nil

            // Interpolate
            if let prev = prevAnchor, let next = nextAnchor {
                let gapSize = next.pos - prev.pos
                if gapSize <= 3 {
                    // Small gap: nearest neighbor
                    let distToPrev = i - prev.pos
                    let distToNext = next.pos - i
                    corrected[i] = distToPrev <= distToNext ? prev.val : next.val
                } else {
                    // Linear interpolation
                    let t = Float(i - prev.pos) / Float(next.pos - prev.pos)
                    corrected[i] = prev.val + Int(t * Float(next.val - prev.val))
                }
            } else if let prev = prevAnchor {
                // After last anchor: clamp to last anchor value
                corrected[i] = prev.val
            } else if let next = nextAnchor {
                // Before first anchor: clamp to first anchor value
                corrected[i] = next.val
            }

            i += 1
        }

        // Final pass: ensure nondecreasing order.
        for i in 1..<corrected.count {
            if corrected[i] < corrected[i - 1] {
                corrected[i] = corrected[i - 1]
            }
        }

        return corrected
    }

    /// Find a longest nondecreasing subsequence with Qwen's earliest-index ties.
    /// Prefix maxima over timestamp ranks keep the search O(n log n).
    static func longestNondecreasingSubsequencePositions(_ arr: [Int]) -> [Int] {
        guard !arr.isEmpty else { return [] }

        let values = Set(arr).sorted()
        let ranks = Dictionary(uniqueKeysWithValues: values.enumerated().map { ($0.element, $0.offset + 1) })
        var tree = [(length: Int, index: Int)](repeating: (0, -1), count: values.count + 1)
        var parent = [Int](repeating: -1, count: arr.count)
        var longest = (length: 0, index: -1)

        for (i, value) in arr.enumerated() {
            let rank = ranks[value]!
            var predecessor = (length: 0, index: -1)
            var query = rank
            while query > 0 {
                let candidate = tree[query]
                if candidate.length > predecessor.length
                    || (candidate.length == predecessor.length && candidate.index < predecessor.index) {
                    predecessor = candidate
                }
                query -= query & -query
            }
            parent[i] = predecessor.index
            let length = predecessor.length + 1

            var update = rank
            while update < tree.count {
                // Input order already makes the stored index earliest on a tie.
                if length > tree[update].length { tree[update] = (length, i) }
                update += update & -update
            }
            if length > longest.length { longest = (length, i) }
        }

        var positions: [Int] = []
        var idx = longest.index
        while idx != -1 {
            positions.append(idx)
            idx = parent[idx]
        }

        positions.reverse()
        return positions
    }
}
