import Foundation
import AudioCommon

// MARK: - RTTM Format

/// RTTM (Rich Transcription Time Marked) segment for standard diarization evaluation.
public struct RTTMSegment: Sendable {
    public let filename: String
    public let startTime: Float
    public let duration: Float
    public let speakerLabel: String

    public init(filename: String, startTime: Float, duration: Float, speakerLabel: String) {
        self.filename = filename
        self.startTime = startTime
        self.duration = duration
        self.speakerLabel = speakerLabel
    }

    /// Format as standard RTTM line: `SPEAKER <file> 1 <start> <dur> <NA> <NA> <speaker> <NA> <NA>`
    public var rttmLine: String {
        let s = String(format: "%.3f", startTime)
        let d = String(format: "%.3f", duration)
        return "SPEAKER \(filename) 1 \(s) \(d) <NA> <NA> \(speakerLabel) <NA> <NA>"
    }
}

/// Convert diarization result to RTTM format.
public func toRTTM(segments: [DiarizedSegment], filename: String) -> [RTTMSegment] {
    segments.map { seg in
        RTTMSegment(
            filename: filename,
            startTime: seg.startTime,
            duration: seg.duration,
            speakerLabel: "speaker_\(seg.speakerId)"
        )
    }
}

/// Write RTTM segments to string.
public func formatRTTM(_ rttmSegments: [RTTMSegment]) -> String {
    rttmSegments.map(\.rttmLine).joined(separator: "\n")
}

// MARK: - DER Computation

/// Diarization Error Rate result.
public struct DERResult: Sendable {
    /// Total scored speech duration in seconds
    public let totalSpeech: Float
    /// False alarm duration (non-speech classified as speech)
    public let falseAlarm: Float
    /// Missed speech duration
    public let missedSpeech: Float
    /// Speaker confusion duration (wrong speaker assigned)
    public let confusion: Float

    /// Diarization Error Rate = (FA + Miss + Confusion) / TotalSpeech
    public var der: Float {
        guard totalSpeech > 0 else { return 0 }
        return (falseAlarm + missedSpeech + confusion) / totalSpeech
    }

    /// Diarization Error Rate as percentage
    public var derPercent: Float { der * 100 }
}

/// Compute Diarization Error Rate between reference and hypothesis.
///
/// Uses frame-level scoring with configurable resolution and collar.
/// Collar applies forgiveness around reference segment boundaries.
///
/// - Parameters:
///   - reference: reference (ground truth) segments
///   - hypothesis: hypothesis (system output) segments
///   - collar: forgiveness collar in seconds around boundaries (default 0.25s)
///   - resolution: scoring resolution in seconds (default 0.01s = 10ms)
/// - Returns: DER breakdown
public func computeDER(
    reference: [DiarizedSegment],
    hypothesis: [DiarizedSegment],
    collar: Float = 0.25,
    resolution: Float = 0.01
) -> DERResult {
    guard !reference.isEmpty else {
        let hTotal = hypothesis.reduce(Float(0)) { $0 + $1.duration }
        return DERResult(totalSpeech: 0, falseAlarm: hTotal, missedSpeech: 0, confusion: 0)
    }

    // Find time range
    let allSegments: [DiarizedSegment] = reference + hypothesis
    let maxTime = allSegments.map(\.endTime).max()!
    let numFrames = Int(ceil(maxTime / resolution))
    guard numFrames > 0 else {
        return DERResult(totalSpeech: 0, falseAlarm: 0, missedSpeech: 0, confusion: 0)
    }

    // Build collar mask: frames near reference boundaries are excluded from scoring
    var collarMask = [Bool](repeating: false, count: numFrames)
    if collar > 0 {
        for seg in reference {
            let startFrame = Int(seg.startTime / resolution)
            let endFrame = Int(seg.endTime / resolution)
            let collarFrames = Int(collar / resolution)

            for f in max(0, startFrame - collarFrames)..<min(numFrames, startFrame + collarFrames) {
                collarMask[f] = true
            }
            for f in max(0, endFrame - collarFrames)..<min(numFrames, endFrame + collarFrames) {
                collarMask[f] = true
            }
        }
    }

    // Build per-frame speaker sets for reference and hypothesis
    // Use sorted arrays of speaker IDs (faster than Set for small N)
    let refSpeakers = buildFrameSpeakers(segments: reference, numFrames: numFrames, resolution: resolution)
    let hypSpeakers = buildFrameSpeakers(segments: hypothesis, numFrames: numFrames, resolution: resolution)

    // Score each frame
    var totalSpeech: Float = 0
    var falseAlarm: Float = 0
    var missedSpeech: Float = 0
    var confusion: Float = 0

    for f in 0..<numFrames {
        if collarMask[f] { continue }

        let refCount = refSpeakers[f].count
        let hypCount = hypSpeakers[f].count

        if refCount == 0 && hypCount == 0 { continue }

        if refCount == 0 {
            // No reference speech, any hypothesis is false alarm
            falseAlarm += Float(hypCount) * resolution
            continue
        }

        // Reference speech exists — count it
        totalSpeech += Float(refCount) * resolution

        if hypCount == 0 {
            // All reference speech is missed
            missedSpeech += Float(refCount) * resolution
            continue
        }

        // Both have speakers — compute exact match by speaker ID
        let matched = countExactMatched(ref: refSpeakers[f], hyp: hypSpeakers[f])
        let unmatchedRef = refCount - matched  // ref speakers with no matching hyp
        let unmatchedHyp = hypCount - matched  // hyp speakers with no matching ref
        // Confusion: min of unmatched ref/hyp (wrong speaker assigned)
        let conf = min(unmatchedRef, unmatchedHyp)
        // Missed: unmatched ref beyond confusion
        let missed = unmatchedRef - conf
        // False alarm: unmatched hyp beyond confusion
        let fa = unmatchedHyp - conf

        missedSpeech += Float(missed) * resolution
        confusion += Float(conf) * resolution
        falseAlarm += Float(fa) * resolution
    }

    return DERResult(
        totalSpeech: totalSpeech,
        falseAlarm: falseAlarm,
        missedSpeech: missedSpeech,
        confusion: confusion
    )
}

// MARK: - Optimal Speaker Mapping

/// Compute DER with optimal 1-to-1 speaker mapping between reference and hypothesis.
///
/// Uses an exact maximum-overlap assignment between reference and hypothesis
/// speakers, then scores the remapped hypothesis once. The assignment is
/// polynomial in the speaker count and supports unequal speaker sets.
public func computeDERWithOptimalMapping(
    reference: [DiarizedSegment],
    hypothesis: [DiarizedSegment],
    collar: Float = 0.25,
    resolution: Float = 0.01
) -> DERResult {
    let refSpeakers = Set(reference.map(\.speakerId)).sorted()
    let hypSpeakers = Set(hypothesis.map(\.speakerId)).sorted()

    guard !refSpeakers.isEmpty, !hypSpeakers.isEmpty else {
        return computeDER(reference: reference, hypothesis: hypothesis,
                         collar: collar, resolution: resolution)
    }

    return assignedOptimalMapping(
        reference: reference, hypothesis: hypothesis,
        refSpeakers: refSpeakers, hypSpeakers: hypSpeakers,
        collar: collar, resolution: resolution
    )
}

// MARK: - RTTM Parsing

/// Parse RTTM file content into DiarizedSegments.
public func parseRTTM(_ content: String) -> [DiarizedSegment] {
    var segments = [DiarizedSegment]()
    var speakerMap = [String: Int]()
    var nextId = 0

    for line in content.split(separator: "\n") {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty, trimmed.hasPrefix("SPEAKER") else { continue }

        let parts = trimmed.split(whereSeparator: \.isWhitespace).map(String.init)
        guard parts.count >= 8 else { continue }

        guard let start = Float(parts[3]),
              let dur = Float(parts[4]) else { continue }

        let speaker = parts[7]
        if speakerMap[speaker] == nil {
            speakerMap[speaker] = nextId
            nextId += 1
        }

        segments.append(DiarizedSegment(
            startTime: start,
            endTime: start + dur,
            speakerId: speakerMap[speaker]!
        ))
    }

    return segments.sorted { $0.startTime < $1.startTime }
}

// MARK: - Internals

private func buildFrameSpeakers(
    segments: [DiarizedSegment],
    numFrames: Int,
    resolution: Float
) -> [[Int]] {
    var result = [[Int]](repeating: [], count: numFrames)

    for seg in segments {
        let startFrame = max(0, Int(seg.startTime / resolution))
        let endFrame = min(numFrames, Int(seg.endTime / resolution))

        for f in startFrame..<endFrame {
            if !result[f].contains(seg.speakerId) {
                result[f].append(seg.speakerId)
            }
        }
    }

    return result
}

/// Count speakers present in both ref and hyp by exact ID match.
private func countExactMatched(ref: [Int], hyp: [Int]) -> Int {
    var matched = 0
    for rSpk in ref {
        if hyp.contains(rSpk) {
            matched += 1
        }
    }
    return matched
}

private func assignedOptimalMapping(
    reference: [DiarizedSegment],
    hypothesis: [DiarizedSegment],
    refSpeakers: [Int],
    hypSpeakers: [Int],
    collar: Float,
    resolution: Float
) -> DERResult {
    let allSegs: [DiarizedSegment] = reference + hypothesis
    let maxTime = allSegs.map(\.endTime).max()!
    let numFrames = Int(ceil(maxTime / resolution))

    let refFrames = buildFrameSpeakers(segments: reference, numFrames: numFrames, resolution: resolution)
    let hypFrames = buildFrameSpeakers(segments: hypothesis, numFrames: numFrames, resolution: resolution)

    var excluded = [Bool](repeating: false, count: numFrames)
    if collar > 0 {
        let collarFrames = Int(collar / resolution)
        for segment in reference {
            let start = Int(segment.startTime / resolution)
            let end = Int(segment.endTime / resolution)
            for frame in max(0, start - collarFrames)..<min(
                numFrames, start + collarFrames)
            {
                excluded[frame] = true
            }
            for frame in max(0, end - collarFrames)..<min(
                numFrames, end + collarFrames)
            {
                excluded[frame] = true
            }
        }
    }

    // overlap[r][h] is scored time shared by a reference and hypothesis
    // speaker. Maximizing its one-to-one sum minimizes speaker confusion.
    var overlap = [[Int]](repeating: [Int](repeating: 0, count: hypSpeakers.count), count: refSpeakers.count)
    let refIndex = Dictionary(uniqueKeysWithValues: refSpeakers.enumerated().map { ($1, $0) })
    let hypIndex = Dictionary(uniqueKeysWithValues: hypSpeakers.enumerated().map { ($1, $0) })

    for f in 0..<numFrames where !excluded[f] {
        for rSpk in refFrames[f] {
            guard let ri = refIndex[rSpk] else { continue }
            for hSpk in hypFrames[f] {
                guard let hi = hypIndex[hSpk] else { continue }
                overlap[ri][hi] += 1
            }
        }
    }

    var mapping = [Int: Int]()
    for (refIndex, hypIndex) in maximumWeightAssignment(overlap).enumerated() {
        guard let hypIndex else { continue }
        mapping[hypSpeakers[hypIndex]] = refSpeakers[refIndex]
    }

    let maximumSpeakerID = max(refSpeakers.max() ?? 0, hypSpeakers.max() ?? 0)
    let unmappedBase = maximumSpeakerID < Int.max - hypSpeakers.count
        ? maximumSpeakerID + 1
        : Int.min / 2
    let unmapped = Dictionary(uniqueKeysWithValues: hypSpeakers.enumerated().map {
        ($1, unmappedBase + $0)
    })
    let remapped = hypothesis.map { seg in
        DiarizedSegment(
            startTime: seg.startTime,
            endTime: seg.endTime,
            speakerId: mapping[seg.speakerId] ?? unmapped[seg.speakerId]!
        )
    }

    return computeDER(reference: reference, hypothesis: remapped,
                     collar: collar, resolution: resolution)
}

/// Exact maximum-weight bipartite assignment. Rows map to optional columns.
/// A square zero-padded Hungarian solve lets either side contain more speakers.
private func maximumWeightAssignment(_ weights: [[Int]]) -> [Int?] {
    let rowCount = weights.count
    let columnCount = weights.first?.count ?? 0
    guard rowCount > 0, columnCount > 0 else {
        return [Int?](repeating: nil, count: rowCount)
    }

    let size = max(rowCount, columnCount)
    let maximumWeight = weights.lazy.flatMap { $0 }.max() ?? 0
    var rowPotential = [Int](repeating: 0, count: size + 1)
    var columnPotential = [Int](repeating: 0, count: size + 1)
    var matchedRow = [Int](repeating: 0, count: size + 1)
    var previousColumn = [Int](repeating: 0, count: size + 1)
    let infinity = Int.max / 4

    func cost(row: Int, column: Int) -> Int {
        guard row <= rowCount, column <= columnCount else {
            return maximumWeight
        }
        return maximumWeight - weights[row - 1][column - 1]
    }

    for row in 1...size {
        matchedRow[0] = row
        var column = 0
        var minimum = [Int](repeating: infinity, count: size + 1)
        var used = [Bool](repeating: false, count: size + 1)

        repeat {
            used[column] = true
            let currentRow = matchedRow[column]
            var delta = infinity
            var nextColumn = 0
            for candidate in 1...size where !used[candidate] {
                let reducedCost = cost(row: currentRow, column: candidate)
                    - rowPotential[currentRow] - columnPotential[candidate]
                if reducedCost < minimum[candidate] {
                    minimum[candidate] = reducedCost
                    previousColumn[candidate] = column
                }
                if minimum[candidate] < delta {
                    delta = minimum[candidate]
                    nextColumn = candidate
                }
            }
            for candidate in 0...size {
                if used[candidate] {
                    rowPotential[matchedRow[candidate]] += delta
                    columnPotential[candidate] -= delta
                } else {
                    minimum[candidate] -= delta
                }
            }
            column = nextColumn
        } while matchedRow[column] != 0

        repeat {
            let prior = previousColumn[column]
            matchedRow[column] = matchedRow[prior]
            column = prior
        } while column != 0
    }

    var assignment = [Int?](repeating: nil, count: rowCount)
    for column in 1...size {
        let row = matchedRow[column]
        if row > 0, row <= rowCount, column <= columnCount {
            assignment[row - 1] = column - 1
        }
    }
    return assignment
}
