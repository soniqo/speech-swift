import AudioCommon

/// Shared helpers for diarization post-processing, used by both
/// PyannoteDiarizationPipeline and SortformerDiarizer.
enum DiarizationHelpers {

    /// Merge adjacent segments from the same speaker when the gap is below `minSilence`.
    ///
    /// Segments are grouped per-speaker, merged within each group, then sorted globally.
    static func mergeSegments(
        _ segments: [DiarizedSegment],
        minSilence: Float
    ) -> [DiarizedSegment] {
        guard !segments.isEmpty else { return [] }

        var bySpeaker = [Int: [DiarizedSegment]]()
        for seg in segments {
            bySpeaker[seg.speakerId, default: []].append(seg)
        }

        var merged = [DiarizedSegment]()
        for (spk, spkSegs) in bySpeaker {
            let sorted = spkSegs.sorted { $0.startTime < $1.startTime }
            var current = sorted[0]

            for i in 1..<sorted.count {
                let next = sorted[i]
                if next.startTime - current.endTime < minSilence {
                    current = DiarizedSegment(
                        startTime: current.startTime,
                        endTime: next.endTime,
                        speakerId: spk
                    )
                } else {
                    merged.append(current)
                    current = next
                }
            }
            merged.append(current)
        }

        merged.sort { $0.startTime < $1.startTime }
        return merged
    }

    /// Remap speaker IDs to contiguous 0-based range, preserving order of first appearance.
    static func compactSpeakerIds(_ segments: [DiarizedSegment]) -> [DiarizedSegment] {
        let usedIds = Set(segments.map(\.speakerId)).sorted()
        let idMap = Dictionary(uniqueKeysWithValues: usedIds.enumerated().map { ($1, $0) })
        return segments.map {
            DiarizedSegment(
                startTime: $0.startTime,
                endTime: $0.endTime,
                speakerId: idMap[$0.speakerId] ?? $0.speakerId
            )
        }
    }

    /// Linear interpolation resampling.
    static func resample(_ audio: [Float], from sourceSR: Int, to targetSR: Int) -> [Float] {
        guard sourceSR != targetSR else { return audio }
        let ratio = Double(targetSR) / Double(sourceSR)
        let outputLen = Int(Double(audio.count) * ratio)
        var output = [Float](repeating: 0, count: outputLen)

        for i in 0..<outputLen {
            let srcPos = Double(i) / ratio
            let srcIdx = Int(srcPos)
            let frac = Float(srcPos - Double(srcIdx))

            if srcIdx + 1 < audio.count {
                output[i] = audio[srcIdx] * (1 - frac) + audio[srcIdx + 1] * frac
            } else if srcIdx < audio.count {
                output[i] = audio[srcIdx]
            }
        }

        return output
    }
}
