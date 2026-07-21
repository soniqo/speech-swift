import SpeechVAD

/// Pooled diarization metrics for recordings with one reference speaker count.
///
/// Grouping by the reference RTTM count keeps two-, three-, and four-speaker
/// conditions visible instead of hiding capacity regressions in one corpus-wide
/// DER. Percentages are duration-weighted from the underlying DER components.
public struct DiarizationSpeakerCountResult: Codable, Equatable, Sendable {
    public let referenceSpeakerCount: Int
    public let files: Int
    public let derPercent: Double
    public let missPercent: Double
    public let falseAlarmPercent: Double
    public let speakerErrorPercent: Double
    public let totalSpeechSeconds: Double
    public let missedSpeechSeconds: Double
    public let falseAlarmSeconds: Double
    public let confusionSeconds: Double
    public let speakerCountAccuracyPercent: Double
}

/// Incrementally aggregates DER components by the number of speakers in each
/// reference recording.
public struct DiarizationSpeakerCountAccumulator: Sendable {
    private struct Bucket: Sendable {
        var files = 0
        var correctSpeakerCounts = 0
        var totalSpeech: Float = 0
        var missedSpeech: Float = 0
        var falseAlarm: Float = 0
        var confusion: Float = 0
    }

    private var buckets: [Int: Bucket] = [:]

    public init() {}

    public mutating func add(
        referenceSpeakerCount: Int,
        predictedSpeakerCount: Int,
        score: DERResult
    ) {
        guard referenceSpeakerCount > 0 else { return }
        var bucket = buckets[referenceSpeakerCount, default: Bucket()]
        bucket.files += 1
        if predictedSpeakerCount == referenceSpeakerCount {
            bucket.correctSpeakerCounts += 1
        }
        bucket.totalSpeech += score.totalSpeech
        bucket.missedSpeech += score.missedSpeech
        bucket.falseAlarm += score.falseAlarm
        bucket.confusion += score.confusion
        buckets[referenceSpeakerCount] = bucket
    }

    public var results: [DiarizationSpeakerCountResult] {
        buckets.keys.sorted().compactMap { speakerCount in
            guard let bucket = buckets[speakerCount] else { return nil }
            let totalSpeech = Double(bucket.totalSpeech)
            func speechPercent(_ seconds: Float) -> Double {
                totalSpeech > 0 ? Double(seconds) / totalSpeech * 100 : 0
            }
            let errorSeconds = bucket.missedSpeech + bucket.falseAlarm + bucket.confusion
            return DiarizationSpeakerCountResult(
                referenceSpeakerCount: speakerCount,
                files: bucket.files,
                derPercent: speechPercent(errorSeconds),
                missPercent: speechPercent(bucket.missedSpeech),
                falseAlarmPercent: speechPercent(bucket.falseAlarm),
                speakerErrorPercent: speechPercent(bucket.confusion),
                totalSpeechSeconds: totalSpeech,
                missedSpeechSeconds: Double(bucket.missedSpeech),
                falseAlarmSeconds: Double(bucket.falseAlarm),
                confusionSeconds: Double(bucket.confusion),
                speakerCountAccuracyPercent: Double(bucket.correctSpeakerCounts)
                    / Double(bucket.files) * 100)
        }
    }
}
