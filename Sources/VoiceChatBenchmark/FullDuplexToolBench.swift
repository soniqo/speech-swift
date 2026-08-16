import ArgumentParser
import AudioCommon
import CryptoKit
import Foundation
import VoiceChat

private struct FullDuplexToolScenario {
    let schemaVersion: Int
    let id: String
    let transcript: String
    let audioURL: URL
    let audioSHA256: String?
    let expectedTool: String?
    let expectedArguments: FullDuplexExpectedArguments?
    let expectedReply: FullDuplexExpectedReply?
    let expectedAudio: FullDuplexExpectedAudio?
    let replyRequired: Bool
    let availableToolsJSON: String
    let providerResponseJSON: String?
    let providerDelayMilliseconds: Double

    static func load(from url: URL) throws -> FullDuplexToolScenario {
        let data = try Data(contentsOf: url)
        guard let object = try JSONSerialization.jsonObject(with: data)
            as? [String: Any] else {
            throw ValidationError("tool scenario must be a JSON object")
        }

        func requiredString(_ key: String) throws -> String {
            guard let value = object[key] as? String, !value.isEmpty else {
                throw ValidationError("tool scenario is missing `\(key)`")
            }
            return value
        }

        let schemaVersion = (object["schema_version"] as? NSNumber)?.intValue ?? 1
        guard schemaVersion == 1 else {
            throw ValidationError(
                "unsupported tool-scenario schema version \(schemaVersion)")
        }
        let audioPath = try requiredString("audio")
        let audioURL = URL(
            fileURLWithPath: audioPath,
            relativeTo: url.deletingLastPathComponent())
            .standardizedFileURL
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw ValidationError("scenario audio does not exist: \(audioURL.path)")
        }
        let availableToolsJSON: String
        if let toolsPath = object["available_tools"] as? String {
            let toolsURL = URL(
                fileURLWithPath: toolsPath,
                relativeTo: url.deletingLastPathComponent())
                .standardizedFileURL
            availableToolsJSON = try String(
                contentsOf: toolsURL,
                encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines)
        } else if let tools = object["available_tools"] as? [[String: Any]],
                  !tools.isEmpty {
            let toolsData = try JSONSerialization.data(
                withJSONObject: tools,
                options: [.withoutEscapingSlashes])
            guard let encoded = String(data: toolsData, encoding: .utf8) else {
                throw ValidationError("could not encode scenario tools as UTF-8")
            }
            availableToolsJSON = encoded
        } else {
            throw ValidationError(
                "tool scenario requires a tool array or JSON-file path in `available_tools`")
        }

        let expectedTool = (object["expected_tool"] as? String)
            .flatMap { $0.isEmpty ? nil : $0 }
        let expectedArguments = try FullDuplexExpectedArguments.parse(
            object["expected_arguments"])
        let expectedReply = try FullDuplexExpectedReply.parse(
            object["expected_reply"])
        let expectedAudio = try FullDuplexExpectedAudio.parse(
            object["expected_audio"])
        let replyRequired: Bool
        if let rawReplyRequired = object["reply_required"] {
            guard let value = rawReplyRequired as? Bool else {
                throw ValidationError("`reply_required` must be a boolean")
            }
            replyRequired = value
        } else {
            replyRequired = true
        }
        if expectedTool == nil, expectedArguments != nil {
            throw ValidationError(
                "a scenario without `expected_tool` cannot expect arguments")
        }
        if !replyRequired, expectedTool == nil {
            throw ValidationError(
                "`reply_required: false` requires an expected tool")
        }
        if !replyRequired, expectedReply != nil {
            throw ValidationError(
                "a first-step-only scenario cannot expect a spoken reply")
        }
        if !replyRequired, expectedAudio != nil {
            throw ValidationError(
                "a first-step-only scenario cannot expect spoken-audio quality")
        }
        var providerResponseJSON: String?
        if let responsePath = object["provider_response"] as? String {
            let responseURL = URL(
                fileURLWithPath: responsePath,
                relativeTo: url.deletingLastPathComponent())
                .standardizedFileURL
            providerResponseJSON = try String(
                contentsOf: responseURL,
                encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines)
        } else if let response = object["provider_response"],
                  !(response is NSNull) {
            let responseData = try JSONSerialization.data(
                withJSONObject: response,
                options: [.withoutEscapingSlashes])
            providerResponseJSON = String(data: responseData, encoding: .utf8)
        }
        if expectedTool != nil, providerResponseJSON == nil {
            throw ValidationError(
                "a scenario with `expected_tool` requires `provider_response`")
        }

        return FullDuplexToolScenario(
            schemaVersion: schemaVersion,
            id: try requiredString("id"),
            transcript: try requiredString("transcript"),
            audioURL: audioURL,
            audioSHA256: object["audio_sha256"] as? String,
            expectedTool: expectedTool,
            expectedArguments: expectedArguments,
            expectedReply: expectedReply,
            expectedAudio: expectedAudio,
            replyRequired: replyRequired,
            availableToolsJSON: availableToolsJSON,
            providerResponseJSON: providerResponseJSON,
            providerDelayMilliseconds:
                (object["provider_delay_ms"] as? NSNumber)?.doubleValue ?? 200)
    }
}

struct FullDuplexExpectedAudio {
    let minimumActiveSpeechMilliseconds: Double?
    let maximumInternalPauseMilliseconds: Double?
    let maximumClippedSampleFraction: Double?
    let maximumSuspectOnsetTransients: Int?
    let maximumSuspectFrameBoundaries: Int?

    static func parse(_ value: Any?) throws -> FullDuplexExpectedAudio? {
        guard let value, !(value is NSNull) else { return nil }
        guard let object = value as? [String: Any] else {
            throw ValidationError("`expected_audio` must be a JSON object")
        }
        let allowed = Set([
            "minimum_active_speech_ms",
            "maximum_internal_pause_ms",
            "maximum_clipped_sample_fraction",
            "maximum_suspect_onset_transients",
            "maximum_suspect_frame_boundaries",
        ])
        let unknown = Set(object.keys).subtracting(allowed)
        guard unknown.isEmpty else {
            throw ValidationError(
                "unsupported expected-audio constraints: "
                    + unknown.sorted().joined(separator: ", "))
        }

        func nonnegativeDouble(_ key: String) throws -> Double? {
            guard let raw = object[key] else { return nil }
            guard let number = raw as? NSNumber else {
                throw ValidationError("`expected_audio.\(key)` must be numeric")
            }
            let value = number.doubleValue
            guard value.isFinite, value >= 0 else {
                throw ValidationError(
                    "`expected_audio.\(key)` must be finite and nonnegative")
            }
            return value
        }

        let minimumActive = try nonnegativeDouble(
            "minimum_active_speech_ms")
        let maximumPause = try nonnegativeDouble(
            "maximum_internal_pause_ms")
        let maximumClipping = try nonnegativeDouble(
            "maximum_clipped_sample_fraction")
        if let maximumClipping, maximumClipping > 1 {
            throw ValidationError(
                "`expected_audio.maximum_clipped_sample_fraction` must not exceed 1")
        }
        func nonnegativeInteger(_ key: String) throws -> Int? {
            guard let raw = object[key] else { return nil }
            guard let number = raw as? NSNumber,
                  number.doubleValue.rounded() == number.doubleValue,
                  number.intValue >= 0 else {
                throw ValidationError(
                    "`expected_audio.\(key)` must be a nonnegative integer")
            }
            return number.intValue
        }
        let maximumOnsetTransients = try nonnegativeInteger(
            "maximum_suspect_onset_transients")
        let maximumBoundaries = try nonnegativeInteger(
            "maximum_suspect_frame_boundaries")
        guard minimumActive != nil || maximumPause != nil
                || maximumClipping != nil || maximumOnsetTransients != nil
                || maximumBoundaries != nil else {
            throw ValidationError("`expected_audio` has no constraints")
        }
        return FullDuplexExpectedAudio(
            minimumActiveSpeechMilliseconds: minimumActive,
            maximumInternalPauseMilliseconds: maximumPause,
            maximumClippedSampleFraction: maximumClipping,
            maximumSuspectOnsetTransients: maximumOnsetTransients,
            maximumSuspectFrameBoundaries: maximumBoundaries)
    }

    func mismatches(actual: FullDuplexAcousticAnalysis) -> [String] {
        var mismatches: [String] = []
        if let minimumActiveSpeechMilliseconds,
           actual.activeSpeechMilliseconds < minimumActiveSpeechMilliseconds {
            mismatches.append("active_speech_too_short")
        }
        if let maximumInternalPauseMilliseconds,
           actual.maximumInternalPauseMilliseconds
            > maximumInternalPauseMilliseconds {
            mismatches.append("internal_pause_too_long")
        }
        if let maximumClippedSampleFraction,
           actual.clippedSampleFraction > maximumClippedSampleFraction {
            mismatches.append("excessive_clipping")
        }
        if let maximumSuspectOnsetTransients,
           actual.suspectOnsetTransientCount
            > maximumSuspectOnsetTransients {
            mismatches.append("onset_transient")
        }
        if let maximumSuspectFrameBoundaries,
           actual.suspectFrameBoundaryCount
            > maximumSuspectFrameBoundaries {
            mismatches.append("frame_boundary_discontinuity")
        }
        if actual.nonFiniteSampleCount > 0 {
            mismatches.append("non_finite_audio")
        }
        return mismatches
    }
}

struct FullDuplexExpectedReply {
    let containsAllWords: Set<String>
    let containsAnyWords: Set<String>
    let absentWords: Set<String>

    static func parse(_ value: Any?) throws -> FullDuplexExpectedReply? {
        guard let value, !(value is NSNull) else { return nil }
        guard let object = value as? [String: Any] else {
            throw ValidationError("`expected_reply` must be a JSON object")
        }
        let allowed = Set([
            "contains_all_words", "contains_any_words", "absent_words",
        ])
        let unknown = Set(object.keys).subtracting(allowed)
        guard unknown.isEmpty else {
            throw ValidationError(
                "unsupported expected-reply constraints: "
                    + unknown.sorted().joined(separator: ", "))
        }

        func words(_ key: String) throws -> Set<String> {
            guard let raw = object[key] else { return [] }
            guard let values = raw as? [String], !values.isEmpty,
                  values.allSatisfy({ !$0.isEmpty }) else {
                throw ValidationError("`expected_reply.\(key)` must be strings")
            }
            let normalized = Set(values.flatMap(Self.normalizedWords))
            guard !normalized.isEmpty else {
                throw ValidationError(
                    "`expected_reply.\(key)` must contain words")
            }
            return normalized
        }

        let containsAll = try words("contains_all_words")
        let containsAny = try words("contains_any_words")
        let absent = try words("absent_words")
        guard !containsAll.isEmpty || !containsAny.isEmpty || !absent.isEmpty else {
            throw ValidationError("`expected_reply` has no constraints")
        }
        guard containsAll.isDisjoint(with: absent) else {
            throw ValidationError(
                "reply words cannot be both required and absent")
        }
        guard containsAny.isEmpty || !containsAny.isSubset(of: absent) else {
            throw ValidationError(
                "every alternative reply word is marked absent")
        }
        return FullDuplexExpectedReply(
            containsAllWords: containsAll,
            containsAnyWords: containsAny,
            absentWords: absent)
    }

    func mismatches(actual: String) -> [String] {
        let actualWords = Set(Self.normalizedWords(actual))
        var mismatches = containsAllWords.subtracting(actualWords)
            .sorted()
            .map { "missing:\($0)" }
        if !containsAnyWords.isEmpty,
           containsAnyWords.isDisjoint(with: actualWords) {
            mismatches.append(
                "missing_any:" + containsAnyWords.sorted().joined(separator: "|"))
        }
        mismatches.append(contentsOf: absentWords.intersection(actualWords)
            .sorted()
            .map { "unexpected:\($0)" })
        return mismatches
    }

    private static func normalizedWords(_ text: String) -> [String] {
        text.lowercased().split {
            !$0.isLetter && !$0.isNumber
        }.map(String.init)
    }
}

struct FullDuplexExpectedArguments {
    let required: Set<String>
    let absent: Set<String>
    let equals: [String: Any]
    let stringContainsWords: [String: Set<String>]

    static func parse(_ value: Any?) throws -> FullDuplexExpectedArguments? {
        guard let value, !(value is NSNull) else { return nil }
        guard let object = value as? [String: Any] else {
            throw ValidationError("`expected_arguments` must be a JSON object")
        }
        let allowed = Set([
            "required", "absent", "equals", "string_contains_words",
        ])
        let unknown = Set(object.keys).subtracting(allowed)
        guard unknown.isEmpty else {
            throw ValidationError(
                "unsupported expected-argument constraints: "
                    + unknown.sorted().joined(separator: ", "))
        }

        func stringSet(_ key: String) throws -> Set<String> {
            guard let raw = object[key] else { return [] }
            guard let values = raw as? [String], !values.isEmpty,
                  values.allSatisfy({ !$0.isEmpty }) else {
                throw ValidationError("`expected_arguments.\(key)` must be strings")
            }
            return Set(values)
        }

        let required = try stringSet("required")
        let absent = try stringSet("absent")
        guard required.isDisjoint(with: absent) else {
            throw ValidationError(
                "expected arguments cannot be both required and absent")
        }
        let equals: [String: Any]
        if let raw = object["equals"] {
            guard let values = raw as? [String: Any] else {
                throw ValidationError(
                    "`expected_arguments.equals` must be a JSON object")
            }
            equals = values
        } else {
            equals = [:]
        }

        var contains: [String: Set<String>] = [:]
        if let raw = object["string_contains_words"] {
            guard let values = raw as? [String: Any] else {
                throw ValidationError(
                    "`expected_arguments.string_contains_words` must be an object")
            }
            for (key, rawWords) in values {
                guard let words = rawWords as? [String], !words.isEmpty,
                      words.allSatisfy({ !$0.isEmpty }) else {
                    throw ValidationError(
                        "expected words for `\(key)` must be non-empty strings")
                }
                contains[key] = Set(words.map { $0.lowercased() })
            }
        }
        guard !required.isEmpty || !absent.isEmpty || !equals.isEmpty
                || !contains.isEmpty else {
            throw ValidationError("`expected_arguments` has no constraints")
        }
        let constrained = Set(equals.keys).union(contains.keys)
        guard constrained.isDisjoint(with: absent) else {
            throw ValidationError(
                "absent arguments cannot also have value constraints")
        }
        return FullDuplexExpectedArguments(
            required: required,
            absent: absent,
            equals: equals,
            stringContainsWords: contains)
    }

    func mismatches(actual: [String: Any]?) -> [String] {
        let actual = actual ?? [:]
        var mismatches: [String] = []
        for key in required.sorted() where Self.isMissing(actual[key]) {
            mismatches.append("missing:\(key)")
        }
        for key in absent.sorted() where !Self.isMissing(actual[key]) {
            mismatches.append("unexpected:\(key)")
        }
        for key in equals.keys.sorted() {
            guard let expected = equals[key],
                  let value = actual[key],
                  Self.canonicalJSON(value) == Self.canonicalJSON(expected)
            else {
                mismatches.append("value:\(key)")
                continue
            }
        }
        for key in stringContainsWords.keys.sorted() {
            guard let value = actual[key] as? String else {
                mismatches.append("words:\(key)")
                continue
            }
            let actualWords = Set(value.lowercased().split {
                !$0.isLetter && !$0.isNumber
            }.map(String.init))
            if !stringContainsWords[key, default: []]
                .isSubset(of: actualWords) {
                mismatches.append("words:\(key)")
            }
        }
        return mismatches
    }

    private static func isMissing(_ value: Any?) -> Bool {
        guard let value, !(value is NSNull) else { return true }
        if let text = value as? String {
            return text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        return false
    }

    private static func canonicalJSON(_ value: Any) -> Data? {
        try? JSONSerialization.data(
            withJSONObject: ["value": value],
            options: [.sortedKeys, .withoutEscapingSlashes])
    }
}

struct FullDuplexNativeToolCall {
    let name: String?
    let arguments: [String: Any]?
    let argumentsJSON: String?
    let validJSON: Bool
}

private struct FullDuplexProviderTiming: Sendable {
    let startedAtNanoseconds: UInt64
    let completedAtNanoseconds: UInt64
    let responseAcceptedAtNanoseconds: UInt64

    var serviceMilliseconds: Double {
        FullDuplexToolClock.milliseconds(
            from: startedAtNanoseconds,
            to: completedAtNanoseconds)
    }

    var responseQueueWaitMilliseconds: Double {
        FullDuplexToolClock.milliseconds(
            from: completedAtNanoseconds,
            to: responseAcceptedAtNanoseconds)
    }
}

struct FullDuplexMicrophonePush: Sendable {
    let startedAtNanoseconds: UInt64
    let completedAtNanoseconds: UInt64
    let deadlineNanoseconds: UInt64

    var elapsedMilliseconds: Double {
        FullDuplexToolClock.milliseconds(
            from: startedAtNanoseconds,
            to: completedAtNanoseconds)
    }

    var behindMilliseconds: Double {
        // `deadlineNanoseconds` is when this 80 ms microphone frame becomes
        // available. It has until the next frame arrives to finish without
        // accumulating capture backlog.
        let nextFrameDeadline = deadlineNanoseconds
            + UInt64(VoiceChatSession.frameMilliseconds) * 1_000_000
        return max(0, FullDuplexToolClock.milliseconds(
            from: nextFrameDeadline,
            to: completedAtNanoseconds))
    }

    func overlaps(_ start: UInt64, _ end: UInt64) -> Bool {
        completedAtNanoseconds >= start && startedAtNanoseconds <= end
    }
}

private enum FullDuplexToolClock {
    static func milliseconds(from start: UInt64, to end: UInt64) -> Double {
        if end >= start {
            return Double(end - start) / 1_000_000
        }
        return -Double(start - end) / 1_000_000
    }

    static func adding(milliseconds: Double, to value: UInt64) -> UInt64 {
        value + UInt64(max(0, milliseconds) * 1_000_000)
    }
}

private struct FullDuplexToolPhaseReport: Codable {
    let elapsedMilliseconds: Double
    let tokenSteps: Int?
    let tokensPerSecond: Double?
    let modelMilliseconds: Double?
    let languageCacheMilliseconds: Double?
    let speechCacheMilliseconds: Double?
    let interleavingMilliseconds: Double?
    let prefillBatches: Int?
}

private struct FullDuplexProviderReport: Codable {
    let configuredDelayMilliseconds: Double
    let serviceMilliseconds: Double?
    let responseQueueWaitMilliseconds: Double?
}

private struct FullDuplexLatencyReport: Codable {
    let requestStartToInputEndMilliseconds: Double
    let inputEndToToolCallStartMilliseconds: Double?
    let inputEndToToolCallCompleteMilliseconds: Double?
    let inputEndToFirstTextMilliseconds: Double?
    let inputEndToFirstAudioFrameReadyMilliseconds: Double?
    let inputEndToFirstSustainedAudioMilliseconds: Double?
    let inputEndToReplyCompleteMilliseconds: Double?
    let requestStartToFirstSustainedAudioMilliseconds: Double?
    let requestStartToReplyCompleteMilliseconds: Double?
    let audibleOnsetOffsetWithinFrameMilliseconds: Double?
}

private struct FullDuplexMicrophonePhaseReport: Codable {
    let frameCount: Int
    let foregroundRealTimeFactor: Double
    let serviceP50Milliseconds: Double
    let serviceP95Milliseconds: Double
    let serviceMaximumMilliseconds: Double
}

private struct FullDuplexRealtimeReport: Codable {
    let frameMilliseconds: Int
    let frameCount: Int
    let foregroundRealTimeFactor: Double
    let wallClockRealTimeFactor: Double
    let maximumBehindMilliseconds: Double
    let deadlineMissesOverFiveMilliseconds: Int
    let allFrames: FullDuplexMicrophonePhaseReport
    let normalFrames: FullDuplexMicrophonePhaseReport
    let toolFrames: FullDuplexMicrophonePhaseReport
    let toolDecodeFrames: FullDuplexMicrophonePhaseReport
    let providerFrames: FullDuplexMicrophonePhaseReport
    let resultSyncFrames: FullDuplexMicrophonePhaseReport
}

struct FullDuplexAcousticAnalysis: Codable, Equatable {
    let sampleCount: Int
    let durationMilliseconds: Double
    let nonFiniteSampleCount: Int
    let peakAbsoluteAmplitude: Double
    let clippedSampleCount: Int
    let clippedSampleFraction: Double
    let activeSpeechMilliseconds: Double
    let speechSpanMilliseconds: Double
    let internalPauseCount: Int
    let maximumInternalPauseMilliseconds: Double
    let totalInternalPauseMilliseconds: Double
    let onsetAnalysisSpanMilliseconds: Double
    let maximumOnsetJump: Double
    let p99SteadySpeechJump: Double
    let maximumOnsetToSteadyP99Ratio: Double
    let suspectOnsetTransientCount: Int
    let frameBoundaryCount: Int
    let maximumFrameBoundaryJump: Double
    let p95FrameBoundaryJump: Double
    let p99WithinFrameJump: Double
    let maximumBoundaryToWithinP99Ratio: Double
    let suspectFrameBoundaryCount: Int
}

private struct FullDuplexAudioQualityReport: Codable {
    /// Frames exactly as queued by the live player. Boundary diagnostics on
    /// this stream expose clicks introduced by frame-by-frame stitching.
    let livePlayback: FullDuplexAcousticAnalysis
    /// One continuous codec decode of the complete generated-code history.
    /// Comparing it with live playback separates codec/model artifacts from
    /// live frame-boundary artifacts.
    let offlineRender: FullDuplexAcousticAnalysis
}

enum FullDuplexAcousticAnalyzer {
    static let analysisWindowMilliseconds = 10.0
    static let onsetAnalysisMilliseconds = 50.0
    static let minimumInternalPauseMilliseconds = 120.0
    static let clippingAmplitude = 0.999
    static let minimumSuspectOnsetJump = 0.05
    static let suspectOnsetRatio = 6.0
    static let minimumSuspectBoundaryJump = 0.05
    static let suspectBoundaryRatio = 6.0

    static func analyze(
        samples: [Float],
        sampleRate: Int,
        frameBoundaries: [Int],
        thresholdDBFS: Double
    ) -> FullDuplexAcousticAnalysis {
        guard sampleRate > 0, !samples.isEmpty else {
            return FullDuplexAcousticAnalysis(
                sampleCount: samples.count,
                durationMilliseconds: 0,
                nonFiniteSampleCount: samples.filter { !$0.isFinite }.count,
                peakAbsoluteAmplitude: 0,
                clippedSampleCount: 0,
                clippedSampleFraction: 0,
                activeSpeechMilliseconds: 0,
                speechSpanMilliseconds: 0,
                internalPauseCount: 0,
                maximumInternalPauseMilliseconds: 0,
                totalInternalPauseMilliseconds: 0,
                onsetAnalysisSpanMilliseconds: 0,
                maximumOnsetJump: 0,
                p99SteadySpeechJump: 0,
                maximumOnsetToSteadyP99Ratio: 0,
                suspectOnsetTransientCount: 0,
                frameBoundaryCount: 0,
                maximumFrameBoundaryJump: 0,
                p95FrameBoundaryJump: 0,
                p99WithinFrameJump: 0,
                maximumBoundaryToWithinP99Ratio: 0,
                suspectFrameBoundaryCount: 0)
        }

        let finiteSamples = samples.map { $0.isFinite ? $0 : 0 }
        let nonFiniteCount = samples.count(where: { !$0.isFinite })
        let magnitudes = finiteSamples.map { abs(Double($0)) }
        let peak = magnitudes.max() ?? 0
        let clipped = magnitudes.count(where: { $0 >= clippingAmplitude })
        let duration = Double(samples.count) * 1_000 / Double(sampleRate)

        let windowSamples = max(
            1,
            Int((Double(sampleRate) * analysisWindowMilliseconds / 1_000)
                .rounded()))
        let threshold = pow(10, thresholdDBFS / 20)
        var activeWindows: [Bool] = []
        var windowDurations: [Double] = []
        for start in stride(from: 0, to: finiteSamples.count, by: windowSamples) {
            let end = min(finiteSamples.count, start + windowSamples)
            let energy = finiteSamples[start ..< end].reduce(0.0) {
                $0 + Double($1 * $1)
            }
            let rms = sqrt(energy / Double(max(1, end - start)))
            activeWindows.append(rms >= threshold)
            windowDurations.append(
                Double(end - start) * 1_000 / Double(sampleRate))
        }
        let firstActive = activeWindows.firstIndex(of: true)
        let lastActive = activeWindows.lastIndex(of: true)
        let activeMilliseconds = zip(activeWindows, windowDurations).reduce(0) {
            $0 + ($1.0 ? $1.1 : 0)
        }
        let speechSpanMilliseconds: Double
        var pauseDurations: [Double] = []
        if let firstActive, let lastActive {
            speechSpanMilliseconds = windowDurations[firstActive ... lastActive]
                .reduce(0, +)
            var index = firstActive + 1
            while index < lastActive {
                guard !activeWindows[index] else {
                    index += 1
                    continue
                }
                let start = index
                while index < lastActive, !activeWindows[index] {
                    index += 1
                }
                let pause = windowDurations[start ..< index].reduce(0, +)
                if pause >= minimumInternalPauseMilliseconds {
                    pauseDurations.append(pause)
                }
            }
        } else {
            speechSpanMilliseconds = 0
        }

        let onsetWindowSamples = max(
            1,
            Int((Double(sampleRate) * onsetAnalysisMilliseconds / 1_000)
                .rounded()))
        let firstActiveSample = firstActive.map { $0 * windowSamples }
        let lastActiveSampleExclusive = lastActive.map {
            min(finiteSamples.count, ($0 + 1) * windowSamples)
        }
        let onsetEnd = firstActiveSample.map {
            min(finiteSamples.count, $0 + onsetWindowSamples)
        }
        var onsetJumps: [Double] = []
        var steadySpeechJumps: [Double] = []
        if let onsetEnd, onsetEnd > 1 {
            onsetJumps.reserveCapacity(onsetEnd - 1)
            for index in 1 ..< onsetEnd {
                onsetJumps.append(abs(Double(
                    finiteSamples[index] - finiteSamples[index - 1])))
            }
        }
        if let onsetEnd, let lastActiveSampleExclusive,
           lastActiveSampleExclusive > onsetEnd {
            steadySpeechJumps.reserveCapacity(
                lastActiveSampleExclusive - onsetEnd)
            for index in onsetEnd ..< lastActiveSampleExclusive {
                steadySpeechJumps.append(abs(Double(
                    finiteSamples[index] - finiteSamples[index - 1])))
            }
        }
        onsetJumps.sort()
        steadySpeechJumps.sort()
        let maximumOnsetJump = onsetJumps.last ?? 0
        let p99SteadySpeechJump = percentile(
            steadySpeechJumps,
            fraction: 0.99)
        let onsetRatio = p99SteadySpeechJump > 0
            ? maximumOnsetJump / p99SteadySpeechJump
            : (maximumOnsetJump > 0 ? maximumOnsetJump / 1e-9 : 0)
        let suspectOnsetThreshold = max(
            minimumSuspectOnsetJump,
            p99SteadySpeechJump * suspectOnsetRatio)
        let suspectOnsetTransients = onsetJumps.count(where: {
            $0 > suspectOnsetThreshold
        })

        let validBoundaries = Array(Set(frameBoundaries.filter {
            $0 > 0 && $0 < finiteSamples.count
        })).sorted()
        let boundarySet = Set(validBoundaries)
        var boundaryJumps: [Double] = []
        var withinFrameJumps: [Double] = []
        if finiteSamples.count > 1 {
            boundaryJumps.reserveCapacity(validBoundaries.count)
            withinFrameJumps.reserveCapacity(
                finiteSamples.count - 1 - validBoundaries.count)
            for index in 1 ..< finiteSamples.count {
                let jump = abs(Double(
                    finiteSamples[index] - finiteSamples[index - 1]))
                if boundarySet.contains(index) {
                    boundaryJumps.append(jump)
                } else {
                    withinFrameJumps.append(jump)
                }
            }
        }
        boundaryJumps.sort()
        withinFrameJumps.sort()
        let p95Boundary = percentile(boundaryJumps, fraction: 0.95)
        let p99Within = percentile(withinFrameJumps, fraction: 0.99)
        let maximumBoundary = boundaryJumps.last ?? 0
        let boundaryRatio = p99Within > 0
            ? maximumBoundary / p99Within
            : (maximumBoundary > 0 ? maximumBoundary / 1e-9 : 0)
        let suspectThreshold = max(
            minimumSuspectBoundaryJump,
            p99Within * suspectBoundaryRatio)
        let suspectBoundaries = boundaryJumps.count(where: {
            $0 > suspectThreshold
        })

        return FullDuplexAcousticAnalysis(
            sampleCount: samples.count,
            durationMilliseconds: duration,
            nonFiniteSampleCount: nonFiniteCount,
            peakAbsoluteAmplitude: peak,
            clippedSampleCount: clipped,
            clippedSampleFraction: Double(clipped) / Double(samples.count),
            activeSpeechMilliseconds: activeMilliseconds,
            speechSpanMilliseconds: speechSpanMilliseconds,
            internalPauseCount: pauseDurations.count,
            maximumInternalPauseMilliseconds: pauseDurations.max() ?? 0,
            totalInternalPauseMilliseconds: pauseDurations.reduce(0, +),
            onsetAnalysisSpanMilliseconds: firstActive == nil
                ? 0
                : Double(onsetEnd ?? 0) * 1_000 / Double(sampleRate),
            maximumOnsetJump: maximumOnsetJump,
            p99SteadySpeechJump: p99SteadySpeechJump,
            maximumOnsetToSteadyP99Ratio: onsetRatio,
            suspectOnsetTransientCount: suspectOnsetTransients,
            frameBoundaryCount: boundaryJumps.count,
            maximumFrameBoundaryJump: maximumBoundary,
            p95FrameBoundaryJump: p95Boundary,
            p99WithinFrameJump: p99Within,
            maximumBoundaryToWithinP99Ratio: boundaryRatio,
            suspectFrameBoundaryCount: suspectBoundaries)
    }

    private static func percentile(
        _ sortedValues: [Double],
        fraction: Double
    ) -> Double {
        guard !sortedValues.isEmpty else { return 0 }
        let position = Double(sortedValues.count - 1) * fraction
        let lower = Int(position.rounded(.down))
        let upper = Int(position.rounded(.up))
        guard lower != upper else { return sortedValues[lower] }
        let weight = position - Double(lower)
        return sortedValues[lower] * (1 - weight)
            + sortedValues[upper] * weight
    }
}

private struct FullDuplexToolOutcomeReport: Codable {
    let status: String
    let success: Bool
    let expectedTool: String?
    let actualTool: String?
    let callPayloadIsValidJSON: Bool
    let nativeCallPayload: String?
    let actualArgumentsJSON: String?
    let argumentsMatch: Bool?
    let argumentMismatches: [String]
    let replyMatch: Bool?
    let replyMismatches: [String]
    let audioMatch: Bool?
    let audioMismatches: [String]
    let referenceTranscript: String
    let recognizedTranscript: String
    let assistantReply: String
    let firstTextObserved: Bool
    let firstSustainedAudioObserved: Bool
    let replyCompleted: Bool
}

private struct FullDuplexScenarioReport: Codable {
    let id: String
    let schemaVersion: Int
    let audioFile: String
    let audioSHA256: String?
    let audioDurationSeconds: Double
    let providerKind: String
    let replyRequired: Bool
}

private struct FullDuplexRuntimeReport: Codable {
    let modelBundle: String
    let quantization: String
    let speechIterations: Int
    let speechContextFrames: Int
    let functionCallEndpointFrames: Int
    let audibilityThresholdDBFS: Double
    let loadMilliseconds: Double
}

private struct FullDuplexToolBenchmarkReport: Codable {
    let schemaVersion: Int
    let benchmark: String
    let generatedAtUTC: String
    let scenario: FullDuplexScenarioReport
    let runtime: FullDuplexRuntimeReport
    let outcome: FullDuplexToolOutcomeReport
    let latency: FullDuplexLatencyReport
    let nativeToolDecode: FullDuplexToolPhaseReport?
    let provider: FullDuplexProviderReport
    let toolResultSync: FullDuplexToolPhaseReport?
    let realtime: FullDuplexRealtimeReport
    let audioQuality: FullDuplexAudioQualityReport
}

extension VoiceChatBench {
    func runFullDuplexToolScenario(
        root: URL,
        scenarioURL: URL
    ) async throws {
        let scenario = try FullDuplexToolScenario.load(from: scenarioURL)
        let configuredProviderDelay = providerDelayMs
            ?? scenario.providerDelayMilliseconds
        guard configuredProviderDelay.isFinite,
              configuredProviderDelay >= 0,
              configuredProviderDelay <= 10_000 else {
            throw ValidationError("provider delay must be between 0 and 10000 ms")
        }
        guard speechIterations > 0, speechIterations <= 32 else {
            throw ValidationError("--speech-iterations must be between 1 and 32")
        }
        guard toolTimeoutSeconds.isFinite,
              toolTimeoutSeconds >= 2,
              toolTimeoutSeconds <= 60 else {
            throw ValidationError("--tool-timeout-seconds must be between 2 and 60")
        }
        guard audibilityThresholdDbfs.isFinite,
              audibilityThresholdDbfs >= -100,
              audibilityThresholdDbfs <= -10 else {
            throw ValidationError(
                "--audibility-threshold-dbfs must be between -100 and -10")
        }
        if let functionCallEndpointFrames {
            guard functionCallEndpointFrames > 0,
                  functionCallEndpointFrames
                    <= VoiceChatTurnTakingParameters
                        .nvidiaTurnTakingFallbackFrames else {
                throw ValidationError(
                    "--function-call-endpoint-frames must be between 1 and 40")
            }
        }
        if let expectedSHA256 = scenario.audioSHA256 {
            let actualSHA256 = try Self.fileSHA256(scenario.audioURL)
            guard actualSHA256.caseInsensitiveCompare(expectedSHA256)
                == .orderedSame else {
                throw ValidationError(
                    "scenario audio SHA-256 mismatch: \(actualSHA256) != \(expectedSHA256)")
            }
        }

        var samples = try AudioFileLoader.load(
            url: scenario.audioURL,
            targetSampleRate: VoiceChatSession.inputSampleRate)
        guard !samples.isEmpty else {
            throw ValidationError("scenario audio is empty")
        }
        let frameSize = VoiceChatSession.inputSamplesPerFrame
        let frameCount = (samples.count + frameSize - 1) / frameSize
        samples.append(contentsOf: repeatElement(
            0,
            count: frameCount * frameSize - samples.count))
        let audioDurationSeconds = Double(samples.count)
            / Double(VoiceChatSession.inputSampleRate)

        print("loading complete VoiceChat bundle...")
        let loadStarted = DispatchTime.now().uptimeNanoseconds
        let fullModel = try await VoiceChatModel.load(
            from: root,
            progressHandler: { progress, stage in
                print(String(format: "  [%3.0f%%] %@", progress * 100, stage))
            })
        let prompt = try VoiceChatSession.toolCallingSystemPrompt(
            availableToolsJSON: scenario.availableToolsJSON)
        let retainedSpeechFrames = speechContextFrames ?? 250
        var toolTurnTaking = VoiceChatTurnTakingParameters
            .functionCallingRealtime
        if let functionCallEndpointFrames {
            toolTurnTaking.functionCallEndOfUtteranceFrames =
                functionCallEndpointFrames
        }
        let session = try await fullModel.startSession(
            systemPrompt: prompt,
            speech: .init(
                iterations: speechIterations,
                recentContextFrames: retainedSpeechFrames,
                realtimeIdleOptimization: true),
            streamUserTranscript: true,
            turnTaking: toolTurnTaking,
            functionCallingEnabled: true)
        let loadMilliseconds = FullDuplexToolClock.milliseconds(
            from: loadStarted,
            to: DispatchTime.now().uptimeNanoseconds)

        let frameNanoseconds = UInt64(VoiceChatSession.frameMilliseconds)
            * 1_000_000
        let silence = [Float](repeating: 0, count: frameSize)
        let benchmarkStarted = DispatchTime.now().uptimeNanoseconds
        var nextDeadline = benchmarkStarted
        var inputEnded = benchmarkStarted
        var pushes: [FullDuplexMicrophonePush] = []
        var callPayload: String?
        var callCompletedAt: UInt64?
        var providerTask: Task<FullDuplexProviderTiming, Error>?
        var firstTextAt: UInt64?
        var firstAudioFrameReadyAt: UInt64?
        var firstSustainedAudioAt: UInt64?
        var audibleOffsetMilliseconds: Double?
        var replyCompletedAt: UInt64?
        var sawAssistantContent = false
        var livePlaybackSamples: [Float] = []
        var livePlaybackFrameBoundaries: [Int] = []

        func startProvider(
            responseJSON: String
        ) -> Task<FullDuplexProviderTiming, Error> {
            Task {
                let started = DispatchTime.now().uptimeNanoseconds
                try await Task.sleep(nanoseconds: UInt64(
                    configuredProviderDelay * 1_000_000))
                let completed = DispatchTime.now().uptimeNanoseconds
                try await session.injectFunctionResponse(responseJSON)
                let accepted = DispatchTime.now().uptimeNanoseconds
                return FullDuplexProviderTiming(
                    startedAtNanoseconds: started,
                    completedAtNanoseconds: completed,
                    responseAcceptedAtNanoseconds: accepted)
            }
        }

        func observe(
            _ events: [VoiceChatFrameEvent],
            at observedAt: UInt64
        ) {
            for event in events {
                if event.playbackRequired, !event.audio.isEmpty {
                    if !livePlaybackSamples.isEmpty {
                        livePlaybackFrameBoundaries.append(
                            livePlaybackSamples.count)
                    }
                    livePlaybackSamples.append(contentsOf: event.audio)
                }
                if callPayload == nil, let payload = event.functionCall {
                    callPayload = payload
                    callCompletedAt = observedAt
                    let call = Self.parseNativeToolCall(payload)
                    let response = Self.benchmarkProviderResponse(
                        call: call,
                        expectedTool: scenario.expectedTool,
                        expectedArguments: scenario.expectedArguments,
                        expectedResponseJSON: scenario.providerResponseJSON)
                    providerTask = startProvider(responseJSON: response)
                }
                if firstTextAt == nil, event.speaking {
                    firstTextAt = observedAt
                    sawAssistantContent = true
                } else if event.speaking {
                    sawAssistantContent = true
                }
                if sawAssistantContent,
                   firstAudioFrameReadyAt == nil,
                   event.playbackRequired,
                   !event.audio.isEmpty {
                    firstAudioFrameReadyAt = observedAt
                }
                if sawAssistantContent,
                   firstSustainedAudioAt == nil,
                   event.playbackRequired,
                   let onset = Self.sustainedAudioOnsetMilliseconds(
                       event.audio,
                       sampleRate: VoiceChatSession.outputSampleRate,
                       thresholdDBFS: audibilityThresholdDbfs)
                {
                    audibleOffsetMilliseconds = onset
                    firstSustainedAudioAt = FullDuplexToolClock.adding(
                        milliseconds: onset,
                        to: observedAt)
                }
                if sawAssistantContent,
                   event.textToken == fullModel.tokenizer.eosID,
                   replyCompletedAt == nil {
                    replyCompletedAt = observedAt
                }
            }
        }

        func waitForDeadline(_ deadline: UInt64) async throws {
            let now = DispatchTime.now().uptimeNanoseconds
            if deadline > now {
                try await Task.sleep(nanoseconds: deadline - now)
            }
        }

        func push(_ frame: [Float]) async throws {
            nextDeadline += frameNanoseconds
            try await waitForDeadline(nextDeadline)
            let started = DispatchTime.now().uptimeNanoseconds
            let events = try await session.pushAudio(frame)
            let completed = DispatchTime.now().uptimeNanoseconds
            pushes.append(FullDuplexMicrophonePush(
                startedAtNanoseconds: started,
                completedAtNanoseconds: completed,
                deadlineNanoseconds: nextDeadline))
            observe(events, at: completed)
        }

        for start in stride(from: 0, to: samples.count, by: frameSize) {
            try await push(Array(samples[start ..< start + frameSize]))
        }
        // Audio reaches the microphone on the real-time deadline even if the
        // model is behind when it finishes processing that final frame.
        inputEnded = nextDeadline

        let timeoutAt = FullDuplexToolClock.adding(
            milliseconds: toolTimeoutSeconds * 1_000,
            to: inputEnded)
        while DispatchTime.now().uptimeNanoseconds < timeoutAt,
              replyCompletedAt == nil {
            try await push(silence)
            if Self.firstStepScenarioIsComplete(
                replyRequired: scenario.replyRequired,
                callObserved: callPayload != nil,
                responseMetricsObserved:
                    await session.functionResponseMetrics() != nil) {
                break
            }
        }

        let providerTiming: FullDuplexProviderTiming?
        if let providerTask {
            if DispatchTime.now().uptimeNanoseconds >= timeoutAt {
                providerTask.cancel()
            }
            providerTiming = try? await providerTask.value
        } else {
            providerTiming = nil
        }

        let callMetrics = await session.functionCallDecodeMetrics()
        let responseMetrics = await session.functionResponseMetrics()
        let recognizedTranscript = await session.userTranscript()
        let assistantReply = await session.reply()
        // This post-run decode is deliberately outside every latency and RTF
        // timestamp. It gives the benchmark a continuous reference waveform to
        // compare with the exact frame sequence that live playback received.
        let offlineRenderedAudio = await session.renderedAudio()
        let offlineFrameBoundaries = stride(
            from: VoiceChatSession.outputSamplesPerFrame,
            to: offlineRenderedAudio.count,
            by: VoiceChatSession.outputSamplesPerFrame).map { $0 }
        let liveAudioQuality = FullDuplexAcousticAnalyzer.analyze(
            samples: livePlaybackSamples,
            sampleRate: VoiceChatSession.outputSampleRate,
            frameBoundaries: livePlaybackFrameBoundaries,
            thresholdDBFS: audibilityThresholdDbfs)
        let offlineAudioQuality = FullDuplexAcousticAnalyzer.analyze(
            samples: offlineRenderedAudio,
            sampleRate: VoiceChatSession.outputSampleRate,
            frameBoundaries: offlineFrameBoundaries,
            thresholdDBFS: audibilityThresholdDbfs)
        let callInfo = Self.parseNativeToolCall(callPayload)
        let argumentMismatches: [String]
        let argumentsMatch: Bool?
        if let expectedArguments = scenario.expectedArguments,
           callInfo.validJSON,
           callInfo.name == scenario.expectedTool {
            argumentMismatches = expectedArguments.mismatches(
                actual: callInfo.arguments)
            argumentsMatch = argumentMismatches.isEmpty
        } else {
            argumentMismatches = []
            argumentsMatch = nil
        }
        let replyMismatches: [String]
        let replyMatch: Bool?
        if let expectedReply = scenario.expectedReply,
           replyCompletedAt != nil {
            replyMismatches = expectedReply.mismatches(actual: assistantReply)
            replyMatch = replyMismatches.isEmpty
        } else {
            replyMismatches = []
            replyMatch = nil
        }
        let audioMismatches: [String]
        let audioMatch: Bool?
        if let expectedAudio = scenario.expectedAudio,
           replyCompletedAt != nil {
            audioMismatches = expectedAudio.mismatches(actual: liveAudioQuality)
            audioMatch = audioMismatches.isEmpty
        } else {
            audioMismatches = []
            audioMatch = nil
        }
        let callStartAt = callCompletedAt.flatMap { completed in
            callMetrics.map { metrics in
                completed - UInt64(max(0, metrics.elapsedMilliseconds) * 1_000_000)
            }
        }
        let resultSyncStart = providerTiming?.responseAcceptedAtNanoseconds
        let resultSyncEnd: UInt64?
        if let resultSyncStart, let responseMetrics {
            resultSyncEnd = FullDuplexToolClock.adding(
                milliseconds: responseMetrics.elapsedMilliseconds,
                to: resultSyncStart)
        } else {
            resultSyncEnd = nil
        }

        let status: String
        if let expected = scenario.expectedTool {
            if callPayload == nil {
                status = firstTextAt == nil
                    ? "timeout_no_tool_call"
                    : "tool_selection_miss"
            } else if !callInfo.validJSON {
                status = "malformed_tool_call"
            } else if callInfo.name != expected {
                status = "wrong_tool"
            } else if !argumentMismatches.isEmpty {
                status = "wrong_arguments"
            } else if !scenario.replyRequired, responseMetrics == nil {
                status = "result_sync_incomplete"
            } else if scenario.replyRequired, firstSustainedAudioAt == nil {
                status = "timeout_no_audible_reply"
            } else if scenario.replyRequired, replyCompletedAt == nil {
                status = "reply_incomplete"
            } else if !replyMismatches.isEmpty {
                status = "wrong_reply"
            } else if !audioMismatches.isEmpty {
                status = "audio_quality_regression"
            } else {
                status = "completed"
            }
        } else if callPayload != nil {
            status = "unexpected_tool"
        } else if firstSustainedAudioAt == nil {
            status = "timeout_no_audible_reply"
        } else if replyCompletedAt == nil {
            status = "reply_incomplete"
        } else if !replyMismatches.isEmpty {
            status = "wrong_reply"
        } else if !audioMismatches.isEmpty {
            status = "audio_quality_regression"
        } else {
            status = "completed"
        }

        let callPhasePushes = Self.overlappingPushes(
            pushes,
            start: callStartAt,
            end: callCompletedAt)
        let providerPhasePushes = Self.overlappingPushes(
            pushes,
            start: providerTiming?.startedAtNanoseconds,
            end: providerTiming?.completedAtNanoseconds)
        let resultPhasePushes = Self.overlappingPushes(
            pushes,
            start: resultSyncStart,
            end: resultSyncEnd)
        let toolPhaseStarts = Set(
            (callPhasePushes + providerPhasePushes + resultPhasePushes)
                .map(\.startedAtNanoseconds))
        let toolPhasePushes = pushes.filter {
            toolPhaseStarts.contains($0.startedAtNanoseconds)
        }
        let normalPhasePushes = pushes.filter {
            !toolPhaseStarts.contains($0.startedAtNanoseconds)
        }

        let report = FullDuplexToolBenchmarkReport(
            schemaVersion: 1,
            benchmark: "voicechat_full_duplex_tools",
            generatedAtUTC: ISO8601DateFormatter().string(from: Date()),
            scenario: FullDuplexScenarioReport(
                id: scenario.id,
                schemaVersion: scenario.schemaVersion,
                audioFile: scenario.audioURL.lastPathComponent,
                audioSHA256: scenario.audioSHA256,
                audioDurationSeconds: audioDurationSeconds,
                providerKind: "deterministic_delay",
                replyRequired: scenario.replyRequired),
            runtime: FullDuplexRuntimeReport(
                modelBundle: root.lastPathComponent,
                quantization: Self.quantizationLabel(root: root),
                speechIterations: speechIterations,
                speechContextFrames: retainedSpeechFrames,
                functionCallEndpointFrames:
                    toolTurnTaking.functionCallEndOfUtteranceFrames
                        ?? toolTurnTaking.endOfUtteranceFrames,
                audibilityThresholdDBFS: audibilityThresholdDbfs,
                loadMilliseconds: loadMilliseconds),
            outcome: FullDuplexToolOutcomeReport(
                status: status,
                success: status == "completed",
                expectedTool: scenario.expectedTool,
                actualTool: callInfo.name,
                callPayloadIsValidJSON: callInfo.validJSON,
                nativeCallPayload: callPayload,
                actualArgumentsJSON: callInfo.argumentsJSON,
                argumentsMatch: argumentsMatch,
                argumentMismatches: argumentMismatches,
                replyMatch: replyMatch,
                replyMismatches: replyMismatches,
                audioMatch: audioMatch,
                audioMismatches: audioMismatches,
                referenceTranscript: scenario.transcript,
                recognizedTranscript: recognizedTranscript,
                assistantReply: assistantReply,
                firstTextObserved: firstTextAt != nil,
                firstSustainedAudioObserved: firstSustainedAudioAt != nil,
                replyCompleted: replyCompletedAt != nil),
            latency: FullDuplexLatencyReport(
                requestStartToInputEndMilliseconds:
                    FullDuplexToolClock.milliseconds(
                        from: benchmarkStarted,
                        to: inputEnded),
                inputEndToToolCallStartMilliseconds: callStartAt.map {
                    FullDuplexToolClock.milliseconds(from: inputEnded, to: $0)
                },
                inputEndToToolCallCompleteMilliseconds: callCompletedAt.map {
                    FullDuplexToolClock.milliseconds(from: inputEnded, to: $0)
                },
                inputEndToFirstTextMilliseconds: firstTextAt.map {
                    FullDuplexToolClock.milliseconds(from: inputEnded, to: $0)
                },
                inputEndToFirstAudioFrameReadyMilliseconds:
                    firstAudioFrameReadyAt.map {
                        FullDuplexToolClock.milliseconds(
                            from: inputEnded,
                            to: $0)
                    },
                inputEndToFirstSustainedAudioMilliseconds:
                    firstSustainedAudioAt.map {
                        FullDuplexToolClock.milliseconds(
                            from: inputEnded,
                            to: $0)
                    },
                inputEndToReplyCompleteMilliseconds: replyCompletedAt.map {
                    FullDuplexToolClock.milliseconds(from: inputEnded, to: $0)
                },
                requestStartToFirstSustainedAudioMilliseconds:
                    firstSustainedAudioAt.map {
                        FullDuplexToolClock.milliseconds(
                            from: benchmarkStarted,
                            to: $0)
                    },
                requestStartToReplyCompleteMilliseconds: replyCompletedAt.map {
                    FullDuplexToolClock.milliseconds(
                        from: benchmarkStarted,
                        to: $0)
                },
                audibleOnsetOffsetWithinFrameMilliseconds:
                    audibleOffsetMilliseconds),
            nativeToolDecode: callMetrics.map {
                FullDuplexToolPhaseReport(
                    elapsedMilliseconds: $0.elapsedMilliseconds,
                    tokenSteps: $0.tokenSteps,
                    tokensPerSecond: $0.tokensPerSecond,
                    modelMilliseconds: $0.modelMilliseconds,
                    languageCacheMilliseconds: nil,
                    speechCacheMilliseconds: $0.speechCacheMilliseconds,
                    interleavingMilliseconds: $0.interleavingMilliseconds,
                    prefillBatches: nil)
            },
            provider: FullDuplexProviderReport(
                configuredDelayMilliseconds: configuredProviderDelay,
                serviceMilliseconds: providerTiming?.serviceMilliseconds,
                responseQueueWaitMilliseconds:
                    providerTiming?.responseQueueWaitMilliseconds),
            toolResultSync: responseMetrics.map {
                FullDuplexToolPhaseReport(
                    elapsedMilliseconds: $0.elapsedMilliseconds,
                    tokenSteps: $0.tokenSteps,
                    tokensPerSecond: $0.tokensPerSecond,
                    modelMilliseconds: nil,
                    languageCacheMilliseconds: $0.languageCacheMilliseconds,
                    speechCacheMilliseconds: $0.speechCacheMilliseconds,
                    interleavingMilliseconds: $0.interleavingMilliseconds,
                    prefillBatches: $0.prefillBatches)
            },
            realtime: FullDuplexRealtimeReport(
                frameMilliseconds: VoiceChatSession.frameMilliseconds,
                frameCount: pushes.count,
                foregroundRealTimeFactor:
                    pushes.map(\.elapsedMilliseconds).reduce(0, +)
                        / Double(max(1, pushes.count * VoiceChatSession.frameMilliseconds)),
                wallClockRealTimeFactor:
                    FullDuplexToolClock.milliseconds(
                        from: benchmarkStarted,
                        to: pushes.last?.completedAtNanoseconds
                            ?? benchmarkStarted)
                        / Double(max(
                            1,
                            pushes.count * VoiceChatSession.frameMilliseconds)),
                maximumBehindMilliseconds:
                    pushes.map(\.behindMilliseconds).max() ?? 0,
                deadlineMissesOverFiveMilliseconds:
                    pushes.filter { $0.behindMilliseconds > 5 }.count,
                allFrames: Self.microphonePhaseReport(pushes),
                normalFrames: Self.microphonePhaseReport(normalPhasePushes),
                toolFrames: Self.microphonePhaseReport(toolPhasePushes),
                toolDecodeFrames: Self.microphonePhaseReport(callPhasePushes),
                providerFrames: Self.microphonePhaseReport(providerPhasePushes),
                resultSyncFrames: Self.microphonePhaseReport(
                    resultPhasePushes)),
            audioQuality: FullDuplexAudioQualityReport(
                livePlayback: liveAudioQuality,
                offlineRender: offlineAudioQuality))

        Self.printToolBenchmarkSummary(report)
        if let output {
            let encoder = JSONEncoder()
            encoder.keyEncodingStrategy = .convertToSnakeCase
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(report).write(
                to: URL(fileURLWithPath: output),
                options: .atomic)
            print("wrote \(output)")
        }
        if let responseOutput {
            try WAVWriter.write(
                samples: offlineRenderedAudio,
                sampleRate: VoiceChatSession.outputSampleRate,
                to: URL(fileURLWithPath: responseOutput))
            print("wrote \(responseOutput)")
        }
    }

    private static func sustainedAudioOnsetMilliseconds(
        _ samples: [Float],
        sampleRate: Int,
        thresholdDBFS: Double
    ) -> Double? {
        guard !samples.isEmpty, sampleRate > 0 else { return nil }
        let threshold = pow(10, thresholdDBFS / 20)
        let window = max(1, sampleRate / 100) // 10 ms
        guard samples.count >= window else { return nil }
        for start in stride(from: 0, through: samples.count - window, by: window) {
            let end = start + window
            let energy = samples[start ..< end].reduce(0.0) {
                $0 + Double($1 * $1)
            }
            let rms = sqrt(energy / Double(window))
            if rms >= threshold {
                return Double(start) * 1_000 / Double(sampleRate)
            }
        }
        return nil
    }

    private static func fileSHA256(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let block = try handle.read(upToCount: 1_048_576), !block.isEmpty {
            hasher.update(data: block)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static func parseNativeToolCall(
        _ payload: String?
    ) -> FullDuplexNativeToolCall {
        guard let payload else {
            return FullDuplexNativeToolCall(
                name: nil, arguments: nil, argumentsJSON: nil, validJSON: false)
        }
        let json = payload
            .replacingOccurrences(of: "<TOOLCALL>", with: "")
            .replacingOccurrences(of: "</TOOLCALL>", with: "")
        guard let data = json.data(using: .utf8),
              let calls = try? JSONSerialization.jsonObject(with: data)
                as? [[String: Any]],
              calls.count == 1,
              let call = calls.first,
              let name = call["name"] as? String,
              !name.isEmpty else {
            return FullDuplexNativeToolCall(
                name: nil, arguments: nil, argumentsJSON: nil, validJSON: false)
        }
        let arguments: [String: Any]
        if let object = call["arguments"] as? [String: Any] {
            arguments = object
        } else if let string = call["arguments"] as? String,
                  let argumentData = string.data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: argumentData)
                    as? [String: Any] {
            arguments = object
        } else if call["arguments"] == nil || call["arguments"] is NSNull {
            arguments = [:]
        } else {
            return FullDuplexNativeToolCall(
                name: name, arguments: nil, argumentsJSON: nil, validJSON: false)
        }
        let argumentData = try? JSONSerialization.data(
            withJSONObject: arguments,
            options: [.sortedKeys, .withoutEscapingSlashes])
        return FullDuplexNativeToolCall(
            name: name,
            arguments: arguments,
            argumentsJSON: argumentData.flatMap { String(data: $0, encoding: .utf8) },
            validJSON: argumentData != nil)
    }

    static func benchmarkProviderResponse(
        call: FullDuplexNativeToolCall,
        expectedTool: String?,
        expectedArguments: FullDuplexExpectedArguments?,
        expectedResponseJSON: String?
    ) -> String {
        guard call.validJSON,
              call.name == expectedTool,
              expectedArguments?.mismatches(actual: call.arguments).isEmpty
                ?? true else {
            return #"{"error":"unexpected benchmark tool","ok":false}"#
        }
        return expectedResponseJSON
            ?? #"{"error":"missing benchmark response","ok":false}"#
    }

    static func firstStepScenarioIsComplete(
        replyRequired: Bool,
        callObserved: Bool,
        responseMetricsObserved: Bool
    ) -> Bool {
        !replyRequired && callObserved && responseMetricsObserved
    }

    private static func quantizationLabel(root: URL) -> String {
        let configURL = root.appendingPathComponent("llm/config.json")
        guard let data = try? Data(contentsOf: configURL),
              let config = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any],
              let quantization = config["quantization"] as? [String: Any],
              let bits = (quantization["bits"] as? NSNumber)?.intValue,
              let groupSize = (quantization["group_size"] as? NSNumber)?.intValue
        else { return "unknown" }
        let headBits = (quantization["head_bits"] as? NSNumber)?.intValue
        return headBits.map { "\(bits)-bit g\(groupSize), head \($0)-bit" }
            ?? "\(bits)-bit g\(groupSize)"
    }

    private static func overlappingPushes(
        _ pushes: [FullDuplexMicrophonePush],
        start: UInt64?,
        end: UInt64?
    ) -> [FullDuplexMicrophonePush] {
        guard let start, let end else { return [] }
        return pushes.filter { $0.overlaps(start, end) }
    }

    private static func microphonePhaseReport(
        _ pushes: [FullDuplexMicrophonePush]
    ) -> FullDuplexMicrophonePhaseReport {
        let values = pushes.map(\.elapsedMilliseconds).sorted()
        return FullDuplexMicrophonePhaseReport(
            frameCount: values.count,
            foregroundRealTimeFactor: values.reduce(0, +)
                / Double(max(
                    1,
                    values.count * VoiceChatSession.frameMilliseconds)),
            serviceP50Milliseconds: percentile(values, fraction: 0.50),
            serviceP95Milliseconds: percentile(values, fraction: 0.95),
            serviceMaximumMilliseconds: values.last ?? 0)
    }

    private static func percentile(
        _ sortedValues: [Double],
        fraction: Double
    ) -> Double {
        guard !sortedValues.isEmpty else { return 0 }
        let index = Int((Double(sortedValues.count - 1) * fraction).rounded(.up))
        return sortedValues[min(sortedValues.count - 1, max(0, index))]
    }

    private static func printToolBenchmarkSummary(
        _ report: FullDuplexToolBenchmarkReport
    ) {
        func value(_ milliseconds: Double?) -> String {
            milliseconds.map { String(format: "%.0f ms", $0) } ?? "n/a"
        }
        func realTimeFactor(_ phase: FullDuplexMicrophonePhaseReport) -> String {
            guard phase.frameCount > 0 else { return "n/a" }
            return String(format: "%.2f", phase.foregroundRealTimeFactor)
        }
        print("")
        print("scenario        \(report.scenario.id)")
        print("outcome         \(report.outcome.status)")
        print("transcript      \(String(reflecting: report.outcome.recognizedTranscript))")
        print("tool            \(report.outcome.actualTool ?? "none")")
        print("tool start      \(value(report.latency.inputEndToToolCallStartMilliseconds)) after input end")
        print("tool complete   \(value(report.latency.inputEndToToolCallCompleteMilliseconds)) after input end")
        if let phase = report.nativeToolDecode {
            print(String(
                format: "native decode   %.0f ms, %d steps, %.1f tok/s",
                phase.elapsedMilliseconds,
                phase.tokenSteps ?? 0,
                phase.tokensPerSecond ?? 0))
        }
        print("provider        \(value(report.provider.serviceMilliseconds))")
        if let phase = report.toolResultSync {
            print(String(
                format: "result sync     %.0f ms, %d tokens, %d batches",
                phase.elapsedMilliseconds,
                phase.tokenSteps ?? 0,
                phase.prefillBatches ?? 0))
        }
        print("first text      \(value(report.latency.inputEndToFirstTextMilliseconds)) after input end")
        print("first speech    \(value(report.latency.inputEndToFirstSustainedAudioMilliseconds)) after input end")
        print("reply complete  \(value(report.latency.inputEndToReplyCompleteMilliseconds)) after input end")
        let normalRTF = realTimeFactor(report.realtime.normalFrames)
        let toolRTF = realTimeFactor(report.realtime.toolFrames)
        let averageRTF = String(
            format: "%.2f", report.realtime.foregroundRealTimeFactor)
        let wallRTF = String(
            format: "%.2f", report.realtime.wallClockRealTimeFactor)
        let frameP95 = String(
            format: "%.1f", report.realtime.allFrames.serviceP95Milliseconds)
        let maximumBehind = String(
            format: "%.1f", report.realtime.maximumBehindMilliseconds)
        print(
            "live frames     normal RTF \(normalRTF), tool RTF \(toolRTF), "
                + "avg RTF \(averageRTF), wall RTF \(wallRTF), "
                + "p95 \(frameP95) ms, max behind \(maximumBehind) ms")
        let liveAudio = report.audioQuality.livePlayback
        print(String(
            format: "live audio      %.0f ms active, longest pause %.0f ms, "
                + "clipped %.4f%%, onset pops %d, suspect joins %d/%d",
            liveAudio.activeSpeechMilliseconds,
            liveAudio.maximumInternalPauseMilliseconds,
            liveAudio.clippedSampleFraction * 100,
            liveAudio.suspectOnsetTransientCount,
            liveAudio.suspectFrameBoundaryCount,
            liveAudio.frameBoundaryCount))
        let offlineAudio = report.audioQuality.offlineRender
        print(String(
            format: "offline audio   %.0f ms active, longest pause %.0f ms, "
                + "clipped %.4f%%, onset pops %d",
            offlineAudio.activeSpeechMilliseconds,
            offlineAudio.maximumInternalPauseMilliseconds,
            offlineAudio.clippedSampleFraction * 100,
            offlineAudio.suspectOnsetTransientCount))
        print("assistant       \(String(reflecting: report.outcome.assistantReply))")
    }
}
