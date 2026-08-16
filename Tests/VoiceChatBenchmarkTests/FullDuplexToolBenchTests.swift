import XCTest

@testable import VoiceChatBenchmark

final class FullDuplexToolBenchTests: XCTestCase {
    func testBenchmarkParsesFunctionEndpointOverride() throws {
        let root = try VoiceChatBench.parseAsRoot([
            "--model", "/tmp/voicechat",
            "--function-call-endpoint-frames", "8",
        ])
        let command = try XCTUnwrap(root as? VoiceChatBench)

        XCTAssertEqual(command.functionCallEndpointFrames, 8)
    }

    func testBacklogStartsAtTheNextFrameDeadline() {
        let deadline: UInt64 = 1_000_000_000
        let healthy = FullDuplexMicrophonePush(
            startedAtNanoseconds: deadline,
            completedAtNanoseconds: deadline + 70_000_000,
            deadlineNanoseconds: deadline)
        let late = FullDuplexMicrophonePush(
            startedAtNanoseconds: deadline,
            completedAtNanoseconds: deadline + 105_000_000,
            deadlineNanoseconds: deadline)

        XCTAssertEqual(healthy.elapsedMilliseconds, 70, accuracy: 0.000_001)
        XCTAssertEqual(healthy.behindMilliseconds, 0, accuracy: 0.000_001)
        XCTAssertEqual(late.elapsedMilliseconds, 105, accuracy: 0.000_001)
        XCTAssertEqual(late.behindMilliseconds, 25, accuracy: 0.000_001)
    }

    func testArgumentExpectationsScoreRequiredAbsentAndSemanticValues() throws {
        let expectation = try XCTUnwrap(FullDuplexExpectedArguments.parse([
            "required": ["name", "due_date"],
            "absent": ["list", "priority"],
            "equals": ["completed": true],
            "string_contains_words": ["name": ["call", "dentist"]],
        ]))

        XCTAssertEqual(expectation.mismatches(actual: [
            "name": "Call the dentist",
            "due_date": "2026-08-20 20:00",
            "completed": true,
        ]), [])
        XCTAssertEqual(expectation.mismatches(actual: [
            "name": "Call the doctor",
            "list": "Work",
            "completed": false,
        ]), [
            "missing:due_date",
            "unexpected:list",
            "value:completed",
            "words:name",
        ])
    }

    func testReplyExpectationsAllowNaturalWordingAndRejectMisleadingClaims() throws {
        let expectation = try XCTUnwrap(FullDuplexExpectedReply.parse([
            "contains_all_words": ["phone", "John"],
            "contains_any_words": ["reminder", "reminders"],
            "absent_words": ["cannot"],
        ]))

        XCTAssertEqual(
            expectation.mismatches(
                actual: "You have one reminder named Phone John."),
            [])
        XCTAssertEqual(
            expectation.mismatches(
                actual: "I cannot find any matching item."),
            [
                "missing:john",
                "missing:phone",
                "missing_any:reminder|reminders",
                "unexpected:cannot",
            ])
    }

    func testReplyExpectationsRejectContradictoryConstraints() {
        XCTAssertThrowsError(try FullDuplexExpectedReply.parse([
            "contains_all_words": ["reminder"],
            "absent_words": ["reminder"],
        ]))
    }

    func testReplyExpectationsAllowNoAnyOfConstraint() throws {
        let expectation = try XCTUnwrap(FullDuplexExpectedReply.parse([
            "contains_all_words": ["soniqo"],
        ]))

        XCTAssertEqual(
            expectation.mismatches(actual: "I am Soniqo."),
            [])
    }

    func testNativeToolCallParsesObjectAndStringArguments() {
        let object = VoiceChatBench.parseNativeToolCall(
            #"<TOOLCALL>[{"name":"update_reminder","arguments":{"id":"r1","completed":true}}]</TOOLCALL>"#)
        XCTAssertTrue(object.validJSON)
        XCTAssertEqual(object.name, "update_reminder")
        XCTAssertEqual(object.arguments?["id"] as? String, "r1")
        XCTAssertEqual(object.arguments?["completed"] as? Bool, true)

        let string = VoiceChatBench.parseNativeToolCall(
            #"<TOOLCALL>[{"name":"create_reminder","arguments":"{\"name\":\"Buy milk\"}"}]</TOOLCALL>"#)
        XCTAssertTrue(string.validJSON)
        XCTAssertEqual(string.arguments?["name"] as? String, "Buy milk")
    }

    func testExpectedArgumentsRejectContradictoryConstraints() {
        XCTAssertThrowsError(try FullDuplexExpectedArguments.parse([
            "required": ["name"],
            "absent": ["name"],
        ]))
    }

    func testWrongToolNeverReceivesTheExpectedToolsSuccessResponse() {
        let wrongCall = VoiceChatBench.parseNativeToolCall(
            #"<TOOLCALL>[{"name":"update_reminder","arguments":{"id":"r1","completed":true}}]</TOOLCALL>"#)
        let response = VoiceChatBench.benchmarkProviderResponse(
            call: wrongCall,
            expectedTool: "list_reminders",
            expectedArguments: nil,
            expectedResponseJSON: #"{"ok":true,"tool":"list_reminders"}"#)

        XCTAssertEqual(
            response,
            #"{"error":"unexpected benchmark tool","ok":false}"#)
    }

    func testWrongArgumentsNeverReceiveTheExpectedToolsSuccessResponse() throws {
        let call = VoiceChatBench.parseNativeToolCall(
            #"<TOOLCALL>[{"name":"list_reminders","arguments":{"search":"phone"}}]</TOOLCALL>"#)
        let expectedArguments = try XCTUnwrap(
            FullDuplexExpectedArguments.parse([
                "string_contains_words": ["search": ["phone", "john"]],
            ]))
        let response = VoiceChatBench.benchmarkProviderResponse(
            call: call,
            expectedTool: "list_reminders",
            expectedArguments: expectedArguments,
            expectedResponseJSON: #"{"ok":true,"tool":"list_reminders"}"#)

        XCTAssertEqual(
            response,
            #"{"error":"unexpected benchmark tool","ok":false}"#)
    }

    func testFirstStepOnlyScenarioStopsAfterResultSynchronization() {
        XCTAssertFalse(VoiceChatBench.firstStepScenarioIsComplete(
            replyRequired: false,
            callObserved: true,
            responseMetricsObserved: false))
        XCTAssertFalse(VoiceChatBench.firstStepScenarioIsComplete(
            replyRequired: true,
            callObserved: true,
            responseMetricsObserved: true))
        XCTAssertTrue(VoiceChatBench.firstStepScenarioIsComplete(
            replyRequired: false,
            callObserved: true,
            responseMetricsObserved: true))
    }

    func testAcousticAnalysisFindsInternalPauseWithoutInventingClicks() {
        let sampleRate = 1_000
        var firstSpeech = [Float](repeating: 0.1, count: 200)
        firstSpeech[firstSpeech.count - 1] = 0
        var secondSpeech = [Float](repeating: 0.1, count: 200)
        secondSpeech[0] = 0
        let samples = firstSpeech + [Float](repeating: 0, count: 300)
            + secondSpeech

        let analysis = FullDuplexAcousticAnalyzer.analyze(
            samples: samples,
            sampleRate: sampleRate,
            frameBoundaries: [100, 200, 300, 400, 500, 600],
            thresholdDBFS: -40)

        XCTAssertEqual(analysis.activeSpeechMilliseconds, 400, accuracy: 0.001)
        XCTAssertEqual(analysis.speechSpanMilliseconds, 700, accuracy: 0.001)
        XCTAssertEqual(analysis.internalPauseCount, 1)
        XCTAssertEqual(
            analysis.maximumInternalPauseMilliseconds, 300, accuracy: 0.001)
        XCTAssertEqual(analysis.clippedSampleCount, 0)
        XCTAssertEqual(analysis.nonFiniteSampleCount, 0)
        XCTAssertEqual(analysis.suspectOnsetTransientCount, 0)
        XCTAssertEqual(analysis.suspectFrameBoundaryCount, 0)
    }

    func testAcousticAnalysisFlagsAnIsolatedOnsetTransient() {
        let sampleRate = 1_000
        var samples = [Float](repeating: 0, count: 100)
        samples[90] = 0.9
        samples += (0 ..< 300).map {
            Float(0.1 * sin(Double($0) * 2 * .pi / 100))
        }

        let analysis = FullDuplexAcousticAnalyzer.analyze(
            samples: samples,
            sampleRate: sampleRate,
            frameBoundaries: [100, 200, 300],
            thresholdDBFS: -40)

        XCTAssertEqual(analysis.suspectOnsetTransientCount, 2)
        XCTAssertGreaterThan(analysis.maximumOnsetJump, 0.8)
        XCTAssertGreaterThan(analysis.maximumOnsetToSteadyP99Ratio, 6)
    }

    func testAcousticAnalysisFlagsAnIsolatedLiveFrameDiscontinuity() {
        let sampleRate = 1_000
        var samples = (0 ..< 400).map {
            Float(0.1 * sin(Double($0) * 2 * .pi / 100))
        }
        samples[200] = 0.9

        let analysis = FullDuplexAcousticAnalyzer.analyze(
            samples: samples,
            sampleRate: sampleRate,
            frameBoundaries: [100, 200, 300],
            thresholdDBFS: -40)

        XCTAssertEqual(analysis.frameBoundaryCount, 3)
        XCTAssertEqual(analysis.suspectFrameBoundaryCount, 1)
        XCTAssertGreaterThan(analysis.maximumFrameBoundaryJump, 0.8)
        XCTAssertEqual(analysis.clippedSampleCount, 0)
    }

    func testExpectedAudioScoresClippingPausesAndBoundaryArtifacts() throws {
        let expectation = try XCTUnwrap(FullDuplexExpectedAudio.parse([
            "minimum_active_speech_ms": 500,
            "maximum_internal_pause_ms": 200,
            "maximum_clipped_sample_fraction": 0,
            "maximum_suspect_onset_transients": 0,
            "maximum_suspect_frame_boundaries": 0,
        ]))
        let analysis = FullDuplexAcousticAnalysis(
            sampleCount: 1_000,
            durationMilliseconds: 1_000,
            nonFiniteSampleCount: 1,
            peakAbsoluteAmplitude: 1,
            clippedSampleCount: 2,
            clippedSampleFraction: 0.002,
            activeSpeechMilliseconds: 400,
            speechSpanMilliseconds: 900,
            internalPauseCount: 1,
            maximumInternalPauseMilliseconds: 300,
            totalInternalPauseMilliseconds: 300,
            onsetAnalysisSpanMilliseconds: 50,
            maximumOnsetJump: 0.8,
            p99SteadySpeechJump: 0.02,
            maximumOnsetToSteadyP99Ratio: 40,
            suspectOnsetTransientCount: 1,
            frameBoundaryCount: 10,
            maximumFrameBoundaryJump: 0.8,
            p95FrameBoundaryJump: 0.2,
            p99WithinFrameJump: 0.02,
            maximumBoundaryToWithinP99Ratio: 40,
            suspectFrameBoundaryCount: 1)

        XCTAssertEqual(expectation.mismatches(actual: analysis), [
            "active_speech_too_short",
            "internal_pause_too_long",
            "excessive_clipping",
            "onset_transient",
            "frame_boundary_discontinuity",
            "non_finite_audio",
        ])
    }

    func testExpectedAudioRejectsInvalidThresholds() {
        XCTAssertThrowsError(try FullDuplexExpectedAudio.parse([
            "maximum_clipped_sample_fraction": 1.1,
        ]))
        XCTAssertThrowsError(try FullDuplexExpectedAudio.parse([
            "maximum_suspect_frame_boundaries": -1,
        ]))
        XCTAssertThrowsError(try FullDuplexExpectedAudio.parse([
            "maximum_suspect_onset_transients": 1.5,
        ]))
    }
}
