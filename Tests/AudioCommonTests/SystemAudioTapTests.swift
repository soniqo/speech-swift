#if os(macOS)
import CoreAudio
import XCTest
@testable import AudioCommon

final class SystemAudioTapTests: XCTestCase {

    // MARK: Exclusion list

    func testOwnProcessExcludedByDefault() {
        let tap = SystemAudioTap()
        XCTAssertEqual(tap.effectiveExcludedPIDs(), [getpid()])
    }

    func testExplicitPIDsPrecedeOwnProcessAndDeduplicate() {
        let tap = SystemAudioTap(excludeCurrentProcess: true, excludedPIDs: [42, 42, getpid(), 7])
        XCTAssertEqual(tap.effectiveExcludedPIDs(), [42, getpid(), 7])
    }

    func testOptOutOfOwnProcessExclusion() {
        let tap = SystemAudioTap(excludeCurrentProcess: false, excludedPIDs: [])
        XCTAssertEqual(tap.effectiveExcludedPIDs(), [])
    }

    // MARK: Aggregate composition

    func testAggregateContainsOnlyTheTap() {
        let uuid = UUID()
        let composition = SystemAudioTap.aggregateComposition(
            tapUUID: uuid, aggregateUID: "agg-uid")

        // The whole point of the tap-only aggregate: no sub-device list, so the
        // output device's own input streams can never leak into the capture.
        XCTAssertNil(composition[kAudioAggregateDeviceSubDeviceListKey])
        XCTAssertNil(composition[kAudioAggregateDeviceMainSubDeviceKey])

        XCTAssertEqual(composition[kAudioAggregateDeviceUIDKey] as? String, "agg-uid")
        XCTAssertEqual(composition[kAudioAggregateDeviceIsPrivateKey] as? Bool, true)
        XCTAssertEqual(composition[kAudioAggregateDeviceIsStackedKey] as? Bool, false)
        XCTAssertEqual(composition[kAudioAggregateDeviceTapAutoStartKey] as? Bool, true)

        let tapList = composition[kAudioAggregateDeviceTapListKey] as? [[String: Any]]
        XCTAssertEqual(tapList?.count, 1)
        XCTAssertEqual(tapList?.first?[kAudioSubTapUIDKey] as? String, uuid.uuidString)
        XCTAssertEqual(tapList?.first?[kAudioSubTapDriftCompensationKey] as? Bool, true)
    }

    // MARK: Mono mixdown

    func testInterleavedStereoMixdownAveragesFrames() {
        let mono = SystemAudioTap.monoMixdown(interleaved: [1, 3, -2, 2, 0.5, 0.5], channels: 2)
        XCTAssertEqual(mono, [2, 0, 0.5])
    }

    func testInterleavedMonoPassesThrough() {
        let samples: [Float] = [0.1, -0.2, 0.3]
        XCTAssertEqual(SystemAudioTap.monoMixdown(interleaved: samples, channels: 1), samples)
    }

    func testInterleavedDropsTrailingPartialFrame() {
        let mono = SystemAudioTap.monoMixdown(interleaved: [1, 1, 1], channels: 2)
        XCTAssertEqual(mono, [1])
    }

    func testPlanarMixdownAveragesChannels() {
        let mono = SystemAudioTap.monoMixdown(planar: [[1, 2], [3, 4]])
        XCTAssertEqual(mono, [2, 3])
    }

    func testPlanarMixdownToleratesLengthMismatch() {
        let mono = SystemAudioTap.monoMixdown(planar: [[2, 2, 2], [4]])
        XCTAssertEqual(mono, [3, 1, 1])
    }

    func testPlanarSingleChannelPassesThrough() {
        XCTAssertEqual(SystemAudioTap.monoMixdown(planar: [[5, 6]]), [5, 6])
    }

    func testEmptyInputsProduceEmptyMono() {
        XCTAssertEqual(SystemAudioTap.monoMixdown(interleaved: [], channels: 2), [])
        XCTAssertEqual(SystemAudioTap.monoMixdown(planar: []), [])
    }

    // MARK: Silence accounting

    func testNonSilentCountUsesThreshold() {
        let samples: [Float] = [0, 1e-7, -1e-7, 0.01, -0.5, 0]
        XCTAssertEqual(SystemAudioTap.nonSilentCount(samples, threshold: 1e-6), 2)
        XCTAssertEqual(SystemAudioTap.nonSilentCount([], threshold: 1e-6), 0)
    }

    // MARK: Capture timestamps

    func testValidHostTimeIsPreserved() {
        var timestamp = AudioTimeStamp()
        timestamp.mHostTime = 42
        timestamp.mFlags = .hostTimeValid
        XCTAssertEqual(SystemAudioTap.hostTime(from: timestamp), 42)
    }

    func testInvalidHostTimeIsNil() {
        var timestamp = AudioTimeStamp()
        timestamp.mHostTime = 42
        timestamp.mFlags = []
        XCTAssertNil(SystemAudioTap.hostTime(from: timestamp))
    }

    // MARK: OSStatus rendering

    func testDescribeOSStatusRendersFourCharCode() {
        // kAudioHardwareIllegalOperationError = 'nope' — the classic status when
        // tap creation is rejected.
        XCTAssertEqual(SystemAudioTap.describeOSStatus(1_852_797_029), "'nope' (1852797029)")
    }

    func testDescribeOSStatusFallsBackToDecimal() {
        XCTAssertEqual(SystemAudioTap.describeOSStatus(-50), "-50")
        XCTAssertEqual(SystemAudioTap.describeOSStatus(0), "0")
    }

    // MARK: Lifecycle guards (no hardware touched)

    func testStopWithoutStartIsIdempotent() {
        let tap = SystemAudioTap()
        tap.stop()
        tap.stop()
        if case .stopped = tap.captureState {} else {
            XCTFail("expected stopped state, got \(tap.captureState)")
        }
        XCTAssertEqual(tap.framesCaptured, 0)
        XCTAssertEqual(tap.nonSilentFrames, 0)
        XCTAssertEqual(tap.tapSampleRate, 0)
    }
}
#endif
