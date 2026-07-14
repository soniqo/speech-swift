#if canImport(CoreML)
import AudioCommon
import CoreML
import XCTest
@testable import SpeechVAD

final class Community1DiarizationTests: XCTestCase {
    func testPublishedDefaults() {
        let config = Community1Config.default
        XCTAssertEqual(Community1Config.sampleRate, 16_000)
        XCTAssertEqual(Community1Config.chunkSamples, 160_000)
        XCTAssertEqual(Community1Config.framesPerChunk, 589)
        XCTAssertEqual(config.clusteringThreshold, 0.6, accuracy: 1e-7)
        XCTAssertEqual(config.fa, 0.07, accuracy: 1e-7)
        XCTAssertEqual(config.fb, 0.8, accuracy: 1e-7)
        XCTAssertEqual(config.minimumActiveRatio, 0.2, accuracy: 1e-7)
    }

    func testOfficialChunkStartsIncludePaddedOrphan() {
        XCTAssertEqual(
            Community1DiarizationPipeline.chunkStarts(sampleCount: 80_000),
            [0]
        )
        XCTAssertEqual(
            Community1DiarizationPipeline.chunkStarts(sampleCount: 160_000),
            [0]
        )
        XCTAssertEqual(
            Community1DiarizationPipeline.chunkStarts(sampleCount: 168_000),
            [0, 16_000]
        )
        XCTAssertEqual(
            Community1DiarizationPipeline.chunkStarts(sampleCount: 176_000),
            [0, 16_000]
        )
    }

    func testSpeakerCountUsesOverlapAverageAndNearestEvenRounding() {
        var chunk = Array(
            repeating: [Float](repeating: 0, count: 3),
            count: Community1Config.framesPerChunk
        )
        chunk[0][0] = 1
        chunk[1][0] = 1
        chunk[1][1] = 1
        let count = Community1DiarizationPipeline.speakerCount([chunk])
        XCTAssertEqual(count.count, 594)
        XCTAssertEqual(count[0], 1)
        XCTAssertEqual(count[1], 2)
        XCTAssertEqual(count[588], 0)
        XCTAssertEqual(count[593], 0)
    }

    func testConstrainedAssignmentUsesEachClusterAtMostOnce() {
        let scores = [
            [0.9, 0.8],
            [0.85, 0.1],
            [0.2, 0.7],
        ]
        let assignment = Community1Clustering.constrainedArgmax(scores)
        XCTAssertEqual(assignment, [1, 0, -2])
    }

    func testCentroidLinkageCut() {
        let embeddings: [[Float]] = [
            [1, 0], [0.98, 0.02],
            [-1, 0], [-0.98, 0.02],
        ]
        XCTAssertEqual(
            Community1Clustering.centroidLinkageLabels(embeddings, threshold: 0.2),
            [0, 0, 1, 1]
        )
    }

    func testCentroidLinkageScalesToRollingWindowEmbeddingCounts() {
        let embeddings: [[Float]] = (0..<300).map { sample in
            var embedding = [Float](
                repeating: 0,
                count: Community1Config.embeddingDimension
            )
            embedding[sample % 3] = 1
            for dimension in 3..<embedding.count {
                embedding[dimension] = Float((sample * 17 + dimension * 13) % 11) * 0.0001
            }
            return embedding
        }

        let start = Date()
        let labels = Community1Clustering.centroidLinkageLabels(
            embeddings,
            threshold: 0.1
        )

        XCTAssertEqual(Set(labels).count, 3)
        XCTAssertLessThan(Date().timeIntervalSince(start), 5.0)
    }

    func testVBxMatchesReferenceFixture() {
        let labels = [0, 0, 1, 1]
        let features = [
            [1.0, 0.2], [0.9, 0.1], [-0.8, 0.1], [-1.0, 0.3],
        ]
        let result = Community1Clustering.vbx(
            initialLabels: labels,
            features: features,
            phi: [2.0, 0.5],
            config: .default
        )
        let expectedPriors = [0.500150422801, 0.499849577199]
        XCTAssertEqual(result.priors[0], expectedPriors[0], accuracy: 1e-9)
        XCTAssertEqual(result.priors[1], expectedPriors[1], accuracy: 1e-9)
        XCTAssertEqual(result.responsibilities[0][0], 0.500152474379, accuracy: 1e-9)
        XCTAssertEqual(result.responsibilities[3][1], 0.499851716677, accuracy: 1e-9)
    }

    func testTimelineUsesReceptiveFieldCenters() {
        var binary = Array(repeating: [Float](repeating: 0, count: 1), count: 5)
        binary[1][0] = 1
        binary[2][0] = 1
        let segments = Community1DiarizationPipeline.toSegments(binary)
        XCTAssertEqual(segments.count, 1)
        XCTAssertEqual(
            segments[0].startTime,
            Community1Config.frameDuration / 2 + Community1Config.frameStep,
            accuracy: 1e-7
        )
        XCTAssertEqual(
            segments[0].endTime,
            Community1Config.frameDuration / 2 + 3 * Community1Config.frameStep,
            accuracy: 1e-7
        )
    }
}

final class E2ECommunity1DiarizationTests: XCTestCase {
    func testPublishedBundleOnVoxConverseProbe() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let bundlePath = environment["COMMUNITY1_COREML_BUNDLE"],
              let audioPath = environment["COMMUNITY1_TEST_AUDIO"],
              let referencePath = environment["COMMUNITY1_TEST_RTTM"] else {
            throw XCTSkip(
                "set COMMUNITY1_COREML_BUNDLE, COMMUNITY1_TEST_AUDIO, and COMMUNITY1_TEST_RTTM"
            )
        }
        let bundle = URL(fileURLWithPath: bundlePath)
        let audioURL = URL(fileURLWithPath: audioPath)
        let referenceURL = URL(fileURLWithPath: referencePath)
        guard FileManager.default.fileExists(atPath: bundle.path),
              FileManager.default.fileExists(atPath: audioURL.path),
              FileManager.default.fileExists(atPath: referenceURL.path) else {
            throw XCTSkip("set COMMUNITY1_COREML_BUNDLE, COMMUNITY1_TEST_AUDIO, and COMMUNITY1_TEST_RTTM")
        }

        let pipeline = try Community1DiarizationPipeline.fromLocal(
            directory: bundle,
            computeUnits: .cpuAndNeuralEngine
        )
        let audio = try AudioFileLoader.load(url: audioURL, targetSampleRate: 16_000)
        let result = try pipeline.diarize(audio: audio, sampleRate: 16_000)
        let reference = parseRTTM(try String(contentsOf: referenceURL, encoding: .utf8))
        let score = computeDERWithOptimalMapping(
            reference: reference,
            hypothesis: result.segments,
            collar: 0.25,
            resolution: 0.01
        )

        XCTAssertEqual(result.numSpeakers, 3)
        XCTAssertLessThan(score.derPercent, 8.0)
    }
}
#endif
