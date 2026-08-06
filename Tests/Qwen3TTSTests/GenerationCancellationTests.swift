import Foundation
import XCTest
@testable import Qwen3TTS

final class GenerationCancellationTests: XCTestCase {

    func testCancellationStopsBeforeTheNextTokenStep() {
        var checkpointCount = 0
        var completedSteps: [Int] = []

        XCTAssertThrowsError(
            try runQwen3TTSGenerationLoop(
                0..<10,
                checkCancellation: {
                    checkpointCount += 1
                    if checkpointCount == 4 {
                        throw CancellationError()
                    }
                },
                body: { step in
                    completedSteps.append(step)
                    return true
                })
        ) { error in
            XCTAssertTrue(error is CancellationError)
        }

        XCTAssertEqual(checkpointCount, 4)
        XCTAssertEqual(completedSteps, [0, 1, 2])
    }

    func testNaturalTerminationDoesNotRunAnExtraCheckpoint() {
        var checkpointCount = 0
        var completedSteps: [Int] = []

        runQwen3TTSGenerationLoop(
            0..<10,
            checkCancellation: { checkpointCount += 1 },
            body: { step in
                completedSteps.append(step)
                return step < 2
            })

        XCTAssertEqual(checkpointCount, 3)
        XCTAssertEqual(completedSteps, [0, 1, 2])
    }

    func testSwiftTaskCancellationPropagatesAsCancellationError() async {
        let gate = SuspensionGate()
        let task = Task<Int, Error> {
            await gate.wait()
            var completedSteps = 0
            try runQwen3TTSGenerationLoop(
                0..<10,
                checkCancellation: { try Task.checkCancellation() },
                body: { _ in
                    completedSteps += 1
                    return true
                })
            return completedSteps
        }

        await gate.waitUntilEntered()
        task.cancel()
        await gate.open()

        do {
            _ = try await task.value
            XCTFail("A cancelled generation task must not begin another token step")
        } catch is CancellationError {
            // Expected cooperative cancellation.
        } catch {
            XCTFail("Expected CancellationError, received \(error)")
        }
    }

    func testPublicAsyncGenerateRejectsPrecancelledTaskBeforeTokenizerAccess() async {
        let gate = SuspensionGate()
        let task = Task<[Float], Error> {
            await gate.wait()
            let model = Qwen3TTSModel()
            return try await model.generate(
                text: "Cancellation must win before tokenizer access.",
                language: "english")
        }

        await gate.waitUntilEntered()
        task.cancel()
        await gate.open()

        do {
            _ = try await task.value
            XCTFail("A pre-cancelled async generation must not access the unset tokenizer")
        } catch is CancellationError {
            // Expected before the unset tokenizer can be accessed.
        } catch {
            XCTFail("Expected CancellationError, received \(error)")
        }
    }

    func testAsyncGenerationPathsUseTheCooperativeLoop() throws {
        let sourceDirectory = packageRoot
            .appendingPathComponent("Sources", isDirectory: true)
            .appendingPathComponent("Qwen3TTS", isDirectory: true)
        let modelSource = try String(
            contentsOf: sourceDirectory.appendingPathComponent("Qwen3TTS.swift"),
            encoding: .utf8)
        let protocolSource = try String(
            contentsOf: sourceDirectory.appendingPathComponent("Qwen3TTS+Protocols.swift"),
            encoding: .utf8)

        XCTAssertEqual(
            occurrences(of: "runQwen3TTSGenerationLoop(", in: modelSource),
            2,
            "Single-item and streaming token loops must use the cooperative runner")
        XCTAssertEqual(
            occurrences(of: "for iterIdx in 1..<safeMaxTokens", in: modelSource),
            1,
            "Only the synchronous batch loop may remain outside the async cancellation contract")
        XCTAssertTrue(protocolSource.contains("return try synthesizeCheckingCancellation("))
        XCTAssertTrue(modelSource.contains("continuation.onTermination = { @Sendable _ in task.cancel() }"))
    }

    private func occurrences(of needle: String, in source: String) -> Int {
        source.components(separatedBy: needle).count - 1
    }

    private var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }
}

private actor SuspensionGate {
    private var entered = false
    private var isOpen = false
    private var entryWaiters: [CheckedContinuation<Void, Never>] = []
    private var openWaiters: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        entered = true
        let entryWaiters = self.entryWaiters
        self.entryWaiters.removeAll()
        for waiter in entryWaiters { waiter.resume() }

        guard !isOpen else { return }
        await withCheckedContinuation { continuation in
            openWaiters.append(continuation)
        }
    }

    func waitUntilEntered() async {
        guard !entered else { return }
        await withCheckedContinuation { continuation in
            entryWaiters.append(continuation)
        }
    }

    func open() {
        isOpen = true
        let openWaiters = self.openWaiters
        self.openWaiters.removeAll()
        for waiter in openWaiters { waiter.resume() }
    }
}
