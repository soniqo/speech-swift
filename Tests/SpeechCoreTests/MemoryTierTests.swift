import XCTest
@testable import SpeechCore

final class MemoryTierTests: XCTestCase {

    func testDetectReturnsValidTier() {
        let tier = MemoryTier.detect()
        XCTAssertNotNil(tier)
        // On macOS test machine with 8GB+, should be .full
        #if os(macOS)
        let totalGB = Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024 * 1024)
        if totalGB >= 8 {
            XCTAssertEqual(tier, .full)
        }
        #endif
    }

    func testTierProperties() {
        // Full tier: everything enabled, no auto-unload
        XCTAssertTrue(MemoryTier.full.useCoreMLASR)
        XCTAssertTrue(MemoryTier.full.useCoreMLTTS)
        XCTAssertTrue(MemoryTier.full.useLLM)
        XCTAssertFalse(MemoryTier.full.autoUnload)

        // Standard: CoreML + auto-unload
        XCTAssertTrue(MemoryTier.standard.useCoreMLASR)
        XCTAssertTrue(MemoryTier.standard.useCoreMLTTS)
        XCTAssertTrue(MemoryTier.standard.useLLM)
        XCTAssertTrue(MemoryTier.standard.autoUnload)

        // Constrained: Apple Speech, CoreML TTS, LLM, auto-unload
        XCTAssertFalse(MemoryTier.constrained.useCoreMLASR)
        XCTAssertTrue(MemoryTier.constrained.useCoreMLTTS)
        XCTAssertTrue(MemoryTier.constrained.useLLM)
        XCTAssertTrue(MemoryTier.constrained.autoUnload)

        // Minimal: only system TTS, no LLM
        XCTAssertFalse(MemoryTier.minimal.useCoreMLASR)
        XCTAssertFalse(MemoryTier.minimal.useCoreMLTTS)
        XCTAssertFalse(MemoryTier.minimal.useLLM)
        XCTAssertTrue(MemoryTier.minimal.autoUnload)
    }

    func testDescriptionNotEmpty() {
        for tier in [MemoryTier.full, .standard, .constrained, .minimal] {
            XCTAssertFalse(tier.description.isEmpty)
        }
    }
}
