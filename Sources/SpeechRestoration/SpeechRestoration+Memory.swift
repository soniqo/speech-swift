import AudioCommon

extension SpeechRestorer: ModelMemoryManageable {
    public var isLoaded: Bool { _isLoaded }

    public func unload() {
        guard _isLoaded else { return }
        predictor = nil
        vocoder = nil
        _isLoaded = false
    }

    public var memoryFootprint: Int {
        guard _isLoaded else { return 0 }
        // Measured peak RSS (single-artifact process, NOTES.md):
        //   fp16 ≈ 1711 MB, int8-palette ≈ 1321 MB. Report the on-disk weight
        //   size as a rough resident-weight estimate.
        switch variant {
        case .fp16: return 713_000_000
        case .int8: return 407_000_000
        }
    }
}
